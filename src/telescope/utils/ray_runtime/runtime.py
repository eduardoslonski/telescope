"""Ray-native runtime for orchestrating inference and trainer workers."""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, TextIO

import ray
import requests
import torch.distributed as dist
from ray.util import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from telescope.utils import config, paths
from telescope.utils.tlog import get_logger, setup_logging
from telescope.trainer.backends import create_backend
from telescope.trainer.metrics import GPUTimelineLogger, create_timeline_tracker
from telescope.trainer.metrics.torch_memory_logger import TorchMemoryLogger
from telescope.trainer.weight_sync import (
    broadcast_weights_to_inference,
    prepare_weights_for_broadcast,
    send_weights_to_inference,
    setup_inference_communicator,
    setup_inference_communicator_for_group,
)

_log = get_logger("orchestrator")
_inference_log = get_logger("inference")
_trainer_log = get_logger("trainer")

_VALID_PLACEMENT_STRATEGIES = {"PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"}
_SETUP_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "NCCL_SOCKET_IFNAME",
    "NCCL_DEBUG",
    "VIRTUAL_ENV",
    "UV_PROJECT_ENVIRONMENT",
    "TELESCOPE_RUN_DIR",
    "TELESCOPE_CHECKPOINT_DIR",
    "PYTHONPATH",
)

# Environment keys captured per GPU-worker (inference & trainer actors).
_GPU_ENV_KEYS = (
    # CUDA / device
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_MODULE_LOADING",
    "TORCH_CUDA_ARCH_LIST",
    # NCCL / networking
    "NCCL_SOCKET_IFNAME",
    "NCCL_DEBUG",
    "NCCL_P2P_DISABLE",
    "NCCL_P2P_LEVEL",
    "NCCL_IB_DISABLE",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_SHM_DISABLE",
    "NCCL_BUFFSIZE",
    "NCCL_NTHREADS",
    "NCCL_NSOCKS_PERTHREAD",
    "NCCL_SOCKET_NTHREADS",
    "NCCL_CROSS_NIC",
    "NCCL_ALGO",
    "NCCL_PROTO",
    # Distributed training
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    # Misc
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
)


def _collect_visible_gpu_hardware() -> list[dict[str, Any]]:
    """Collect hardware details for every CUDA-visible GPU on this process.

    Uses ``nvidia-smi`` as the **primary** data source (always available
    on GPU nodes, no Python CUDA init needed) and ``torch.cuda`` as
    **enrichment** (compute-capability, SM count, exact byte-level memory,
    CUDA runtime version).

    Returns one dict per visible GPU, ordered by local CUDA index.
    """
    _NA = ("[N/A]", "N/A", "[Not Supported]", "")

    # ── 1. nvidia-smi: primary source ───────────────────────────────────
    smi_fields = (
        "index", "name", "uuid", "serial", "pci.bus_id",
        "pcie.link.gen.current", "pcie.link.gen.max",
        "pcie.link.width.current", "pcie.link.width.max",
        "memory.total", "memory.free", "memory.used",
        "power.draw", "power.limit", "power.max_limit", "power.default_limit",
        "clocks.current.graphics", "clocks.current.memory",
        "clocks.max.graphics", "clocks.max.memory",
        "temperature.gpu", "temperature.memory",
        "fan.speed",
        "compute_mode", "persistence_mode",
        "mig.mode.current",
        "ecc.mode.current",
        "ecc.errors.corrected.volatile.total",
        "ecc.errors.uncorrected.volatile.total",
        "vbios_version", "driver_version",
    )
    smi_rows: dict[int, dict[str, str]] = {}
    try:
        smi_result = subprocess.run(
            ["nvidia-smi",
             f"--query-gpu={','.join(smi_fields)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if smi_result.returncode == 0:
            for line in smi_result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < len(smi_fields):
                    parts.extend([""] * (len(smi_fields) - len(parts)))
                row = dict(zip(smi_fields, parts))
                idx_str = row.get("index", "")
                if idx_str.isdigit():
                    smi_rows[int(idx_str)] = row
    except Exception:
        pass

    # Determine which physical GPUs are visible to this process.
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        # Numeric indices (most common).  UUID-based values are left to
        # the torch fallback below.
        parts = [s.strip() for s in cuda_visible.split(",") if s.strip()]
        if all(p.isdigit() for p in parts):
            phys_indices = [int(p) for p in parts]
        else:
            phys_indices = sorted(smi_rows.keys())
    else:
        phys_indices = sorted(smi_rows.keys())

    # Build one entry per visible GPU from nvidia-smi.
    gpus: list[dict[str, Any]] = []
    for local_idx, phys_idx in enumerate(phys_indices):
        row = smi_rows.get(phys_idx, {})
        gpu: dict[str, Any] = {
            "local_index": local_idx,
            "physical_index": phys_idx,
        }

        if row:
            # ── identification ──────────────────────────────────────────
            for smi_key, out_key in (
                ("name", "name"),
                ("uuid", "uuid"),
                ("pci.bus_id", "pci_bus_id"),
                ("vbios_version", "vbios"),
                ("driver_version", "driver_version"),
            ):
                val = row.get(smi_key, "")
                if val and val not in _NA:
                    gpu[out_key] = val
            serial = row.get("serial", "")
            if serial and serial not in _NA:
                gpu["serial"] = serial

            # ── memory (MiB from nvidia-smi) ────────────────────────────
            for smi_key, out_key in (
                ("memory.total", "memory_total_mib"),
                ("memory.free", "memory_free_mib"),
                ("memory.used", "memory_used_mib"),
            ):
                try:
                    gpu[out_key] = int(float(row.get(smi_key, "")))
                except (ValueError, TypeError):
                    pass

            # ── PCIe ────────────────────────────────────────────────────
            for smi_key, out_key in (
                ("pcie.link.gen.current", "pcie_gen"),
                ("pcie.link.gen.max", "pcie_gen_max"),
                ("pcie.link.width.current", "pcie_width"),
                ("pcie.link.width.max", "pcie_width_max"),
            ):
                val = row.get(smi_key, "")
                if val.isdigit():
                    gpu[out_key] = int(val)

            # ── power (watts) ───────────────────────────────────────────
            for smi_key, out_key in (
                ("power.draw", "power_draw_w"),
                ("power.limit", "power_limit_w"),
                ("power.max_limit", "power_max_w"),
                ("power.default_limit", "power_default_w"),
            ):
                try:
                    gpu[out_key] = round(float(row[smi_key]), 1)
                except (KeyError, ValueError):
                    pass

            # ── clocks (MHz) ────────────────────────────────────────────
            for smi_key, out_key in (
                ("clocks.current.graphics", "clock_graphics_mhz"),
                ("clocks.current.memory", "clock_memory_mhz"),
                ("clocks.max.graphics", "clock_max_graphics_mhz"),
                ("clocks.max.memory", "clock_max_memory_mhz"),
            ):
                val = row.get(smi_key, "")
                if val.isdigit():
                    gpu[out_key] = int(val)

            # ── thermals & fan ──────────────────────────────────────────
            for smi_key, out_key in (
                ("temperature.gpu", "temperature_gpu_c"),
                ("temperature.memory", "temperature_memory_c"),
            ):
                val = row.get(smi_key, "")
                if val.isdigit():
                    gpu[out_key] = int(val)
            fan = row.get("fan.speed", "")
            if fan and fan not in _NA:
                try:
                    gpu["fan_speed_pct"] = int(fan)
                except ValueError:
                    pass

            # ── modes ───────────────────────────────────────────────────
            for smi_key, out_key in (
                ("compute_mode", "compute_mode"),
                ("persistence_mode", "persistence_mode"),
                ("mig.mode.current", "mig_mode"),
            ):
                val = row.get(smi_key, "")
                if val and val not in _NA:
                    gpu[out_key] = val

            # ── ECC ─────────────────────────────────────────────────────
            ecc_mode = row.get("ecc.mode.current", "")
            if ecc_mode and ecc_mode not in _NA:
                gpu["ecc_mode"] = ecc_mode
            for smi_key, out_key in (
                ("ecc.errors.corrected.volatile.total", "ecc_errors_corrected"),
                ("ecc.errors.uncorrected.volatile.total", "ecc_errors_uncorrected"),
            ):
                val = row.get(smi_key, "")
                if val.isdigit():
                    gpu[out_key] = int(val)

        gpus.append(gpu)

    # ── 2. torch.cuda: enrichment ───────────────────────────────────────
    try:
        import torch

        if torch.cuda.is_available():
            num_torch_gpus = torch.cuda.device_count()
            for idx in range(min(num_torch_gpus, len(gpus))):
                props = torch.cuda.get_device_properties(idx)
                gpus[idx]["memory_total_bytes"] = props.total_mem
                gpus[idx]["memory_gb"] = round(props.total_mem / (1024**3), 2)
                gpus[idx]["compute_capability"] = f"{props.major}.{props.minor}"
                gpus[idx]["multi_processor_count"] = props.multi_processor_count
                gpus[idx].setdefault("name", props.name)
            # CUDA runtime & torch versions.
            cuda_ver = getattr(torch.version, "cuda", None)
            torch_ver = getattr(torch, "__version__", None)
            for gpu in gpus:
                if cuda_ver:
                    gpu["cuda_runtime_version"] = str(cuda_ver)
                if torch_ver:
                    gpu["torch_version"] = str(torch_ver)
    except Exception:
        pass

    # If nvidia-smi failed and torch also returned nothing, try a minimal
    # torch-only fallback so we at least get device names & memory.
    if not gpus:
        try:
            import torch

            if torch.cuda.is_available():
                for idx in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(idx)
                    gpus.append({
                        "local_index": idx,
                        "name": props.name,
                        "memory_total_bytes": props.total_mem,
                        "memory_gb": round(props.total_mem / (1024**3), 2),
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multi_processor_count": props.multi_processor_count,
                    })
        except Exception:
            pass

    return gpus


def _worker_env_snapshot() -> dict[str, str]:
    """Capture environment variables relevant to GPU workers."""
    snapshot: dict[str, str] = {}
    for key in _GPU_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            snapshot[key] = value
    return snapshot


def _parse_cuda_visible_devices(value: str) -> list[int]:
    """Parse ``CUDA_VISIBLE_DEVICES`` into a list of physical GPU indices."""
    if not value:
        return []
    parts = [s.strip() for s in value.split(",") if s.strip()]
    indices: list[int] = []
    for p in parts:
        if p.isdigit():
            indices.append(int(p))
    return indices


def _validate_placement_strategy(name: str, strategy: str) -> str:
    strategy = strategy.upper().strip()
    if strategy not in _VALID_PLACEMENT_STRATEGIES:
        valid = ", ".join(sorted(_VALID_PLACEMENT_STRATEGIES))
        raise ValueError(f"{name} must be one of: {valid}. Got: {strategy!r}")
    return strategy


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _resolve_cluster_node_index(node_ip: str, ray_node_id: str) -> int:
    """Resolve a stable node index (0..N-1), ordered by node_ip then ray node id."""
    try:
        alive_nodes = [node for node in ray.nodes() if bool(node.get("Alive"))]
    except Exception:
        return -1

    ordered_nodes = sorted(
        {
            (
                str(node.get("NodeManagerAddress") or ""),
                str(node.get("NodeID") or ""),
            )
            for node in alive_nodes
        },
        key=lambda item: (item[0], item[1]),
    )

    ray_node_id_str = str(ray_node_id or "")
    if ray_node_id_str:
        for idx, (_, candidate_ray_node_id) in enumerate(ordered_nodes):
            if candidate_ray_node_id == ray_node_id_str:
                return idx

    node_ip_str = str(node_ip or "")
    if node_ip_str:
        for idx, (candidate_node_ip, _) in enumerate(ordered_nodes):
            if candidate_node_ip == node_ip_str:
                return idx

    if len(ordered_nodes) == 1:
        return 0
    return -1


def _build_stdout_filename(worker_type: str, node_id: int, rank: int) -> str:
    node_part = str(node_id) if int(node_id) >= 0 else "unknown"
    rank_part = str(rank) if int(rank) >= 0 else "unknown"
    return f"{worker_type}_node_{node_part}_rank_{rank_part}.stdout"


class _TeeStream:
    """Mirror stream writes to both an original stream and a file stream."""

    def __init__(self, primary: TextIO, mirror: TextIO):
        self._primary = primary
        self._mirror = mirror

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)

    def write(self, data: str) -> int:
        text = data if isinstance(data, str) else str(data)
        primary_written = self._primary.write(text)
        self._mirror.write(text)
        return primary_written if isinstance(primary_written, int) else len(text)

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()


def _redirect_telescope_console_stream(stream: TextIO) -> None:
    """
    Point Telescope console handlers to the supplied stream.

    Keep file handlers untouched and only retarget console stream handlers.
    """
    root_logger = logging.getLogger("telescope")
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setStream(stream)


def _build_default_ray_runtime_env(user_runtime_env: Any) -> dict[str, Any]:
    """
    Build a Ray runtime_env that reuses the current Python env for workers.

    This avoids per-worker virtualenv recreation/download storms when running
    under uv and keeps worker interpreter resolution aligned with the driver.
    """
    runtime_env = dict(user_runtime_env) if isinstance(user_runtime_env, dict) else {}

    # Exclude large files from the working_dir package to avoid hitting
    # Ray's 512MB upload limit (checkpoints, .git, caches, etc.)
    default_excludes = [
        "checkpoints*",
        "converted*",
        "*.safetensors",
        "*.pt",
        "*.bin",
        ".git",
        "__pycache__",
        "*.pyc",
        "logs",
        "wandb",
    ]
    existing = list(runtime_env.get("excludes") or [])
    runtime_env["excludes"] = existing + [e for e in default_excludes if e not in existing]

    if bool(config.cfg.ray_pin_py_executable):
        runtime_env.setdefault("py_executable", sys.executable)

    env_vars = dict(runtime_env.get("env_vars") or {})
    if bool(config.cfg.ray_propagate_active_venv):
        active_venv = os.environ.get("VIRTUAL_ENV")
        if active_venv:
            env_vars.setdefault("VIRTUAL_ENV", active_venv)
            env_vars.setdefault("UV_PROJECT_ENVIRONMENT", active_venv)
    if bool(config.cfg.ray_propagate_run_dir):
        run_dir = os.environ.get("TELESCOPE_RUN_DIR") or os.getcwd()
        env_vars.setdefault("TELESCOPE_RUN_DIR", run_dir)
    ckpt_dir = os.environ.get("TELESCOPE_CHECKPOINT_DIR")
    if ckpt_dir:
        env_vars.setdefault("TELESCOPE_CHECKPOINT_DIR", ckpt_dir)
    runtime_env["env_vars"] = env_vars

    return runtime_env


def resolve_worker_count(explicit_count: int | None, role: str) -> int:
    """Resolve worker count for a role from explicit config."""
    if explicit_count is None:
        raise ValueError(
            f"{role} worker count is not set. "
            f"Please set {role.upper()}_NUM_WORKERS in `telescope.config`."
        )
    count = int(explicit_count)
    if count < 1:
        raise ValueError(f"{role} worker count must be >= 1, got {count}")
    return count


def init_ray_cluster() -> dict[str, Any]:
    """Attach to a Ray cluster and return basic runtime info."""
    if not ray.is_initialized():
        from telescope.utils.tlog import is_debug_mode

        address = config.cfg.ray_address
        auto_start_local = bool(config.cfg.ray_auto_start_local)
        namespace = config.cfg.ray_namespace
        log_to_driver = bool(config.cfg.ray_log_to_driver)
        # Suppress Ray worker log forwarding in non-debug mode — this is the
        # single biggest noise reduction (stops all pid=... prefixed lines).
        if not is_debug_mode():
            log_to_driver = False
        user_runtime_env = config.cfg.ray_runtime_env
        runtime_env = _build_default_ray_runtime_env(user_runtime_env)

        runtime_env_hook = os.environ.get("RAY_RUNTIME_ENV_HOOK")
        disable_runtime_env_hook = bool(
            config.cfg.ray_disable_runtime_env_hook
        )
        if runtime_env_hook and disable_runtime_env_hook:
            _log.warning(
                f"RAY_RUNTIME_ENV_HOOK={runtime_env_hook!r} detected; disabling it for "
                "Telescope launch to avoid per-worker uv env recreation."
            )
            os.environ.pop("RAY_RUNTIME_ENV_HOOK", None)
            runtime_env_hook = None
        elif runtime_env_hook:
            _log.warning(
                "RAY_RUNTIME_ENV_HOOK is set. This can trigger per-worker uv env "
                "creation and large dependency downloads."
            )

        base_kwargs: dict[str, Any] = {
            "namespace": namespace,
            "log_to_driver": log_to_driver,
            "include_dashboard": False,
            "ignore_reinit_error": True,
        }
        if runtime_env:
            base_kwargs["runtime_env"] = runtime_env

        # _system_config can only be set when starting a new cluster, not when
        # connecting to an existing one.
        local_kwargs = {**base_kwargs, "_system_config": {"enable_metrics_collection": False}}

        address_str = "" if address is None else str(address).strip()
        if not address_str:
            ray.init(**local_kwargs)
        else:
            try:
                ray.init(address=address_str, **base_kwargs)
            except Exception as exc:  # pragma: no cover - runtime integration error path
                should_fallback_local = auto_start_local and address_str.lower() == "auto"
                if should_fallback_local:
                    _log.warning(
                        "No running Ray cluster found at address='auto'; "
                        "starting a local Ray runtime for one-command launch."
                    )
                    ray.init(**local_kwargs)
                else:
                    raise RuntimeError(
                        "Failed to initialize Ray cluster.\n"
                        "Start Ray first (e.g. `ray start --head --num-gpus=<N>`) and "
                        "then run Telescope.\n"
                        "Or enable local auto-start with "
                        "`config.cfg.ray_auto_start_local = True`."
                    ) from exc

    runtime_context = ray.get_runtime_context()
    node_id = ""
    try:
        node_id = str(runtime_context.get_node_id())
    except Exception:
        node_id = ""
    return {
        "node_id": node_id,
        "node_ip": ray.util.get_node_ip_address(),
        "resources": ray.cluster_resources(),
    }


def _serialize_resource_map(resource_map: dict[str, Any]) -> dict[str, float]:
    serialized: dict[str, float] = {}
    for key, value in (resource_map or {}).items():
        try:
            serialized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return serialized


def _runtime_env_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for key in _SETUP_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            snapshot[key] = value
    return snapshot


def _nest_setup_system_info(flat: dict[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for key, value in (flat or {}).items():
        if not isinstance(key, str) or not key.startswith("setup/"):
            continue
        parts = key.split("/")[1:]  # drop "setup"
        current = nested
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return nested


def _int_or_default(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _compact_nested_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in (payload or {}).items():
        if isinstance(value, dict):
            nested = _compact_nested_payload(value)
            if nested:
                compact[key] = nested
            continue
        if isinstance(value, list):
            if value:
                compact[key] = value
            continue
        if value is not None and value != "":
            compact[key] = value
    return compact


def _build_node_hardware(system_info: dict[str, Any]) -> dict[str, Any]:
    """Build the ``hardware`` subtree for a node from nested system info."""
    gpu_raw = dict(system_info.get("gpu") or {})

    # ── GPU interconnect ────────────────────────────────────────────────
    interconnect: dict[str, Any] = {}
    for key in ("nvlink_available", "nvlink_connections", "nvlink_speed_per_link_gbps",
                "nvlink_links_per_gpu", "topology", "topology_matrix"):
        if key in gpu_raw:
            short = (
                key.replace("nvlink_available", "nvlink")
                .replace("nvlink_speed_per_link_gbps", "speed_gbps")
                .replace("nvlink_connections", "connections")
                .replace("nvlink_links_per_gpu", "links_per_gpu")
            )
            interconnect[short] = gpu_raw.pop(key)
    if interconnect:
        gpu_raw["interconnect"] = interconnect

    # ── Per-GPU devices (dict-of-dicts → sorted list) ───────────────────
    devices_raw = gpu_raw.pop("devices", None)
    if isinstance(devices_raw, dict) and devices_raw:
        devices_list = [
            dict(devices_raw[k]) for k in sorted(devices_raw, key=lambda k: int(k))
        ]
        gpu_raw["devices"] = devices_list

    # ── Rename memory key for clarity ───────────────────────────────────
    if "memory_total_gb" in gpu_raw:
        gpu_raw["memory_gb"] = gpu_raw.pop("memory_total_gb")

    # ── CPU ──────────────────────────────────────────────────────────────
    cpu_raw = dict(system_info.get("cpu") or {})
    if "physical_cores" in cpu_raw:
        cpu_raw["cores"] = cpu_raw.pop("physical_cores")

    # ── Memory ───────────────────────────────────────────────────────────
    memory_raw = dict(system_info.get("memory") or {})

    # ── Network hardware (InfiniBand / RDMA) ───────────────────────────
    network_raw = dict(system_info.get("network") or {})
    # hostname and ip_address are already top-level on the node — keep only
    # hardware-specific sub-fields like infiniband.
    network_raw.pop("hostname", None)
    network_raw.pop("ip_address", None)

    return _compact_nested_payload({
        "gpu": gpu_raw,
        "cpu": cpu_raw,
        "memory": memory_raw,
        "disk": dict(system_info.get("disk") or {}),
        "network": network_raw,
    })


def _build_node_software(
    system_info: dict[str, Any],
    runtime_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Build the ``software`` subtree for a node."""
    return _compact_nested_payload({
        "os": dict(system_info.get("os") or {}),
        "python": {
            "version": str(runtime_snapshot.get("python_version") or ""),
            "executable": str(runtime_snapshot.get("python_executable") or ""),
        },
        "packages": dict(system_info.get("package_versions") or {}),
        "container": dict(system_info.get("container") or {}),
        "environment": dict(runtime_snapshot.get("environment") or {}),
    })


@ray.remote(num_cpus=0)
def _collect_node_setup_snapshot() -> dict[str, Any]:
    """Collect per-node hardware/software information (runs on each Ray node)."""
    import platform

    from telescope.orchestrator.loggers.system_info import get_system_info

    runtime_context = ray.get_runtime_context()
    ray_node_id = ""
    try:
        ray_node_id = str(runtime_context.get_node_id())
    except Exception:
        ray_node_id = ""

    system_info_flat = get_system_info()
    return {
        "collected_at": time.time(),
        "hostname": socket.gethostname(),
        "node_ip": ray.util.get_node_ip_address(),
        "ray_node_id": ray_node_id,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "environment": _runtime_env_snapshot(),
        "system_info": _nest_setup_system_info(system_info_flat),
    }


def collect_cluster_setup(
    ray_runtime_info: dict[str, Any] | None,
    inference_server_infos: list[dict[str, Any]] | None = None,
    trainer_runtime_infos: list[dict[str, Any]] | None = None,
    num_inference_servers: int | None = None,
    num_trainer_ranks: int | None = None,
    inference_tp_size: int = 1,
) -> dict[str, Any]:
    """
    Capture a role-centric cluster setup snapshot for W&B.

    The payload is a tree structured by **role → nodes → gpus** so that
    a reader can reconstruct the full deployment from the snapshot alone:

    - ``inference``: all inference GPU assignments, grouped by node
      - ``inference.nodes[].gpus[]``: one entry per inference GPU with
        server identity, TP group membership, and endpoint URL
    - ``trainer``: all trainer GPU assignments, grouped by node
      - ``trainer.nodes[].gpus[]``: one entry per trainer GPU with
        global rank, parallelism group ranks (DP/TP/PP/CP/EP), and device
    - ``cluster``: physical cluster metadata
      - ``cluster.nodes[]``: one entry per physical node with hardware,
        software, and Ray resource info
    """
    if not ray.is_initialized():
        return {}

    inference_server_infos = list(inference_server_infos or [])
    trainer_runtime_infos = list(trainer_runtime_infos or [])
    inferred_tp_size = max(1, int(inference_tp_size or 1))
    alive_nodes = [n for n in ray.nodes() if bool(n.get("Alive"))]

    # ── collect per-node snapshots in parallel ──────────────────────────
    node_refs: list[tuple[str, str, Any, dict[str, Any], dict[str, Any]]] = []
    for node in alive_nodes:
        node_ip = str(node.get("NodeManagerAddress") or "")
        ray_node_id = str(node.get("NodeID") or "")
        options: dict[str, Any] = {"num_cpus": 0}
        node_resource_key = f"node:{node_ip}"
        if node_resource_key in (node.get("Resources") or {}):
            options["resources"] = {node_resource_key: 0.001}
        ref = _collect_node_setup_snapshot.options(**options).remote()
        node_refs.append((ray_node_id, node_ip, ref, dict(node.get("Resources") or {}), dict(node)))

    snapshots: list[dict[str, Any]] = []
    for ray_node_id, node_ip, ref, resources, raw_node in node_refs:
        try:
            snapshot = dict(ray.get(ref, timeout=45))
        except Exception as exc:  # pragma: no cover
            snapshot = {
                "collection_error": str(exc),
                "node_ip": node_ip,
                "ray_node_id": ray_node_id,
                "hostname": "",
                "environment": {},
                "system_info": {},
            }
        snapshot["node_ip"] = str(snapshot.get("node_ip") or node_ip)
        snapshot["ray_node_id"] = str(snapshot.get("ray_node_id") or ray_node_id)
        snapshot["hostname"] = str(snapshot.get("hostname") or "")
        snapshot["ray_resources"] = _serialize_resource_map(resources)
        snapshot["ray_node_raw"] = raw_node
        snapshots.append(snapshot)

    # ── assign stable numeric node IDs (0..N-1), sorted by IP ──────────
    snapshots.sort(key=lambda s: (str(s.get("node_ip", "")), str(s.get("ray_node_id", ""))))
    node_id_by_ip: dict[str, int] = {}
    node_id_by_ray_node_id: dict[str, int] = {}

    # Build cluster.nodes[] — the physical node descriptions.
    cluster_nodes: list[dict[str, Any]] = []
    for node_idx, snapshot in enumerate(snapshots):
        node_ip = str(snapshot.get("node_ip") or "")
        ray_node_id = str(snapshot.get("ray_node_id") or "")
        node_id_by_ip[node_ip] = node_idx
        if ray_node_id:
            node_id_by_ray_node_id[ray_node_id] = node_idx

        system_info = dict(snapshot.get("system_info") or {})
        runtime_snapshot = {
            "python_executable": str(snapshot.get("python_executable") or ""),
            "python_version": str(snapshot.get("python_version") or ""),
            "platform": str(snapshot.get("platform") or ""),
            "environment": dict(snapshot.get("environment") or {}),
        }
        raw_node = dict(snapshot.get("ray_node_raw") or {})

        cluster_nodes.append({
            "node_id": node_idx,
            "hostname": str(snapshot.get("hostname") or ""),
            "ip": node_ip,
            "ray_node_id": ray_node_id,
            "is_driver": False,  # set below
            "hardware": _build_node_hardware(system_info),
            "software": _build_node_software(system_info, runtime_snapshot),
            "ray": _compact_nested_payload({
                "alive": bool(raw_node.get("Alive", True)),
                "labels": dict(raw_node.get("Labels") or {}),
                "resources": dict(snapshot.get("ray_resources") or {}),
            }),
            "collected_at": float(snapshot.get("collected_at", 0.0)),
            "collection_error": str(snapshot.get("collection_error") or ""),
        })

    # ── resolve helpers ─────────────────────────────────────────────────
    def resolve_node_id(*, node_id: Any = None, node_ip: Any = None, ray_node_id: Any = None) -> int:
        try:
            parsed = int(node_id)
            if parsed >= 0:
                return parsed
        except (TypeError, ValueError):
            pass
        rid = str(ray_node_id or "")
        if rid and rid in node_id_by_ray_node_id:
            return int(node_id_by_ray_node_id[rid])
        ip = str(node_ip or "")
        if ip and ip in node_id_by_ip:
            return int(node_id_by_ip[ip])
        return -1

    cluster_node_by_id: dict[int, dict[str, Any]] = {
        _int_or_default(n.get("node_id"), default=-1): n
        for n in cluster_nodes
        if _int_or_default(n.get("node_id"), default=-1) >= 0
    }

    # Helper to extract short node identity for role sub-trees.
    def _node_identity(node_id: int) -> dict[str, Any]:
        cn = cluster_node_by_id.get(node_id, {})
        return {
            "node_id": node_id,
            "hostname": str(cn.get("hostname") or ""),
            "ip": str(cn.get("ip") or ""),
        }

    # ── mark driver node ────────────────────────────────────────────────
    runtime_info = dict(ray_runtime_info or {})
    driver_node_id = resolve_node_id(
        node_id=runtime_info.get("node_id"),
        node_ip=runtime_info.get("node_ip"),
        ray_node_id=runtime_info.get("node_id"),
    )
    if driver_node_id in cluster_node_by_id:
        cluster_node_by_id[driver_node_id]["is_driver"] = True

    # ── build inference tree: inference → nodes → gpus ──────────────────
    # Each server is expanded into tp_size GPU entries so the tree is
    # truly per-GPU.  For TP=1 (common), this is 1:1 with servers.
    inference_gpus_by_node: dict[int, list[dict[str, Any]]] = {}
    unassigned_inference_gpus: list[dict[str, Any]] = []

    for info in sorted(inference_server_infos, key=lambda i: int(i.get("server_idx", -1))):
        server_idx = int(info.get("server_idx", -1))
        node_id = resolve_node_id(
            node_id=info.get("node_id"),
            node_ip=str(info.get("node_ip") or ""),
            ray_node_id=str(info.get("ray_node_id") or ""),
        )
        tp_group_id = int(info.get("tp_group_id", server_idx))
        tp_size = int(info.get("tp_size", inferred_tp_size))
        port = int(info.get("port", -1))
        url = str(info.get("url") or "")
        hw_list: list[dict[str, Any]] = info.get("gpu_hardware") or []
        worker_env: dict[str, str] = info.get("env") or {}
        # Physical GPU indices visible to this server (from CUDA_VISIBLE_DEVICES).
        phys_indices = _parse_cuda_visible_devices(worker_env.get("CUDA_VISIBLE_DEVICES", ""))

        for tp_rank in range(tp_size):
            # gpu_index = physical GPU index on the node.  Matches the
            # ``gpu_index`` column in the GPU metrics tables so you can
            # join on ``(node_id, gpu_index)``.
            gpu_idx = phys_indices[tp_rank] if tp_rank < len(phys_indices) else -1
            gpu_entry: dict[str, Any] = {
                "gpu_index": gpu_idx,
                "server_idx": server_idx,
                "tp_group_id": tp_group_id,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
                "port": port,
                "url": url,
            }
            # Attach per-GPU hardware (indexed by tp_rank within the server).
            if tp_rank < len(hw_list):
                gpu_entry["hardware"] = hw_list[tp_rank]
            elif hw_list:
                gpu_entry["hardware"] = hw_list[0]
            if worker_env:
                gpu_entry["env"] = worker_env
            if node_id >= 0 and node_id in cluster_node_by_id:
                inference_gpus_by_node.setdefault(node_id, []).append(gpu_entry)
            else:
                gpu_entry.update({
                    "node_id": node_id,
                    "node_ip": str(info.get("node_ip") or ""),
                    "hostname": str(info.get("hostname") or ""),
                    "ray_node_id": str(info.get("ray_node_id") or ""),
                })
                unassigned_inference_gpus.append(gpu_entry)

    inference_nodes: list[dict[str, Any]] = []
    for nid in sorted(inference_gpus_by_node):
        gpus = sorted(
            inference_gpus_by_node[nid],
            key=lambda g: (
                _int_or_default(g.get("gpu_index")),
                _int_or_default(g.get("server_idx")),
                _int_or_default(g.get("tp_rank")),
            ),
        )
        node_entry = _node_identity(nid)
        node_entry["gpus"] = gpus
        inference_nodes.append(node_entry)

    num_servers = int(num_inference_servers or len(inference_server_infos))
    inference_total_gpus = num_servers * inferred_tp_size

    # ── build trainer tree: trainer → nodes → gpus ──────────────────────
    trainer_gpus_by_node: dict[int, list[dict[str, Any]]] = {}
    unassigned_trainer_gpus: list[dict[str, Any]] = []

    for info in sorted(trainer_runtime_infos, key=lambda i: int(i.get("rank", -1))):
        node_id = resolve_node_id(
            node_id=info.get("node_id"),
            node_ip=str(info.get("node_ip") or ""),
            ray_node_id=str(info.get("ray_node_id") or ""),
        )
        trainer_env: dict[str, str] = info.get("env") or {}
        local_rank = int(info.get("local_rank", -1))
        # gpu_index = physical GPU index on the node.  Matches the
        # ``gpu_index`` column in GPU metrics tables (join on
        # ``(node_id, gpu_index)``).
        t_phys = _parse_cuda_visible_devices(trainer_env.get("CUDA_VISIBLE_DEVICES", ""))
        t_gpu_idx = t_phys[local_rank] if 0 <= local_rank < len(t_phys) else (t_phys[0] if t_phys else -1)
        gpu_entry: dict[str, Any] = {
            "gpu_index": t_gpu_idx,
            "rank": int(info.get("rank", -1)),
            "local_rank": local_rank,
            "device": str(info.get("device") or ""),
            "dp_rank": int(info.get("dp_rank", -1)),
            "dp_world_size": int(info.get("dp_world_size", -1)),
        }
        # Include Megatron parallelism ranks when available.
        for pkey in ("tp_rank", "tp_size", "pp_rank", "pp_size",
                      "cp_rank", "cp_size", "ep_rank", "ep_size"):
            val = info.get(pkey)
            if val is not None:
                gpu_entry[pkey] = int(val)
        # Attach per-GPU hardware (each trainer rank owns exactly one GPU).
        trainer_hw: list[dict[str, Any]] = info.get("gpu_hardware") or []
        if trainer_hw:
            gpu_entry["hardware"] = trainer_hw[0]
        if trainer_env:
            gpu_entry["env"] = trainer_env

        if node_id >= 0 and node_id in cluster_node_by_id:
            trainer_gpus_by_node.setdefault(node_id, []).append(gpu_entry)
        else:
            gpu_entry.update({
                "node_id": node_id,
                "node_ip": str(info.get("node_ip") or ""),
                "hostname": str(info.get("hostname") or ""),
                "ray_node_id": str(info.get("ray_node_id") or ""),
            })
            unassigned_trainer_gpus.append(gpu_entry)

    trainer_nodes: list[dict[str, Any]] = []
    for nid in sorted(trainer_gpus_by_node):
        gpus = sorted(
            trainer_gpus_by_node[nid],
            key=lambda g: _int_or_default(g.get("rank")),
        )
        node_entry = _node_identity(nid)
        node_entry["gpus"] = gpus
        trainer_nodes.append(node_entry)

    num_ranks = int(num_trainer_ranks or len(trainer_runtime_infos))

    # Derive parallelism sizes from the first trainer rank (all ranks
    # report consistent sizes).
    first_rank = trainer_runtime_infos[0] if trainer_runtime_infos else {}
    trainer_tp_size = int(first_rank.get("tp_size", config.cfg.megatron_tensor_parallel_size))
    trainer_pp_size = int(first_rank.get("pp_size", config.cfg.megatron_pipeline_parallel_size))
    trainer_cp_size = int(first_rank.get("cp_size", config.cfg.megatron_context_parallel_size))
    trainer_ep_size = int(first_rank.get("ep_size", config.cfg.megatron_expert_parallel_size))
    trainer_dp_size = int(first_rank.get("dp_world_size", 0))
    if trainer_dp_size <= 0 and num_ranks > 0:
        divisor = max(1, trainer_tp_size * trainer_pp_size * trainer_cp_size)
        trainer_dp_size = max(1, num_ranks // divisor)

    # Count total physical GPUs across all nodes from Ray resources.
    total_cluster_gpus = 0
    for cn in cluster_nodes:
        ray_res = (cn.get("ray") or {}).get("resources") or {}
        total_cluster_gpus += int(float(ray_res.get("GPU", 0)))

    # ── build top-level payload ─────────────────────────────────────────
    return {
        "schema_version": "6.0",
        "generated_at_unix_s": time.time(),
        "model": str(config.cfg.model),
        "inference": {
            "num_servers": num_servers,
            "tensor_parallel_size": inferred_tp_size,
            "total_gpus": inference_total_gpus,
            "placement_strategy": str(
                config.cfg.ray_inference_placement_strategy
            ),
            "nodes": inference_nodes,
            "unassigned_gpus": unassigned_inference_gpus,
        },
        "trainer": {
            "backend": str(config.cfg.train_backend),
            "world_size": num_ranks,
            "total_gpus": num_ranks,
            "data_parallel_size": trainer_dp_size,
            "tensor_parallel_size": trainer_tp_size,
            "pipeline_parallel_size": trainer_pp_size,
            "context_parallel_size": trainer_cp_size,
            "expert_parallel_size": trainer_ep_size,
            "placement_strategy": str(
                config.cfg.ray_trainer_placement_strategy
            ),
            "nodes": trainer_nodes,
            "unassigned_gpus": unassigned_trainer_gpus,
        },
        "cluster": {
            "num_nodes": len(cluster_nodes),
            "total_gpus": total_cluster_gpus,
            "driver_node_id": driver_node_id,
            "ray": {
                "address": str(config.cfg.ray_address),
                "namespace": str(config.cfg.ray_namespace),
            },
            "nodes": cluster_nodes,
        },
    }


def _read_proc_cpu_times() -> tuple[int, int] | None:
    try:
        with open("/proc/stat", "r", encoding="utf-8") as file_handle:
            parts = file_handle.readline().split()
        if not parts or parts[0] != "cpu":
            return None
        times = [int(x) for x in parts[1:9]]
        total = sum(times)
        idle = times[3] + times[4]
        return total, idle
    except Exception:
        return None


@ray.remote(num_cpus=0)
def _collect_node_infra_metrics_snapshot() -> dict[str, Any]:
    """
    Collect a point-in-time infra snapshot on one Ray node.

    Returns dict payloads that can be converted into GpuMetricSample/CpuMetricSample.
    """
    timestamp = time.time()
    node_ip = ray.util.get_node_ip_address()
    ray_node_id = str(ray.get_runtime_context().get_node_id())
    hostname = socket.gethostname()
    gpu_samples: list[dict[str, Any]] = []
    cpu_samples: list[dict[str, Any]] = []

    # GPU metrics via NVML.
    try:
        import pynvml

        pynvml.nvmlInit()
        num_gpus = int(pynvml.nvmlDeviceGetCount())
        for gpu_idx in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_samples.append(
                    {
                        "timestamp": timestamp,
                        "node_id": -1,
                        "node_ip": node_ip,
                        "hostname": hostname,
                        "ray_node_id": ray_node_id,
                        "gpu_index": gpu_idx,
                        "metric_name": "gpu_memory_used_gb",
                        "value": float(mem_info.used / (1024**3)),
                    }
                )
            except Exception:
                continue

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_samples.append(
                    {
                        "timestamp": timestamp,
                        "node_id": -1,
                        "node_ip": node_ip,
                        "hostname": hostname,
                        "ray_node_id": ray_node_id,
                        "gpu_index": gpu_idx,
                        "metric_name": "gpu_utilization_percent",
                        "value": float(util.gpu),
                    }
                )
            except Exception:
                pass
        pynvml.nvmlShutdown()
    except Exception:
        pass

    # CPU utilization (sampled from /proc/stat over a short interval).
    first = _read_proc_cpu_times()
    time.sleep(0.05)
    second = _read_proc_cpu_times()
    if first is not None and second is not None:
        first_total, first_idle = first
        second_total, second_idle = second
        total_delta = second_total - first_total
        idle_delta = second_idle - first_idle
        if total_delta > 0:
            busy_percent = 100.0 * (1.0 - (idle_delta / total_delta))
            cpu_samples.append(
                {
                    "timestamp": timestamp,
                    "node_id": -1,
                    "node_ip": node_ip,
                    "hostname": hostname,
                    "ray_node_id": ray_node_id,
                    "metric_name": "cpu_utilization_percent",
                    "value": float(max(0.0, busy_percent)),
                }
            )

    # System memory from /proc/meminfo.
    try:
        mem_values: dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                parts = line.split()
                if len(parts) < 2:
                    continue
                key = parts[0].rstrip(":")
                mem_values[key] = int(parts[1])
        total_gb = float(mem_values.get("MemTotal", 0) / (1024**2))
        available_gb = float(mem_values.get("MemAvailable", 0) / (1024**2))
        used_gb = max(0.0, total_gb - available_gb)
        percent = (100.0 * used_gb / total_gb) if total_gb > 0 else 0.0
        cpu_samples.extend(
            [
                {
                    "timestamp": timestamp,
                    "node_id": -1,
                    "node_ip": node_ip,
                    "hostname": hostname,
                    "ray_node_id": ray_node_id,
                    "metric_name": "system_memory_used_gb",
                    "value": used_gb,
                },
                {
                    "timestamp": timestamp,
                    "node_id": -1,
                    "node_ip": node_ip,
                    "hostname": hostname,
                    "ray_node_id": ray_node_id,
                    "metric_name": "system_memory_available_gb",
                    "value": available_gb,
                },
                {
                    "timestamp": timestamp,
                    "node_id": -1,
                    "node_ip": node_ip,
                    "hostname": hostname,
                    "ray_node_id": ray_node_id,
                    "metric_name": "system_memory_percent",
                    "value": percent,
                },
            ]
        )
    except Exception:
        pass

    return {
        "timestamp": timestamp,
        "node_id": -1,
        "node_ip": node_ip,
        "hostname": hostname,
        "ray_node_id": ray_node_id,
        "gpu": gpu_samples,
        "cpu": cpu_samples,
    }


def collect_cluster_infra_metrics_samples(
    exclude_node_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect one infra sample per Ray node (GPU + CPU metrics)."""
    if not ray.is_initialized():
        return [], []

    excluded = {str(node_id) for node_id in (exclude_node_ids or set()) if node_id}
    alive_nodes = [n for n in ray.nodes() if bool(n.get("Alive"))]
    refs = []
    for node in alive_nodes:
        node_ip = str(node.get("NodeManagerAddress") or "")
        if node_ip in excluded:
            continue
        options: dict[str, Any] = {"num_cpus": 0}
        node_resource_key = f"node:{node_ip}"
        if node_resource_key in (node.get("Resources") or {}):
            options["resources"] = {node_resource_key: 0.001}
        refs.append(_collect_node_infra_metrics_snapshot.options(**options).remote())

    gpu_samples: list[dict[str, Any]] = []
    cpu_samples: list[dict[str, Any]] = []
    for ref in refs:
        try:
            payload = dict(ray.get(ref, timeout=15))
        except Exception:
            continue
        gpu_samples.extend(payload.get("gpu", []) or [])
        cpu_samples.extend(payload.get("cpu", []) or [])
    return gpu_samples, cpu_samples


@ray.remote(max_restarts=0)
class InferenceServerActor:
    """One Ray actor that owns one vLLM inference server process (one or more GPUs with TP)."""

    def __init__(self, server_idx: int, bind_host: str = "0.0.0.0", model: str | None = None, config_data: dict | None = None):
        if config_data is not None:
            from telescope.utils.config import install_config
            install_config(config_data)
        setup_logging()
        self.server_idx = int(server_idx)
        self.bind_host = bind_host
        self._model = model
        self.node_ip = ray.util.get_node_ip_address()
        self.hostname = socket.gethostname()
        self.ray_node_id = str(ray.get_runtime_context().get_node_id())
        self.node_id = _resolve_cluster_node_index(self.node_ip, self.ray_node_id)
        self.port = _pick_free_port()
        self.tp_size = max(1, int(config.cfg.inference_tensor_parallel_size))
        self.tp_group_id = self.server_idx
        self._process: subprocess.Popen | None = None
        # Capture GPU hardware *before* vLLM subprocess takes the device.
        self._gpu_hardware: list[dict[str, Any]] = _collect_visible_gpu_hardware()
        self._worker_env: dict[str, str] = _worker_env_snapshot()

    @property
    def public_url(self) -> str:
        return f"http://{self.node_ip}:{self.port}"

    def _server_info_payload(self) -> dict[str, Any]:
        """Build the common info dict returned by ``start()``."""
        return {
            "server_idx": self.server_idx,
            "url": self.public_url,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "hostname": self.hostname,
            "ray_node_id": self.ray_node_id,
            "tp_group_id": self.tp_group_id,
            "tp_size": self.tp_size,
            "port": self.port,
            "gpu_hardware": self._gpu_hardware,
            "env": self._worker_env,
            "subprocess_pid": self._process.pid if self._process else None,
        }

    def start(self, otlp_endpoint: str | None = None, startup_timeout_s: int = 600) -> dict[str, Any]:
        """Spawn vLLM server process and wait until ready."""
        if self._process is not None and self._process.poll() is None:
            return self._server_info_payload()

        paths.ensure_stdout_dirs()
        log_file = paths.STDOUT_INFERENCE_DIR / _build_stdout_filename(
            worker_type="inference",
            node_id=self.node_id,
            rank=self.server_idx,
        )

        cmd = [
            sys.executable,
            "-m",
            "telescope.inference.server",
            "--host",
            self.bind_host,
            "--port",
            str(self.port),
        ]
        if self._model:
            cmd.extend(["--model", self._model])
        env = dict(os.environ)
        if otlp_endpoint:
            env["TELESCOPE_OTLP_TRACES_ENDPOINT"] = otlp_endpoint
        if bool(config.cfg.enable_vllm_tracing):
            env["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "http/protobuf"
            env["OTEL_BSP_SCHEDULE_DELAY"] = "200"

        with open(log_file, "w") as file_handle:
            self._process = subprocess.Popen(
                cmd,
                stdout=file_handle,
                stderr=file_handle,
                stdin=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )

        self._wait_until_ready(timeout_s=startup_timeout_s)
        _inference_log.info(
            f"Ray inference server {self.server_idx} started on {self.public_url}",
            rank=self.server_idx,
        )
        info = self._server_info_payload()
        info["ready_time"] = time.time()
        return info

    def _wait_until_ready(self, timeout_s: int = 600) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"Inference server {self.server_idx} exited early with code "
                    f"{self._process.returncode}"
                )
            if self.check_ready():
                return
            time.sleep(1.0)
        raise TimeoutError(
            f"Inference server {self.server_idx} did not become ready within {timeout_s}s"
        )

    def check_ready(self) -> bool:
        try:
            response = requests.get(f"http://127.0.0.1:{self.port}/v1/models", timeout=2.0)
            return 200 <= response.status_code < 500
        except requests.RequestException:
            return False

    def init_broadcast(self, host: str, port: int, world_size: int, rank: int, group: str = "full") -> bool:
        payload = {
            "host": host,
            "port": int(port),
            "world_size": int(world_size),
            "rank": int(rank),
            "group": group,
        }
        response = requests.post(
            f"http://127.0.0.1:{self.port}/init_broadcast",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        return True

    def load_weights(self, group: str = "full") -> dict[str, Any]:
        start_time = time.time()
        response = requests.get(
            f"http://127.0.0.1:{self.port}/load_weights",
            params={"group": group},
            timeout=900,
        )
        response.raise_for_status()
        end_time = time.time()
        return {
            "server": self.server_idx,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "hostname": self.hostname,
            "ray_node_id": self.ray_node_id,
            "tp_group_id": self.tp_group_id,
            "tp_size": self.tp_size,
            "start_time": start_time,
            "end_time": end_time,
        }

    def drain_torch_memory_samples(self) -> list[dict[str, Any]]:
        """Collect current torch allocator metrics from the vLLM server workers."""
        try:
            response = requests.get(f"http://127.0.0.1:{self.port}/torch_memory", timeout=5.0)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []

        raw_samples: list[Any]
        if isinstance(payload, dict):
            raw_samples = list(payload.get("samples", []) or [])
        elif isinstance(payload, list):
            raw_samples = list(payload)
        else:
            return []

        normalized: list[dict[str, Any]] = []
        for sample in raw_samples:
            if not isinstance(sample, dict):
                continue
            item = dict(sample)
            item.setdefault("source", "torch_inference")
            item.setdefault("rank", -1)
            item.setdefault("node_id", self.node_id)
            item.setdefault("node_ip", self.node_ip)
            item.setdefault("hostname", self.hostname)
            item.setdefault("ray_node_id", self.ray_node_id)
            item.setdefault("server", self.server_idx)
            item.setdefault("tp_group_id", self.tp_group_id)
            item.setdefault("tp_size", self.tp_size)
            normalized.append(item)
        return normalized

    def get_subprocess_pid(self) -> int | None:
        """Return the PID of the vLLM subprocess, or None if not started."""
        return self._process.pid if self._process else None

    def stop(self) -> bool:
        if self._process is None or self._process.poll() is not None:
            return True

        pgid = os.getpgid(self._process.pid)
        os.killpg(pgid, signal.SIGINT)
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            self._process.wait(timeout=10)
        return True


@ray.remote(max_restarts=0, concurrency_groups={"default": 1, "signal": 1})
class TrainerRayActor:
    """One trainer rank hosted in one Ray actor with one GPU.

    Uses two concurrency groups so that lightweight signal methods (like
    ``eval_done``) can execute in a separate thread while ``train_step``
    is running.  This lets the orchestrator deliver eval completion
    signals without waiting for queued train steps to finish.
    """

    def __init__(self, rank: int, world_size: int, config_data: dict | None = None):
        if config_data is not None:
            from telescope.utils.config import install_config
            install_config(config_data)
        setup_logging()
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.node_ip = ray.util.get_node_ip_address()
        self.hostname = socket.gethostname()
        self.ray_node_id = str(ray.get_runtime_context().get_node_id())
        self.node_id = _resolve_cluster_node_index(self.node_ip, self.ray_node_id)
        self._backend = None
        self._communicator = None
        self._inference_actors: list[Any] = []
        self._torch_memory_logger: TorchMemoryLogger | None = None
        self._stdout_file_handle: TextIO | None = None
        self._stdout_tee_stream: _TeeStream | None = None
        self._original_stdout: TextIO | None = None
        self._original_stderr: TextIO | None = None
        self._start_stdout_capture()

    def get_node_ip_and_free_port(self) -> tuple[str, int]:
        return ray.util.get_node_ip_address(), _pick_free_port()

    def _start_stdout_capture(self) -> None:
        """Mirror actor stdout/stderr to logs/stdout/trainer/trainer_node_*_rank_*.stdout."""
        if self._stdout_file_handle is not None:
            return

        paths.ensure_stdout_dirs()
        log_file = paths.STDOUT_TRAINER_DIR / _build_stdout_filename(
            worker_type="trainer",
            node_id=self.node_id,
            rank=self.rank,
        )
        self._stdout_file_handle = open(log_file, "a", buffering=1, encoding="utf-8")
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._stdout_tee_stream = _TeeStream(self._original_stdout, self._stdout_file_handle)
        sys.stdout = self._stdout_tee_stream
        sys.stderr = self._stdout_tee_stream
        _redirect_telescope_console_stream(self._stdout_tee_stream)

    def _stop_stdout_capture(self) -> None:
        if self._stdout_file_handle is None:
            return

        if self._original_stdout is not None:
            _redirect_telescope_console_stream(self._original_stdout)
            sys.stdout = self._original_stdout
        if self._original_stderr is not None:
            sys.stderr = self._original_stderr
        try:
            self._stdout_file_handle.flush()
        finally:
            self._stdout_file_handle.close()
        self._stdout_file_handle = None
        self._stdout_tee_stream = None
        self._original_stdout = None
        self._original_stderr = None

    def initialize(self, master_addr: str, master_port: int) -> dict[str, Any]:
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = "0"

        if config.cfg.train_backend == "megatron":
            os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        self._backend = create_backend()
        runtime = self._backend.init()
        runtime.update(
            {
                "node_id": self.node_id,
                "node_ip": self.node_ip,
                "hostname": self.hostname,
                "ray_node_id": self.ray_node_id,
                "gpu_hardware": _collect_visible_gpu_hardware(),
                "env": _worker_env_snapshot(),
            }
        )
        self._start_torch_memory_logger(runtime)
        _trainer_log.info(
            f"Trainer actor initialized (rank={runtime['rank']}/{runtime['world_size']})",
            rank=runtime["rank"],
        )
        runtime["ready_time"] = time.time()
        return runtime

    def _start_torch_memory_logger(self, runtime: dict[str, Any]) -> None:
        """Start torch memory sampling for this trainer actor."""
        if self._torch_memory_logger is not None:
            return

        local_rank = int(runtime.get("local_rank", 0))
        self._torch_memory_logger = TorchMemoryLogger(
            rank=int(runtime.get("rank", self.rank)),
            local_rank=local_rank,
            node_id=runtime.get("node_id", self.node_id),
            sample_interval_ms=config.cfg.torch_memory_sample_interval_seconds * 1000,
        )
        self._torch_memory_logger.start()

    def _stop_torch_memory_logger(self) -> None:
        if self._torch_memory_logger is None:
            return
        self._torch_memory_logger.stop()
        self._torch_memory_logger = None

    def drain_torch_memory_samples(self) -> list[dict[str, Any]]:
        """
        Drain buffered torch memory samples for this rank.

        Samples are generated inside the actor on the configured
        ``TORCH_MEMORY_SAMPLE_INTERVAL_SECONDS`` cadence and drained by the
        orchestrator on its metrics loop cadence.
        """
        if self._torch_memory_logger is None:
            return []
        try:
            return self._torch_memory_logger.drain_samples()
        except Exception:
            return []

    def configure_weight_broadcast(
        self,
        master_addr: str,
        master_port: int,
        num_inference_servers: int,
        inference_actors: list[Any],
        training_only_port: int | None = None,
        num_training_servers: int | None = None,
        training_inference_actors: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Configure NCCL weight broadcast groups.

        When *training_only_port* is provided the trainer also creates a
        second communicator that covers only the non-eval inference servers
        (the "training_only" group).
        """
        if self._backend is None:
            raise RuntimeError("Trainer backend not initialized")

        if self._backend.is_weight_broadcast_rank:
            self._inference_actors = list(inference_actors)
            self._training_inference_actors = list(training_inference_actors or inference_actors)
        else:
            self._inference_actors = []
            self._training_inference_actors = []

        # Full group communicator (trainer + ALL inference servers)
        self._communicator = setup_inference_communicator(
            rank=self._backend.rank,
            device=self._backend.device,
            master_address=master_addr,
            master_port=master_port,
            num_inference_servers=num_inference_servers,
        )

        # Training-only group communicator (trainer + non-eval servers)
        self._communicator_training_only = None
        if training_only_port is not None and num_training_servers is not None and num_training_servers > 0:
            self._communicator_training_only = setup_inference_communicator_for_group(
                rank=self._backend.rank,
                device=self._backend.device,
                master_address=master_addr,
                master_port=training_only_port,
                num_servers_in_group=num_training_servers,
                group_name="training_only",
            )

        return {
            "rank": self._backend.rank,
            "is_weight_broadcast_rank": self._backend.is_weight_broadcast_rank,
        }

    # ------------------------------------------------------------------
    # Eval schedule & blocking
    # ------------------------------------------------------------------

    def set_eval_schedule(self, eval_steps: set[int]) -> bool:
        """Store which training steps are eval steps."""
        self._eval_steps: set[int] = set(eval_steps)
        self._eval_active: bool = False
        self._eval_done_event: Any = None  # threading.Event, created lazily
        _trainer_log.info(f"Eval schedule set: {sorted(self._eval_steps)[:20]}...", rank=self.rank)
        return True

    @ray.method(concurrency_group="signal")
    def eval_done(self) -> bool:
        """Called by orchestrator when eval finishes — unblocks the trainer.

        Runs in the ``signal`` concurrency group so it can execute while
        ``train_step`` is running in the ``default`` group.  The flag
        write (``_eval_active = False``) is a simple Python attribute
        assignment, which is atomic under CPython's GIL.
        """
        self._eval_active = False
        evt = getattr(self, "_eval_done_event", None)
        if evt is not None:
            evt.set()
        _trainer_log.info("eval_done signal received", rank=self.rank)
        return True

    def _wait_for_eval_done(self):
        """Block until orchestrator calls ``eval_done``."""
        import threading
        if getattr(self, "_eval_done_event", None) is None:
            self._eval_done_event = threading.Event()
        else:
            self._eval_done_event.clear()
        _trainer_log.info("Trainer blocking — waiting for eval_done signal", rank=self.rank)
        self._eval_done_event.wait()

    # ------------------------------------------------------------------
    # Weight broadcast
    # ------------------------------------------------------------------

    def _broadcast_weights_impl(
        self,
        communicator,
        group_name: str,
        actors: list,
        step: int | None = None,
        tracker=None,
    ) -> list:
        """Core weight broadcast: gather → fire inference → NCCL send.

        Returns Ray ObjectRefs for deferred collection of inference timing.
        The trainer GPU is free as soon as ``barrier_post`` completes.
        """
        is_broadcast_rank = self._backend.is_weight_broadcast_rank

        # Phase 1: Gather weights (collective) + barrier
        state_dict = prepare_weights_for_broadcast(
            self._backend, step=step, tracker=tracker,
        )

        # NOW fire inference — gather is done, broadcast imminent.
        # Inference serves requests during the gather phase above instead
        # of blocking in comm.broadcast() waiting for the trainer.
        load_refs = []
        if is_broadcast_rank:
            load_refs = [actor.load_weights.remote(group=group_name) for actor in actors]

        # Phase 2: NCCL broadcast + final barrier
        broadcast_weights_to_inference(
            self._backend, state_dict, communicator, step=step, tracker=tracker,
        )

        # Return ObjectRefs — don't block on ray.get.
        # Timing is collected by the orchestrator asynchronously.
        return load_refs

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def train_step(self, step: int, trainer_data: dict) -> dict[str, Any]:
        if self._backend is None:
            raise RuntimeError("Trainer backend not initialized")
        local_rank = int(getattr(self._backend, "local_rank", 0))

        eval_steps = getattr(self, "_eval_steps", set())
        eval_active = getattr(self, "_eval_active", False)
        is_eval_step = step in eval_steps

        # If this is an eval step and a previous eval is still active, block.
        if is_eval_step and eval_active:
            _trainer_log.info(
                f"Step {step} is an eval step but eval is active — blocking",
                step=step, rank=self.rank,
            )
            self._wait_for_eval_done()

        _trainer_log.info("Starting RL training step", step=step, rank=self.rank)

        timeline_logger: GPUTimelineLogger | None = None
        if bool(config.cfg.track_gpu_events):
            timeline_logger = GPUTimelineLogger(rank=self._backend.rank)
        tracker = create_timeline_tracker(timeline_logger)
        tracker.start_step()

        is_broadcast_rank = self._backend.is_weight_broadcast_rank

        train_start = time.perf_counter()
        metrics = self._backend.train_step(trainer_data, tracker=tracker)
        train_time = time.perf_counter() - train_start

        # Re-read eval_active: eval_done() may have run in the "signal"
        # concurrency group during the training computation above, clearing
        # the flag.  Checking here lets us switch to the full NCCL group
        # immediately rather than waiting another step.
        eval_active = getattr(self, "_eval_active", False)

        # Decide which NCCL group to use for weight broadcast
        use_full_group = is_eval_step or not eval_active
        communicator = self._communicator if use_full_group else (self._communicator_training_only or self._communicator)
        group_name = "full" if use_full_group else "training_only"
        actors_for_broadcast = self._inference_actors if use_full_group else self._training_inference_actors

        if is_broadcast_rank:
            _trainer_log.info(
                f"Starting weight sync to inference (group={group_name})",
                step=step, rank=self.rank,
            )
        else:
            _trainer_log.info(
                "Waiting for weight sync from broadcast rank",
                step=step, rank=self.rank,
            )
        with tracker.track("weight_broadcast"):
            load_refs = self._broadcast_weights_impl(
                communicator=communicator,
                group_name=group_name,
                actors=actors_for_broadcast,
                step=step,
                tracker=tracker,
            )

        # After broadcasting on an eval step, mark eval active
        if is_eval_step:
            self._eval_active = True

        if is_broadcast_rank:
            _trainer_log.info(
                f"Weight sync complete (group={group_name}, "
                f"inference_servers={len(load_refs)})",
                step=step, rank=self.rank,
            )

        with tracker.track("finalize_timeline", cpu=True):
            if timeline_logger is not None:
                timeline_logger.finalize_step()

        _grad = metrics.get("grad_norm")
        _grad_str = "n/a" if _grad is None else f"{float(_grad):.4f}"
        _trainer_log.info(
            "Completed trainer step "
            f"{step} | grad_norm={_grad_str} "
            f"train_time_s={train_time:.3f}",
            step=step, rank=self.rank,
        )

        # Step metrics: dynamically iterate over all metrics from the backend.
        # Users can add custom metrics by returning extra keys from train_step().
        # Keys with "/" are split into section/group/metric; fewer parts leave earlier fields empty.
        if is_broadcast_rank:
            step_metrics = []
            for key, value in metrics.items():
                parts = key.split("/")
                if len(parts) >= 3:
                    section, group, metric = parts[0], parts[1], "/".join(parts[2:])
                elif len(parts) == 2:
                    section, group, metric = parts[0], "", parts[1]
                else:
                    section, group, metric = "", "", key
                step_metrics.append({"step": step, "metric": metric, "value": float(value), "section": section, "group": group})
        else:
            step_metrics = []

        timeline_events = (
            timeline_logger.get_serializable_events()
            if timeline_logger is not None
            else []
        )
        for event in timeline_events:
            event["node_id"] = self.node_id
            event["node_ip"] = self.node_ip
            event["hostname"] = self.hostname
            event["ray_node_id"] = self.ray_node_id
            event["local_rank"] = local_rank
        return {
            "step": step,
            "rank": self._backend.rank,
            "local_rank": local_rank,
            "node_id": self.node_id,
            "train_time_s": train_time,
            "train_metrics": metrics,
            "step_metrics": step_metrics,
            "timeline_events": timeline_events,
            "weight_sync_load_refs": load_refs,
            "eval_step": is_eval_step,
            "weight_sync_group": group_name,
            "step_end_time": time.time(),
        }

    def save_checkpoint(self, step: int) -> dict[str, Any]:
        """Save a checkpoint at the given step. All actors must call simultaneously."""
        if self._backend is None:
            raise RuntimeError("Trainer backend not initialized")
        local_rank = int(getattr(self._backend, "local_rank", 0))

        timeline_logger: GPUTimelineLogger | None = None
        if bool(config.cfg.track_gpu_events):
            timeline_logger = GPUTimelineLogger(rank=self._backend.rank)
        tracker = create_timeline_tracker(timeline_logger)
        tracker.start_step()

        start = time.perf_counter()
        with tracker.track("checkpoint", cpu=True):
            self._backend.save_checkpoint(step=step, ckpt_dir=paths.CHECKPOINT_DIR, tracker=tracker)
        elapsed = time.perf_counter() - start

        if timeline_logger is not None:
            timeline_logger.finalize_step()

        timeline_events = (
            timeline_logger.get_serializable_events()
            if timeline_logger is not None
            else []
        )
        for event in timeline_events:
            event["node_id"] = self.node_id
            event["node_ip"] = self.node_ip
            event["hostname"] = self.hostname
            event["ray_node_id"] = self.ray_node_id
            event["local_rank"] = local_rank

        return {
            "rank": self._backend.rank,
            "local_rank": local_rank,
            "node_id": self.node_id,
            "step": step,
            "checkpoint_time_s": elapsed,
            "timeline_events": timeline_events,
            "save_end_time": time.time(),
        }

    def load_checkpoint(self, step: int) -> dict[str, Any]:
        """Load a checkpoint at the given step. All actors must call simultaneously."""
        if self._backend is None:
            raise RuntimeError("Trainer backend not initialized")

        start = time.perf_counter()
        self._backend.load_checkpoint(step=step, ckpt_dir=paths.CHECKPOINT_DIR)
        elapsed = time.perf_counter() - start

        return {
            "rank": self._backend.rank,
            "step": step,
            "load_time_s": elapsed,
        }

    def broadcast_weights(self) -> list[dict[str, Any]]:
        """Broadcast current model weights to inference servers. All actors must call simultaneously."""
        if self._backend is None:
            raise RuntimeError("Trainer backend not initialized")
        load_refs = self._broadcast_weights_impl(
            communicator=self._communicator,
            group_name="full",
            actors=self._inference_actors,
        )
        # For standalone calls (checkpoint restore), resolve immediately.
        return ray.get(load_refs) if load_refs else []

    def shutdown(self) -> bool:
        self._stop_torch_memory_logger()
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            self._stop_stdout_capture()
        return True


@dataclass
class RayInferenceGroup:
    """Placement-group managed collection of inference actors."""

    num_servers: int
    gpus_per_server: int
    cpus_per_worker: float
    placement_strategy: str
    startup_timeout_s: int = 600
    bind_host: str = "0.0.0.0"
    model: str | None = None

    def __post_init__(self):
        self.placement_strategy = _validate_placement_strategy(
            "RAY_INFERENCE_PLACEMENT_STRATEGY", self.placement_strategy
        )
        self._pg = None
        self.actors: list[Any] = []
        self.server_infos: list[dict[str, Any]] = []
        self._start_refs: list[Any] = []

    @property
    def server_urls(self) -> list[str]:
        return [info["url"] for info in self.server_infos]

    def launch(self, otlp_endpoint: str | None = None) -> None:
        """Create placement group, spawn actors, and fire async start calls.

        This is the first phase of a two-phase startup.  Call ``wait_ready()``
        afterwards to collect results.
        """
        from telescope.utils.config import get_config
        config_data = get_config().model_dump()

        bundles = [
            {"GPU": float(self.gpus_per_server), "CPU": float(self.cpus_per_worker)}
            for _ in range(self.num_servers)
        ]
        self._pg = placement_group(bundles=bundles, strategy=self.placement_strategy)
        ray.get(self._pg.ready(), timeout=self.startup_timeout_s)

        actors = []
        for idx in range(self.num_servers):
            scheduling = PlacementGroupSchedulingStrategy(
                placement_group=self._pg,
                placement_group_bundle_index=idx,
                placement_group_capture_child_tasks=True,
            )
            actor = InferenceServerActor.options(
                num_gpus=float(self.gpus_per_server),
                num_cpus=float(self.cpus_per_worker),
                scheduling_strategy=scheduling,
            ).remote(server_idx=idx, bind_host=self.bind_host, model=self.model, config_data=config_data)
            actors.append(actor)

        self.actors = actors
        self._start_refs = [
            actor.start.remote(
                otlp_endpoint=otlp_endpoint,
                startup_timeout_s=self.startup_timeout_s,
            )
            for actor in self.actors
        ]

    def wait_ready(self) -> list[dict[str, Any]]:
        """Block until all inference actors finish starting and return server infos."""
        infos = ray.get(self._start_refs)
        self._start_refs = []
        self.server_infos = sorted(infos, key=lambda item: item["server_idx"])
        return self.server_infos

    def init_broadcast(self, host: str, port: int, world_size: int) -> None:
        tp_size = self.gpus_per_server
        refs = [
            actor.init_broadcast.remote(
                host=host,
                port=port,
                world_size=world_size,
                rank=idx * tp_size + 1,
            )
            for idx, actor in enumerate(self.actors)
        ]
        ray.get(refs)

    def drain_torch_memory_samples(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """Drain torch allocator samples from all inference actors."""
        if not self.actors:
            return []
        refs = [actor.drain_torch_memory_samples.remote() for actor in self.actors]
        per_actor_samples = ray.get(refs, timeout=timeout)
        merged: list[dict[str, Any]] = []
        for samples in per_actor_samples:
            if samples:
                merged.extend(samples)
        return merged

    def stop(self) -> None:
        if self.actors:
            # Collect subprocess PIDs before killing actors so we can ensure
            # vLLM processes are cleaned up even if the graceful stop fails.
            subprocess_pids = self._collect_subprocess_pids()
            try:
                ray.get([actor.stop.remote() for actor in self.actors], timeout=30)
            except Exception as exc:  # pragma: no cover - best-effort shutdown
                _log.warning(f"Inference actor shutdown encountered errors: {exc}")
            self._kill_actors()
            # Safety net: directly kill any vLLM subprocesses that survived.
            # They run in their own process groups (start_new_session=True),
            # so killing the Ray actor alone does not terminate them.
            for pid in subprocess_pids:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        if self._pg is not None:
            remove_placement_group(self._pg)
            self._pg = None

    def force_kill(self) -> None:
        """Force-kill all actors immediately without graceful shutdown.

        Kills the Ray actors first, then directly kills the vLLM subprocess
        process groups using PIDs collected at startup.  This avoids relying
        on the actor being responsive (its concurrency group may be saturated).
        """
        subprocess_pids = self._collect_subprocess_pids()
        self._kill_actors()
        # Kill vLLM subprocess process groups directly.  The subprocesses were
        # spawned with start_new_session=True so PGID == subprocess PID.
        for pid in subprocess_pids:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if self._pg is not None:
            remove_placement_group(self._pg)
            self._pg = None

    def _collect_subprocess_pids(self) -> list[int]:
        """Best-effort collection of vLLM subprocess PIDs.

        Uses server_infos when available (normal case).  If interrupted before
        wait_ready() populated server_infos, falls back to collecting any
        already-completed start refs and querying actors directly.
        """
        # Normal case: server_infos already populated after wait_ready().
        pids = [
            info["subprocess_pid"]
            for info in self.server_infos
            if info.get("subprocess_pid")
        ]
        if pids:
            return pids

        # Early interrupt: try to harvest any already-completed start refs.
        for ref in self._start_refs:
            try:
                info = ray.get(ref, timeout=0)
                if info and info.get("subprocess_pid"):
                    pids.append(info["subprocess_pid"])
            except Exception:
                pass
        self._start_refs = []

        # For actors whose start() hasn't returned yet, ask for PID directly.
        if len(pids) < len(self.actors):
            for actor in self.actors:
                try:
                    pid = ray.get(actor.get_subprocess_pid.remote(), timeout=2)
                    if pid and pid not in pids:
                        pids.append(pid)
                except Exception:
                    pass

        return pids

    def _kill_actors(self) -> None:
        for actor in self.actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        self.actors = []


@dataclass
class RayTrainerGroup:
    """Placement-group managed distributed trainer actors."""

    world_size: int
    cpus_per_worker: float
    placement_strategy: str
    startup_timeout_s: int = 900

    def __post_init__(self):
        self.placement_strategy = _validate_placement_strategy(
            "RAY_TRAINER_PLACEMENT_STRATEGY", self.placement_strategy
        )
        self._pg = None
        self.actors: list[Any] = []
        self.runtime_infos: list[dict[str, Any]] = []
        self.master_addr: str | None = None
        self.master_port: int | None = None
        self._init_refs: list[Any] = []

    def launch(self) -> None:
        """Create placement group, spawn actors, and fire async initialize calls.

        This is the first phase of a two-phase startup.  Call ``wait_ready()``
        afterwards to collect results.
        """
        from telescope.utils.config import get_config
        config_data = get_config().model_dump()

        bundles = [{"GPU": 1, "CPU": float(self.cpus_per_worker)} for _ in range(self.world_size)]
        self._pg = placement_group(bundles=bundles, strategy=self.placement_strategy)
        ray.get(self._pg.ready(), timeout=self.startup_timeout_s)

        actors = []
        for rank in range(self.world_size):
            scheduling = PlacementGroupSchedulingStrategy(
                placement_group=self._pg,
                placement_group_bundle_index=rank,
                placement_group_capture_child_tasks=True,
            )
            actor = TrainerRayActor.options(
                num_gpus=1,
                num_cpus=float(self.cpus_per_worker),
                scheduling_strategy=scheduling,
            ).remote(rank=rank, world_size=self.world_size, config_data=config_data)
            actors.append(actor)

        self.actors = actors
        self.master_addr, self.master_port = ray.get(
            self.actors[0].get_node_ip_and_free_port.remote()
        )
        self._init_refs = [
            actor.initialize.remote(master_addr=self.master_addr, master_port=self.master_port)
            for actor in self.actors
        ]

    def wait_ready(self) -> list[dict[str, Any]]:
        """Block until all trainer actors finish initializing and return runtime infos."""
        self.runtime_infos = ray.get(self._init_refs)
        self._init_refs = []
        return self.runtime_infos

    def allocate_weight_broadcast_endpoint(self) -> tuple[str, int]:
        if not self.actors:
            raise RuntimeError("Trainer actors not started")
        return ray.get(self.actors[0].get_node_ip_and_free_port.remote())

    def submit_train_step(self, step: int, trainer_data_per_rank: list[dict]) -> list[Any]:
        num_actors = len(self.actors)
        num_shards = len(trainer_data_per_rank)
        if num_shards < 1:
            raise ValueError("trainer_data_per_rank must contain at least one shard")

        if num_shards == num_actors:
            data_by_actor = trainer_data_per_rank
        elif self.runtime_infos and len(self.runtime_infos) == num_actors:
            # Megatron path: trainer_data_per_rank is keyed by DP rank; TP/PP ranks
            # sharing a DP rank must receive the same trainer shard.
            data_by_actor = []
            for idx, info in enumerate(self.runtime_infos):
                dp_rank = int(info.get("dp_rank", -1))
                if dp_rank < 0 or dp_rank >= num_shards:
                    raise ValueError(
                        "Trainer data shard count does not match trainer DP mapping: "
                        f"actor_idx={idx}, dp_rank={dp_rank}, num_shards={num_shards}"
                    )
                data_by_actor.append(trainer_data_per_rank[dp_rank])
        else:
            raise ValueError(
                "Trainer data shard count does not match trainer world size and "
                "runtime dp mapping is unavailable: "
                f"num_shards={num_shards}, num_actors={num_actors}"
            )
        # Explicitly put data into the object store so references survive until
        # remote workers fetch them (prevents ReferenceCountingAssertionError
        # in multi-node clusters).
        data_refs = [ray.put(data_by_actor[idx]) for idx in range(num_actors)]
        return [
            actor.train_step.remote(step=step, trainer_data=data_refs[idx])
            for idx, actor in enumerate(self.actors)
        ]

    @staticmethod
    def wait_step_results(step_refs: list[Any]) -> list[dict[str, Any]]:
        return ray.get(step_refs)

    def drain_torch_memory_samples(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """Drain buffered torch memory samples from all trainer actors."""
        if not self.actors:
            return []
        refs = [actor.drain_torch_memory_samples.remote() for actor in self.actors]
        per_actor_samples = ray.get(refs, timeout=timeout)
        merged: list[dict[str, Any]] = []
        for samples in per_actor_samples:
            if samples:
                merged.extend(samples)
        return merged

    def save_checkpoint(self, step: int) -> list[dict[str, Any]]:
        """Save checkpoint across all trainer actors. Blocks until complete."""
        refs = [actor.save_checkpoint.remote(step=step) for actor in self.actors]
        return ray.get(refs)

    def load_checkpoint(self, step: int) -> list[dict[str, Any]]:
        """Load checkpoint across all trainer actors. Blocks until complete."""
        refs = [actor.load_checkpoint.remote(step=step) for actor in self.actors]
        return ray.get(refs)

    def broadcast_weights(self) -> list[list[dict[str, Any]]]:
        """Broadcast current model weights to inference servers. Blocks until complete."""
        refs = [actor.broadcast_weights.remote() for actor in self.actors]
        return ray.get(refs)

    def stop(self) -> None:
        if self.actors:
            try:
                ray.get([actor.shutdown.remote() for actor in self.actors], timeout=30)
            except Exception as exc:  # pragma: no cover - best-effort shutdown
                _log.warning(f"Trainer actor shutdown encountered errors: {exc}")
            self._kill_actors()
        if self._pg is not None:
            remove_placement_group(self._pg)
            self._pg = None

    def force_kill(self) -> None:
        """Force-kill all actors immediately without graceful shutdown."""
        self._kill_actors()
        if self._pg is not None:
            remove_placement_group(self._pg)
            self._pg = None

    def _kill_actors(self) -> None:
        for actor in self.actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        self.actors = []


