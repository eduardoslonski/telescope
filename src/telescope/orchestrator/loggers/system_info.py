"""
System information gathering for wandb config logging.

Collects hardware and software setup information:
- GPU (NVIDIA): per-device details (name, memory, UUID, PCIe, power), topology, NVLink
- CPU: model, cores (physical + logical), architecture, frequency, NUMA topology
- Memory: total, available, swap
- Disk: total/free space
- Network: IP, hostname, InfiniBand/RDMA
- Container: detection of Docker/container environment
- Package versions: Python, PyTorch, vLLM, CUDA, NCCL, flash-attn, etc.
"""
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path


def get_system_info() -> dict:
    """
    Gather comprehensive system information.

    Returns a flat dict with hierarchical keys like:
    - setup/gpu/name
    - setup/gpu/devices/0/uuid
    - setup/cpu/model
    - setup/package_versions/pytorch
    """
    info = {}

    # GPU info (NVIDIA)
    info.update(_get_gpu_info())

    # CPU info
    info.update(_get_cpu_info())

    # Memory info
    info.update(_get_memory_info())

    # Disk info
    info.update(_get_disk_info())

    # Network info
    info.update(_get_network_info())

    # Package versions
    info.update(_get_package_versions())

    # OS info
    info.update(_get_os_info())

    # Container detection
    info.update(_get_container_info())

    return info


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

def _get_gpu_info() -> dict:
    """Get NVIDIA GPU information using torch and nvidia-smi.

    Note: This may run inside a Ray task that has no GPU allocation, so
    ``CUDA_VISIBLE_DEVICES`` can be empty and ``torch.cuda.is_available()``
    returns ``False`` even on a GPU node.  We therefore always attempt the
    ``nvidia-smi`` fallback so the setup still reports GPU hardware.
    """
    info: dict = {}
    prefix = "setup/gpu"
    torch_saw_gpus = False

    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch_saw_gpus = True
            info[f"{prefix}/available"] = True
            info[f"{prefix}/count"] = torch.cuda.device_count()

            props = torch.cuda.get_device_properties(0)
            info[f"{prefix}/name"] = props.name
            info[f"{prefix}/memory_total_gb"] = round(props.total_memory / (1024**3), 2)
            info[f"{prefix}/compute_capability"] = f"{props.major}.{props.minor}"
            info[f"{prefix}/multi_processor_count"] = props.multi_processor_count

            all_names = [
                torch.cuda.get_device_properties(i).name
                for i in range(torch.cuda.device_count())
            ]
            if len(set(all_names)) == 1:
                info[f"{prefix}/all_devices"] = f"{len(all_names)}x {all_names[0]}"
            else:
                info[f"{prefix}/all_devices"] = ", ".join(all_names)
    except ImportError:
        pass
    except Exception as e:
        info[f"{prefix}/torch_error"] = str(e)

    # Always attempt nvidia-smi — it is not affected by CUDA_VISIBLE_DEVICES
    # and works even when this process has no GPU allocation from Ray.
    nvidia_smi_info = _get_nvidia_smi_info()
    info.update(nvidia_smi_info)

    # Per-GPU detailed info via nvidia-smi.
    per_gpu_info = _get_per_gpu_details()
    info.update(per_gpu_info)

    # If torch couldn't see GPUs, try nvidia-smi for count/name/memory.
    if not torch_saw_gpus:
        smi_gpu_info = _get_gpu_info_via_smi()
        if smi_gpu_info:
            info[f"{prefix}/available"] = True
            for key, value in smi_gpu_info.items():
                info.setdefault(key, value)
        else:
            info.setdefault(f"{prefix}/available", False)

    return info


def _get_gpu_info_via_smi() -> dict:
    """Fallback GPU detection via nvidia-smi (ignores CUDA_VISIBLE_DEVICES)."""
    info = {}
    prefix = "setup/gpu"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,count",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if not lines:
            return {}
        info[f"{prefix}/count"] = len(lines)
        # Parse first GPU as representative.
        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) >= 1:
            info[f"{prefix}/name"] = parts[0]
        if len(parts) >= 2:
            try:
                mem_mib = float(parts[1])
                info[f"{prefix}/memory_total_gb"] = round(mem_mib / 1024, 2)
            except ValueError:
                pass
        all_names = [p.split(",")[0].strip() for p in lines]
        if len(set(all_names)) == 1:
            info[f"{prefix}/all_devices"] = f"{len(all_names)}x {all_names[0]}"
        else:
            info[f"{prefix}/all_devices"] = ", ".join(all_names)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    except Exception:
        pass
    return info


_PER_GPU_QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "serial",
    "pci.bus_id",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
    "memory.total",
    "memory.used",
    "memory.free",
    "power.limit",
    "power.max_limit",
    "power.default_limit",
    "compute_mode",
    "persistence_mode",
    "temperature.gpu",
    "clocks.max.graphics",
    "clocks.max.memory",
    "vbios_version",
    "ecc.mode.current",
)


def _get_per_gpu_details() -> dict:
    """Collect per-GPU details via nvidia-smi --query-gpu (one row per GPU)."""
    info: dict = {}
    prefix = "setup/gpu"
    try:
        query = ",".join(_PER_GPU_QUERY_FIELDS)
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if not lines:
            return {}

        num_fields = len(_PER_GPU_QUERY_FIELDS)
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < num_fields:
                parts.extend([""] * (num_fields - len(parts)))
            row = dict(zip(_PER_GPU_QUERY_FIELDS, parts))

            idx = row.get("index", "").strip()
            if not idx.isdigit():
                continue
            dp = f"{prefix}/devices/{idx}"

            info[f"{dp}/name"] = row.get("name", "")
            _set_if(info, f"{dp}/uuid", row.get("uuid"))
            _set_if(info, f"{dp}/serial", row.get("serial"))
            _set_if(info, f"{dp}/pci_bus_id", row.get("pci.bus_id"))
            _set_if(info, f"{dp}/vbios", row.get("vbios_version"))

            # PCIe
            _set_int(info, f"{dp}/pcie_gen", row.get("pcie.link.gen.current"))
            _set_int(info, f"{dp}/pcie_gen_max", row.get("pcie.link.gen.max"))
            _set_int(info, f"{dp}/pcie_width", row.get("pcie.link.width.current"))
            _set_int(info, f"{dp}/pcie_width_max", row.get("pcie.link.width.max"))

            # Memory (MiB → GiB)
            _set_float_mib_to_gb(info, f"{dp}/memory_total_gb", row.get("memory.total"))
            _set_float_mib_to_gb(info, f"{dp}/memory_used_gb", row.get("memory.used"))
            _set_float_mib_to_gb(info, f"{dp}/memory_free_gb", row.get("memory.free"))

            # Power (watts)
            _set_float(info, f"{dp}/power_limit_w", row.get("power.limit"))
            _set_float(info, f"{dp}/power_max_w", row.get("power.max_limit"))
            _set_float(info, f"{dp}/power_default_w", row.get("power.default_limit"))

            # Clocks (MHz)
            _set_int(info, f"{dp}/clock_max_graphics_mhz", row.get("clocks.max.graphics"))
            _set_int(info, f"{dp}/clock_max_memory_mhz", row.get("clocks.max.memory"))

            # Temperature (°C)
            _set_int(info, f"{dp}/temperature_c", row.get("temperature.gpu"))

            # Modes
            _set_if(info, f"{dp}/compute_mode", row.get("compute_mode"))
            _set_if(info, f"{dp}/persistence_mode", row.get("persistence_mode"))
            _set_if(info, f"{dp}/ecc", row.get("ecc.mode.current"))

        # Promote shared PCIe gen to top-level gpu field (useful summary).
        pcie_gens = [
            info.get(f"{prefix}/devices/{i}/pcie_gen")
            for i in range(len(lines))
            if info.get(f"{prefix}/devices/{i}/pcie_gen") is not None
        ]
        if pcie_gens:
            info[f"{prefix}/pcie_gen"] = min(pcie_gens)

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    except Exception:
        pass
    return info


def _get_nvidia_smi_info() -> dict:
    """Get driver, NVLink topology, and bandwidth info from nvidia-smi."""
    info: dict = {}
    prefix = "setup/gpu"

    try:
        # Driver version
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            info[f"{prefix}/driver_version"] = result.stdout.strip().split("\n")[0]

        # GPU topology — check for NVLink connections
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            topo_output = result.stdout
            info[f"{prefix}/topology_matrix"] = topo_output.strip()

            nvlink_matches = re.findall(r'NV(\d+)', topo_output)
            if nvlink_matches:
                nvlink_count = max(int(m) for m in nvlink_matches)
                info[f"{prefix}/nvlink_available"] = True
                info[f"{prefix}/nvlink_connections"] = nvlink_count
                info[f"{prefix}/topology"] = "nvlink"
            elif "PIX" in topo_output or "PXB" in topo_output:
                info[f"{prefix}/nvlink_available"] = False
                info[f"{prefix}/topology"] = "pcie_switch"
            elif "PHB" in topo_output:
                info[f"{prefix}/nvlink_available"] = False
                info[f"{prefix}/topology"] = "pcie_host_bridge"
            elif "SYS" in topo_output or "NODE" in topo_output:
                info[f"{prefix}/nvlink_available"] = False
                info[f"{prefix}/topology"] = "pcie_numa"
            else:
                info[f"{prefix}/nvlink_available"] = False
                info[f"{prefix}/topology"] = "unknown"

        # NVLink bandwidth info
        result = subprocess.run(
            ["nvidia-smi", "nvlink", "--status"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            speeds = re.findall(r'Link \d+: (\d+) GB/s', result.stdout)
            if speeds:
                link_speed = int(speeds[0])
                gpu_count = max(1, info.get(f"{prefix}/count", 1))
                num_links = len(speeds) // gpu_count
                info[f"{prefix}/nvlink_speed_per_link_gbps"] = link_speed
                info[f"{prefix}/nvlink_links_per_gpu"] = num_links

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

def _get_cpu_info() -> dict:
    """Get CPU information including NUMA topology."""
    info: dict = {}
    prefix = "setup/cpu"

    info[f"{prefix}/logical_cores"] = os.cpu_count()
    info[f"{prefix}/architecture"] = platform.machine()

    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()

        # Model name
        for line in cpuinfo.split("\n"):
            if "model name" in line:
                info[f"{prefix}/model"] = line.split(":")[1].strip()
                break

        # Physical cores (count unique core ids across all sockets)
        physical_ids = set()
        core_ids = set()
        current_physical_id = None
        for line in cpuinfo.split("\n"):
            stripped = line.strip()
            if stripped.startswith("physical id"):
                current_physical_id = stripped.split(":")[1].strip()
            elif stripped.startswith("core id") and current_physical_id is not None:
                core_id = stripped.split(":")[1].strip()
                core_ids.add((current_physical_id, core_id))
                physical_ids.add(current_physical_id)
                current_physical_id = None
        if core_ids:
            info[f"{prefix}/physical_cores"] = len(core_ids)
            info[f"{prefix}/sockets"] = len(physical_ids)
        else:
            # Fallback: assume logical = physical
            info[f"{prefix}/physical_cores"] = os.cpu_count()

        # CPU frequency
        for line in cpuinfo.split("\n"):
            if "cpu MHz" in line:
                try:
                    mhz = float(line.split(":")[1].strip())
                    info[f"{prefix}/frequency_mhz"] = round(mhz, 1)
                except ValueError:
                    pass
                break

    except Exception:
        info[f"{prefix}/physical_cores"] = os.cpu_count()

    # NUMA topology
    info.update(_get_numa_info())

    return info


def _get_numa_info() -> dict:
    """Detect NUMA topology (node count, CPUs per node, GPU affinity)."""
    info: dict = {}
    prefix = "setup/cpu/numa"

    try:
        result = subprocess.run(
            ["numactl", "--hardware"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            output = result.stdout
            # "available: 2 nodes (0-1)"
            avail_match = re.search(r'available:\s*(\d+)\s*node', output)
            if avail_match:
                info[f"{prefix}/num_nodes"] = int(avail_match.group(1))

            # Per-node memory: "node 0 size: 123456 MB"
            for match in re.finditer(r'node\s+(\d+)\s+size:\s+(\d+)\s+MB', output):
                node_idx = match.group(1)
                mem_mb = int(match.group(2))
                info[f"{prefix}/node_{node_idx}_memory_gb"] = round(mem_mb / 1024, 2)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    except Exception:
        pass

    # Fallback: count NUMA nodes via sysfs
    if f"{prefix}/num_nodes" not in info:
        try:
            numa_nodes = list(Path("/sys/devices/system/node").glob("node[0-9]*"))
            if numa_nodes:
                info[f"{prefix}/num_nodes"] = len(numa_nodes)
        except Exception:
            pass

    # GPU-to-NUMA affinity (which NUMA node each GPU is closest to)
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            # Parse NUMA affinity from topology output
            # Look for lines like "GPU0  GPU1  ... NUMA Affinity"
            for line in result.stdout.split("\n"):
                match = re.match(r'^GPU(\d+)\s+.*\s+(\d+)\s*$', line.strip())
                if match:
                    gpu_idx = match.group(1)
                    numa_node = match.group(2)
                    info[f"{prefix}/gpu_{gpu_idx}_affinity"] = int(numa_node)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def _get_memory_info() -> dict:
    """Get system memory information (total, available, swap)."""
    info: dict = {}
    prefix = "setup/memory"

    try:
        meminfo: dict[str, int] = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val_parts = parts[1].strip().split()
                    if val_parts:
                        try:
                            meminfo[key] = int(val_parts[0])  # kB
                        except ValueError:
                            pass

        if "MemTotal" in meminfo:
            info[f"{prefix}/total_gb"] = round(meminfo["MemTotal"] / (1024**2), 2)
        if "MemAvailable" in meminfo:
            info[f"{prefix}/available_gb"] = round(meminfo["MemAvailable"] / (1024**2), 2)
        if "SwapTotal" in meminfo:
            swap_gb = round(meminfo["SwapTotal"] / (1024**2), 2)
            info[f"{prefix}/swap_total_gb"] = swap_gb
            if "SwapFree" in meminfo:
                info[f"{prefix}/swap_free_gb"] = round(meminfo["SwapFree"] / (1024**2), 2)
        if "HugePages_Total" in meminfo:
            hp_total = meminfo["HugePages_Total"]
            if hp_total > 0:
                info[f"{prefix}/hugepages_total"] = hp_total
                info[f"{prefix}/hugepages_free"] = meminfo.get("HugePages_Free", 0)
                hp_size_kb = meminfo.get("Hugepagesize", 2048)
                info[f"{prefix}/hugepage_size_kb"] = hp_size_kb

    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

def _get_disk_info() -> dict:
    """Get disk space information for the workspace."""
    info: dict = {}
    prefix = "setup/disk"

    try:
        usage = shutil.disk_usage(Path.cwd())
        info[f"{prefix}/total_gb"] = round(usage.total / (1024**3), 2)
        info[f"{prefix}/free_gb"] = round(usage.free / (1024**3), 2)
        info[f"{prefix}/used_percent"] = round((usage.used / usage.total) * 100, 1)
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def _get_network_info() -> dict:
    """Get network information including InfiniBand/RDMA detection."""
    info: dict = {}
    prefix = "setup/network"

    try:
        info[f"{prefix}/hostname"] = socket.gethostname()

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            info[f"{prefix}/ip_address"] = s.getsockname()[0]
        except Exception:
            info[f"{prefix}/ip_address"] = socket.gethostbyname(socket.gethostname())
        finally:
            s.close()
    except Exception:
        pass

    # InfiniBand / RDMA detection (critical for multi-node training)
    info.update(_get_infiniband_info())

    return info


def _get_infiniband_info() -> dict:
    """Detect InfiniBand devices and their status."""
    info: dict = {}
    prefix = "setup/network/infiniband"

    try:
        # Check for IB devices via sysfs
        ib_path = Path("/sys/class/infiniband")
        if not ib_path.exists():
            info[f"{prefix}/available"] = False
            return info

        devices = sorted(d.name for d in ib_path.iterdir() if d.is_dir())
        if not devices:
            info[f"{prefix}/available"] = False
            return info

        info[f"{prefix}/available"] = True
        info[f"{prefix}/num_devices"] = len(devices)
        info[f"{prefix}/devices"] = ", ".join(devices)

        # Per-device details from sysfs
        for dev_idx, dev_name in enumerate(devices):
            dp = f"{prefix}/device_{dev_idx}"
            info[f"{dp}/name"] = dev_name

            # Read device attributes
            dev_path = ib_path / dev_name
            for attr_name, key in [
                ("fw_ver", "firmware"),
                ("board_id", "board_id"),
                ("hca_type", "type"),
            ]:
                attr_file = dev_path / attr_name
                if attr_file.exists():
                    try:
                        info[f"{dp}/{key}"] = attr_file.read_text().strip()
                    except Exception:
                        pass

            # Port info (rate, state)
            ports_path = dev_path / "ports"
            if ports_path.exists():
                for port_dir in sorted(ports_path.iterdir()):
                    port_num = port_dir.name
                    rate_file = port_dir / "rate"
                    state_file = port_dir / "state"
                    if rate_file.exists():
                        try:
                            info[f"{dp}/port_{port_num}_rate"] = rate_file.read_text().strip()
                        except Exception:
                            pass
                    if state_file.exists():
                        try:
                            raw = state_file.read_text().strip()
                            # e.g. "4: ACTIVE"
                            info[f"{dp}/port_{port_num}_state"] = raw.split(":")[-1].strip()
                        except Exception:
                            pass

    except Exception:
        pass

    # Try ibstat for a more readable summary
    try:
        result = subprocess.run(
            ["ibstat", "-s"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            info[f"{prefix}/ibstat_summary"] = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------

def _get_package_versions() -> dict:
    """Get versions of key packages."""
    info: dict = {}
    prefix = "setup/package_versions"

    # Python version
    info[f"{prefix}/python"] = sys.version.split()[0]

    # PyTorch and CUDA
    try:
        import torch
        info[f"{prefix}/pytorch"] = torch.__version__
        info[f"{prefix}/cuda"] = torch.version.cuda or "N/A"
        info[f"{prefix}/cudnn"] = (
            str(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else "N/A"
        )
    except Exception:
        pass

    # NCCL
    try:
        import torch
        if hasattr(torch.cuda, "nccl") and hasattr(torch.cuda.nccl, "version"):
            nccl_ver = torch.cuda.nccl.version()
            if isinstance(nccl_ver, tuple):
                info[f"{prefix}/nccl"] = ".".join(str(v) for v in nccl_ver)
            else:
                info[f"{prefix}/nccl"] = str(nccl_ver)
    except Exception:
        pass

    # vLLM
    try:
        import vllm
        info[f"{prefix}/vllm"] = vllm.__version__
    except Exception:
        pass

    # Transformers
    try:
        import transformers
        info[f"{prefix}/transformers"] = transformers.__version__
    except Exception:
        pass

    # Datasets
    try:
        import datasets
        info[f"{prefix}/datasets"] = datasets.__version__
    except Exception:
        pass

    # wandb
    try:
        import wandb
        info[f"{prefix}/wandb"] = wandb.__version__
    except Exception:
        pass

    # Flash Attention
    try:
        import flash_attn
        info[f"{prefix}/flash_attn"] = flash_attn.__version__
    except Exception:
        pass

    # Flash Attention 3 (Hopper)
    try:
        import flash_attn_3
        info[f"{prefix}/flash_attn_3"] = flash_attn_3.__version__
    except Exception:
        pass

    # NVIDIA Apex
    try:
        import apex
        info[f"{prefix}/apex"] = apex.__version__
    except Exception:
        pass

    # Transformer Engine
    try:
        import transformer_engine
        info[f"{prefix}/transformer_engine"] = transformer_engine.__version__
    except Exception:
        pass

    # Megatron-Core (if present) – import can trigger CUDA JIT compilation
    # that may fail with arbitrary errors, so catch broadly.
    try:
        import megatron.core
        info[f"{prefix}/megatron_core"] = megatron.core.__version__
    except Exception:
        pass

    # DeepSpeed (if present)
    try:
        import deepspeed
        info[f"{prefix}/deepspeed"] = deepspeed.__version__
    except Exception:
        pass

    # Ray
    try:
        import ray
        info[f"{prefix}/ray"] = ray.__version__
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# OS
# ---------------------------------------------------------------------------

def _get_os_info() -> dict:
    """Get operating system information."""
    info: dict = {}
    prefix = "setup/os"

    info[f"{prefix}/system"] = platform.system()
    info[f"{prefix}/release"] = platform.release()
    info[f"{prefix}/version"] = platform.version()
    info[f"{prefix}/platform"] = platform.platform()

    return info


# ---------------------------------------------------------------------------
# Container detection
# ---------------------------------------------------------------------------

def _get_container_info() -> dict:
    """Detect whether running inside a container (Docker, Kubernetes, etc.)."""
    info: dict = {}
    prefix = "setup/container"

    is_container = False
    runtime = ""

    # Docker: /.dockerenv file
    if Path("/.dockerenv").exists():
        is_container = True
        runtime = "docker"

    # Kubernetes: SERVICE_HOST env var or /var/run/secrets/kubernetes.io
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        is_container = True
        runtime = "kubernetes"
    elif Path("/var/run/secrets/kubernetes.io").exists():
        is_container = True
        runtime = "kubernetes"

    # cgroup check (works for Docker/containerd)
    if not is_container:
        try:
            with open("/proc/1/cgroup", "r") as f:
                cgroup_content = f.read()
            if "docker" in cgroup_content or "containerd" in cgroup_content:
                is_container = True
                runtime = "docker"
            elif "kubepods" in cgroup_content:
                is_container = True
                runtime = "kubernetes"
        except Exception:
            pass

    # cgroup v2 check
    if not is_container:
        try:
            with open("/proc/self/mountinfo", "r") as f:
                mountinfo = f.read()
            if "docker" in mountinfo or "containerd" in mountinfo:
                is_container = True
                runtime = "docker"
        except Exception:
            pass

    info[f"{prefix}/is_container"] = is_container
    if runtime:
        info[f"{prefix}/runtime"] = runtime

    # Container memory/CPU limits (cgroup)
    if is_container:
        info.update(_get_cgroup_limits())

    return info


def _get_cgroup_limits() -> dict:
    """Read cgroup resource limits (useful for containers)."""
    info: dict = {}
    prefix = "setup/container"

    # Memory limit (cgroup v1 and v2)
    for path in [
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",      # v1
        "/sys/fs/cgroup/memory.max",                         # v2
    ]:
        try:
            raw = Path(path).read_text().strip()
            if raw != "max" and raw != "9223372036854771712":
                limit_bytes = int(raw)
                info[f"{prefix}/memory_limit_gb"] = round(limit_bytes / (1024**3), 2)
                break
        except Exception:
            continue

    # CPU quota (cgroup v1)
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if quota > 0 and period > 0:
            info[f"{prefix}/cpu_limit_cores"] = round(quota / period, 2)
    except Exception:
        pass

    # CPU quota (cgroup v2)
    try:
        raw = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
        parts = raw.split()
        if len(parts) == 2 and parts[0] != "max":
            quota = int(parts[0])
            period = int(parts[1])
            if quota > 0 and period > 0:
                info[f"{prefix}/cpu_limit_cores"] = round(quota / period, 2)
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_if(info: dict, key: str, value: str | None) -> None:
    """Set key if value is non-empty and not '[N/A]'."""
    if value and value.strip() and value.strip() not in ("[N/A]", "N/A", "[Not Supported]"):
        info[key] = value.strip()


def _set_int(info: dict, key: str, value: str | None) -> None:
    """Set key as int if parseable."""
    if not value:
        return
    try:
        info[key] = int(value.strip())
    except (ValueError, TypeError):
        pass


def _set_float(info: dict, key: str, value: str | None) -> None:
    """Set key as float (rounded to 2 decimals) if parseable."""
    if not value:
        return
    try:
        info[key] = round(float(value.strip()), 2)
    except (ValueError, TypeError):
        pass


def _set_float_mib_to_gb(info: dict, key: str, value: str | None) -> None:
    """Set key as float GiB (converted from MiB) if parseable."""
    if not value:
        return
    try:
        info[key] = round(float(value.strip()) / 1024, 2)
    except (ValueError, TypeError):
        pass
