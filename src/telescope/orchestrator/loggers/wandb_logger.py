"""
W&B logging manager for the orchestrator.

The orchestrator owns the wandb run and manages all logging:
- Scalar metrics (loss, rewards, etc.) are read from trainer files and logged
- Events (GPU timeline, orchestrator events) are logged via EventLogger to zip archives
- Rollouts are logged to parquet files per training step
- System metrics (GPU, CPU) are collected by SystemMetricsLogger
- vLLM metrics (requests, KV cache, latencies) are collected by VllmMetricsLogger

All event and metrics data is consolidated into unified zip files in the events/ folder:
- events/tail.zip: Last 60 seconds (orchestrator, trainer, inference, gpu, cpu, vllm parquet files)
- events/block_live.zip: Current 30-minute block
- events/block_*.zip: Finalized 30-minute blocks
"""
import fnmatch
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import zipfile

from telescope import __version__, schema_version, table_schema_versions
from telescope.utils import config
from telescope.utils.tlog import get_logger
from telescope.orchestrator.loggers.event_logger import EventLogger
from telescope.orchestrator.loggers.system_metrics_logger import (
    CpuMetricSample,
    GpuMetricSample,
    SystemMetricsLogger,
)
from telescope.orchestrator.loggers.vllm_metrics_logger import VllmMetricsLogger

_log = get_logger("orchestrator")

def _get_required_positive_int_config(name: str) -> int:
    raw_value = getattr(config.cfg, name, None)
    if raw_value is None:
        raise ValueError(f"{name} must be set in `telescope.config`.")

    value = int(raw_value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


def _get_git_commit() -> str | None:
    """Get the short git commit hash of the current HEAD, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_repo_root() -> Path:
    """Resolve the repository root used for code snapshot uploads."""
    default_root = Path(config.__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=default_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_root = Path(result.stdout.strip())
            if git_root.exists():
                return git_root
    except Exception:
        pass
    return default_root


class WandbLogger:
    """
    Wandb logging manager for the orchestrator.
    
    The orchestrator owns the wandb run. The trainer writes metrics to files,
    and the orchestrator reads them and logs to wandb.
    """

    def __init__(self):
        self.run = None
        self.event_logger = EventLogger()
        self.system_metrics_logger = SystemMetricsLogger(
            collection_interval_seconds=float(
                config.cfg.system_metrics_collection_interval_seconds
            )
        )
        self.vllm_metrics_logger = VllmMetricsLogger()
        self.last_logged_step = -1
        self.start_time = time.time()
        self.num_trainer_ranks = _get_required_positive_int_config("trainer_num_workers")
        self.driver_node_id: int = -1
        self.driver_node_ip: str = ""
        self.driver_hostname: str = ""
        self.driver_ray_node_id: str = ""
        self.setup: dict = {}
        self._node_identity_by_id: dict[int, dict] = {}
        self._node_id_by_node_ip: dict[str, int] = {}
        self._node_id_by_ray_node_id: dict[str, int] = {}
        self._inference_server_infos: list[dict] = []
        self._inference_server_by_idx: dict[int, dict] = {}
        self._inference_server_by_url: dict[str, dict] = {}
        self._trainer_runtime_by_rank: dict[int, dict] = {}

    def _upload_code_snapshot(self):
        """
        Upload a filtered zip of source files to W&B.

        The snapshot includes all files except those matching exclude patterns
        or exceeding the per-file size cap. Patterns are matched against
        individual directory and file names using fnmatch (supports *, ?, []).
        """
        if self.run is None or not config.cfg.wandb_upload_code:
            return

        max_file_size_mb = float(config.cfg.wandb_code_max_file_size_mb)
        if max_file_size_mb <= 0:
            _log.warning("Skipping code upload: WANDB_CODE_MAX_FILE_SIZE_MB must be > 0")
            return
        max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

        exclude_patterns = list(config.cfg.wandb_code_exclude_patterns)

        def _is_excluded(name: str) -> bool:
            return any(fnmatch.fnmatch(name, pat) for pat in exclude_patterns)

        repo_root = _get_repo_root()
        included_files: list[tuple[Path, str, int]] = []
        skipped_large: list[dict[str, int | str]] = []

        for dirpath, dirnames, filenames in os.walk(repo_root):
            dirnames[:] = sorted(d for d in dirnames if not _is_excluded(d))

            for filename in sorted(filenames):
                if _is_excluded(filename):
                    continue

                file_path = Path(dirpath) / filename
                if file_path.is_symlink():
                    continue

                try:
                    size_bytes = file_path.stat().st_size
                except OSError:
                    continue

                rel_path = file_path.relative_to(repo_root).as_posix()
                if size_bytes > max_file_size_bytes:
                    skipped_large.append({"path": rel_path, "size_bytes": size_bytes})
                    continue

                included_files.append((file_path, rel_path, size_bytes))

        if not included_files:
            _log.warning(f"No files matched snapshot filters in {repo_root}")
            return

        archive_relpath = "code/source.zip"
        metadata_relpath = "code/metadata.json"
        archive_path = Path(self.run.dir) / archive_relpath
        metadata_path = Path(self.run.dir) / metadata_relpath
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        total_code_bytes = sum(size for _, _, size in included_files)
        manifest = {
            "repo_root": str(repo_root),
            "max_file_size_bytes": max_file_size_bytes,
            "exclude_patterns": sorted(exclude_patterns),
            "included_file_count": len(included_files),
            "included_total_bytes": total_code_bytes,
            "skipped_too_large_count": len(skipped_large),
            "skipped_too_large_files": skipped_large[:200],
        }

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for abs_path, rel_path, _ in included_files:
                zf.write(abs_path, arcname=rel_path)
        metadata_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        self.run.save(str(archive_path), base_path=self.run.dir, policy="now")
        self.run.save(str(metadata_path), base_path=self.run.dir, policy="now")
        self.run.summary.update(
            {
                "code/archive_path": archive_relpath,
                "code/metadata_path": metadata_relpath,
                "code/file_count": len(included_files),
                "code/total_bytes": total_code_bytes,
                "code/skipped_too_large_count": len(skipped_large),
                "code/max_file_size_bytes": max_file_size_bytes,
            }
        )
        _log.info(
            f"Uploaded code snapshot to W&B ({len(included_files)} files, "
            f"{total_code_bytes / (1024 * 1024):.2f} MiB)"
        )

    def initialize(self):
        """Initialize wandb run if enabled."""
        if not config.cfg.use_wandb:
            return

        import wandb

        # Keep wandb run config in sync with telescope config automatically.
        run_config = config.cfg.model_dump()

        # Parse tags from config or environment variable
        tags = list(config.cfg.wandb_tags) if config.cfg.wandb_tags else []
        env_tags = os.environ.get("WANDB_TAGS", "")
        if env_tags:
            # Override with environment variable if set (comma-separated)
            tags = [tag.strip() for tag in env_tags.split(",") if tag.strip()]

        # Always append schema version, per-table schema versions, package version, and git commit to tags
        tags.append(f"schema_version:{schema_version}")
        for table_name, table_ver in table_schema_versions.items():
            tags.append(f"schema_version_{table_name}:{table_ver}")
        tags.append(f"version:{__version__}")
        git_commit = _get_git_commit()
        if git_commit:
            tags.append(f"commit:{git_commit}")
            run_config["git_commit"] = git_commit
        run_config["version"] = __version__
        run_config["schema_version"] = schema_version
        run_config["table_schema_versions"] = dict(table_schema_versions)

        init_kwargs = {
            "project": os.environ.get("WANDB_PROJECT", config.cfg.wandb_project),
            "name": os.environ.get("WANDB_RUN_NAME", config.cfg.wandb_run_name),
            "config": run_config,
        }

        # Add tags if provided
        if tags:
            init_kwargs["tags"] = tags

        # Suppress wandb's stdout banner in non-debug mode
        from telescope.utils.tlog import is_debug_mode
        if not is_debug_mode():
            init_kwargs["settings"] = wandb.Settings(silent=True)

        self.run = wandb.init(**init_kwargs)
        try:
            self._upload_code_snapshot()
        except Exception as e:
            _log.error(f"Failed to upload code snapshot: {e}")
        
        # Initialize all loggers
        self.event_logger.initialize(self.run)
        self.driver_hostname = socket.gethostname()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            self.driver_node_ip = s.getsockname()[0]
            s.close()
        except Exception:
            self.driver_node_ip = socket.gethostbyname(self.driver_hostname)
        self.system_metrics_logger.initialize(
            self.run,
            node_id=self.driver_node_id,
            node_ip=self.driver_node_ip,
            hostname=self.driver_hostname,
            ray_node_id=self.driver_ray_node_id,
        )
        self.vllm_metrics_logger.initialize(self.run)
        
        # Connect metrics loggers to event logger for consolidated uploads
        self.event_logger.set_metrics_loggers(
            system_metrics_logger=self.system_metrics_logger,
            vllm_metrics_logger=self.vllm_metrics_logger,
        )

        run_url = getattr(self.run, "url", None) or ""
        if run_url:
            _log.info(f"W&B run: {self.run.name} ({run_url})")
        else:
            _log.info(f"W&B run: {self.run.name}")

    def update_setup(self, setup: dict):
        """Store full cluster setup JSON in W&B summary."""
        if self.run is None or not setup:
            return

        self.setup = dict(setup)
        setup_json = json.dumps(self.setup, sort_keys=True)
        self.run.summary["setup"] = setup_json
        self.run.summary["setup_schema_version"] = str(self.setup.get("schema_version", ""))

        self._rebuild_node_identity_indices()

        # Keep lookup maps in sync.
        self.set_inference_servers(self._extract_inference_servers())
        self.set_trainer_runtime_infos(self._extract_trainer_ranks())

        driver_info = self._extract_driver_info()
        (
            self.driver_node_id,
            self.driver_node_ip,
            self.driver_hostname,
            self.driver_ray_node_id,
        ) = self._resolve_node_identity(
            node_id=self._as_int(driver_info.get("node_id", -1)),
            node_ip=str(driver_info.get("node_ip", "")),
            hostname=str(driver_info.get("hostname", "")),
            ray_node_id=str(driver_info.get("ray_node_id", "")),
        )
        self.system_metrics_logger.set_node_identity(
            node_id=self.driver_node_id,
            node_ip=self.driver_node_ip,
            hostname=self.driver_hostname,
            ray_node_id=self.driver_ray_node_id,
        )

    def _extract_inference_servers(self) -> list[dict]:
        """
        Extract inference server list from the setup payload.

        Schema v6: servers are derived from ``inference.nodes[].gpus[]``.
        Multiple GPU entries may share a ``server_idx`` (TP > 1); we
        deduplicate to one entry per logical server and enrich with the
        parent node's identity so downstream code can use it.
        """
        inference_section = dict(self.setup.get("inference") or {})
        seen_server_idxs: set[int] = set()
        servers: list[dict] = []
        for node in inference_section.get("nodes", []) or []:
            node_id = self._as_int(node.get("node_id", -1))
            node_ip = str(node.get("ip") or node.get("node_ip") or "")
            hostname = str(node.get("hostname", ""))
            ray_node_id = str(node.get("ray_node_id", ""))
            for gpu in node.get("gpus", []) or []:
                sidx = self._as_int(gpu.get("server_idx", -1))
                if sidx in seen_server_idxs:
                    continue
                seen_server_idxs.add(sidx)
                normalized = dict(gpu)
                normalized.setdefault("node_id", node_id)
                normalized.setdefault("node_ip", node_ip)
                normalized.setdefault("hostname", hostname)
                normalized.setdefault("ray_node_id", ray_node_id)
                servers.append(normalized)
        return sorted(servers, key=lambda item: self._as_int(item.get("server_idx", -1)))

    def _extract_trainer_ranks(self) -> list[dict]:
        """
        Extract trainer rank list from the setup payload.

        Schema v6: ranks live under ``trainer.nodes[].gpus[]`` — enriched
        with parent node identity.
        """
        trainer_section = dict(self.setup.get("trainer") or {})
        ranks: list[dict] = []
        for node in trainer_section.get("nodes", []) or []:
            node_id = self._as_int(node.get("node_id", -1))
            node_ip = str(node.get("ip") or node.get("node_ip") or "")
            hostname = str(node.get("hostname", ""))
            ray_node_id = str(node.get("ray_node_id", ""))
            for gpu in node.get("gpus", []) or []:
                normalized = dict(gpu)
                normalized.setdefault("node_id", node_id)
                normalized.setdefault("node_ip", node_ip)
                normalized.setdefault("hostname", hostname)
                normalized.setdefault("ray_node_id", ray_node_id)
                ranks.append(normalized)
        return sorted(ranks, key=lambda item: self._as_int(item.get("rank", -1)))

    def _extract_driver_info(self) -> dict:
        """
        Find the driver node's identity from the setup payload.

        Schema v6: physical nodes live under ``cluster.nodes[]``.
        Checks ``cluster.driver_node_id`` first, then falls back to
        looking for a node with ``is_driver == true``.
        """
        cluster = dict(self.setup.get("cluster") or {})
        driver_node_id = self._as_int(cluster.get("driver_node_id", -1))
        cluster_nodes = cluster.get("nodes", []) or []

        for node in cluster_nodes:
            nid = self._as_int(node.get("node_id", -1))
            if nid >= 0 and nid == driver_node_id:
                return {
                    "node_id": nid,
                    "node_ip": str(node.get("ip") or node.get("node_ip") or ""),
                    "hostname": str(node.get("hostname", "")),
                    "ray_node_id": str(node.get("ray_node_id", "")),
                }

        # Fallback: find node flagged as driver.
        for node in cluster_nodes:
            if bool(node.get("is_driver", False)):
                return {
                    "node_id": self._as_int(node.get("node_id", -1)),
                    "node_ip": str(node.get("ip") or node.get("node_ip") or ""),
                    "hostname": str(node.get("hostname", "")),
                    "ray_node_id": str(node.get("ray_node_id", "")),
                }
        return {}

    def _rebuild_node_identity_indices(self):
        self._node_identity_by_id = {}
        self._node_id_by_node_ip = {}
        self._node_id_by_ray_node_id = {}

        # Schema v6: physical nodes live under ``cluster.nodes[]``.
        cluster = dict(self.setup.get("cluster") or {})
        for node in cluster.get("nodes", []) or []:
            node_id = self._as_int(node.get("node_id", -1))
            node_ip = str(node.get("ip") or node.get("node_ip") or "")
            hostname = str(node.get("hostname", ""))
            ray_node_id = str(node.get("ray_node_id", ""))
            identity = {
                "node_id": node_id,
                "node_ip": node_ip,
                "hostname": hostname,
                "ray_node_id": ray_node_id,
            }
            if node_id >= 0:
                self._node_identity_by_id[node_id] = identity
            if node_ip:
                self._node_id_by_node_ip[node_ip] = node_id
            if ray_node_id:
                self._node_id_by_ray_node_id[ray_node_id] = node_id

    def _resolve_node_identity(
        self,
        node_id: int = -1,
        node_ip: str = "",
        hostname: str = "",
        ray_node_id: str = "",
    ) -> tuple[int, str, str, str]:
        resolved_node_id = self._as_int(node_id, default=-1)
        resolved_node_ip = str(node_ip or "")
        resolved_hostname = str(hostname or "")
        resolved_ray_node_id = str(ray_node_id or "")

        if resolved_node_id < 0 and resolved_ray_node_id:
            resolved_node_id = self._as_int(
                self._node_id_by_ray_node_id.get(resolved_ray_node_id, -1),
                default=-1,
            )
        if resolved_node_id < 0 and resolved_node_ip:
            resolved_node_id = self._as_int(
                self._node_id_by_node_ip.get(resolved_node_ip, -1),
                default=-1,
            )

        identity = self._node_identity_by_id.get(resolved_node_id, {})
        if identity:
            resolved_node_ip = str(identity.get("node_ip") or resolved_node_ip)
            resolved_hostname = str(identity.get("hostname") or resolved_hostname)
            resolved_ray_node_id = str(identity.get("ray_node_id") or resolved_ray_node_id)

        return resolved_node_id, resolved_node_ip, resolved_hostname, resolved_ray_node_id

    def update_env_summary(self, scheduler):
        """
        Update wandb run summary with environment details (JSON string).
        
        Args:
            scheduler: A Scheduler or MultiEnvScheduler instance.
        """
        if self.run is None:
            return

        from telescope.orchestrator.scheduler import MultiEnvScheduler

        if isinstance(scheduler, MultiEnvScheduler):
            envs = scheduler.environments
        else:
            envs = [scheduler.environment]

        env_details = []
        for env in envs:
            info = {
                "name": env.name,
                "is_multi_turn": env.is_multi_turn,
                "dataset_size": len(env),
            }
            if env.is_multi_turn:
                info["max_turns"] = getattr(env, "max_turns", None)
            if env.metrics_ranges:
                info["metrics_ranges"] = env.metrics_ranges
            env_details.append(info)

        self.run.summary["env_details"] = json.dumps(env_details)
        _log.debug(f"Environment summary logged to wandb: {env_details}")

    def update_eval_env_summary(self, eval_runner, eval_configs):
        """Update wandb run summary with eval environment details + pass@k metric ranges."""
        if self.run is None:
            return

        eval_env_details = []
        for ec in eval_configs:
            env_entry = eval_runner.envs.get(ec.name)
            if env_entry is None:
                continue
            env, samples = env_entry
            info = {
                "name": ec.name,
                "is_multi_turn": env.is_multi_turn,
                "dataset_size": len(samples),
            }
            base_ranges = dict(env.metrics_ranges) if env.metrics_ranges else {}
            pk = ec.pass_k
            for metric_name in (pk.at_k.metrics or []):
                for k in (pk.at_k.k or []):
                    base_ranges[f"pass@{k}/{metric_name}"] = {"min": 0, "max": 1}
            for metric_name in (pk.pow_k.metrics or []):
                for k in (pk.pow_k.k or []):
                    base_ranges[f"pass^{k}/{metric_name}"] = {"min": 0, "max": 1}
            if base_ranges:
                info["metrics_ranges"] = base_ranges
            eval_env_details.append(info)

        self.run.summary["eval_env_details"] = json.dumps(eval_env_details)
        _log.info(f"Eval environment summary logged to wandb: {eval_env_details}")

    def update_model_architecture_summary(self):
        """Log the full model architecture string to W&B summary.

        Instantiates the model on the meta device (zero memory) to obtain
        the ``nn.Module`` repr, following the same approach as the
        ``save_model_architecture_to_file`` helper in HF Transformers.
        """
        if self.run is None:
            return

        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM

            hf_config = AutoConfig.from_pretrained(
                config.cfg.model, trust_remote_code=True
            )
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(hf_config)
            self.run.summary["model_architecture"] = str(model)
            _log.debug("Model architecture logged to wandb summary")
        except Exception as exc:
            _log.warning(f"Failed to log model architecture: {exc}")

    def log_orchestrator_timeline_event(self, event_type: str, timestamp: float = None, step: int = -1):
        """
        Log an instant event to the timeline (orchestrator events).
        
        Used for orchestrator lifecycle events like:
        - inference_processes_start
        - trainer_process_start
        - inference_servers_ready
        - trainer_ready
        - weight_update
        - save_batch
        - inference_call
        """
        if not config.cfg.use_wandb:
            return

        ts = timestamp if timestamp is not None else time.time()
        self.event_logger.log_instant_event(
            event_type=event_type,
            source="orchestrator",
            timestamp=ts,
            step=step,
            node_id=self.driver_node_id,
            node_ip=self.driver_node_ip,
            hostname=self.driver_hostname,
            ray_node_id=self.driver_ray_node_id,
        )

    def log_inference_event(
        self,
        event_type: str,
        server: int,
        start_time: float,
        end_time: float,
        node_id: int = -1,
        node_ip: str = "",
        hostname: str = "",
        ray_node_id: str = "",
        tp_group_id: int = -1,
        tp_size: int = 1,
        prompt_tokens: int = 0,
        rollout_tokens: int = 0,
        group_id: int = -1,
        sample_id: int = -1,
        vllm_request_id: str = "",
        queue_time: float = 0.0,
        time_to_first_token: float = 0.0,
        prefill_time: float = 0.0,
        decode_time: float = 0.0,
        inference_time: float = 0.0,
        e2e_latency: float = 0.0,
        vllm_max_tokens: int = 0,
        is_eval: bool = False,
        is_canceled: bool = False,
        compute_reward_time: float = 0.0,
        step: int = -1,
        off_policy_steps: int = 0,
        server_lane: int = -1,
    ):
        """
        Log an inference event (request or weight broadcast).

        Args:
            event_type: "request" or "weight_broadcast"
            server: Server index (0, 1, ...)
            start_time: Event start timestamp
            end_time: Event end timestamp
            prompt_tokens: Number of prompt tokens (for requests)
            rollout_tokens: Number of generated tokens (for requests)
            group_id: Request group ID (shared by all rollouts in a group)
            sample_id: Run-wide unique sample index (requests only)
            vllm_request_id: vLLM request ID (e.g. "cmpl-abc123")
            queue_time: Time in vLLM queue (from OTLP span)
            time_to_first_token: TTFT from vLLM (from OTLP span)
            prefill_time: Model prefill time (from OTLP span)
            decode_time: Model decode time (from OTLP span)
            inference_time: prefill + decode (from OTLP span)
            e2e_latency: End-to-end latency inside vLLM (from OTLP span)
            vllm_max_tokens: max_tokens param for this request
            is_eval: Whether this is an eval request (not training)
            is_canceled: Whether this request was cancelled before completing
            compute_reward_time: Time for compute_reward or compute_eval_metrics (seconds)
            step: Training step (populated for weight_broadcast events, -1 for requests)
            off_policy_steps: Number of weight updates since this rollout was dispatched
        """
        if not config.cfg.use_wandb:
            return

        server_idx = self._as_int(server, default=-1)
        server_info = self._inference_server_by_idx.get(server_idx, {})
        (
            resolved_node_id,
            resolved_node_ip,
            resolved_hostname,
            resolved_ray_node_id,
        ) = self._resolve_node_identity(
            node_id=self._as_int(node_id, default=-1),
            node_ip=str(node_ip or server_info.get("node_ip") or ""),
            hostname=str(hostname or server_info.get("hostname") or ""),
            ray_node_id=str(ray_node_id or server_info.get("ray_node_id") or ""),
        )
        resolved_tp_group_id = self._as_int(
            tp_group_id
            if tp_group_id != -1
            else server_info.get("tp_group_id", server_idx if server_idx >= 0 else -1)
        )
        server_tp_size = self._as_int(server_info.get("tp_size", 1), default=1)
        resolved_tp_size = self._as_int(
            tp_size
            if tp_size > 0
            else max(1, server_tp_size),
            default=1,
        )

        self.event_logger.log_inference_event(
            event_type=event_type,
            server=server_idx,
            start_time=start_time,
            end_time=end_time,
            node_id=resolved_node_id,
            node_ip=resolved_node_ip,
            hostname=resolved_hostname,
            ray_node_id=resolved_ray_node_id,
            tp_group_id=resolved_tp_group_id,
            tp_size=resolved_tp_size,
            prompt_tokens=prompt_tokens,
            rollout_tokens=rollout_tokens,
            group_id=group_id,
            sample_id=sample_id,
            vllm_request_id=vllm_request_id,
            queue_time=queue_time,
            time_to_first_token=time_to_first_token,
            prefill_time=prefill_time,
            decode_time=decode_time,
            inference_time=inference_time,
            e2e_latency=e2e_latency,
            vllm_max_tokens=vllm_max_tokens,
            is_eval=is_eval,
            is_canceled=is_canceled,
            compute_reward_time=compute_reward_time,
            step=step,
            off_policy_steps=off_policy_steps,
            server_lane=server_lane,
        )

    def log_discarded_rollout(
        self,
        discard_reason: str,
        trainer_step: int,
        inference_step: int,
        group_id: int,
        sample_idx: int,
        prompt: str,
        turns: list[dict] | None,
        reward: float,
        advantage: float,
        env: str = "",
        sample_metrics: dict[str, float] | None = None,
        golden_answers: dict[str, str | None] | None = None,
        info_turns: list[dict] | None = None,
        tokens_prompt: int = 0,
        system_prompt: str = "",
        tokens_system_prompt: int = 0,
        total_tokens: int = 0,
        raw_string: str = "",
        compute_reward_time: float = 0.0,
    ):
        """
        Log a discarded rollout sample.

        These are rollouts that were not sent to the trainer due to:
        - max_async: Async level exceeded the maximum allowed
        - zero_advantage: All samples in the group had zero advantage

        Args:
            discard_reason: Why the rollout was discarded (e.g. "max_async", "zero_advantage")
            trainer_step: Current trainer step at the time of discarding
            inference_step: Current inference step (what step this rollout would have been for)
            group_id: Request group ID (shared by all completions with same prompt)
            sample_idx: Run-wide unique sample index (starts at 0, grows throughout run)
            prompt: Input prompt text (for the prompts_discarded table)
            turns: List of turn dicts with keys:
                   - "turn_order": int (0, 1, 2, ...)
                   - "turn_type": str ("model" or env-provided type)
                   - "content": str (the text content)
                   - "tokens": int (number of tokens for this turn)
            reward: Total reward
            advantage: Computed advantage
            env: Environment name (e.g. "countdown", "coding")
            sample_metrics: Dict of per-sample metric names to float values
            golden_answers: Dict mapping golden answer keys to their values
            info_turns: List of per-turn text info dicts (see EventLogger.log_rollout)
            tokens_prompt: Number of prompt tokens (stored in prompts_discarded table)
            system_prompt: The system message (if any)
            tokens_system_prompt: Number of tokens in the system message
            total_tokens: Total tokens that would have been passed to trainer
            raw_string: Decoded raw input that would have been passed to trainer
            compute_reward_time: Time in seconds for compute_reward() call
        """
        if not config.cfg.use_wandb:
            return

        self.event_logger.log_discarded_rollout(
            discard_reason=discard_reason,
            trainer_step=trainer_step,
            inference_step=inference_step,
            group_id=group_id,
            sample_idx=sample_idx,
            prompt=prompt,
            turns=turns,
            reward=reward,
            advantage=advantage,
            env=env,
            sample_metrics=sample_metrics,
            golden_answers=golden_answers,
            info_turns=info_turns,
            tokens_prompt=tokens_prompt,
            system_prompt=system_prompt,
            tokens_system_prompt=tokens_system_prompt,
            total_tokens=total_tokens,
            raw_string=raw_string,
            compute_reward_time=compute_reward_time,
        )

    def set_inference_servers(self, server_infos: list[dict]):
        """
        Register full inference server metadata (url, node, tp mapping).
        """
        raw_infos = sorted(server_infos or [], key=lambda item: int(item.get("server_idx", -1)))
        infos = []
        for info in raw_infos:
            node_id, node_ip, hostname, ray_node_id = self._resolve_node_identity(
                node_id=self._as_int(info.get("node_id", -1)),
                node_ip=str(info.get("node_ip", "")),
                hostname=str(info.get("hostname", "")),
                ray_node_id=str(info.get("ray_node_id", "")),
            )
            normalized = dict(info)
            normalized["node_id"] = node_id
            normalized["node_ip"] = node_ip
            normalized["hostname"] = hostname
            normalized["ray_node_id"] = ray_node_id
            infos.append(normalized)

        self._inference_server_infos = infos
        self._inference_server_by_idx = {
            int(info.get("server_idx", -1)): info for info in infos
        }
        self._inference_server_by_url = {
            str(info.get("url", "")): info for info in infos if info.get("url")
        }
        if infos:
            self.vllm_metrics_logger.set_server_infos(infos)
        else:
            self.vllm_metrics_logger.set_server_urls([])

    def set_trainer_runtime_infos(self, runtime_infos: list[dict]):
        """Register trainer rank->node/local_rank mapping for metric/event enrichment."""
        by_rank: dict[int, dict] = {}
        for info in (runtime_infos or []):
            if info.get("rank") is None:
                continue
            rank = int(info.get("rank", -1))
            node_id, node_ip, hostname, ray_node_id = self._resolve_node_identity(
                node_id=self._as_int(info.get("node_id", -1)),
                node_ip=str(info.get("node_ip", "")),
                hostname=str(info.get("hostname", "")),
                ray_node_id=str(info.get("ray_node_id", "")),
            )
            normalized = dict(info)
            normalized["node_id"] = node_id
            normalized["node_ip"] = node_ip
            normalized["hostname"] = hostname
            normalized["ray_node_id"] = ray_node_id
            by_rank[rank] = normalized
        self._trainer_runtime_by_rank = by_rank

    @staticmethod
    def _as_int(value, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _resolve_trainer_location(
        self,
        rank: int,
        local_rank: int = -1,
        node_id: int = -1,
        node_ip: str = "",
        hostname: str = "",
        ray_node_id: str = "",
    ) -> tuple[int, int, int, str, str, str]:
        """Resolve trainer rank → (local_rank, node_id, gpu_index, node_ip, hostname, ray_node_id)."""
        try:
            rank_int = int(rank)
        except (TypeError, ValueError):
            rank_int = -1
        info = self._trainer_runtime_by_rank.get(rank_int, {})
        try:
            resolved_local_rank = int(local_rank)
        except (TypeError, ValueError):
            resolved_local_rank = -1
        if resolved_local_rank < 0:
            resolved_local_rank = int(info.get("local_rank", -1))
        resolved_node_id, resolved_node_ip, resolved_hostname, resolved_ray_node_id = (
            self._resolve_node_identity(
                node_id=self._as_int(node_id, default=self._as_int(info.get("node_id", -1))),
                node_ip=str(node_ip or info.get("node_ip", "")),
                hostname=str(hostname or info.get("hostname", "")),
                ray_node_id=str(ray_node_id or info.get("ray_node_id", "")),
            )
        )
        # gpu_index = physical GPU index on the node, parsed from
        # CUDA_VISIBLE_DEVICES at the rank's local_rank offset.
        resolved_gpu_index = self._as_int(info.get("gpu_index", -1))
        if resolved_gpu_index < 0:
            cvd = (info.get("env") or {}).get("CUDA_VISIBLE_DEVICES", "")
            if cvd:
                parts = [s.strip() for s in cvd.split(",") if s.strip().isdigit()]
                lr = max(0, resolved_local_rank)
                if lr < len(parts):
                    resolved_gpu_index = int(parts[lr])
                elif parts:
                    resolved_gpu_index = int(parts[0])
        return (
            resolved_local_rank,
            resolved_node_id,
            resolved_gpu_index,
            resolved_node_ip,
            resolved_hostname,
            resolved_ray_node_id,
        )

    def set_trainer_steps_done(self, trainer_steps_done: int):
        """Update trainer progress metadata used by event summary uploads."""
        self.event_logger.set_trainer_steps_done(trainer_steps_done)

    def log_trainer_metrics_payload(self, step: int, payload: dict) -> bool:
        """
        Log trainer metrics/events that are already aggregated in memory.

        This is used by Ray mode where trainer ranks return per-step payloads
        directly to the orchestrator instead of writing file markers.
        """
        if self.run is None:
            return False

        step_metrics = payload.get("step_metrics", [])
        for sm in step_metrics:
            self.event_logger.log_step_metric(
                step=sm.get("step", step),
                metric=sm.get("metric", ""),
                value=sm.get("value", 0.0),
                section=sm.get("section", ""),
                group=sm.get("group", ""),
            )

        # Log to wandb native charts for quick debugging via wandb UI
        if step_metrics:
            wandb_dict = {}
            for sm in step_metrics:
                parts = [p for p in (sm.get("section", ""), sm.get("group", ""), sm.get("metric", "")) if p]
                key = "/".join(parts) if parts else "unknown"
                wandb_dict[key] = sm.get("value", 0.0)
            self.run.log(wandb_dict, step=step)

        torch_memory_samples = payload.get("torch_memory_samples", [])
        torch_gpu_metrics = self._coerce_torch_memory_samples(torch_memory_samples)
        if torch_gpu_metrics:
            self.event_logger.add_gpu_metrics(torch_gpu_metrics)

        if config.cfg.track_gpu_events:
            timeline_events = payload.get("timeline_events", [])
            for event in timeline_events:
                rank = self._as_int(event.get("rank", 0), default=0)
                local_rank, node_id, gpu_index, node_ip, hostname, ray_node_id = self._resolve_trainer_location(
                    rank=rank,
                    local_rank=self._as_int(event.get("local_rank", -1)),
                    node_id=self._as_int(event.get("node_id", -1)),
                    node_ip=str(event.get("node_ip", "")),
                    hostname=str(event.get("hostname", "")),
                    ray_node_id=str(event.get("ray_node_id", "")),
                )
                self.event_logger.log_event(
                    event_type=event.get("event_type", "unknown"),
                    source="trainer",
                    step=step,
                    rank=rank,
                    local_rank=local_rank,
                    node_id=node_id,
                    gpu_index=gpu_index,
                    node_ip=node_ip,
                    hostname=hostname,
                    ray_node_id=ray_node_id,
                    start_time=event.get("start_time", 0),
                    end_time=event.get("end_time", 0),
                    parent=event.get("parent"),
                    depth=event.get("depth", 0),
                    microbatch=event.get("microbatch", -1),
                    minibatch=event.get("minibatch", -1),
                )

        self.last_logged_step = max(self.last_logged_step, step)
        return bool(step_metrics or torch_gpu_metrics)

    def log_rollout_reward_metrics(self, step: int, batch: list[dict]) -> None:
        """Log reward statistics from a rollout batch to wandb native charts."""
        if self.run is None:
            return

        all_rewards: list[float] = []
        for group in batch:
            rewards = group.get("rewards", [])
            all_rewards.extend(rewards)

        if not all_rewards:
            return

        mean = sum(all_rewards) / len(all_rewards)

        self.run.log({"rollout/reward_mean": mean}, step=step)

    def _coerce_torch_memory_samples(self, samples: list[dict]) -> list[GpuMetricSample]:
        gpu_metrics = []
        for sample in samples:
            try:
                rank = self._as_int(sample.get("rank", -1))
                local_rank, node_id, _gpu_index, node_ip, hostname, ray_node_id = self._resolve_trainer_location(
                    rank=rank,
                    local_rank=self._as_int(sample.get("local_rank", -1)),
                    node_id=self._as_int(sample.get("node_id", -1)),
                    node_ip=str(sample.get("node_ip", "")),
                    hostname=str(sample.get("hostname", "")),
                    ray_node_id=str(sample.get("ray_node_id", "")),
                )
                # gpu_index comes from the sample itself (torch already knows the device ordinal)
                gpu_metrics.append(
                    GpuMetricSample(
                        timestamp=float(sample["timestamp"]),
                        gpu_index=int(sample.get("gpu_index", _gpu_index)),
                        metric_name=str(sample["metric_name"]),
                        value=float(sample["value"]),
                        node_id=node_id,
                        node_ip=node_ip,
                        hostname=hostname,
                        ray_node_id=ray_node_id,
                        rank=rank,
                        local_rank=local_rank,
                        source=str(sample.get("source", "torch_trainer")),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return gpu_metrics

    def log_torch_memory_samples(self, samples: list[dict]) -> bool:
        """
        Log torch CUDA memory samples collected outside the per-step payload path.
        """
        if self.run is None:
            return False
        gpu_metrics = self._coerce_torch_memory_samples(samples)
        if not gpu_metrics:
            return False
        self.event_logger.add_gpu_metrics(gpu_metrics)
        return True

    def _collect_ray_cluster_infra_metrics(self):
        """
        Collect node-level GPU/CPU metrics across Ray nodes.

        This complements local SystemMetricsLogger samples so infra metrics stay
        multi-node aware (especially when workers are spread across hosts).
        """
        try:
            import ray
        except Exception:
            return
        if not ray.is_initialized():
            return

        try:
            from telescope.utils.ray_runtime.runtime import collect_cluster_infra_metrics_samples

            exclude = {self.driver_node_ip} if self.driver_node_ip else set()
            gpu_samples, cpu_samples = collect_cluster_infra_metrics_samples(
                exclude_node_ids=exclude,
            )
        except Exception as exc:
            _log.debug(f"Skipping Ray infra metrics collection: {exc}")
            return

        gpu_metrics: list[GpuMetricSample] = []
        for sample in gpu_samples:
            try:
                node_id, node_ip, hostname, ray_node_id = self._resolve_node_identity(
                    node_id=self._as_int(sample.get("node_id", -1)),
                    node_ip=str(sample.get("node_ip", "")),
                    hostname=str(sample.get("hostname", "")),
                    ray_node_id=str(sample.get("ray_node_id", "")),
                )
                gpu_metrics.append(
                    GpuMetricSample(
                        timestamp=float(sample["timestamp"]),
                        node_id=node_id,
                        node_ip=node_ip,
                        hostname=hostname,
                        ray_node_id=ray_node_id,
                        gpu_index=int(sample["gpu_index"]),
                        metric_name=str(sample["metric_name"]),
                        value=float(sample["value"]),
                        source="ray_node_system",
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        if gpu_metrics:
            self.event_logger.add_gpu_metrics(gpu_metrics)

        cpu_metrics: list[CpuMetricSample] = []
        for sample in cpu_samples:
            try:
                node_id, node_ip, hostname, ray_node_id = self._resolve_node_identity(
                    node_id=self._as_int(sample.get("node_id", -1)),
                    node_ip=str(sample.get("node_ip", "")),
                    hostname=str(sample.get("hostname", "")),
                    ray_node_id=str(sample.get("ray_node_id", "")),
                )
                cpu_metrics.append(
                    CpuMetricSample(
                        timestamp=float(sample["timestamp"]),
                        node_id=node_id,
                        node_ip=node_ip,
                        hostname=hostname,
                        ray_node_id=ray_node_id,
                        metric_name=str(sample["metric_name"]),
                        value=float(sample["value"]),
                        source="ray_node_system",
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        if cpu_metrics:
            self.event_logger.add_cpu_metrics(cpu_metrics)

    def log_pending_metrics(self):
        """
        Check for any metrics files that haven't been logged yet and log them.
        Called periodically by the orchestrator.
        """
        if self.run is None:
            return

        self._collect_ray_cluster_infra_metrics()

    async def start_event_upload_loop(self):
        """
        Start the background event upload and metrics collection loops.
        
        EventLogger handles all uploads (events + metrics consolidated in same zips).
        SystemMetricsLogger and VllmMetricsLogger only collect data.
        """
        if config.cfg.use_wandb and self.run is not None:
            # Start collection loops for system and vLLM metrics
            await self.system_metrics_logger.start()
            await self.vllm_metrics_logger.start()
            # Start consolidated upload loop (EventLogger pulls from all sources)
            await self.event_logger.start_upload_loop()

    async def stop_event_upload_loop(self):
        """Stop the background event upload and metrics collection loops."""
        # Stop upload loop first
        if self.event_logger._upload_task is not None:
            await self.event_logger.stop_upload_loop()
        # Then stop collection loops
        if self.system_metrics_logger._collection_task is not None:
            await self.system_metrics_logger.stop()
        if self.vllm_metrics_logger._collection_task is not None:
            await self.vllm_metrics_logger.stop()

    # ------------------------------------------------------------------
    # Eval logging
    # ------------------------------------------------------------------

    def log_eval_prompt(self, prompt):
        """Log an eval prompt to the event logger (deduplicated)."""
        try:
            self.event_logger.log_eval_prompt(prompt)
        except Exception as exc:
            _log.warning(f"Failed to log eval prompt: {exc}")

    def log_eval_rollout(self, rollout):
        """Log an eval rollout to the event logger."""
        try:
            self.event_logger.log_eval_rollout(rollout)
        except Exception as exc:
            _log.warning(f"Failed to log eval rollout: {exc}")

    def finish(self):
        """Finish the wandb run."""
        if self.run is not None:
            self.event_logger.finish()
            self.system_metrics_logger.finish()
            self.vllm_metrics_logger.finish()
            self.run.finish()
