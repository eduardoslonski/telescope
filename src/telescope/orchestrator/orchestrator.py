"""
Main orchestrator for RL post-training.

The orchestrator coordinates:
1. Inference servers (vLLM) - generate completions
2. Trainer (FSDP or Megatron) - update model weights
3. W&B logging - metrics and events for frontend UI

Architecture:
- Inference and training run asynchronously
- Inference can be ahead of trainer (batches queue up)
- Weights are broadcast from trainer to inference via NCCL
"""
import asyncio
import collections
import itertools
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import httpx
import ray

from telescope.utils import config, paths
from telescope.orchestrator.batch_processor import preprocess_batch
from telescope.orchestrator.eval_runner import (
    EvalConfig,
    EvalRunner,
    PassKConfig,
    PassKEntry,
    compute_eval_schedule,
)
from telescope.orchestrator.loggers.event_logger import (
    EnvResponseRecord,
    EvalGenerationRollout,
    EvalPrompt,
    GenerationRecord,
)
from telescope.orchestrator import generate as generate_module
from telescope.orchestrator.generate import (
    SampleLifecycleCallbacks,
    compute_advantages,
    get_chat_template_kwargs,
    init_interleaved_tokenizer,
    process_group,
    process_multiturn_sample,
    process_sample,
)
from telescope.orchestrator.loggers import WandbLogger
from telescope.orchestrator.loggers.otlp_receiver import OtlpReceiver
from telescope.orchestrator.scheduler import create_scheduler
from telescope.utils.ray_runtime.runtime import (
    RayInferenceGroup,
    RayTrainerGroup,
    _pick_free_port,
    collect_cluster_setup,
    init_ray_cluster,
    resolve_worker_count,
)
from telescope.utils.tlog import get_logger, is_debug_mode, setup_logging
from telescope.utils.tlog.logger import setup_file_tailers
from telescope.utils.tlog.noise_filter import suppress_third_party_noise

_log = get_logger("orchestrator")


class InflightRolloutInfo:
    """Mutable metadata for an in-flight rollout group."""
    __slots__ = ("sample_off_policy_steps", "sample_timings", "tasks", "server_url", "sample_ids")

    def __init__(self, server_url: str):
        self.sample_off_policy_steps: dict[int, int] = {}  # sample_idx -> off_policy_steps
        self.sample_timings: dict[int, dict] = {}  # sample_idx -> timing dict (shared with generate_completion)
        self.tasks: list[asyncio.Task] = []
        self.server_url: str = server_url
        self.sample_ids: dict[int, int] = {}  # sample_idx -> pre-assigned run-wide sample_id


class Orchestrator:
    """
    Coordinates rollout and training for RL post-training.
    
    The orchestrator:
    - Starts and manages inference/trainer processes
    - Dispatches prompts to inference servers (round-robin)
    - Collects completions and batches them for trainer
    - Monitors trainer progress and triggers weight updates
    - Logs everything to W&B for the frontend UI
    """

    def __init__(self):
        _log.debug("Initializing Orchestrator...")
        
        # Config
        self.max_concurrent_prompts_per_server = config.cfg.max_concurrent_prompts_per_server
        self.batch_size = config.cfg.prompts_batch_size_for_trainer

        self.num_inference_workers = resolve_worker_count(
            explicit_count=config.cfg.inference_num_workers,
            role="inference",
        )
        self.inference_tp_size = max(1, int(config.cfg.inference_tensor_parallel_size))
        if self.num_inference_workers % self.inference_tp_size != 0:
            raise ValueError(
                "INFERENCE_NUM_WORKERS must be divisible by INFERENCE_TENSOR_PARALLEL_SIZE. "
                f"Got workers={self.num_inference_workers}, tp={self.inference_tp_size}."
            )
        self.num_inference_servers = self.num_inference_workers // self.inference_tp_size
        self.num_trainer_ranks = resolve_worker_count(
            explicit_count=config.cfg.trainer_num_workers,
            role="trainer",
        )

        backend = str(config.cfg.train_backend).strip().lower()
        if backend not in {"fsdp", "megatron"}:
            raise ValueError(
                f"Unsupported TRAIN_BACKEND={backend!r}. Choose 'fsdp' or 'megatron'."
            )
        if backend == "megatron":
            try:
                import megatron.core  # noqa: F401
            except ImportError:
                raise ImportError(
                    "train_backend is set to 'megatron' but megatron-core is not installed. "
                    "Install with: uv sync --extra megatron\n"
                    "Or switch to train_backend: 'fsdp' in your config."
                ) from None
            tp_size = max(1, int(config.cfg.megatron_tensor_parallel_size))
            pp_size = max(1, int(config.cfg.megatron_pipeline_parallel_size))
            cp_size = max(1, int(config.cfg.megatron_context_parallel_size))
            model_parallel = tp_size * pp_size * cp_size
            if self.num_trainer_ranks % model_parallel != 0:
                raise ValueError(
                    "TRAINER_NUM_WORKERS must be divisible by "
                    "MEGATRON_TENSOR_PARALLEL_SIZE * MEGATRON_PIPELINE_PARALLEL_SIZE * MEGATRON_CONTEXT_PARALLEL_SIZE "
                    "when TRAIN_BACKEND='megatron'. "
                    f"Got workers={self.num_trainer_ranks}, tp={tp_size}, pp={pp_size}, "
                    f"cp={cp_size}, product={model_parallel}."
                )
        self.train_backend = backend
        self.num_trainer_data_shards = self.num_trainer_ranks
        
        _log.debug(
            f"Config: inference_workers={self.num_inference_workers}, "
            f"inference_servers={self.num_inference_servers}, "
            f"inference_tp={self.inference_tp_size}, "
            f"trainer_ranks={self.num_trainer_ranks}, backend={self.train_backend}"
        )
        _log.debug(f"Config: max_concurrent_prompts_per_server={self.max_concurrent_prompts_per_server}, batch_size={self.batch_size}")
        _log.debug(f"Config: group_size={config.cfg.group_size}")
        _log.debug(f"Config: max_async_rollout={config.cfg.max_async_rollout}")

        # State
        self.trainer_step = 0  # Current trainer step (last completed)
        self.inference_step = 0  # Current inference step (can be ahead)
        self.inference_model_step = -1  # Model version inference servers have (-1 = base model, no training)
        self.num_steps = 0
        self.bucket: list = []  # Completed groups waiting for trainer
        self.active_count = 0
        self.rollout_done: asyncio.Event = None
        self.waiting_for_trainer: bool = False  # True when blocked by async limit
        self._checkpoint_pending: bool = False  # True when a checkpoint step was submitted but not yet saved
        self._pending_batches: list[tuple[int, list[dict]]] = []  # Buffered (step, trainer_data) waiting for checkpoint
        self.next_request_id = 0  # Counter for request group IDs
        self.next_sample_idx = 0  # Counter for unique sample_idx across entire run
        # Per-server lane slots: each server gets max_concurrent_prompts_per_server slots.
        # Slot N -> lanes [N*GROUP_SIZE, (N+1)*GROUP_SIZE).
        # Keyed by server URL and populated after inference servers are started.
        self.available_server_lane_slots: dict[str, set[int]] = {}
        self.prefetch_enabled = config.cfg.enable_prompt_prefetch
        self.prefetch_buffer_size = max(0, int(config.cfg.prompt_prefetch_buffer_size))
        self.prefetch_queue: asyncio.Queue | None = None
        self.prefetch_task: asyncio.Task | None = None
        # Individual sample lanes: each sample gets its own lane slot instead of
        # sharing one slot per GROUP_SIZE group.
        self.individual_sample_lanes = config.cfg.enable_individual_sample_lanes
        self.free_lane_after_generation = config.cfg.free_lane_after_generation and self.individual_sample_lanes
        # Tracks partially-completed groups when individual_sample_lanes is enabled.
        # Maps request_id -> {prompt_data, server_url, model_step, lane_slots,
        #                      samples: [dict|None]*GROUP_SIZE, remaining: int,
        #                      dispatched: int, is_multiturn: bool, env: object|None}
        self.pending_individual_groups: dict[int, dict] = {}
        # Per-server queues of individual samples waiting for a free lane.
        # Each entry: (request_id, sample_idx).  Populated after inference
        # servers are started.
        self.individual_sample_queues: dict[str, collections.deque] = {}
        # Per-rollout off-policy tracking (keyed by request_id)
        self.inflight_rollout_info: dict[int, InflightRolloutInfo] = {}

        # Ray runtime
        self.ray_runtime_info: dict | None = None
        self.inference_group: RayInferenceGroup | None = None
        self.trainer_group: RayTrainerGroup | None = None
        self.trainer_runtime_infos: list[dict] = []
        self.pending_train_step_refs: dict[int, list] = {}
        self._step_batches: dict[int, list[dict]] = {}  # batch per step for reward logging

        # Inference endpoints (OpenAI-compatible URLs used by rollout generation)
        self.server_urls: list = []
        self.server_cycle = None

        # Data
        self.scheduler = None
        self.eos_token = None
        self.tokenizer = None  # For multi-turn chat template formatting

        # HTTP client for inference requests (created in run())
        self.http_client: httpx.AsyncClient | None = None

        # vLLM tracing (per-request timing via OTLP)
        self.otlp_receiver: OtlpReceiver | None = None

        # Logging
        self.wandb_logger = WandbLogger()

        # Eval state
        self.eval_configs: list[EvalConfig] = self._parse_eval_configs()
        self.eval_num_servers = max(0, int(config.cfg.eval_num_servers)) if self.eval_configs else 0
        self.eval_schedule: dict[int, list[EvalConfig]] = {}
        self.eval_runner: EvalRunner | None = None
        self.eval_server_urls: list[str] = []
        self.training_server_urls: list[str] = []
        self.eval_server_model_step: int = -1
        self.eval_active: bool = False
        self._pending_eval_task: asyncio.Task | None = None
        # Per-server asyncio tasks for tracking in-flight training prompts
        self.active_tasks_by_server: dict[str, set[asyncio.Task]] = {}

    @staticmethod
    def _parse_eval_configs() -> list[EvalConfig]:
        raw = config.cfg.evals or []
        configs = []
        for item in raw:
            raw_pass_k = item.pass_k
            if isinstance(raw_pass_k, dict):
                raw_at_k = raw_pass_k.get("at_k", {})
                raw_pow_k = raw_pass_k.get("pow_k", {})
                pass_k_cfg = PassKConfig(
                    at_k=PassKEntry(
                        metrics=list(raw_at_k.get("metrics", [])),
                        k=list(raw_at_k.get("k", [1])),
                    ),
                    pow_k=PassKEntry(
                        metrics=list(raw_pow_k.get("metrics", [])),
                        k=list(raw_pow_k.get("k", [])),
                    ),
                )
            else:
                pass_k_cfg = PassKConfig(at_k=PassKEntry(k=list(raw_pass_k)))
            # Build eval sampling_params: start from training defaults,
            # overlay the eval's sampling overrides.
            eval_sampling = dict(config.cfg.get_sampling_params())
            eval_sampling.update(item.get_sampling_overrides())
            configs.append(EvalConfig(
                name=item.name,
                eval_every=item.eval_every,
                pass_k=pass_k_cfg,
                num_samples=item.num_samples,
                separate_eval_samples=item.separate_eval_samples,
                kwargs=dict(item.kwargs),
                sampling_params=eval_sampling,
            ))
        return configs

    # =========================================================================
    # Runtime Management
    # =========================================================================

    def start_processes(self):
        """
        Start Ray runtime: scheduler, inference actors, trainer actors.
        """
        _log.debug("=== Starting orchestrator Ray runtime ===")
        num_steps = config.cfg.number_of_steps
        _log.debug(f"Total training steps: {num_steps}")
        _log.debug(f"Model: {config.cfg.model}")
        _log.debug(f"Sampling params: {config.cfg.get_sampling_params()}")
        _log.debug(f"Max model length: {config.cfg.max_model_len}")
        
        self.wandb_logger.log_orchestrator_timeline_event("ray_init_start")
        self.ray_runtime_info = init_ray_cluster()
        self.wandb_logger.log_orchestrator_timeline_event("ray_init_done")
        _log.debug(f"Connected to Ray cluster (driver node_ip={self.ray_runtime_info['node_ip']})")
        _log.info("Ray cluster connected")

        # Start OTLP receiver for vLLM tracing (must start BEFORE inference servers)
        otlp_endpoint = None
        if config.cfg.enable_vllm_tracing:
            otlp_port = config.cfg.otlp_receiver_port
            advertised_host = self.ray_runtime_info["node_ip"] if self.ray_runtime_info else "127.0.0.1"
            self.otlp_receiver = OtlpReceiver(
                port=otlp_port,
                host="0.0.0.0",
                advertised_host=advertised_host,
            )
            self.otlp_receiver.start()
            otlp_endpoint = self.otlp_receiver.endpoint_url
            _log.debug(f"OTLP receiver started on {otlp_endpoint} for vLLM tracing")

        inference_cpus = float(config.cfg.ray_inference_cpus_per_worker)
        trainer_cpus = float(config.cfg.ray_trainer_cpus_per_worker)
        placement_timeout = int(config.cfg.ray_placement_timeout_s)

        # --- Launch both groups in parallel (non-blocking heavy work) ---
        # Kick off worker launches BEFORE scheduler creation so that placement
        # group allocation and actor initialization overlap with tokenizer +
        # dataset loading below.
        _log.info(f"Starting inference servers ({self.num_inference_servers})...")
        self.wandb_logger.log_orchestrator_timeline_event("inference_processes_start")
        self.inference_group = RayInferenceGroup(
            num_servers=self.num_inference_servers,
            gpus_per_server=self.inference_tp_size,
            cpus_per_worker=inference_cpus,
            placement_strategy=config.cfg.ray_inference_placement_strategy,
            startup_timeout_s=placement_timeout,
            bind_host=config.cfg.inference_host,
            model=config.cfg.model,
        )

        _log.info(f"Starting trainer ({self.num_trainer_ranks} ranks, {self.train_backend})...")
        self.wandb_logger.log_orchestrator_timeline_event("trainer_process_start")
        self.trainer_group = RayTrainerGroup(
            world_size=self.num_trainer_ranks,
            cpus_per_worker=trainer_cpus,
            placement_strategy=config.cfg.ray_trainer_placement_strategy,
            startup_timeout_s=placement_timeout,
        )

        # Run both launch() calls concurrently — each blocks on PG allocation,
        # so threading overlaps their placement group waits.
        with ThreadPoolExecutor(max_workers=2) as pool:
            inf_future = pool.submit(self.inference_group.launch, otlp_endpoint=otlp_endpoint)
            train_future = pool.submit(self.trainer_group.launch)
            inf_future.result()
            train_future.result()

        # --- Prepare scheduler while workers are initializing in background ---
        _log.info("Preparing environments and scheduler...")
        self.wandb_logger.log_orchestrator_timeline_event("environments_prepare_start")
        # Compute eval-excluded indices so the scheduler skips them
        train_env_names = {e.name for e in config.cfg.environments}
        excluded_indices: dict[str, set[int]] = {}
        for ec in self.eval_configs:
            if ec.separate_eval_samples and ec.num_samples > 0:
                if ec.name not in train_env_names:
                    _log.warning(
                        f"Eval '{ec.name}' has separate_eval_samples=True but does not "
                        f"match any training environment ({train_env_names}). "
                        f"Eval samples will NOT be excluded from training."
                    )
                    continue
                excluded_indices[ec.name] = set(range(ec.num_samples))
        self.scheduler, self.eos_token, self.tokenizer = create_scheduler(
            excluded_indices=excluded_indices or None,
        )
        _log.debug(f"Scheduler created with eos_token={repr(self.eos_token)}")

        # Initialize interleaved tokenizer for local tokenization (no HTTP calls)
        if config.cfg.interleaved_rollouts and self.tokenizer is not None:
            if hasattr(self.scheduler, 'environments'):
                envs = self.scheduler.environments
            else:
                envs = [self.scheduler.environment]
            any_multi_turn = any(env.is_multi_turn for env in envs)
            if any_multi_turn:
                _log.debug("Initializing InterleavedTokenizer for local multi-turn tokenization")
                init_interleaved_tokenizer(self.tokenizer)
            else:
                _log.debug("Skipping InterleavedTokenizer: no multi-turn environments configured")
        self.wandb_logger.update_env_summary(self.scheduler)
        self.wandb_logger.update_model_architecture_summary()
        self.wandb_logger.log_orchestrator_timeline_event("environments_prepare_done")

        # --- Wait for inference results and set up inference state ---
        self.inference_group.wait_ready()
        self.server_urls = list(self.inference_group.server_urls)
        self.server_cycle = itertools.cycle(self.server_urls)
        # Always use per-sample lane slots (1 slot = 1 vLLM request).
        slots_per_server = self.max_concurrent_prompts_per_server * config.cfg.group_size
        self.available_server_lane_slots = {
            url: set(range(slots_per_server))
            for url in self.server_urls
        }
        self.individual_sample_queues = {
            url: collections.deque() for url in self.server_urls
        }
        self.wandb_logger.set_inference_servers(self.inference_group.server_infos)
        inference_ready_times = [
            info["ready_time"] for info in self.inference_group.server_infos if "ready_time" in info
        ]
        inference_ready_ts = max(inference_ready_times) if inference_ready_times else None
        self.wandb_logger.log_orchestrator_timeline_event("inference_servers_ready", timestamp=inference_ready_ts)
        _log.info(f"Inference servers ready ({self.num_inference_servers})")

        # --- Wait for trainer results and set up trainer state ---
        self.trainer_runtime_infos = self.trainer_group.wait_ready()
        self.wandb_logger.set_trainer_runtime_infos(self.trainer_runtime_infos)
        dp_world_sizes = {
            int(info.get("dp_world_size", -1))
            for info in self.trainer_runtime_infos
            if int(info.get("dp_world_size", -1)) > 0
        }
        if len(dp_world_sizes) == 1:
            self.num_trainer_data_shards = next(iter(dp_world_sizes))
        else:
            dp_ranks = {
                int(info.get("dp_rank", -1))
                for info in self.trainer_runtime_infos
                if int(info.get("dp_rank", -1)) >= 0
            }
            if dp_ranks:
                self.num_trainer_data_shards = len(dp_ranks)
        _log.debug(
            "Trainer data sharding: "
            f"{self.num_trainer_data_shards} shard(s) for backend={self.train_backend}",
        )
        _log.info(f"Trainer ready ({self.num_trainer_ranks} ranks)")

        # Now init weight broadcast (needs both inference and trainer ready)
        _log.info("Initializing weight broadcast...")
        self._init_weight_broadcast()

        # --- Resume from checkpoint (after broadcast init so weights can be synced) ---
        if config.cfg.resume_from_checkpoint:
            self._load_checkpoint_and_resume()

        # --- Eval setup (after broadcast init so NCCL groups are ready) ---
        use_all_servers = config.cfg.eval_start_end_use_all_servers
        has_dedicated_eval = self.eval_num_servers > 0 and self.eval_configs
        has_baseline_or_final = self.eval_configs and use_all_servers and (
            config.cfg.eval_before_training or config.cfg.eval_after_training
        )

        if has_dedicated_eval:
            self.eval_schedule = compute_eval_schedule(
                self.eval_configs, config.cfg.number_of_steps
            )
            _log.debug(
                f"Eval schedule computed: {len(self.eval_schedule)} steps have evals "
                f"(EVAL_NUM_SERVERS={self.eval_num_servers})"
            )
            # Designate last EVAL_NUM_SERVERS as eval servers
            all_urls = list(self.inference_group.server_urls)
            self.training_server_urls = all_urls[: -self.eval_num_servers]
            self.eval_server_urls = all_urls[-self.eval_num_servers :]
            _log.debug(f"Training servers: {self.training_server_urls}")
            _log.debug(f"Eval servers: {self.eval_server_urls}")

            # Send eval schedule to trainer so it knows which steps are eval steps
            eval_step_set = set(self.eval_schedule.keys())
            ray.get([
                actor.set_eval_schedule.remote(eval_step_set)
                for actor in self.trainer_group.actors
            ])
            _log.debug(f"Eval schedule sent to {len(self.trainer_group.actors)} trainer actors")
        else:
            self.training_server_urls = list(self.server_urls)
            self.eval_server_urls = []

        if (has_dedicated_eval or has_baseline_or_final) and self.eval_runner is None:
            self.eval_runner = EvalRunner(self.eval_configs, tokenizer=self.tokenizer)
            self.eval_runner.load_environments()
            self.wandb_logger.update_eval_env_summary(self.eval_runner, self.eval_configs)

        try:
            setup_snapshot = collect_cluster_setup(
                ray_runtime_info=self.ray_runtime_info,
                inference_server_infos=self.inference_group.server_infos,
                trainer_runtime_infos=self.trainer_runtime_infos,
                num_inference_servers=self.num_inference_servers,
                num_trainer_ranks=self.num_trainer_ranks,
                inference_tp_size=self.inference_tp_size,
            )
            self.wandb_logger.update_setup(setup_snapshot)
        except Exception as exc:
            _log.warning(f"Failed to collect/update setup snapshot: {exc}")
        trainer_ready_times = [
            info["ready_time"] for info in self.trainer_runtime_infos if "ready_time" in info
        ]
        trainer_ready_ts = max(trainer_ready_times) if trainer_ready_times else None
        self.wandb_logger.log_orchestrator_timeline_event("trainer_ready", timestamp=trainer_ready_ts)

    def stop_processes(self, force: bool = False):
        """Stop all Ray actors and side services.

        Args:
            force: If True, force-kill all actors immediately without graceful
                shutdown.  Used during interrupt to avoid hanging on stuck NCCL
                operations.  Trainer is killed first so that in-flight weight
                syncs don't deadlock when inference goes away.
        """
        if force:
            # Kill trainer first: if a train_step is mid-weight-sync to
            # inference via NCCL, killing inference first would leave the
            # trainer blocked on a dead NCCL peer indefinitely.
            for name, group in [("trainer", self.trainer_group), ("inference", self.inference_group)]:
                if group is not None:
                    _log.debug(f"Force-killing {name} actors...")
                    try:
                        group.force_kill()
                    except Exception as exc:
                        _log.warning(f"Error while force-killing {name} group: {exc}")
        else:
            if self.inference_group is not None:
                try:
                    samples = self.inference_group.drain_torch_memory_samples(timeout=10)
                    if samples:
                        self.wandb_logger.log_torch_memory_samples(samples)
                except Exception as exc:
                    _log.warning(f"Error while draining inference torch memory samples: {exc}")
                try:
                    self.inference_group.stop()
                except Exception as exc:
                    _log.warning(f"Error while stopping inference group: {exc}")
            if self.trainer_group is not None:
                try:
                    samples = self.trainer_group.drain_torch_memory_samples(timeout=10)
                    if samples:
                        self.wandb_logger.log_torch_memory_samples(samples)
                except Exception as exc:
                    _log.warning(f"Error while draining trainer torch memory samples: {exc}")
                try:
                    self.trainer_group.stop()
                except Exception as exc:
                    _log.warning(f"Error while stopping trainer group: {exc}")
        if self.otlp_receiver is not None:
            try:
                self.otlp_receiver.stop()
            except Exception as exc:
                _log.warning(f"Error while stopping OTLP receiver: {exc}")

    # =========================================================================
    # Main Run Loop
    # =========================================================================

    async def run(self):
        """
        Main training loop.
        
        Runs three concurrent tasks:
        1. Weight update watcher - monitors trainer, triggers vLLM weight updates
        2. Rollout loop - dispatches prompts, collects completions
        3. Metrics logger - reads trainer metrics, logs to W&B
        """

        self.num_steps = config.cfg.number_of_steps

        total_capacity = self.max_concurrent_prompts_per_server * len(self.server_urls)
        _log.section("Training")
        _log.debug(f"Training capacity: {self.max_concurrent_prompts_per_server} concurrent prompts/server ({total_capacity} total across {len(self.server_urls)} servers)")
        _log.debug(f"Run configuration: batch_size={self.batch_size}, group_size={config.cfg.group_size}")
        _log.debug(f"Async rollout limit: {config.cfg.max_async_rollout}")
        _log.debug(f"Inference servers: {self.server_urls}")
        _log.debug(f"Starting state: trainer_step={self.trainer_step}, inference_step={self.inference_step}, inference_model_step={self.inference_model_step}")
        self.wandb_logger.set_trainer_steps_done(self.trainer_step)

        await self.wandb_logger.start_event_upload_loop()
        _log.debug("W&B event upload loop started")

        # Create HTTP client with high connection limits for concurrent requests
        limits = httpx.Limits(max_connections=8192, max_keepalive_connections=0)
        self.http_client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(1200.0))
        _log.debug("HTTP client created (max_connections=8192, timeout=1200s)")

        try:
            # Baseline eval (before any training) — skip on resume since model is already trained
            if config.cfg.eval_before_training and self.trainer_step == 0 and self.eval_runner and self.eval_configs:
                use_all = config.cfg.eval_start_end_use_all_servers
                baseline_urls = (
                    list(self.inference_group.server_urls)
                    if use_all and self.inference_group
                    else list(self.eval_server_urls)
                )
                if baseline_urls:
                    _log.info(
                        f"Running baseline evals on {len(baseline_urls)} servers"
                        + (" (EVAL_START_END_USE_ALL_SERVERS)" if use_all else "")
                    )
                    await self._run_eval_phase(
                        eval_configs=self.eval_configs,
                        step=0,
                        model_step=0,
                        label="baseline",
                        eval_server_urls=baseline_urls,
                    )

            # Pre-warm environment resources (e.g. sandbox pools) so they
            # are ready by the time the first completions need rewards.
            if config.cfg.eager_prepare_resources:
                if hasattr(self.scheduler, 'environments'):
                    _envs = self.scheduler.environments
                else:
                    _envs = [self.scheduler.environment]
                for _env in _envs:
                    await _env.prepare_resources()

            watcher = asyncio.create_task(self._weight_update_watcher())
            rollout = asyncio.create_task(self._rollout_loop())
            metrics = asyncio.create_task(self._metrics_logger())
            main_tasks = [watcher, rollout, metrics]

            try:
                done, pending = await asyncio.wait(
                    main_tasks, return_when=asyncio.FIRST_EXCEPTION,
                )
                # Re-raise the first exception (if any).
                for t in done:
                    if not t.cancelled() and t.exception() is not None:
                        raise t.exception()
                # All tasks finished normally — await any stragglers.
                if pending:
                    await asyncio.gather(*pending)
            finally:
                # Unblock _rollout_loop in case it's waiting on the event.
                if self.rollout_done is not None:
                    self.rollout_done.set()
                for t in main_tasks:
                    if not t.done():
                        t.cancel()
                remaining = [t for t in main_tasks if not t.done()]
                if remaining:
                    # Timeout avoids hanging on un-cancellable to_thread calls.
                    await asyncio.wait(remaining, timeout=5.0)

            # Wait for any in-flight eval task from the last training step
            # before checking EVAL_AFTER_TRAINING or shutting down.
            if self._pending_eval_task is not None:
                await self._pending_eval_task
                self._pending_eval_task = None

            # Final eval (after all training) — skip configs that already
            # ran at this step via the regular eval_every schedule.
            if config.cfg.eval_after_training and self.eval_runner and self.eval_configs:
                final_step = self.trainer_step - 1
                already_scheduled = set()
                for ec in getattr(self, "eval_schedule", {}).get(final_step, []):
                    already_scheduled.add(ec.name)
                final_configs = [
                    ec for ec in self.eval_configs if ec.name not in already_scheduled
                ]
                if not final_configs:
                    _log.info(
                        f"Skipping EVAL_AFTER_TRAINING — all evals already ran "
                        f"at step {final_step + 1} via eval_every schedule"
                    )
                else:
                    use_all = config.cfg.eval_start_end_use_all_servers
                    final_urls = (
                        list(self.inference_group.server_urls)
                        if use_all and self.inference_group
                        else list(self.eval_server_urls)
                    )
                    if final_urls:
                        _log.info(
                            f"Running final evals ({len(final_configs)} configs) on "
                            f"{len(final_urls)} servers"
                            + (" (EVAL_START_END_USE_ALL_SERVERS)" if use_all else "")
                        )
                        await self._run_eval_phase(
                            eval_configs=final_configs,
                            step=final_step + 1,
                            model_step=self.inference_model_step + 1,
                            label="final",
                            eval_server_urls=final_urls,
                        )
        finally:
            # Signal shutdown so in-flight requests don't retry / log warnings.
            generate_module._shutting_down = True
            self.inflight_rollout_info.clear()

            # Cancel all in-flight rollout tasks before closing the HTTP client
            # so they don't wake up to a closed client and flood warnings.
            all_tasks = [
                t
                for tasks in self.active_tasks_by_server.values()
                for t in tasks
                if not t.done()
            ]
            for t in all_tasks:
                t.cancel()
            if all_tasks:
                await asyncio.gather(*all_tasks, return_exceptions=True)

            if self.eval_runner:
                await self.eval_runner.close()
            await self.http_client.aclose()
            await self.wandb_logger.stop_event_upload_loop()

    # =========================================================================
    # Async Tasks
    # =========================================================================

    async def _weight_update_watcher(self):
        """Wait for Ray trainer step completions and advance model step."""
        while self.trainer_step < self.num_steps:
            step_refs = self.pending_train_step_refs.get(self.trainer_step)
            if step_refs is None:
                await asyncio.sleep(0.1)
                continue

            _log.debug(f"Waiting for trainer step {self.trainer_step} to finish", step=self.trainer_step)
            try:
                step_results = await asyncio.to_thread(
                    RayTrainerGroup.wait_step_results,
                    step_refs,
                )
            except Exception as exc:
                _log.exception(f"Trainer step {self.trainer_step} failed: {exc}", step=self.trainer_step)
                raise

            del self.pending_train_step_refs[self.trainer_step]

            _log.info(f"\033[32mStep {self.trainer_step}\033[0m completed")

            # Determine weight sync group from the results
            weight_sync_group = "full"
            is_eval_step = False
            for r in step_results:
                if r.get("weight_sync_group"):
                    weight_sync_group = r["weight_sync_group"]
                if r.get("eval_step"):
                    is_eval_step = True

            await self._update_inference_weights(step_results, weight_sync_group)

            # Checkpoint saving
            completed_step = self.trainer_step
            is_last_step = (completed_step == self.num_steps - 1)
            if self._should_save_checkpoint(completed_step, is_last_step):
                await self._save_checkpoint(completed_step)
                self._flush_pending_batches()

            self.trainer_step += 1
            self.wandb_logger.set_trainer_steps_done(self.trainer_step)

            # If this was an eval step, trigger eval
            if is_eval_step and completed_step in self.eval_schedule:
                self._pending_eval_task = asyncio.create_task(
                    self._handle_eval_step(completed_step)
                )

        _log.info("Training complete")

        # Signal the rollout loop to stop so asyncio.wait() can return.
        # Without this, the rollout loop blocks on rollout_done.wait()
        # while in-flight HTTP requests drain slowly (up to the HTTP
        # timeout), deadlocking the shutdown sequence.
        if self.rollout_done is not None:
            self.rollout_done.set()

    async def _metrics_logger(self):
        """Periodically log trainer metrics to W&B."""
        metrics_interval_s = max(0.1, float(config.cfg.metrics_logger_interval_seconds))
        torch_drain_interval_s = max(0.1, float(config.cfg.ray_torch_memory_drain_interval_seconds))

        sleep_interval_s = max(0.1, min(0.5, metrics_interval_s, torch_drain_interval_s))
        next_metrics_poll = 0.0
        next_torch_drain = 0.0

        while self.trainer_step < self.num_steps:
            now = time.monotonic()
            if now >= next_metrics_poll:
                self.wandb_logger.log_pending_metrics()
                next_metrics_poll = now + metrics_interval_s
            if now >= next_torch_drain:
                await self._drain_ray_torch_memory_metrics()
                await self._drain_ray_inference_torch_memory_metrics()
                next_torch_drain = now + torch_drain_interval_s
            await asyncio.sleep(sleep_interval_s)

        # Final check
        await asyncio.sleep(min(1.0, sleep_interval_s))
        self.wandb_logger.log_pending_metrics()
        await self._drain_ray_torch_memory_metrics()
        await self._drain_ray_inference_torch_memory_metrics()
        _log.debug("Metrics logger completed")

    async def _drain_ray_torch_memory_metrics(self):
        """
        Pull high-frequency torch memory samples from Ray trainer actors.

        Trainer actors sample on the configured
        ``TORCH_MEMORY_SAMPLE_INTERVAL_SECONDS`` cadence and buffer locally;
        orchestrator drains buffered samples on
        ``RAY_TORCH_MEMORY_DRAIN_INTERVAL_SECONDS`` cadence and appends them to
        gpu.parquet.
        """
        if self.trainer_group is None:
            return

        try:
            samples = await asyncio.to_thread(
                self.trainer_group.drain_torch_memory_samples
            )
        except Exception as exc:
            _log.warning(f"Failed to drain torch memory samples from trainers: {exc}")
            return

        if samples:
            self.wandb_logger.log_torch_memory_samples(samples)

    async def _drain_ray_inference_torch_memory_metrics(self):
        """
        Pull torch allocator samples from Ray inference workers.

        Inference workers collect these via vLLM worker-extension RPC, and the
        orchestrator drains/forwards them to gpu.parquet on the same configured
        ``RAY_TORCH_MEMORY_DRAIN_INTERVAL_SECONDS`` cadence.
        """
        if self.inference_group is None:
            return

        try:
            samples = await asyncio.to_thread(
                self.inference_group.drain_torch_memory_samples
            )
        except Exception as exc:
            _log.warning(f"Failed to drain torch memory samples from inference workers: {exc}")
            return

        if samples:
            self.wandb_logger.log_torch_memory_samples(samples)

    async def _rollout_loop(self):
        """Dispatch prompts to inference, collect completions."""
        _log.debug("=== Rollout loop starting ===")
        self.rollout_done = asyncio.Event()
        
        if self._prefetch_active():
            _log.debug(f"Prompt prefetch enabled (buffer size={self.prefetch_buffer_size})")
            self.prefetch_queue = asyncio.Queue(maxsize=self.prefetch_buffer_size)
            self.prefetch_task = asyncio.create_task(self._prefetch_loop())
        else:
            _log.debug("Prompt prefetch disabled")

        # Start initial batch of prompts (fill all server slots).
        # In individual mode each _start_next_prompt dispatches 1 sample, so
        # the bound is per-sample; in group mode it dispatches 1 group.
        if self.individual_sample_lanes:
            total_capacity = self.max_concurrent_prompts_per_server * config.cfg.group_size * len(self.server_urls)
        else:
            total_capacity = self.max_concurrent_prompts_per_server * len(self.server_urls)
        _log.debug(f"Launching up to {total_capacity} initial dispatches...")
        _dispatch_yield_interval = max(1, len(self.server_urls))
        for i in range(total_capacity):
            if not self._start_next_prompt():
                break
            _log.debug(f"Starting initial prompt {i+1}/{total_capacity}")
            if (i + 1) % _dispatch_yield_interval == 0:
                await asyncio.sleep(0)

        _log.debug(f"Started {self.active_count} initial prompts")
        _log.debug(f"Rollout loop initialized: active={self.active_count}, bucket={len(self.bucket)}")
        await self.rollout_done.wait()
        
        if self.prefetch_task is not None:
            self.prefetch_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.prefetch_task
        _log.debug("Rollout loop completed")

    # =========================================================================
    # Rollout Helpers
    # =========================================================================

    def _pick_server_with_capacity(self) -> str | None:
        """Pick a server with available lane slots, round-robin. Returns None if all full."""
        for _ in range(len(self.server_urls)):
            url = next(self.server_cycle)
            slots = self.available_server_lane_slots.get(url)
            if slots and len(slots) >= config.cfg.group_size:
                return url
        return None

    def _pop_first_available_lane_slot(self, available_slots: set[int]) -> int:
        """Return and remove the smallest free lane slot for deterministic reuse."""
        if not available_slots:
            raise RuntimeError(
                "No available lane slots; this indicates inconsistent lane accounting."
            )
        lane_slot = min(available_slots)
        available_slots.remove(lane_slot)
        return lane_slot

    # --- Deferred dispatch: consolidates many _start_next_prompt calls ---
    _dispatch_pending = False
    _dispatch_task: asyncio.Task | None = None

    def _schedule_dispatch(self):
        """Request a dispatch cycle without blocking the caller.

        Multiple calls between event-loop yields are coalesced into a
        single async task that dispatches in batches with periodic yields.
        """
        if self._dispatch_pending:
            return
        self._dispatch_pending = True
        if self._dispatch_task is None or self._dispatch_task.done():
            self._dispatch_task = asyncio.ensure_future(self._deferred_dispatch_loop())

    async def _deferred_dispatch_loop(self):
        """Process pending dispatches with periodic yields."""
        _yield_interval = max(1, len(self.server_urls))
        while self._dispatch_pending:
            self._dispatch_pending = False
            _count = 0
            while self._start_next_prompt():
                _count += 1
                if _count % _yield_interval == 0:
                    await asyncio.sleep(0)
            await asyncio.sleep(0)

    def _start_next_prompt(self) -> bool:
        if self.individual_sample_lanes:
            return self._start_next_individual()

        # --- Group mode ---
        if self._checkpoint_pending:
            return False

        if not self._has_pending_prompt_data():
            _log.debug("No pending prompt data available")
            return False

        async_level = self._compute_async_level(self.inference_model_step)
        if async_level > config.cfg.max_async_rollout:
            if not self.waiting_for_trainer:
                self.waiting_for_trainer = True
                _log.info(f"Rollout paused: async level {async_level} > max {config.cfg.max_async_rollout}")
                self.wandb_logger.log_orchestrator_timeline_event("rollout_paused_max_async", step=self.inference_model_step)
            return False

        server_url = self._pick_server_with_capacity()
        if server_url is None:
            _log.debug("All servers at capacity, cannot dispatch")
            return False

        prompt_data = self._get_next_prompt_data()
        if prompt_data is None:
            return False

        request_id = self.next_request_id
        self.next_request_id += 1
        self.active_count += 1

        server_slots = self.available_server_lane_slots[server_url]
        lane_slots = [
            self._pop_first_available_lane_slot(server_slots)
            for _ in range(config.cfg.group_size)
        ]

        env_name = prompt_data.get("env_name", "unknown")
        _log.debug(
            f"Dispatching request_id={request_id} to {server_url} "
            f"(env={env_name}, active={self.active_count}, "
            f"lane_slots={lane_slots})"
        )
        self.wandb_logger.log_orchestrator_timeline_event("inference_call", group_id=request_id)

        # Pre-assign sample_ids and log start events for each sample in the group
        sample_ids = {}
        now = time.time()
        server_idx = self._get_server_index(server_url)
        server_info = self._get_server_info(server_url)
        for i in range(config.cfg.group_size):
            sample_ids[i] = self.next_sample_idx
            self.next_sample_idx += 1
            self.wandb_logger.event_logger.log_rollout_event(
                event_type="generation",
                phase="start",
                timestamp=now,
                sample_id=sample_ids[i],
                server_id=server_idx,
                generation_idx=0,
                group_id=request_id,
            )

        task = asyncio.create_task(
            self._run_prompt(prompt_data, server_url, request_id, lane_slots)
        )
        self.active_tasks_by_server.setdefault(server_url, set()).add(task)
        task.add_done_callback(lambda t: (self.active_tasks_by_server.get(server_url, set()).discard(t), t.exception() if not t.cancelled() else None))
        info = InflightRolloutInfo(server_url=server_url)
        info.tasks.append(task)
        info.sample_off_policy_steps = {idx: 0 for idx in range(config.cfg.group_size)}
        info.sample_ids = sample_ids
        self.inflight_rollout_info[request_id] = info
        return True

    # ----- Individual sample dispatch -----

    def _start_next_individual(self) -> bool:
        """Dispatch one individual sample, creating a new group if needed.

        Returns True if a sample was dispatched, False otherwise.
        """
        # 1. Try to dispatch a queued sample first.
        if self._dispatch_one_queued_sample():
            return True

        # 2. No queued work — create a new group (if allowed).
        if self._checkpoint_pending:
            return False

        if not self._has_pending_prompt_data():
            return False

        async_level = self._compute_async_level(self.inference_model_step)
        if async_level > config.cfg.max_async_rollout:
            if not self.waiting_for_trainer:
                self.waiting_for_trainer = True
                _log.info(f"Rollout paused: async level {async_level} > max {config.cfg.max_async_rollout}")
                self.wandb_logger.log_orchestrator_timeline_event("rollout_paused_max_async", step=self.inference_model_step)
            return False

        # Need at least 1 free lane on some server.
        server_url = self._pick_server_with_any_capacity()
        if server_url is None:
            return False

        prompt_data = self._get_next_prompt_data()
        if prompt_data is None:
            return False

        request_id = self.next_request_id
        self.next_request_id += 1
        self.active_count += 1

        env = prompt_data.get("env")
        is_multiturn = env is not None and getattr(env, "is_multi_turn", False)

        self.pending_individual_groups[request_id] = {
            "prompt_data": prompt_data,
            "server_url": server_url,
            "model_step": self.inference_model_step,
            "lane_slots": [None] * config.cfg.group_size,
            "samples": [None] * config.cfg.group_size,
            "remaining": config.cfg.group_size,
            "dispatched": 0,
            "is_multiturn": is_multiturn,
            "env": env,
        }

        env_name = prompt_data.get("env_name", "unknown")
        _log.debug(
            f"Created individual group request_id={request_id} on {server_url} "
            f"(env={env_name}, active={self.active_count})"
        )
        self.wandb_logger.log_orchestrator_timeline_event("inference_call", group_id=request_id)
        self.inflight_rollout_info[request_id] = InflightRolloutInfo(server_url=server_url)

        # Enqueue GROUP_SIZE samples for this group.
        queue = self.individual_sample_queues.setdefault(server_url, collections.deque())
        for idx in range(config.cfg.group_size):
            queue.append((request_id, idx))

        # Dispatch one right away (the server has a free lane).
        self._dispatch_one_queued_sample()
        return True

    def _pick_server_with_any_capacity(self) -> str | None:
        """Pick a server with at least 1 free lane slot, round-robin."""
        for _ in range(len(self.server_urls)):
            url = next(self.server_cycle)
            if self.available_server_lane_slots.get(url):
                return url
        return None

    def _dispatch_one_queued_sample(self) -> bool:
        """Pop one sample from per-server queues and dispatch it if a lane is free."""
        for server_url, queue in self.individual_sample_queues.items():
            slots = self.available_server_lane_slots.get(server_url)
            if queue and slots:
                request_id, sample_idx = queue.popleft()
                group = self.pending_individual_groups.get(request_id)
                if group is None:
                    # Group was cleaned up (cancelled) — skip.
                    continue
                lane_slot = self._pop_first_available_lane_slot(slots)
                group["lane_slots"][sample_idx] = lane_slot
                group["dispatched"] += 1

                task = asyncio.create_task(
                    self._run_individual_sample(
                        group["prompt_data"], server_url, request_id,
                        sample_idx, lane_slot,
                    )
                )
                self.active_tasks_by_server.setdefault(server_url, set()).add(task)
                task.add_done_callback(
                    lambda t, url=server_url: (self.active_tasks_by_server.get(url, set()).discard(t), t.exception() if not t.cancelled() else None)
                )
                rollout_info = self.inflight_rollout_info.get(request_id)
                if rollout_info is not None:
                    rollout_info.tasks.append(task)
                    rollout_info.sample_off_policy_steps[sample_idx] = 0
                    # Pre-assign sample_id and log start event
                    sample_id = self.next_sample_idx
                    self.next_sample_idx += 1
                    rollout_info.sample_ids[sample_idx] = sample_id
                    now = time.time()
                    server_idx = self._get_server_index(server_url)
                    server_info = self._get_server_info(server_url)
                    self.wandb_logger.event_logger.log_rollout_event(
                        event_type="generation",
                        phase="start",
                        timestamp=now,
                        sample_id=sample_id,
                        server_id=server_idx,
                        generation_idx=0,
                        group_id=request_id,
                    )
                _log.debug(
                    f"Dispatched individual sample: request_id={request_id}, "
                    f"sample_idx={sample_idx}, lane_slot={lane_slot}, "
                    f"dispatched={group['dispatched']}/{config.cfg.group_size}"
                )
                return True
        return False

    # ----- End individual sample dispatch -----

    def _make_sample_lifecycle(
        self,
        request_id: int,
        sample_idx: int,
        server_url: str,
        lane_slot: int,
        timing: dict,
        free_lane_on_inference_end: bool = False,
    ) -> SampleLifecycleCallbacks:
        """Create per-sample lifecycle callbacks for decoupled event logging.

        The returned callbacks log:
        - on_inference_end: phase="end" inference event at generation-complete time, optionally frees lane
        - on_reward_start: compute_reward_start orchestrator event
        - on_reward_end: compute_reward_end orchestrator event
        """
        server_idx = self._get_server_index(server_url)
        server_info = self._get_server_info(server_url)
        lane_freed = False

        def on_inference_end():
            nonlocal lane_freed
            rollout_info = self.inflight_rollout_info.get(request_id)
            sample_id = rollout_info.sample_ids.get(sample_idx, -1) if rollout_info else -1

            # Log generation end event (vLLM timing now goes in domain table via log_generation_rollout)
            self.wandb_logger.event_logger.log_rollout_event(
                event_type="generation",
                phase="end",
                sample_id=sample_id,
                server_id=server_idx,
                group_id=request_id,
                generation_idx=0,  # single-turn; multi-turn sets this per-turn in generate.py
            )
            timing["inference_end_logged"] = True

            # Optionally free lane slot (individual mode with free_lane_after_generation)
            if free_lane_on_inference_end and not lane_freed:
                lane_freed = True
                if server_url in self.available_server_lane_slots:
                    self.available_server_lane_slots[server_url].add(lane_slot)
                self._schedule_dispatch()

        # sample_id resolved at reward_start time and captured for reward_end,
        # because rollout_info may be popped by _on_rollout_complete before reward_end fires.
        _reward_sample_id = -1

        def on_reward_start():
            nonlocal _reward_sample_id
            rollout_info = self.inflight_rollout_info.get(request_id)
            _reward_sample_id = rollout_info.sample_ids.get(sample_idx, -1) if rollout_info else -1
            self.wandb_logger.event_logger.log_rollout_event(
                event_type="reward",
                phase="start",
                sample_id=_reward_sample_id,
                group_id=request_id,
            )

        def on_reward_end(compute_reward_time: float):
            self.wandb_logger.event_logger.log_rollout_event(
                event_type="reward",
                phase="end",
                sample_id=_reward_sample_id,
                group_id=request_id,
            )

        def on_env_response_start():
            rollout_info = self.inflight_rollout_info.get(request_id)
            sid = rollout_info.sample_ids.get(sample_idx, -1) if rollout_info else -1
            self.wandb_logger.event_logger.log_rollout_event(
                event_type="env_response",
                phase="start",
                sample_id=sid,
                group_id=request_id,
            )

        def on_env_response_end():
            rollout_info = self.inflight_rollout_info.get(request_id)
            sid = rollout_info.sample_ids.get(sample_idx, -1) if rollout_info else -1
            self.wandb_logger.event_logger.log_rollout_event(
                event_type="env_response",
                phase="end",
                sample_id=sid,
                group_id=request_id,
            )

        lifecycle = SampleLifecycleCallbacks(
            on_inference_end=on_inference_end,
            on_reward_start=on_reward_start,
            on_reward_end=on_reward_end,
            on_env_response_start=on_env_response_start,
            on_env_response_end=on_env_response_end,
        )
        # Expose lane_freed state so _run_individual_sample can check it
        lifecycle._lane_freed = lambda: lane_freed  # type: ignore[attr-defined]
        return lifecycle

    async def _run_prompt(
        self,
        prompt_data: dict,
        server_url: str,
        request_id: int,
        lane_slots: list[int],
    ):
        """Run a single prompt rollout (group mode)."""
        env = prompt_data.get("env")
        env_name = prompt_data.get("env_name", "unknown")

        _log.debug(f"Starting rollout group request_id={request_id} on {server_url} (env={env_name}, model_step={self.inference_model_step}, group_size={config.cfg.group_size})")

        # Inject IDs into sample metadata for debug tracing in compute_reward
        sample = prompt_data.get("sample")
        if sample is not None:
            rollout_info = self.inflight_rollout_info.get(request_id)
            sample.metadata["_group_id"] = request_id
            if rollout_info and rollout_info.sample_ids:
                sample.metadata["_sample_ids"] = dict(rollout_info.sample_ids)

        sample_timings = [{} for _ in range(config.cfg.group_size)]
        rollout_info = self.inflight_rollout_info.get(request_id)
        if rollout_info:
            rollout_info.sample_timings = {i: sample_timings[i] for i in range(config.cfg.group_size)}

        # Create per-sample lifecycle callbacks for decoupled event logging
        sample_lifecycles = [
            self._make_sample_lifecycle(
                request_id=request_id,
                sample_idx=i,
                server_url=server_url,
                lane_slot=lane_slots[i] if i < len(lane_slots) else -1,
                timing=sample_timings[i],
                free_lane_on_inference_end=False,  # group mode frees all lanes together
            )
            for i in range(config.cfg.group_size)
        ]

        try:
            result = await process_group(
                prompt_data,
                self.eos_token,
                server_url,
                self.scheduler.compute_reward,
                env=env,
                tokenizer=self.tokenizer,
                http_client=self.http_client,
                sample_timings=sample_timings,
                sample_lifecycles=sample_lifecycles,
            )
        except asyncio.CancelledError:
            self._on_cancelled_group(request_id, server_url, lane_slots, sample_timings)
            raise

        result["model_step"] = self.inference_model_step
        result["group_id"] = request_id
        result["individual_lane_slots"] = lane_slots

        # Eagerly fetch OTLP timing for each sample in the group.
        if "error" not in result:
            await self._fetch_otlp_timing_for_group(result)

        _log.debug(f"Completed rollout request_id={request_id} on {server_url} (model_step={result['model_step']})")

        self._on_rollout_complete(result)

    # ----- Individual sample lane helpers -----

    async def _run_individual_sample(
        self,
        prompt_data: dict,
        server_url: str,
        request_id: int,
        sample_idx: int,
        lane_slot: int,
    ):
        """Run a single sample for individual-lane mode."""
        env = prompt_data.get("env")
        is_multiturn = env is not None and getattr(env, "is_multi_turn", False)

        timing = {}
        rollout_info = self.inflight_rollout_info.get(request_id)
        if rollout_info:
            rollout_info.sample_timings[sample_idx] = timing

        # Create lifecycle callbacks for decoupled event logging + lane freeing.
        lifecycle = self._make_sample_lifecycle(
            request_id=request_id,
            sample_idx=sample_idx,
            server_url=server_url,
            lane_slot=lane_slot,
            timing=timing,
            free_lane_on_inference_end=self.free_lane_after_generation,
        )

        try:
            if is_multiturn:
                result = await process_multiturn_sample(
                    self.http_client,
                    env,
                    prompt_data["sample"],
                    self.eos_token,
                    server_url,
                    self.tokenizer,
                    prompt_data=prompt_data,
                    timing_out=timing,
                    lifecycle=lifecycle,
                )
            else:
                result = await process_sample(
                    self.http_client,
                    prompt_data,
                    self.eos_token,
                    server_url,
                    self.scheduler.compute_reward,
                    self.tokenizer,
                    timing_out=timing,
                    lifecycle=lifecycle,
                )
        except asyncio.CancelledError:
            lane_freed = lifecycle._lane_freed()  # type: ignore[attr-defined]
            now = time.time()
            self._handle_cancelled_sample(
                request_id, sample_idx, server_url, lane_slot,
                timing.get("start_time", now), timing.get("end_time", now),
                lane_already_freed=lane_freed,
            )
            raise
        except Exception as e:
            if not generate_module._shutting_down:
                _log.warning(f"Individual sample error: request_id={request_id}, idx={sample_idx}: {e}")
            result = {"error": "sample_error", "error_message": str(e)}

        # Free lane slot if not already freed by the lifecycle on_inference_end callback.
        lane_freed = lifecycle._lane_freed()  # type: ignore[attr-defined]
        if not lane_freed:
            if server_url in self.available_server_lane_slots:
                self.available_server_lane_slots[server_url].add(lane_slot)

        # Eagerly fetch OTLP timing while still in this sample's async task.
        # The BatchSpanProcessor flushes every ~200ms, so fetching here
        # (right after sample completion) is much more reliable than deferring
        # to batch-save time when spans may have been pruned or not yet flushed.
        # Guard against CancelledError during the OTLP fetch (which contains
        # an asyncio.sleep await point): the generation result is already
        # available, so we must still record it even if the task is cancelled.
        try:
            if "error" not in result:
                await self._fetch_otlp_timing_for_result(result)
        except asyncio.CancelledError:
            pass

        self._on_individual_sample_complete(request_id, sample_idx, result)

    def _handle_cancelled_sample(
        self, request_id: int, sample_idx: int, server_url: str, lane_slot: int,
        start_time: float, end_time: float,
        lane_already_freed: bool = False,
    ):
        """Handle a cancelled individual sample (e.g. during off-policy cancellation or eval drain)."""
        if not lane_already_freed and server_url in self.available_server_lane_slots:
            self.available_server_lane_slots[server_url].add(lane_slot)

        # Log generation end event for the cancelled sample
        if not generate_module._shutting_down:
            rollout_info = self.inflight_rollout_info.get(request_id)
            sample_id = rollout_info.sample_ids.get(sample_idx, -1) if rollout_info else -1
            server_idx = self._get_server_index(server_url)
            self.wandb_logger.event_logger.log_rollout_event(
                event_type="generation",
                phase="end",
                sample_id=sample_id,
                server_id=server_idx,
                group_id=request_id,
                generation_idx=0,
            )

        group = self.pending_individual_groups.get(request_id)
        if group is None:
            return

        group["remaining"] -= 1
        group["samples"][sample_idx] = {"error": "cancelled", "error_message": "Task cancelled"}

        # Lane freed — schedule dispatch (coalesced, non-blocking).
        self._schedule_dispatch()

        if group["remaining"] == 0:
            self.active_count -= 1
            del self.pending_individual_groups[request_id]
            self.inflight_rollout_info.pop(request_id, None)
            _log.debug(f"Cancelled group {request_id} fully resolved (active={self.active_count})")

    async def _fetch_otlp_timing_for_result(self, result: dict):
        """Eagerly fetch OTLP timing data for a single completed sample.

        Called right after the sample finishes, while still inside the
        sample's async task.  A brief ``asyncio.sleep`` gives the
        BatchSpanProcessor time to flush (OTEL_BSP_SCHEDULE_DELAY=200ms).
        The fetched timing is stored in the sample result dict so that
        ``_log_inference_request_events`` can use it later without
        another (likely stale) lookup.
        """
        if self.otlp_receiver is None:
            return

        # Single-turn: "request_timing" (dict); multi-turn: "request_timings" (list)
        timings = result.get("request_timings")
        if timings is None:
            single = result.get("request_timing")
            timings = [single] if single else []

        for timing in timings:
            vllm_request_id = timing.get("vllm_request_id", "")
            if not vllm_request_id:
                continue

            span_key = f"{vllm_request_id}-0"
            # Try immediate lookup — the span may already be available.
            span_data = self.otlp_receiver.get_and_remove(span_key)
            if span_data is None:
                span_data = self.otlp_receiver.get_and_remove(vllm_request_id)
            if span_data is None:
                # Wait for BSP flush (200ms schedule + margin) and retry.
                await asyncio.sleep(0.3)
                span_data = self.otlp_receiver.get_and_remove(span_key)
                if span_data is None:
                    span_data = self.otlp_receiver.get_and_remove(vllm_request_id)
            if span_data is not None:
                timing["otlp_timing"] = span_data

    async def _fetch_otlp_timing_for_group(self, result: dict):
        """Eagerly fetch OTLP timing for all samples in a completed group.

        Called in group-mode ``_run_prompt`` right after ``process_group``
        returns, so each sample's span is looked up close to completion time
        rather than being deferred to batch-save time.
        """
        if self.otlp_receiver is None:
            return

        request_timings = result.get("request_timings", [])
        for timing in request_timings:
            vllm_request_id = timing.get("vllm_request_id", "")
            if not vllm_request_id:
                continue

            span_key = f"{vllm_request_id}-0"
            span_data = self.otlp_receiver.get_and_remove(span_key)
            if span_data is None:
                span_data = self.otlp_receiver.get_and_remove(vllm_request_id)
            if span_data is None:
                await asyncio.sleep(0.3)
                span_data = self.otlp_receiver.get_and_remove(span_key)
                if span_data is None:
                    span_data = self.otlp_receiver.get_and_remove(vllm_request_id)
            if span_data is not None:
                timing["otlp_timing"] = span_data

    def _on_cancelled_group(
        self, request_id: int, server_url: str, lane_slots: list[int],
        sample_timings: list[dict],
    ):
        """Handle a cancelled group rollout (e.g. during off-policy cancellation or eval drain)."""
        # Return lane slots to the pool.
        if server_url in self.available_server_lane_slots:
            self.available_server_lane_slots[server_url].update(lane_slots)

        rollout_info = self.inflight_rollout_info.pop(request_id, None)
        sample_off_policy = rollout_info.sample_off_policy_steps if rollout_info else {}
        self.active_count -= 1
        _log.debug(f"Cancelled group {request_id} (active={self.active_count})")

        # Log inference events for each sample in the group.
        if not generate_module._shutting_down:
            now = time.time()
            server_idx = self._get_server_index(server_url)
            server_info = self._get_server_info(server_url)
            pre_assigned_ids = rollout_info.sample_ids if rollout_info else {}
            for sample_i in range(config.cfg.group_size):
                sample_id = pre_assigned_ids.get(sample_i, -1)
                self.wandb_logger.event_logger.log_rollout_event(
                    event_type="generation",
                    phase="end",
                    sample_id=sample_id,
                    server_id=server_idx,
                    group_id=request_id,
                )

        # Lanes freed — try to dispatch new groups immediately.
        self._try_start_pending_rollouts()

    def _log_individual_sample_end_event(
        self, request_id: int, sample_idx: int, result: dict, group: dict,
    ):
        """Log generation end event for a single completed individual-mode sample."""
        rollout_info = self.inflight_rollout_info.get(request_id)
        sample_id = rollout_info.sample_ids.get(sample_idx, -1) if rollout_info else -1

        server_url = group["server_url"]
        server_idx = self._get_server_index(server_url)

        # Extract timings (single-turn: "request_timing", multi-turn: "request_timings")
        timings = result.get("request_timings")
        if timings is None:
            single = result.get("request_timing")
            timings = [single] if single else []

        for timing in timings:
            if not timing:
                continue
            # Skip if already logged by lifecycle callback at generation-complete time
            if timing.get("inference_end_logged"):
                continue

            self.wandb_logger.event_logger.log_rollout_event(
                event_type="generation",
                phase="end",
                sample_id=sample_id,
                server_id=server_idx,
                group_id=request_id,
                generation_idx=0,
            )

    def _log_individual_sample_error_end_event(
        self, request_id: int, sample_idx: int, group: dict,
    ):
        """Log generation end event for a failed individual-mode sample."""
        rollout_info = self.inflight_rollout_info.get(request_id)
        sample_id = rollout_info.sample_ids.get(sample_idx, -1) if rollout_info else -1

        server_url = group["server_url"]
        server_idx = self._get_server_index(server_url)

        self.wandb_logger.event_logger.log_rollout_event(
            event_type="generation",
            phase="end",
            sample_id=sample_id,
            server_id=server_idx,
            group_id=request_id,
            generation_idx=0,
        )

    def _on_individual_sample_complete(self, request_id: int, sample_idx: int, result: dict):
        """Store individual sample result; assemble group when all samples are done."""
        group = self.pending_individual_groups.get(request_id)
        if group is None:
            _log.debug(f"Ignoring sample result for unknown group {request_id}")
            return

        group["samples"][sample_idx] = result
        group["remaining"] -= 1

        # Log end event for this sample immediately (per-sample, not waiting for group)
        if not generate_module._shutting_down:
            if "error" not in result:
                self._log_individual_sample_end_event(request_id, sample_idx, result, group)
            else:
                self._log_individual_sample_error_end_event(request_id, sample_idx, group)

        _log.debug(
            f"Individual sample complete: request_id={request_id}, "
            f"sample_idx={sample_idx}, remaining={group['remaining']}"
        )

        if group["remaining"] > 0:
            # A lane freed — schedule dispatch (coalesced, non-blocking).
            self._schedule_dispatch()
            return

        # All samples done — assemble the group
        del self.pending_individual_groups[request_id]

        if group["is_multiturn"]:
            assembled = self._assemble_multiturn_group(group)
        else:
            assembled = self._assemble_single_turn_group(group)

        assembled["model_step"] = self.inference_model_step
        assembled["group_id"] = request_id
        # Store actual per-sample lane slots so logging uses the real lanes
        assembled["individual_lane_slots"] = group["lane_slots"]
        # Mark that inference end events were already logged per-sample
        assembled["inference_events_logged_per_sample"] = True

        self._on_rollout_complete(assembled)

    def _assemble_single_turn_group(self, group_info: dict) -> dict:
        """Assemble individual single-turn sample results into a group dict.

        Produces the same format as ``process_group()`` in generate.py.
        """
        prompt_data = group_info["prompt_data"]
        server_url = group_info["server_url"]
        group_samples = group_info["samples"]

        # Error check (same as process_group)
        errors = [s for s in group_samples if s and "error" in s]
        if errors:
            error_sample = errors[0]
            return {
                "error": error_sample["error"],
                "error_message": error_sample.get("error_message", "Unknown error"),
                "prompt_text": prompt_data["prompt"],
                "env_name": prompt_data["env_name"],
                "server_url": server_url,
                "prompt_tokens": error_sample.get("prompt_tokens"),
                "max_tokens": error_sample.get("max_tokens"),
            }

        rewards = [s["reward"] for s in group_samples]
        sample_metrics_list = [s["sample_metrics"] for s in group_samples]
        golden_answers_list = [s["golden_answers"] for s in group_samples]
        info_turns_list = [s["info_turns"] for s in group_samples]
        sample_tags_list = [s.get("sample_tags", {}) for s in group_samples]
        compute_reward_times = [s["compute_reward_time"] for s in group_samples]
        completion_texts = [
            s["data_completion"]["choices"][0].get("text", "")
            for s in group_samples
        ]

        request_timings = []
        for idx, s in enumerate(group_samples):
            timing_with_sample = dict(s["request_timing"])
            timing_with_sample["sample_idx_in_group"] = idx
            request_timings.append(timing_with_sample)

        vllm_logprobs = [s["vllm_logprobs"] for s in group_samples]
        advantages = compute_advantages(rewards)

        turns_list = [
            [{
                "turn_order": 0,
                "turn_type": "model",
                "content": completion_text,
                "tokens": request_timings[i].get("rollout_tokens", 0),
                "stop_reason": group_samples[i]["data_completion"]["choices"][0].get("finish_reason", ""),
            }]
            for i, completion_text in enumerate(completion_texts)
        ]

        env = prompt_data.get("env")
        system_prompt = (env.system_prompt or "") if env else ""
        instruction_prompt = (getattr(env, "instruction_prompt", None) or "") if env else ""

        sample = prompt_data.get("sample")
        raw_question = ""
        if sample:
            raw_question = sample.prompt if isinstance(sample.prompt, str) else sample.metadata.get("question", "")

        if instruction_prompt and raw_question:
            user_prompt = f"{instruction_prompt}\n\n{raw_question}"
        else:
            user_prompt = raw_question

        tokens_system_prompt = 0
        if system_prompt and self.tokenizer is not None:
            tokens_system_prompt = len(self.tokenizer.encode(system_prompt))

        return {
            "prompt_text": user_prompt,
            "env_name": prompt_data["env_name"],
            "prompt_token_ids": [s["data_completion"]["choices"][0]["prompt_token_ids"] for s in group_samples],
            "completion_token_ids": [s["data_completion"]["choices"][0]["token_ids"] for s in group_samples],
            "completion_texts": completion_texts,
            "total_tokens": [s["data_completion"]["usage"]["total_tokens"] for s in group_samples],
            "rewards": rewards,
            "advantages": advantages,
            "sample_metrics": sample_metrics_list,
            "golden_answers": golden_answers_list,
            "info_turns": info_turns_list,
            "sample_tags": sample_tags_list,
            "request_timings": request_timings,
            "vllm_logprobs": vllm_logprobs,
            "server_url": server_url,
            "turns": turns_list,
            "generations": [s["generations"] for s in group_samples],
            "env_responses": [s["env_responses"] for s in group_samples],
            "tool_calls": [s["tool_calls"] for s in group_samples],
            "stop_reasons": [
                s["data_completion"]["choices"][0].get("finish_reason", "")
                for s in group_samples
            ],
            "system_prompt": system_prompt,
            "tokens_system_prompt": tokens_system_prompt,
            "compute_reward_times": compute_reward_times,
        }

    def _assemble_multiturn_group(self, group_info: dict) -> dict:
        """Assemble individual multi-turn sample results into a group dict.

        Produces the same format as ``process_multiturn_group()`` in generate.py.
        """
        prompt_data = group_info["prompt_data"]
        server_url = group_info["server_url"]
        env = group_info["env"]
        group_samples = group_info["samples"]

        # Error check
        errors = [s for s in group_samples if s and "error" in s]
        if errors:
            error_sample = errors[0]
            sample = prompt_data["sample"]
            prompt_text = ""
            if isinstance(sample.prompt, str):
                prompt_text = sample.prompt
            elif isinstance(sample.prompt, list) and sample.prompt:
                prompt_text = sample.prompt[0].get("content", "")
            return {
                "error": error_sample["error"],
                "error_message": error_sample.get("error_message", "Unknown error"),
                "prompt_text": prompt_text,
                "env_name": prompt_data["env_name"],
                "server_url": server_url,
            }

        rewards = [s["reward"] for s in group_samples]
        sample_metrics_list = [s["sample_metrics"] for s in group_samples]
        golden_answers_list = [s["golden_answers"] for s in group_samples]
        info_turns_list = [s["info_turns"] for s in group_samples]
        sample_tags_list = [s.get("sample_tags", {}) for s in group_samples]
        completion_texts = [s["completion_text"] for s in group_samples]
        prompt_texts = [s["prompt_text"] for s in group_samples]
        compute_reward_times = [s["compute_reward_time"] for s in group_samples]

        all_request_timings = []
        for idx, s in enumerate(group_samples):
            sample_crt = s.get("compute_reward_time", 0.0)
            for timing in s["request_timings"]:
                timing_with_sample = dict(timing)
                timing_with_sample["sample_idx_in_group"] = idx
                timing_with_sample["compute_reward_time"] = sample_crt
                all_request_timings.append(timing_with_sample)

        prompt_token_ids = [s["prompt_ids"] for s in group_samples]
        completion_token_ids = [s["completion_ids"] for s in group_samples]
        completion_masks = [s["completion_mask"] for s in group_samples]
        full_token_ids = [s["full_token_ids"] for s in group_samples]
        vllm_logprobs = [s["vllm_logprobs"] for s in group_samples]

        total_tokens = [
            len(s["prompt_ids"]) + len(s["completion_ids"])
            for s in group_samples
        ]

        advantages = compute_advantages(rewards)
        turns_list = [s["turns"] for s in group_samples]

        system_prompt = group_samples[0].get("system_prompt", "") if group_samples else ""
        if not system_prompt:
            system_prompt = env.system_prompt or ""

        user_prompt = prompt_texts[0] if prompt_texts else ""

        tokens_system_prompt = 0
        if system_prompt and self.tokenizer is not None:
            tokens_system_prompt = len(self.tokenizer.encode(system_prompt))

        return {
            "prompt_text": user_prompt,
            "env_name": prompt_data["env_name"],
            "prompt_token_ids": prompt_token_ids,
            "completion_token_ids": completion_token_ids,
            "completion_masks": completion_masks,
            "full_token_ids": full_token_ids,
            "completion_texts": completion_texts,
            "total_tokens": total_tokens,
            "rewards": rewards,
            "advantages": advantages,
            "sample_metrics": sample_metrics_list,
            "golden_answers": golden_answers_list,
            "info_turns": info_turns_list,
            "sample_tags": sample_tags_list,
            "request_timings": all_request_timings,
            "vllm_logprobs": vllm_logprobs,
            "server_url": server_url,
            "is_multiturn": True,
            "num_turns": [s["num_turns"] for s in group_samples],
            "stop_reasons": [s["stop_reason"] for s in group_samples],
            "turns": turns_list,
            "generations": [s["generations"] for s in group_samples],
            "env_responses": [s["env_responses"] for s in group_samples],
            "tool_calls": [s["tool_calls"] for s in group_samples],
            "system_prompt": system_prompt,
            "tokens_system_prompt": tokens_system_prompt,
            "compute_reward_times": compute_reward_times,
        }

    # ----- End individual sample lane helpers -----

    def _on_rollout_complete(self, result: dict):
        """Handle completed rollout."""
        self.active_count -= 1
        request_id = result.get("group_id", -1)
        rollout_info = self.inflight_rollout_info.pop(request_id, None)
        result["off_policy_steps"] = dict(rollout_info.sample_off_policy_steps) if rollout_info else {}
        result["pre_assigned_sample_ids"] = dict(rollout_info.sample_ids) if rollout_info else {}
        model_step = result.get("model_step", -1)

        # Return per-sample lane slots to the pool.
        # In individual_sample_lanes mode the lanes were already freed
        # per-sample inside _run_individual_sample, so skip here.
        if not self.individual_sample_lanes:
            lane_slots = result.get("individual_lane_slots")
            server_url = result.get("server_url")
            if lane_slots and server_url in self.available_server_lane_slots:
                self.available_server_lane_slots[server_url].update(lane_slots)
        
        _log.debug(f"Rollout complete: request_id={request_id}, model_step={model_step}, active={self.active_count}")
        
        # Check if this was an error result (e.g., prompt too long)
        if "error" in result:
            error_type = result["error"]
            error_msg = result.get("error_message", "Unknown error")
            
            if error_type == "prompt_too_long":
                prompt_tokens = result.get("prompt_tokens", "?")
                max_tokens = result.get("max_tokens", "?")
                _log.warning(f"Discarding group: prompt too long ({prompt_tokens} tokens > {max_tokens} max)")
            elif not generate_module._shutting_down:
                _log.warning(f"Discarding group due to error: {error_type} - {error_msg}")
            
            self.wandb_logger.log_orchestrator_timeline_event(f"rollout_discarded_{error_type}", group_id=request_id)
            # Continue to next prompt without adding to bucket
            if self._has_pending_prompt_data():
                self._schedule_dispatch()
            elif self.active_count == 0:
                if self.bucket:
                    self._save_batch()
                if not self._checkpoint_pending:
                    self.rollout_done.set()
            return

        # Check if result is within async limit
        model_step = result.get("model_step", 0)
        async_level = self._compute_async_level(model_step)
        
        if async_level > config.cfg.max_async_rollout:
            # Discard stale rollout
            _log.debug(f"Discarding stale rollout: model_step={model_step}, async_level={async_level}")
            self.wandb_logger.log_orchestrator_timeline_event("rollout_discarded_max_async", step=model_step, group_id=request_id)
            self._log_discarded_rollouts(result, "max_async")
        elif self._should_discard_zero_advantage_group(result):
            # Discard group with all zero advantages
            _log.debug(f"Discarding group with all zero advantages: model_step={model_step}")
            self.wandb_logger.log_orchestrator_timeline_event("rollout_discarded_zero_advantage", step=model_step, group_id=request_id)
            self._log_discarded_rollouts(result, "zero_advantage")
        else:
            self.wandb_logger.log_orchestrator_timeline_event("rollouts_group_done", step=self.inference_step, group_id=request_id)
            # Add to bucket
            self.bucket.append(result)
            # Log kept rollout immediately (don't wait for batch completion)
            self._log_kept_rollout_group(result)
            rewards = result.get("rewards", [])
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            env_name = result.get("env_name", "unknown")
            _log.debug(f"Group added to bucket: request_id={request_id}, bucket_size={len(self.bucket)}/{self.batch_size}, active={self.active_count}")
            _log.debug(f"  env={env_name}, model_step={model_step}, avg_reward={avg_reward:.4f}")

            # Save batch if ready
            if len(self.bucket) >= self.batch_size:
                _log.debug(f"Bucket full ({len(self.bucket)} groups), saving batch for step {self.inference_step}")
                self._save_batch()

        # Continue or finish
        if self._has_pending_prompt_data():
            self._schedule_dispatch()
        elif self.active_count == 0:
            # Only flush remaining bucket if we still need more steps
            if self.bucket and self.inference_step < self.num_steps:
                self._save_batch()
            if not self._checkpoint_pending:
                self.rollout_done.set()

    def _compute_async_level(self, model_step: int) -> int:
        """
        Compute async level for a rollout.
        
        Async level represents how far ahead inference is from trainer:
        - 0 means: trainer finished step N, generating for step N+1 (fully sync)
        - 4 means: trainer finished step N, generating for step N+5
        
        Formula: inference_step - model_step - 1
        (where model_step is the last completed trainer step that inference has)
        """
        return self.inference_step - model_step - 1

    def _get_server_index(self, server_url: str) -> int:
        """Get the server index (0, 1, ...) from the server URL.

        Uses the full inference group URL list (stable across eval
        drain/restore) so that server indices remain consistent.
        """
        if self.inference_group is not None:
            try:
                return list(self.inference_group.server_urls).index(server_url)
            except ValueError:
                pass
        try:
            return self.server_urls.index(server_url)
        except ValueError:
            return -1

    def _get_server_info(self, server_url: str) -> dict:
        """Get full server metadata by URL."""
        if self.inference_group is None:
            return {}
        for info in self.inference_group.server_infos:
            if info.get("url") == server_url:
                return info
        return {}

    def _should_discard_zero_advantage_group(self, result: dict) -> bool:
        """
        Check if group should be discarded due to all-identical advantages.

        Returns True if config is enabled and all advantages are identical
        (no intra-group variation → no learning signal).

        This works regardless of whether advantages are group-normalized
        (all-identical rewards → all zeros), group-mean-centered for
        batch-level normalization (all-identical rewards → all zeros),
        or raw rewards for k>1 batch mode (all-identical → all same value).

        Skipped for single-sample groups (k=1): with one sample there is no
        intra-group comparison to make.  The learning signal for k=1 setups
        comes from batch-level normalization, not from intra-group variation.
        """
        if not config.cfg.discard_group_zero_advantage:
            return False

        advantages = result.get("advantages", [])
        if len(advantages) <= 1:
            return False

        # Check if all advantages are identical (with small tolerance for floating point)
        return all(abs(adv - advantages[0]) < 1e-9 for adv in advantages)

    def _log_discarded_rollouts(self, result: dict, discard_reason: str):
        """
        Log all samples from a discarded rollout group.

        Args:
            result: The rollout result dict from process_group
            discard_reason: Why the group was discarded (e.g. "max_async", "zero_advantage")
        """
        prompt_text = result.get("prompt_text", "")
        env_name = result.get("env_name", "")
        group_id = result.get("group_id", -1)
        turns_list = result.get("turns", [])  # Per-sample turns
        generations_list = result.get("generations", [])
        env_responses_list = result.get("env_responses", [])
        tool_calls_list = result.get("tool_calls", [])
        stop_reasons = result.get("stop_reasons", [])
        rewards = result.get("rewards", [])
        advantages = result.get("advantages", [])
        sample_metrics_list = result.get("sample_metrics", [])
        golden_answers_list = result.get("golden_answers", [])
        info_turns_list = result.get("info_turns", [])
        sample_tags_list = result.get("sample_tags", [])
        system_prompt = result.get("system_prompt", "")
        prompt_token_ids = result.get("prompt_token_ids", [])
        completion_token_ids = result.get("completion_token_ids", [])
        full_token_ids = result.get("full_token_ids", [])  # Full sequence for raw_string
        total_tokens_list = result.get("total_tokens", [])
        compute_reward_times = result.get("compute_reward_times", [])

        # Compute tokens_system_prompt and tokens_prompt if we have a tokenizer
        tokens_system_prompt = 0
        tokens_prompt = 0
        if self.tokenizer is not None:
            if system_prompt:
                tokens_system_prompt = len(self.tokenizer.encode(system_prompt))
            if prompt_text:
                tokens_prompt = len(self.tokenizer.encode(prompt_text))

        num_samples = len(turns_list)
        pre_assigned = result.get("pre_assigned_sample_ids", {})

        for idx in range(num_samples):
            reward = rewards[idx] if idx < len(rewards) else 0.0
            advantage = advantages[idx] if idx < len(advantages) else 0.0
            sample_metrics = sample_metrics_list[idx] if idx < len(sample_metrics_list) else {}
            golden_answers = golden_answers_list[idx] if idx < len(golden_answers_list) else {}
            info_turns = info_turns_list[idx] if idx < len(info_turns_list) else []
            sample_tags = sample_tags_list[idx] if idx < len(sample_tags_list) else {}
            generations = generations_list[idx] if idx < len(generations_list) else []
            env_responses = env_responses_list[idx] if idx < len(env_responses_list) else []
            tool_calls = tool_calls_list[idx] if idx < len(tool_calls_list) else []
            stop_reason = stop_reasons[idx] if idx < len(stop_reasons) else ""

            prompt_ids = prompt_token_ids[idx] if idx < len(prompt_token_ids) else []
            comp_ids = completion_token_ids[idx] if idx < len(completion_token_ids) else []
            full_ids = full_token_ids[idx] if idx < len(full_token_ids) else []
            total_tokens = total_tokens_list[idx] if idx < len(total_tokens_list) else len(prompt_ids) + len(comp_ids)

            raw_string = ""
            if self.tokenizer is not None:
                if full_ids:
                    raw_string = self.tokenizer.decode(full_ids, skip_special_tokens=False)
                elif prompt_ids or comp_ids:
                    raw_string = self.tokenizer.decode(prompt_ids + comp_ids, skip_special_tokens=False)

            sample_id = pre_assigned.get(idx, self.next_sample_idx)
            if idx not in pre_assigned:
                self.next_sample_idx += 1

            compute_reward_time = compute_reward_times[idx] if idx < len(compute_reward_times) else 0.0

            self.wandb_logger.event_logger.log_discarded_generation_rollout(
                discard_reason=discard_reason,
                trainer_step=self.trainer_step,
                inference_step=self.inference_step,
                group_id=group_id,
                sample_id=sample_id,
                prompt=prompt_text,
                generations=generations,
                env_responses=env_responses,
                tool_calls=tool_calls,
                reward=reward,
                advantage=advantage,
                env=env_name,
                sample_metrics=sample_metrics,
                golden_answers=golden_answers,
                info_turns=info_turns,
                sample_tags=sample_tags,
                tokens_prompt=tokens_prompt,
                system_prompt=system_prompt,
                tokens_system_prompt=tokens_system_prompt,
                total_tokens=total_tokens,
                raw_string=raw_string,
                compute_reward_time=compute_reward_time,
                stop_reason=stop_reason,
            )

    def _log_kept_rollout_group(self, result: dict):
        """
        Log all samples from a kept rollout group immediately.

        Called directly when a rollout group is kept, so generations
        appear in wandb files without waiting for batch completion.

        Args:
            result: The rollout result dict from process_group
        """
        prompt_text = result.get("prompt_text", "")
        env_name = result.get("env_name", "")
        group_id = result.get("group_id", -1)
        turns_list = result.get("turns", [])  # Per-sample turns
        generations_list = result.get("generations", [])
        env_responses_list = result.get("env_responses", [])
        tool_calls_list = result.get("tool_calls", [])
        stop_reasons = result.get("stop_reasons", [])
        rewards = result.get("rewards", [])
        advantages = result.get("advantages", [])
        sample_metrics_list = result.get("sample_metrics", [])
        golden_answers_list = result.get("golden_answers", [])
        info_turns_list = result.get("info_turns", [])
        sample_tags_list = result.get("sample_tags", [])
        system_prompt = result.get("system_prompt", "")
        prompt_token_ids = result.get("prompt_token_ids", [])
        completion_token_ids = result.get("completion_token_ids", [])
        full_token_ids = result.get("full_token_ids", [])  # Full sequence for raw_string
        total_tokens_list = result.get("total_tokens", [])
        compute_reward_times = result.get("compute_reward_times", [])

        # Compute tokens_system_prompt and tokens_prompt if we have a tokenizer
        tokens_system_prompt = 0
        tokens_prompt = 0
        if self.tokenizer is not None:
            if system_prompt:
                tokens_system_prompt = len(self.tokenizer.encode(system_prompt))
            if prompt_text:
                tokens_prompt = len(self.tokenizer.encode(prompt_text))

        num_samples = len(turns_list)
        pre_assigned = result.get("pre_assigned_sample_ids", {})

        for idx in range(num_samples):
            reward = rewards[idx] if idx < len(rewards) else 0.0
            advantage = advantages[idx] if idx < len(advantages) else 0.0
            sample_metrics = sample_metrics_list[idx] if idx < len(sample_metrics_list) else {}
            golden_answers = golden_answers_list[idx] if idx < len(golden_answers_list) else {}
            info_turns = info_turns_list[idx] if idx < len(info_turns_list) else []
            sample_tags = sample_tags_list[idx] if idx < len(sample_tags_list) else {}
            generations = generations_list[idx] if idx < len(generations_list) else []
            env_responses = env_responses_list[idx] if idx < len(env_responses_list) else []
            tool_calls = tool_calls_list[idx] if idx < len(tool_calls_list) else []
            stop_reason = stop_reasons[idx] if idx < len(stop_reasons) else ""

            # Enrich generation dicts with vLLM timing from request_timings + OTLP data
            _req_timings = result.get("request_timings", [])
            if not result.get("is_multiturn"):
                _t = _req_timings[idx] if idx < len(_req_timings) else {}
                _otlp = _t.get("otlp_timing", {}) or {}
                if generations:
                    generations[0]["queue_time"] = _otlp.get("queue_time", 0.0)
                    generations[0]["ttft"] = _otlp.get("time_to_first_token", 0.0)
                    generations[0]["prefill_time"] = _otlp.get("prefill_time", 0.0)
                    generations[0]["decode_time"] = _otlp.get("decode_time", 0.0)
                    generations[0]["inference_time"] = _otlp.get("inference_time", 0.0)
                    generations[0]["e2e_latency"] = _otlp.get("e2e_latency", 0.0)
                    generations[0]["vllm_request_id"] = _t.get("vllm_request_id", "")
                    _server_url = result.get("server_url", "")
                    generations[0]["server_id"] = self._get_server_index(_server_url) if _server_url else -1

            prompt_ids = prompt_token_ids[idx] if idx < len(prompt_token_ids) else []
            comp_ids = completion_token_ids[idx] if idx < len(completion_token_ids) else []
            full_ids = full_token_ids[idx] if idx < len(full_token_ids) else []
            total_tokens = total_tokens_list[idx] if idx < len(total_tokens_list) else len(prompt_ids) + len(comp_ids)

            raw_string = ""
            if self.tokenizer is not None:
                if full_ids:
                    raw_string = self.tokenizer.decode(full_ids, skip_special_tokens=False)
                elif prompt_ids or comp_ids:
                    raw_string = self.tokenizer.decode(prompt_ids + comp_ids, skip_special_tokens=False)

            sample_id = pre_assigned.get(idx, self.next_sample_idx)
            if idx not in pre_assigned:
                self.next_sample_idx += 1

            compute_reward_time = compute_reward_times[idx] if idx < len(compute_reward_times) else 0.0

            self.wandb_logger.event_logger.log_generation_rollout(
                step=self.inference_step,
                group_id=group_id,
                sample_id=sample_id,
                prompt=prompt_text,
                generations=generations,
                env_responses=env_responses,
                tool_calls=tool_calls,
                reward=reward,
                advantage=advantage,
                env=env_name,
                sample_metrics=sample_metrics,
                golden_answers=golden_answers,
                info_turns=info_turns,
                sample_tags=sample_tags,
                tokens_prompt=tokens_prompt,
                system_prompt=system_prompt,
                tokens_system_prompt=tokens_system_prompt,
                total_tokens=total_tokens,
                raw_string=raw_string,
                compute_reward_time=compute_reward_time,
                stop_reason=stop_reason,
            )

    def _log_inference_request_events(
        self,
        request_timings: list[dict],
        group_id: int,
        server_url: str,
        sample_idx_map: dict[int, int],
        off_policy_steps: dict[int, int] | None = None,
        individual_lane_slots: list[int] | None = None,
    ):
        """Log generation end events for request timings with exact sample/group mapping."""
        if not request_timings:
            return

        server_idx = self._get_server_index(server_url)
        for timing in request_timings:
            # Skip if already logged by lifecycle callback at generation-complete time
            if timing.get("inference_end_logged"):
                continue
            sample_idx_in_group = timing.get("sample_idx_in_group", -1)
            sample_id = sample_idx_map.get(sample_idx_in_group, -1)

            self.wandb_logger.event_logger.log_rollout_event(
                event_type="generation",
                phase="end",
                sample_id=sample_id,
                server_id=server_idx,
                group_id=group_id,
                generation_idx=0,
            )

    def _try_start_pending_rollouts(self):
        """Try to start new rollouts after weights are updated."""
        while self._has_pending_prompt_data():
            if not self._start_next_prompt():
                break
        
        if self.active_count > 0:
            _log.debug(f"Resumed rollout: {self.active_count} active prompts")

    def _flush_pending_batches(self):
        """Submit batches that were buffered while a checkpoint was in progress.

        Called by the watcher after a checkpoint save completes.  If a
        buffered batch itself needs a checkpoint, re-sets the flag and
        stops — the watcher will checkpoint again on the next iteration.
        """
        self._checkpoint_pending = False
        while self._pending_batches:
            step, trainer_data = self._pending_batches.pop(0)
            self.pending_train_step_refs[step] = self.trainer_group.submit_train_step(
                step=step,
                trainer_data_per_rank=trainer_data,
            )
            _log.info(f"Submitted buffered batch for step {step}")
            is_last = (step == self.num_steps - 1)
            if self._should_save_checkpoint(step, is_last):
                self._checkpoint_pending = True
                _log.info(f"Checkpoint pending at buffered step {step}, pausing again")
                break
        self._try_start_pending_rollouts()

    def _save_batch(self):
        """Schedule batch preprocessing + submission as an async task to avoid blocking event loop."""
        if self.trainer_group is None:
            raise RuntimeError("Trainer group is not initialized")

        batch = self.bucket[:self.batch_size]
        self.bucket = self.bucket[self.batch_size:]

        # Capture state needed by the async task
        _start_idx = self.next_sample_idx  # For preprocess_batch (internal trainer use)
        _step = self.inference_step
        _checkpoint_pending = self._checkpoint_pending

        # Count total samples (sample_ids already pre-allocated at dispatch time)
        total_samples = sum(len(group["completion_token_ids"]) for group in batch)
        self._step_batches[_step] = batch
        self.inference_step += 1

        # Check if checkpoint is needed (must be done before next batch)
        submitted_step = _step
        is_last = (submitted_step == self.num_steps - 1)
        if self._should_save_checkpoint(submitted_step, is_last):
            self._checkpoint_pending = True
            _log.info(f"Checkpoint pending at step {submitted_step}, pausing batch submissions")

        # Schedule heavy work as async task — preprocess_batch runs in thread pool
        # to avoid blocking the event loop. Rollout logging is now done immediately
        # per-group in _log_kept_rollout_group (called when group is added to bucket).
        asyncio.ensure_future(self._save_batch_async(batch, _start_idx, _step, _checkpoint_pending))

    async def _save_batch_async(self, batch, start_sample_idx, step, checkpoint_pending):
        """Run heavy batch preprocessing in thread pool, then submit to trainer."""
        # preprocess_batch does FFD packing, padding, tensor creation — heavy CPU
        trainer_data = await asyncio.get_event_loop().run_in_executor(
            None, preprocess_batch, batch, self.num_trainer_data_shards, start_sample_idx,
        )

        if checkpoint_pending:
            self._pending_batches.append((step, trainer_data))
            _log.info(f"Buffered batch for step {step} (checkpoint pending)")
            return

        # Submit train step to Ray
        self.pending_train_step_refs[step] = self.trainer_group.submit_train_step(
            step=step,
            trainer_data_per_rank=trainer_data,
        )

        self.wandb_logger.log_orchestrator_timeline_event("save_batch", step=step)
        _log.debug(f"Submitted trainer step {step} to Ray actors")
        _log.info(f"Batch submitted to trainer for step {step}")

    def _prefetch_active(self) -> bool:
        return self.prefetch_enabled and self.prefetch_buffer_size > 0

    def _has_pending_prompt_data(self) -> bool:
        """Check if we should keep dispatching prompts.
        
        Keeps going as long as we haven't saved all required batches.
        The scheduler is infinite, so the stop condition is based on
        training steps, not sample availability.
        """
        return self.inference_step < self.num_steps

    def _get_next_prompt_data(self) -> dict | None:
        if self.prefetch_queue is not None:
            try:
                prompt_data = self.prefetch_queue.get_nowait()
                _log.debug(f"Got prompt data from prefetch queue (queue_size={self.prefetch_queue.qsize()})")
                return prompt_data
            except asyncio.QueueEmpty:
                _log.debug("Prefetch queue empty, falling back to scheduler")
        
        if self.inference_step < self.num_steps:
            _log.debug("Requesting sample from scheduler...")
            prompt_data = self.scheduler.get_next_sample()
            env_name = prompt_data.get("env_name", "unknown")
            _log.debug(f"Received sample from scheduler: env={env_name}")
            if self._prefetch_active():
                prompt_data = self._prepare_prompt_data(prompt_data)
            return prompt_data
        
        _log.debug("All training steps dispatched, no more prompts needed")
        return None

    def _prepare_prompt_data(self, prompt_data: dict) -> dict:
        env = prompt_data.get("env")
        if env is None:
            return prompt_data
        
        if getattr(env, "is_multi_turn", False):
            if "prefetched_messages" in prompt_data or "prefetched_prompt_str" in prompt_data:
                return prompt_data
            
            sample = prompt_data.get("sample")
            if sample is None:
                return prompt_data
            
            try:
                messages = env.get_initial_prompt(sample, self.tokenizer)
            except Exception as exc:
                _log.warning(f"Multi-turn prefetch failed to build messages: {exc}")
                return prompt_data
            
            prompt_data["prefetched_messages"] = messages
            if self.tokenizer is None:
                return prompt_data
            
            try:
                chat_kwargs = get_chat_template_kwargs()
                prompt_str = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_kwargs,
                )
                prompt_data["prefetched_prompt_str"] = prompt_str
                prompt_data["prefetched_prompt_token_count"] = len(
                    self.tokenizer.encode(prompt_str)
                )
            except Exception as exc:
                _log.warning(f"Multi-turn prefetch failed to build prompt: {exc}")
            return prompt_data
        
        if "prepared_prompt" in prompt_data:
            return prompt_data
        
        raw_prompt = prompt_data.get("prompt", "")
        try:
            if env is not None and hasattr(env, "format_prompt"):
                chat_kwargs = get_chat_template_kwargs()
                prompt_str = env.format_prompt(
                    raw_prompt,
                    tokenizer=self.tokenizer,
                    chat_template_kwargs=chat_kwargs,
                )
            else:
                prompt_str = raw_prompt
        except Exception as exc:
            _log.warning(f"Prompt prefetch failed; using raw prompt: {exc}")
            return prompt_data
        
        prompt_data["prepared_prompt"] = prompt_str
        if self.tokenizer is not None:
            try:
                prompt_data["prepared_prompt_token_count"] = len(
                    self.tokenizer.encode(prompt_str)
                )
            except Exception as exc:
                _log.warning(f"Prompt token count failed during prefetch: {exc}")
        return prompt_data

    async def _prefetch_loop(self):
        """Prepare prompt data ahead of scheduling to reduce dispatch latency."""
        try:
            while self.inference_step < self.num_steps:
                if self.prefetch_queue is None:
                    return
                prompt_data = self.scheduler.get_next_sample()
                prompt_data = self._prepare_prompt_data(prompt_data)
                await self.prefetch_queue.put(prompt_data)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.warning(f"Prompt prefetch loop stopped: {exc}")

    # =========================================================================
    # Weight Synchronization
    # =========================================================================

    async def _update_inference_weights(self, step_results: list[dict], weight_sync_group: str = "full"):
        """
        Consume completed trainer step results.

        In Ray mode, each trainer step already performs NCCL broadcast to inference.
        This method aggregates the returned metrics/events and advances model state.
        """
        step = self.trainer_step

        # Use the trainer's wall-clock timestamp for the weight_update event
        # so it aligns with trainer/inference events.  In multi-node setups,
        # the delay between the trainer finishing and the orchestrator
        # processing the result via Ray can be significant (~30s).
        trainer_end_times = [
            r["step_end_time"] for r in step_results if "step_end_time" in r
        ]
        weight_update_ts = min(trainer_end_times) if trainer_end_times else None
        self.wandb_logger.log_orchestrator_timeline_event("weight_update", step=step, timestamp=weight_update_ts)

        step_metrics = []
        timeline_events = []
        weight_sync_load_refs = []

        for result in step_results:
            step_metrics.extend(result.get("step_metrics", []))
            timeline_events.extend(result.get("timeline_events", []))
            weight_sync_load_refs.extend(result.get("weight_sync_load_refs", []))

        self.wandb_logger.log_trainer_metrics_payload(
            step=step,
            payload={
                "step_metrics": step_metrics,
                "timeline_events": timeline_events,
            },
        )

        batch = self._step_batches.pop(step, None)
        if batch is not None:
            self.wandb_logger.log_rollout_reward_metrics(step, batch)

        # Resolve inference timing asynchronously.  By this point inference is
        # almost certainly done (the training step took seconds).  Using
        # asyncio.to_thread so the event loop stays responsive.
        if weight_sync_load_refs:
            weight_sync_events = await asyncio.to_thread(ray.get, weight_sync_load_refs)
        else:
            weight_sync_events = []

        for event in weight_sync_events:
            self.wandb_logger.event_logger.log_infra_event(
                event_type="weight_sync",
                phase="start",
                timestamp=event.get("start_time", time.time()),
                server_id=event.get("server", -1),
                step=step,
            )
            self.wandb_logger.event_logger.log_infra_event(
                event_type="weight_sync",
                phase="end",
                timestamp=event.get("end_time", time.time()),
                server_id=event.get("server", -1),
                step=step,
            )

        # Advance model step trackers based on which group got the broadcast
        self.inference_model_step = step
        if weight_sync_group == "full":
            self.eval_server_model_step = step
        # When training_only, eval_server_model_step stays frozen
        _log.debug(
            f"Inference model updated to step {self.inference_model_step} "
            f"(eval_server_model_step={self.eval_server_model_step}, group={weight_sync_group})",
            step=self.inference_model_step,
        )
        _log.info(f"Inference weights updated to step {self.inference_model_step}")

        # Per-rollout off-policy tracking: increment counters and cancel stale rollouts
        if self.inflight_rollout_info:
            max_off_policy = config.cfg.max_off_policy_steps
            cancelled_request_ids = []

            for request_id, info in self.inflight_rollout_info.items():
                for sidx in info.sample_off_policy_steps:
                    # Only increment if the sample's HTTP call hasn't completed yet
                    timing = info.sample_timings.get(sidx, {})
                    if "end_time" not in timing:
                        info.sample_off_policy_steps[sidx] += 1
                max_sample = max(info.sample_off_policy_steps.values()) if info.sample_off_policy_steps else 0
                if max_off_policy >= 0 and max_sample > max_off_policy:
                    cancelled_request_ids.append(request_id)
                    for task in info.tasks:
                        if not task.done():
                            task.cancel()
                    if self.individual_sample_lanes:
                        self._cancel_queued_individual_samples(request_id)

            # Don't pop inflight_rollout_info here — the CancelledError handlers
            # (_on_cancelled_group / _handle_cancelled_sample) will pop it when
            # they run, and they need the data (off_policy_steps, etc.).

            if cancelled_request_ids:
                _log.info(
                    f"Cancelled {len(cancelled_request_ids)} stale rollouts "
                    f"(off_policy_steps > {max_off_policy})"
                )
                self.wandb_logger.log_orchestrator_timeline_event(
                    "rollout_cancelled_off_policy", step=step, timestamp=weight_update_ts
                )

        if self.waiting_for_trainer:
            self.waiting_for_trainer = False
            self.wandb_logger.log_orchestrator_timeline_event("rollout_resumed_max_async", step=self.inference_model_step, timestamp=weight_update_ts)
            self._try_start_pending_rollouts()

    def _cancel_queued_individual_samples(self, request_id: int):
        """Remove queued-but-not-dispatched samples for a cancelled group."""
        for server_url, queue in self.individual_sample_queues.items():
            original_len = len(queue)
            filtered = collections.deque(
                (rid, idx) for rid, idx in queue if rid != request_id
            )
            removed = original_len - len(filtered)
            self.individual_sample_queues[server_url] = filtered
            if removed > 0:
                group = self.pending_individual_groups.get(request_id)
                if group is not None:
                    for _ in range(removed):
                        group["remaining"] -= 1
                        for si in range(len(group["samples"])):
                            if group["samples"][si] is None:
                                group["samples"][si] = {
                                    "error": "cancelled",
                                    "error_message": "Off-policy cancellation",
                                }
                                break

        # If all remaining samples were queued (none running), remaining is now 0
        # and no CancelledError handler will fire to clean up.
        group = self.pending_individual_groups.get(request_id)
        if group is not None and group["remaining"] == 0:
            self.active_count -= 1
            del self.pending_individual_groups[request_id]
            self.inflight_rollout_info.pop(request_id, None)
            _log.debug(f"Cancelled group {request_id} fully resolved via queue removal (active={self.active_count})")

    def _init_weight_broadcast(self):
        """Initialize NCCL broadcast between Ray trainer ranks and inference servers.

        When ``EVAL_NUM_SERVERS > 0``, creates two NCCL groups:
        - **full**: trainer rank 0 + ALL inference servers
        - **training_only**: trainer rank 0 + non-eval inference servers
        Eval-only servers join only the full group.
        """
        if self.inference_group is None or self.trainer_group is None:
            raise RuntimeError("Ray groups must be started before initializing weight broadcast")

        num_inference = len(self.inference_group.actors)
        tp_size = self.inference_tp_size
        total_inference_ranks = num_inference * tp_size
        world_size = 1 + total_inference_ranks
        nccl_host, nccl_port = self.trainer_group.allocate_weight_broadcast_endpoint()
        inference_node_ips = {
            info.get("node_ip")
            for info in self.inference_group.server_infos
            if info.get("node_ip")
        }
        prefer_loopback = config.cfg.ray_broadcast_prefer_loopback_if_single_node
        if (
            prefer_loopback
            and len(inference_node_ips) == 1
            and nccl_host in inference_node_ips
        ):
            _log.debug(
                "All inference servers are colocated with trainer rank0 on one node; "
                "using loopback host 127.0.0.1 for NCCL bootstrap."
            )
            nccl_host = "127.0.0.1"
        else:
            _log.debug(
                "Broadcast topology: "
                f"trainer_rank0_ip={nccl_host}, inference_node_ips={sorted(inference_node_ips)}"
            )

        _log.debug(
            f"Initializing NCCL weight broadcast on {nccl_host}:{nccl_port} "
            f"(world_size={world_size})"
        )

        # Determine which actors are training-only vs eval-only
        num_eval = self.eval_num_servers
        num_training = num_inference - num_eval
        all_actors = list(self.inference_group.actors)
        training_actors = all_actors[:num_training] if num_eval > 0 else all_actors
        eval_actors = all_actors[num_training:] if num_eval > 0 else []

        # --- Full group: ALL inference servers ---
        # Each server gets a base rank; TP workers within offset by their tp_rank.
        inference_refs = [
            actor.init_broadcast.remote(
                host=nccl_host,
                port=nccl_port,
                world_size=world_size,
                rank=idx * tp_size + 1,
                group="full",
            )
            for idx, actor in enumerate(all_actors)
        ]

        # --- Training-only group (only when eval servers exist) ---
        training_only_port = None
        training_only_refs = []
        if num_eval > 0 and num_training > 0:
            _, training_only_port = self.trainer_group.allocate_weight_broadcast_endpoint()
            training_only_world_size = 1 + num_training * tp_size
            _log.debug(
                f"Initializing training-only NCCL group on {nccl_host}:{training_only_port} "
                f"(world_size={training_only_world_size})"
            )
            training_only_refs = [
                actor.init_broadcast.remote(
                    host=nccl_host,
                    port=training_only_port,
                    world_size=training_only_world_size,
                    rank=idx * tp_size + 1,
                    group="training_only",
                )
                for idx, actor in enumerate(training_actors)
            ]

        # --- Trainer side: create communicators for both groups ---
        trainer_refs = [
            actor.configure_weight_broadcast.remote(
                master_addr=nccl_host,
                master_port=nccl_port,
                num_inference_servers=total_inference_ranks,
                inference_actors=all_actors,
                training_only_port=training_only_port,
                num_training_servers=num_training * tp_size if num_eval > 0 else None,
                training_inference_actors=training_actors if num_eval > 0 else None,
            )
            for actor in self.trainer_group.actors
        ]

        timeout_s = int(config.cfg.ray_broadcast_init_timeout_s)
        try:
            ray.get([*inference_refs, *training_only_refs, *trainer_refs], timeout=timeout_s)
        except Exception as exc:
            raise RuntimeError(
                "NCCL weight broadcast initialization failed or timed out. "
                f"host={nccl_host} port={nccl_port} world_size={world_size}"
            ) from exc

        _log.debug(
            f"All {num_inference} inference servers joined full broadcast group"
            + (f" ({num_training} also in training-only group)" if num_eval > 0 else "")
        )

    # =========================================================================
    # Eval Management
    # =========================================================================

    async def _handle_eval_step(self, step: int):
        """Triggered when a training step that is an eval step finishes.

        The ``step`` parameter is the 0-indexed internal training step.
        Externally (logs, saved results) we report as ``step + 1`` so that
        eval numbering is 1-indexed.

        The trainer already broadcast on the full group (eval servers have
        this step's weights) and set ``_eval_active = True`` internally.
        """
        eval_configs_for_step = self.eval_schedule.get(step, [])
        if not eval_configs_for_step:
            return

        display_step = step + 1
        _log.info(
            f"Eval step {display_step}: running {len(eval_configs_for_step)} eval(s) "
            f"({', '.join(ec.name for ec in eval_configs_for_step)})",
            step=display_step,
        )
        self.eval_active = True

        # 1. Remove eval servers from training pool and drain them
        await self._drain_eval_servers()

        # 2. Run all evals concurrently on eval servers
        try:
            await self._run_eval_phase(
                eval_configs=eval_configs_for_step,
                step=display_step,
                model_step=self.eval_server_model_step + 1,
                label="step",
            )
        except Exception as exc:
            _log.exception(f"Eval phase failed at step {display_step}: {exc}")

        # 3. Signal trainer that eval is done so it unblocks
        _log.info("Sending eval_done to trainer actors", step=step)
        await asyncio.to_thread(
            lambda: ray.get([
                actor.eval_done.remote()
                for actor in self.trainer_group.actors
            ])
        )
        self.eval_active = False

        # 4. Restore eval servers after next full-group weight sync
        await self._restore_eval_servers()

    async def _run_eval_phase(
        self,
        eval_configs: list[EvalConfig],
        step: int,
        model_step: int,
        label: str,
        eval_server_urls: list[str] | None = None,
    ):
        """Run a set of evals on the given servers and log results.

        Args:
            eval_server_urls: Servers to use. Defaults to ``self.eval_server_urls``.
        """
        eval_server_urls = eval_server_urls or self.eval_server_urls
        if not self.eval_runner or not eval_server_urls:
            return

        self.wandb_logger.log_orchestrator_timeline_event(f"eval_{label}_start", step=step)
        _log.info(f"Starting eval phase ({label}) at step {step}")

        eval_max_concurrent = self.max_concurrent_prompts_per_server * config.cfg.group_size
        results = await self.eval_runner.run_evals(
            eval_configs=eval_configs,
            eval_server_urls=eval_server_urls,
            max_concurrent_per_server=eval_max_concurrent,
            step=step,
            model_step=model_step,
            http_client=self.http_client,
        )

        for result in results:
            if not isinstance(result, dict):
                continue
            eval_name = result.get("eval_name", "unknown")
            avg_metrics = result.get("avg_metrics", {})
            num_samples = result.get("num_samples", 0)
            num_completions = result.get("num_completions", 0)

            _log.info(
                f"Eval {eval_name} (step={step}): "
                f"{num_samples} samples, {num_completions} completions, "
                f"metrics={avg_metrics}",
                step=step,
            )

            # Log individual sample results as EvalGenerationRollout/EvalPrompt + rollout events
            all_server_urls = list(self.inference_group.server_urls) if self.inference_group else []
            sample_results = result.get("sample_results", [])

            # Allocate globally unique group_ids: one per unique prompt (shared across pass@k completions)
            eval_group_ids: dict[int, int] = {}
            for sr in sample_results:
                if sr.sample_idx not in eval_group_ids:
                    eval_group_ids[sr.sample_idx] = self.next_request_id
                    self.next_request_id += 1

            for sr in sample_results:
                eval_sample_id = self.next_sample_idx
                self.next_sample_idx += 1
                eval_request_id = self.next_request_id
                self.next_request_id += 1
                eval_group_id = eval_group_ids[sr.sample_idx]

                # Log generation end rollout event for eval
                if sr.start_time and sr.end_time and sr.server_url:
                    server_idx = all_server_urls.index(sr.server_url) if sr.server_url in all_server_urls else -1

                    self.wandb_logger.event_logger.log_rollout_event(
                        event_type="generation",
                        phase="end",
                        sample_id=eval_sample_id,
                        server_id=server_idx,
                        group_id=eval_group_id,
                        generation_idx=0,
                    )

                tokens_prompt = 0
                tokens_system_prompt = 0
                if self.tokenizer is not None:
                    if sr.prompt_text:
                        tokens_prompt = len(self.tokenizer.encode(sr.prompt_text))
                    if sr.system_prompt:
                        tokens_system_prompt = len(self.tokenizer.encode(sr.system_prompt))

                self.wandb_logger.log_eval_prompt(EvalPrompt(
                    step=step,
                    eval_name=eval_name,
                    model_step=model_step,
                    sample_idx=sr.sample_idx,
                    env=sr.env_name or eval_name,
                    prompt=sr.prompt_text,
                    tokens_prompt=tokens_prompt,
                    system_prompt=sr.system_prompt,
                    tokens_system_prompt=tokens_system_prompt,
                ))

                # Build generation/env_response records from turns
                eval_generations: list[GenerationRecord] = []
                eval_env_responses: list[EnvResponseRecord] = []
                gen_idx = 0
                for t in sr.turns:
                    turn_type = t.get("turn_type", "model")
                    if turn_type == "model":
                        eval_generations.append(GenerationRecord(
                            generation_idx=gen_idx,
                            content=t.get("content", ""),
                            tokens=t.get("tokens", 0),
                            stop_reason=t.get("stop_reason", ""),
                        ))
                        gen_idx += 1
                    else:
                        eval_env_responses.append(EnvResponseRecord(
                            generation_idx=max(0, gen_idx - 1),
                            content=t.get("content", ""),
                            turn_type=turn_type,
                            tokens=t.get("tokens", 0),
                            response_time=t.get("environment_response_time", 0.0),
                        ))

                self.wandb_logger.log_eval_rollout(EvalGenerationRollout(
                    step=step,
                    eval_name=eval_name,
                    model_step=model_step,
                    sample_idx=sr.sample_idx,
                    completion_idx=sr.completion_idx,
                    env=sr.env_name or eval_name,
                    generations=eval_generations,
                    env_responses=eval_env_responses,
                    sample_metrics=sr.metrics,
                    golden_answers=sr.golden_answers,
                    info_turns=sr.info_turns,
                    sample_tags=sr.sample_tags,
                    compute_eval_metrics_time=sr.compute_eval_metrics_time,
                    stop_reason=sr.stop_reason or "",
                ))

        self.wandb_logger.log_orchestrator_timeline_event(f"eval_{label}_done", step=step)
        _log.info(f"Eval phase ({label}) completed at step {step}")

    async def _drain_eval_servers(self):
        """Remove eval servers from training pool and cancel in-flight tasks."""
        # 1. Remove capacity tracking and rebuild server list FIRST so that
        #    any rollout completing during the drain doesn't dispatch to the
        #    eval server.
        for url in self.eval_server_urls:
            self.available_server_lane_slots.pop(url, None)
            # Resolve queued-but-not-yet-dispatched individual samples so
            # their groups' remaining counts stay consistent.
            drained_queue = self.individual_sample_queues.pop(url, None)
            if drained_queue:
                for request_id, sample_idx in drained_queue:
                    group = self.pending_individual_groups.get(request_id)
                    if group is None:
                        continue
                    group["remaining"] -= 1
                    group["samples"][sample_idx] = {
                        "error": "cancelled",
                        "error_message": "Server drained before dispatch",
                    }
                    if group["remaining"] == 0:
                        self.active_count -= 1
                        del self.pending_individual_groups[request_id]
                        _log.debug(
                            f"Drained group {request_id} fully resolved "
                            f"(active={self.active_count})"
                        )
        self.server_urls = list(self.training_server_urls)
        self.server_cycle = itertools.cycle(self.server_urls)

        # 2. Cancel in-flight tasks on each eval server and properly await
        #    them so active_count stays consistent.
        #    The CancelledError handlers in _run_prompt (_on_cancelled_group)
        #    and _run_individual_sample (_handle_cancelled_sample) handle
        #    active_count decrement and inference event logging.
        for url in self.eval_server_urls:
            tasks = list(self.active_tasks_by_server.get(url, set()))
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                for task in tasks:
                    if task.cancelled():
                        _log.debug(
                            f"Rollout cancelled: server={url}, "
                            f"active={self.active_count}"
                        )
            self.active_tasks_by_server.pop(url, None)

        _log.info(f"Drained {len(self.eval_server_urls)} eval servers from training pool")

    async def _restore_eval_servers(self):
        """Wait for eval servers to get fresh weights and rejoin training pool.

        After eval_done, the trainer re-checks ``_eval_active`` before its next
        broadcast and switches to the full NCCL group.  We poll until the watcher
        advances ``eval_server_model_step`` accordingly.
        """
        if not self.eval_server_urls:
            return

        # If training is already complete, no more weight broadcasts will
        # happen, so there's nothing to wait for — just skip restoration.
        if self.trainer_step >= self.num_steps:
            _log.info("Training complete — skipping eval server restoration")
            return

        _log.info("Waiting for eval servers to receive fresh weights via full-group broadcast")

        # We need a *new* full-group broadcast, not just parity with
        # inference_model_step.  When the eval finishes fast (before any
        # training_only broadcasts), both counters sit at the eval step's
        # value and the old check would exit immediately — dispatching
        # training prompts before the eval server gets fresh weights.
        stale = self.eval_server_model_step
        target = max(self.inference_model_step, stale + 1)
        while self.eval_server_model_step < target:
            await asyncio.sleep(0.2)
            target = max(target, self.inference_model_step)

        # Check async level for eval servers before re-adding
        async_level = self._compute_async_level(self.eval_server_model_step)
        while async_level > config.cfg.max_async_rollout:
            await asyncio.sleep(0.2)
            async_level = self._compute_async_level(self.eval_server_model_step)

        # Re-add eval servers to the training pool
        restore_slots = self.max_concurrent_prompts_per_server * config.cfg.group_size
        for url in self.eval_server_urls:
            if url not in self.available_server_lane_slots:
                self.available_server_lane_slots[url] = set(range(restore_slots))
            if url not in self.individual_sample_queues:
                self.individual_sample_queues[url] = collections.deque()

        self.server_urls = list(self.training_server_urls) + list(self.eval_server_urls)
        self.server_cycle = itertools.cycle(self.server_urls)

        _log.info(
            f"Eval servers restored to training pool "
            f"(eval_server_model_step={self.eval_server_model_step})"
        )

        # Proactively fill ALL available server lanes — not just when
        # waiting_for_trainer.  Without this, the eval server's 32 lanes
        # would only fill one-at-a-time as the training server's rollouts
        # complete, creating a visible "ladder" pattern.
        self.waiting_for_trainer = False
        self._try_start_pending_rollouts()

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def _should_save_checkpoint(self, step: int, is_last_step: bool) -> bool:
        """Determine if a checkpoint should be saved at this step."""
        checkpoint_every = config.cfg.checkpoint_every
        if not checkpoint_every:
            return False
        if is_last_step:
            return True
        return (step + 1) % checkpoint_every == 0

    async def _save_checkpoint(self, step: int) -> None:
        """Trigger checkpoint save across all trainer actors.

        The ``step`` parameter is the 0-indexed internal training step.
        Externally (directory names, metadata, logs) we save as ``step + 1``
        so that checkpoint numbering is 1-indexed (e.g. step_10 after the
        10th completed training step).
        """
        save_step = step + 1
        _log.info(f"Saving checkpoint at step {save_step}", step=save_step)
        self.wandb_logger.log_orchestrator_timeline_event("checkpoint_save_start", step=save_step)
        start = time.time()
        try:
            results = await asyncio.to_thread(
                self.trainer_group.save_checkpoint,
                save_step,
            )
            elapsed = time.time() - start
            save_end_times = [
                r["save_end_time"] for r in results if "save_end_time" in r
            ]
            checkpoint_done_ts = max(save_end_times) if save_end_times else None
            _log.info(f"Checkpoint saved at step {save_step} in {elapsed:.2f}s", step=save_step)
            self.wandb_logger.log_orchestrator_timeline_event("checkpoint_save_done", step=save_step, timestamp=checkpoint_done_ts)

            # Log checkpoint sub-event timings to wandb (same pattern as train_step)
            timeline_events = []
            for result in results:
                timeline_events.extend(result.get("timeline_events", []))
            if timeline_events:
                self.wandb_logger.log_trainer_metrics_payload(
                    step=save_step,
                    payload={"timeline_events": timeline_events},
                )

            # Save orchestrator-level state needed for resume
            if config.cfg.checkpoint_save_training_state:
                self._save_orchestrator_state(save_step)

            self._cleanup_old_checkpoints(save_step)
        except Exception as exc:
            _log.exception(f"Checkpoint save failed at step {save_step}: {exc}", step=save_step)
            # Checkpoint failure should not crash training

    def _save_orchestrator_state(self, step: int) -> None:
        """Save orchestrator-level state to the checkpoint directory.

        Writes ``orchestrator_state.json`` inside the ``step_N/`` directory.
        This captures mutable orchestrator state required for resume that is
        not already saved by the trainer backend (model weights, optimizer).

        Must be called AFTER the trainer backend checkpoint completes
        (so the directory exists) and BEFORE old checkpoint cleanup.
        """
        import json

        ckpt_path = paths.CHECKPOINT_DIR / f"step_{step}"
        if not ckpt_path.is_dir():
            _log.warning(
                f"Checkpoint directory {ckpt_path} does not exist, "
                f"skipping orchestrator state save",
                step=step,
            )
            return

        # Serialize scheduler RNG state (numpy RandomState)
        rng_state = self.scheduler._rng.get_state()
        rng_state_json = {
            "type": rng_state[0],
            "keys": rng_state[1].tolist(),
            "pos": int(rng_state[2]),
            "has_gauss": int(rng_state[3]),
            "cached_gaussian": float(rng_state[4]),
        }

        state = {
            "trainer_step": self.trainer_step,
            "inference_model_step": self.inference_model_step,
            "next_request_id": self.next_request_id,
            "next_sample_idx": self.next_sample_idx,
            "scheduler_current_idx": self.scheduler.current_idx,
            "scheduler_rng_state": rng_state_json,
        }

        state_path = ckpt_path / "orchestrator_state.json"
        state_path.write_text(json.dumps(state, indent=2))
        _log.debug(f"Orchestrator state saved to {state_path}", step=step)

    def _cleanup_old_checkpoints(self, current_step: int) -> None:
        """Remove old checkpoints based on keep_last and keep_every rules.

        A checkpoint is kept if it matches EITHER rule. Deleted only if neither.
        """
        keep_last = config.cfg.checkpoint_keep_last
        keep_every = config.cfg.checkpoint_keep_every

        if keep_last is None and keep_every is None:
            return

        ckpt_dir = paths.CHECKPOINT_DIR
        if not ckpt_dir.exists():
            return

        step_dirs = sorted(
            (
                (int(p.name.split("_")[1]), p)
                for p in ckpt_dir.iterdir()
                if p.is_dir() and p.name.startswith("step_") and not p.name.endswith("_tmp")
            ),
            key=lambda x: x[0],
        )
        if not step_dirs:
            return

        steps_to_keep: set[int] = {current_step}

        if keep_last is not None:
            for step, _ in step_dirs[-keep_last:]:
                steps_to_keep.add(step)

        if keep_every is not None:
            for step, _ in step_dirs:
                if step % keep_every == 0:
                    steps_to_keep.add(step)

        import shutil
        for step, path in step_dirs:
            if step not in steps_to_keep:
                _log.info(f"Removing old checkpoint: {path.name}", step=step)
                shutil.rmtree(path)

    def _load_checkpoint_and_resume(self):
        """Load checkpoint and restore all orchestrator state for resume.

        Called from ``start_processes()`` after weight broadcast NCCL groups
        are initialized.  Performs:
        1. Find latest checkpoint step
        2. Read ``orchestrator_state.json``
        3. Load model + optimizer weights on all trainer ranks
        4. Broadcast checkpoint weights to inference servers
        5. Restore orchestrator counters and scheduler RNG state
        """
        import json
        import numpy as np

        resume_val = config.cfg.resume_from_checkpoint
        if isinstance(resume_val, int) and not isinstance(resume_val, bool):
            # Specific step requested — verify it exists
            ckpt_step = resume_val
            ckpt_dir = paths.CHECKPOINT_DIR / f"step_{ckpt_step}"
            if not ckpt_dir.is_dir():
                raise RuntimeError(
                    f"resume_from_checkpoint={ckpt_step} but checkpoint "
                    f"directory {ckpt_dir} does not exist"
                )
        else:
            # True — use latest checkpoint
            ckpt_step = self._get_checkpoint_step()
            if ckpt_step is None:
                raise RuntimeError(
                    "resume_from_checkpoint is True but no checkpoint found "
                    f"in {paths.CHECKPOINT_DIR}"
                )

        ckpt_path = paths.CHECKPOINT_DIR / f"step_{ckpt_step}"
        state_file = ckpt_path / "orchestrator_state.json"
        if not state_file.exists():
            raise RuntimeError(
                f"Checkpoint at step {ckpt_step} is missing orchestrator_state.json"
            )

        state = json.loads(state_file.read_text())
        _log.info(f"Resuming from checkpoint at step {ckpt_step}")

        # Load model + optimizer weights on all trainer ranks (collective)
        _log.info("Loading trainer checkpoint weights...")
        self.trainer_group.load_checkpoint(ckpt_step)
        _log.info("Trainer checkpoint loaded")

        # Broadcast checkpoint weights to inference servers
        _log.info("Broadcasting checkpoint weights to inference servers...")
        self.trainer_group.broadcast_weights()
        _log.info("Inference servers updated with checkpoint weights")

        # Restore orchestrator counters
        self.trainer_step = state["trainer_step"] + 1  # Resume from next step
        self.inference_step = self.trainer_step  # In-flight rollouts are lost; restart in sync
        self.inference_model_step = state["inference_model_step"]
        self.next_request_id = state["next_request_id"]
        self.next_sample_idx = state["next_sample_idx"]

        # Restore scheduler state
        self.scheduler.current_idx = state["scheduler_current_idx"]
        rng = state["scheduler_rng_state"]
        self.scheduler._rng.set_state((
            rng["type"],
            np.array(rng["keys"], dtype=np.uint32),
            rng["pos"],
            rng["has_gauss"],
            rng["cached_gaussian"],
        ))

        _log.info(
            f"Resumed: trainer_step={self.trainer_step}, "
            f"inference_model_step={self.inference_model_step}, "
            f"scheduler_idx={self.scheduler.current_idx}"
        )

    def _get_checkpoint_step(self) -> int | None:
        """Get latest checkpoint step."""
        if not paths.CHECKPOINT_DIR.exists():
            return None
        step_dirs = [
            p for p in paths.CHECKPOINT_DIR.glob("step_*")
            if not p.name.endswith("_tmp")
        ]
        if not step_dirs:
            return None
        return max(int(p.name.split("_")[1]) for p in step_dirs)



def main():
    """Entry point for orchestrator."""
    # Clear old log/stdout files before initializing logging (so each run starts fresh).
    # We do this BEFORE setup_logging() to avoid deleting files that handlers point to.
    # Skip on resume to preserve logs from the previous run segment.
    if not config.cfg.resume_from_checkpoint and paths.LOGS_DIR.exists():
        for old_file in paths.LOGS_DIR.glob("**/*.log"):
            old_file.unlink(missing_ok=True)
        for old_file in paths.LOGS_DIR.glob("**/*.stdout"):
            old_file.unlink(missing_ok=True)

    # Initialize logging system (debug flag was set by config_loader)
    debug = bool(config.cfg.debug)
    setup_logging(debug=debug)
    setup_file_tailers(paths.LOGS_DIR)

    # Suppress third-party noise unless --debug
    if not debug:
        suppress_third_party_noise()

    _log.banner("Telescope")

    # Clean startup summary (always shown)
    _log.info(f"Model: {config.cfg.model}")
    _log.info(f"Steps: {config.cfg.number_of_steps} | Batch: {config.cfg.prompts_batch_size_for_trainer} | Group: {config.cfg.group_size}")

    # Fail fast if no environments are configured (before spinning up Ray/W&B/workers)
    if not config.cfg.environments:
        _log.error(
            "No environments configured. "
            "You must specify at least one environment in your config, e.g.:\n"
            "  environments:\n"
            '    - name: "countdown"\n'
            "      weight: 1.0\n"
            "      reward_min: 0.0\n"
            "      reward_max: 2.0"
        )
        raise SystemExit(1)

    _log.debug("=== Orchestrator starting ===")
    _log.debug(f"Environments: {config.cfg.environments}")
    _log.debug(
        "Ray worker config: "
        f"inference={config.cfg.inference_num_workers}, "
        f"trainer={config.cfg.trainer_num_workers}"
    )
    _log.debug(f"Max concurrent prompts per server: {config.cfg.max_concurrent_prompts_per_server}")
    _log.debug(f"Max async rollout: {config.cfg.max_async_rollout}")

    _log.section("Initializing")

    start_time = time.time()
    orchestrator = Orchestrator()
    _log.debug(f"Orchestrator instance created in {time.time() - start_time:.2f}s")

    _interrupted = False
    _failed = False
    try:
        _log.debug("Initializing W&B logger...")
        orchestrator.wandb_logger.initialize()
        orchestrator.wandb_logger.log_orchestrator_timeline_event("orchestrator_start", start_time)
        orchestrator.wandb_logger.log_orchestrator_timeline_event("wandb_initialized")
        _log.debug("W&B logger initialized")

        _log.debug("Starting Ray runtime...")
        runtime_start = time.time()
        orchestrator.start_processes()
        _log.debug(f"Ray runtime ready in {time.time() - runtime_start:.2f}s")
        orchestrator.wandb_logger.log_orchestrator_timeline_event("training_loop_start")

        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        _log.warning("Interrupted by user")
        generate_module._shutting_down = True
        _interrupted = True
    except Exception as exc:
        _log.error(f"Training failed: {exc}")
        generate_module._shutting_down = True
        _failed = True
    else:
        _failed = False
    finally:
        # Allow one more Ctrl+C to force-exit immediately during cleanup.
        signal.signal(signal.SIGINT, lambda *_: os._exit(1))
        _log.debug("Cleaning up...")
        orchestrator.stop_processes(force=_interrupted or _failed)
        orchestrator.wandb_logger.finish()
        _log.debug(f"Orchestrator finished, total runtime: {time.time() - start_time:.2f}s")

    if _failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
