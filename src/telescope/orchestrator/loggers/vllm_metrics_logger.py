"""
vLLM metrics logging for real-time inference server monitoring.

This module collects metrics from vLLM's /metrics endpoint (Prometheus format)
and provides them to EventLogger for unified upload to W&B.

Metrics are collected every 1 second and simplified to single values per metric:
- Gauges: Direct values (requests running/waiting, cache usage)
- Counters: Cumulative totals (tokens processed, requests completed)
- Histograms: Mean values computed from sum/count (latencies)
"""
from __future__ import annotations

import asyncio
import re
import threading
import time
from dataclasses import dataclass

import aiohttp

from telescope.utils.tlog import get_logger

_log = get_logger("orchestrator")


# Gauge metrics - single value per scrape (vLLM v1 names)
GAUGE_METRICS = {
    "num_requests_running",      # Requests currently being processed
    "num_requests_waiting",      # Requests waiting in queue
    "kv_cache_usage_perc",       # KV cache utilization (0-1)
}

# Counter metrics - cumulative totals (we sum across labels like finished_reason)
COUNTER_METRICS = {
    "request_success_total",       # Total successful requests
    "prompt_tokens_total",         # Total input tokens processed
    "generation_tokens_total",     # Total output tokens generated
    "num_preemptions_total",       # Total preemptions (want this low)
    "prefix_cache_hits_total",     # Cache hits (tokens)
    "prefix_cache_queries_total",  # Cache queries (tokens)
}

# Histogram metrics - we compute mean from _sum/_count
HISTOGRAM_METRICS = {
    "time_to_first_token_seconds",       # Time to first token (TTFT)
    "e2e_request_latency_seconds",       # End-to-end request latency
    "inter_token_latency_seconds",       # Inter-token latency (ITL)
    "request_queue_time_seconds",        # Time spent waiting in queue (WAITING phase)
    "request_inference_time_seconds",    # Time spent in RUNNING phase (prefill + decode)
    "request_prefill_time_seconds",      # Time spent in PREFILL phase
    "request_decode_time_seconds",       # Time spent in DECODE phase
    "iteration_tokens_total",            # Total tokens per engine step (batching density)
    "request_generation_tokens",         # Generation tokens per request (distribution)
    "request_prompt_tokens",             # Prompt tokens per request (distribution)
}


@dataclass
class VllmMetricSample:
    """Single vLLM metric sample."""
    timestamp: float
    server: int
    metric_name: str
    value: float
    node_id: int = -1
    node_ip: str = ""
    hostname: str = ""
    ray_node_id: str = ""
    tp_group_id: int = -1
    tp_size: int = 1


class VllmMetricsLogger:
    """
    Thread-safe vLLM metrics logger that collects metrics from /metrics endpoints
    every 1 second and provides data to EventLogger for unified upload.
    
    Collects simplified metrics suitable for UI dashboards:

    Gauges:
    - requests_running: Current batch size (in-flight requests)
    - requests_waiting: Queue depth (waiting to be scheduled)
    - cache_usage: KV cache utilization %

    Counters:
    - requests_total: Cumulative successful requests
    - prompt_tokens_total: Cumulative input tokens
    - generation_tokens_total: Cumulative output tokens
    - preemptions_total: Cumulative preemptions (KV cache evictions)
    - cache_hits_total: Cumulative prefix cache hits
    - cache_queries_total: Cumulative prefix cache queries

    Histogram means:
    - ttft_mean: Mean time to first token (seconds)
    - e2e_latency_mean: Mean end-to-end latency (seconds)
    - itl_mean: Mean inter-token latency (seconds)
    - queue_time_mean: Mean time waiting in queue (seconds)
    - inference_time_mean: Mean time in RUNNING phase (seconds)
    - prefill_time_mean: Mean time in PREFILL phase (seconds)
    - decode_time_mean: Mean time in DECODE phase (seconds)
    - iteration_tokens_mean: Mean tokens per engine step (batching density)
    - generation_tokens_per_request_mean: Mean generation tokens per request
    - prompt_tokens_per_request_mean: Mean prompt tokens per request
    
    Usage:
        logger = VllmMetricsLogger()
        logger.initialize(wandb_run)
        logger.set_server_urls(["http://localhost:8100", "http://localhost:8101"])
        
        await logger.start()
        metrics = logger.get_and_clear_metrics()
        await logger.stop()
    """

    COLLECTION_INTERVAL_SECONDS = 1.0

    @staticmethod
    def _as_int(value, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def __init__(self):
        self.run = None
        self._lock = threading.Lock()
        self._server_urls: list[str] = []
        self._server_metadata: dict[int, dict] = {}
        self._metrics: list[VllmMetricSample] = []
        self._stop_event: asyncio.Event | None = None
        self._collection_task: asyncio.Task | None = None
        self._run_start_time: float = time.time()
        self._session: aiohttp.ClientSession | None = None

    def initialize(self, wandb_run):
        """Initialize with a W&B run object."""
        self.run = wandb_run
        self._run_start_time = time.time()
        _log.debug(f"VllmMetricsLogger initialized with run {wandb_run.name if wandb_run else 'None'}")

    def set_server_urls(self, urls: list[str]):
        """Set the list of vLLM server URLs to poll."""
        self._server_urls = urls
        self._server_metadata = {}
        _log.debug(f"VllmMetricsLogger monitoring {len(urls)} servers")

    def set_server_infos(self, server_infos: list[dict]):
        """
        Set full server metadata (url, node, TP group) for richer metrics rows.
        """
        ordered = sorted(server_infos, key=lambda item: int(item.get("server_idx", -1)))
        self._server_urls = [str(info.get("url", "")) for info in ordered]
        self._server_metadata = {
            int(info.get("server_idx", -1)): {
                "node_id": self._as_int(info.get("node_id", -1), default=-1),
                "node_ip": str(info.get("node_ip") or ""),
                "hostname": str(info.get("hostname") or ""),
                "ray_node_id": str(info.get("ray_node_id") or ""),
                "tp_group_id": int(info.get("tp_group_id", info.get("server_idx", -1))),
                "tp_size": int(info.get("tp_size", 1)),
            }
            for info in ordered
        }
        _log.debug(f"VllmMetricsLogger monitoring {len(self._server_urls)} servers with metadata")

    @property
    def num_servers(self) -> int:
        """Return the number of servers being monitored."""
        return len(self._server_urls)

    def _parse_prometheus_metrics(self, text: str, server_idx: int, timestamp: float) -> list[VllmMetricSample]:
        """
        Parse Prometheus metrics and return simplified single-value metrics.
        
        - Gauges: Take value directly
        - Counters: Sum across all labels (e.g., sum all finished_reasons)
        - Histograms: Compute mean from _sum/_count
        """
        # Accumulators for counters (sum across labels)
        counter_sums: dict[str, float] = {}
        
        # Accumulators for histogram sum/count
        histogram_sums: dict[str, float] = {}
        histogram_counts: dict[str, float] = {}
        
        # Direct gauge values
        gauge_values: dict[str, float] = {}
        
        for line in text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse: metric_name{labels} value OR metric_name value
            match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{[^}]*\}\s+(.+)$', line)
            if not match:
                match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+(.+)$', line)
            if not match:
                continue
            
            metric_name = match.group(1)
            value_str = match.group(2)
            
            # Remove vllm: prefix
            if metric_name.startswith("vllm:"):
                metric_name = metric_name[5:]
            
            # Parse value, skip NaN/Inf
            try:
                if value_str.lower() in ('+inf', 'inf', '-inf', 'nan'):
                    continue
                value = float(value_str)
            except ValueError:
                continue
            
            # Skip bucket data entirely
            if "_bucket" in metric_name:
                continue
            
            # Handle gauges
            if metric_name in GAUGE_METRICS:
                gauge_values[metric_name] = value
                continue
            
            # Handle counters (sum across labels)
            for counter_name in COUNTER_METRICS:
                if metric_name == counter_name:
                    counter_sums[counter_name] = counter_sums.get(counter_name, 0) + value
                    break
            
            # Handle histogram _sum and _count
            for hist_name in HISTOGRAM_METRICS:
                if metric_name == f"{hist_name}_sum":
                    histogram_sums[hist_name] = value
                    break
                elif metric_name == f"{hist_name}_count":
                    histogram_counts[hist_name] = value
                    break
        
        server_meta = self._server_metadata.get(server_idx, {})
        node_id = self._as_int(server_meta.get("node_id", -1), default=-1)
        node_ip = str(server_meta.get("node_ip") or "")
        hostname = str(server_meta.get("hostname") or "")
        ray_node_id = str(server_meta.get("ray_node_id") or "")
        tp_group_id = int(server_meta.get("tp_group_id", server_idx))
        tp_size = int(server_meta.get("tp_size", 1))

        # Build output samples
        samples = []
        
        # Add gauges with friendly names
        gauge_name_map = {
            "num_requests_running": "requests_running",
            "num_requests_waiting": "requests_waiting",
            "kv_cache_usage_perc": "cache_usage",
        }
        for metric, value in gauge_values.items():
            samples.append(VllmMetricSample(
                timestamp=timestamp,
                server=server_idx,
                metric_name=gauge_name_map.get(metric, metric),
                value=value,
                node_id=node_id,
                node_ip=node_ip,
                hostname=hostname,
                ray_node_id=ray_node_id,
                tp_group_id=tp_group_id,
                tp_size=tp_size,
            ))
        
        # Add counters
        counter_name_map = {
            "request_success_total": "requests_total",
            "prompt_tokens_total": "prompt_tokens_total",
            "generation_tokens_total": "generation_tokens_total",
            "num_preemptions_total": "preemptions_total",
            "prefix_cache_hits_total": "cache_hits_total",
            "prefix_cache_queries_total": "cache_queries_total",
        }
        for metric, value in counter_sums.items():
            samples.append(VllmMetricSample(
                timestamp=timestamp,
                server=server_idx,
                metric_name=counter_name_map.get(metric, metric),
                value=value,
                node_id=node_id,
                node_ip=node_ip,
                hostname=hostname,
                ray_node_id=ray_node_id,
                tp_group_id=tp_group_id,
                tp_size=tp_size,
            ))
        
        # Add histogram means
        hist_name_map = {
            "time_to_first_token_seconds": "ttft_mean",
            "e2e_request_latency_seconds": "e2e_latency_mean",
            "inter_token_latency_seconds": "itl_mean",
            "request_queue_time_seconds": "queue_time_mean",
            "request_inference_time_seconds": "inference_time_mean",
            "request_prefill_time_seconds": "prefill_time_mean",
            "request_decode_time_seconds": "decode_time_mean",
            "iteration_tokens_total": "iteration_tokens_mean",
            "request_generation_tokens": "generation_tokens_per_request_mean",
            "request_prompt_tokens": "prompt_tokens_per_request_mean",
        }
        for hist_name in HISTOGRAM_METRICS:
            if hist_name in histogram_sums and hist_name in histogram_counts:
                count = histogram_counts[hist_name]
                if count > 0:
                    mean = histogram_sums[hist_name] / count
                    samples.append(VllmMetricSample(
                        timestamp=timestamp,
                        server=server_idx,
                        metric_name=hist_name_map.get(hist_name, f"{hist_name}_mean"),
                        value=mean,
                        node_id=node_id,
                        node_ip=node_ip,
                        hostname=hostname,
                        ray_node_id=ray_node_id,
                        tp_group_id=tp_group_id,
                        tp_size=tp_size,
                    ))
        
        return samples

    async def _collect_from_server(self, server_idx: int, url: str, timestamp: float) -> list[VllmMetricSample]:
        """Collect metrics from a single vLLM server."""
        try:
            async with self._session.get(f"{url}/metrics", timeout=aiohttp.ClientTimeout(total=1.0)) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    return self._parse_prometheus_metrics(text, server_idx, timestamp)
        except Exception:
            pass
        return []

    async def _collect_all_metrics(self):
        """Collect metrics from all vLLM servers."""
        if not self._server_urls or self._session is None:
            return
        
        timestamp = time.time()
        
        tasks = [
            self._collect_from_server(idx, url, timestamp)
            for idx, url in enumerate(self._server_urls)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_samples = []
        for result in results:
            if isinstance(result, list):
                all_samples.extend(result)
        
        if all_samples:
            with self._lock:
                self._metrics.extend(all_samples)

    def get_and_clear_metrics(self) -> list[VllmMetricSample]:
        """Get all collected metrics and clear the buffer."""
        with self._lock:
            metrics = list(self._metrics)
            self._metrics = []
        return metrics


    async def _collection_loop(self):
        """Background loop that collects metrics every second."""
        while not self._stop_event.is_set():
            try:
                await self._collect_all_metrics()
            except Exception as e:
                _log.error(f"vLLM metrics collection error: {e}")

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.COLLECTION_INTERVAL_SECONDS
                )
            except asyncio.TimeoutError:
                pass

    async def start(self):
        """Start the collection loop."""
        self._stop_event = asyncio.Event()
        self._session = aiohttp.ClientSession()
        self._collection_task = asyncio.create_task(self._collection_loop())
        _log.debug("vLLM metrics collection started")

    async def stop(self):
        """Stop the collection loop."""
        if self._stop_event:
            self._stop_event.set()
        
        if self._collection_task:
            await self._collection_task
            self._collection_task = None
        
        if self._session:
            await self._session.close()
            self._session = None
        
        _log.info("vLLM metrics collection stopped")

    def finish(self):
        """Finalize and clean up resources."""
        _log.info("VllmMetricsLogger finished")
