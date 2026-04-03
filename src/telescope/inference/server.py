"""
Inference server using vLLM.

Provides OpenAI-compatible API for completions plus custom endpoints
for weight synchronization with the trainer.
"""
import argparse
import os
from contextlib import asynccontextmanager

import uvloop
from fastapi import Request
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    build_async_engine_client_from_engine_args,
    init_app_state,
    setup_server,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser

from telescope.utils import config
from telescope.utils.tlog import get_logger, setup_logging

_log = get_logger("inference")


def build_base_args(host: str = None, port: int = None, model: str = None):
    """Build vLLM server arguments."""
    parser = FlexibleArgumentParser(description="OpenAI-compatible vllm server")
    parser = make_arg_parser(parser)
    args = parser.parse_args([])

    args.host = host or config.cfg.inference_host
    args.port = port or config.cfg.inference_base_port
    args.model = model or config.cfg.model
    args.gpu_memory_utilization = config.cfg.gpu_memory_utilization
    args.max_model_len = config.cfg.max_model_len
    args.tensor_parallel_size = max(1, int(config.cfg.inference_tensor_parallel_size))
    args.worker_extension_cls = "telescope.inference.worker.NCCLWeightUpdateWorker"
    if config.cfg.max_num_seqs is not None:
        args.max_num_seqs = config.cfg.max_num_seqs
    if config.cfg.max_num_batched_tokens is not None:
        args.max_num_batched_tokens = config.cfg.max_num_batched_tokens
    args.enable_prefix_caching = False
    args.override_generation_config = {"max_new_tokens": None}

    # Scheduling policy: "priority" enables turn-aware scheduling for multi-turn
    args.scheduling_policy = config.cfg.vllm_scheduling_policy

    # vLLM tracing configuration (per-request timing via OpenTelemetry)
    if config.cfg.enable_vllm_tracing:
        otlp_override = os.environ.get("TELESCOPE_OTLP_TRACES_ENDPOINT")
        otlp_port = config.cfg.otlp_receiver_port
        args.otlp_traces_endpoint = otlp_override or f"http://localhost:{otlp_port}"
        _log.info(f"vLLM tracing enabled, OTLP endpoint: {args.otlp_traces_endpoint}")
    
    return args


def attach_control_routes(app, engine_client):
    """Attach custom control routes for weight synchronization."""

    @app.get("/testing")
    async def _testing():
        return {"status": "ok"}

    @app.get("/collective_test")
    async def _collective_test():
        await engine_client.collective_rpc("collective_test")
        return {"status": "ok"}

    @app.post("/init_broadcast")
    async def _init_broadcast(request: Request):
        """Initialize NCCL broadcast for weight updates."""
        data = await request.json()
        host = data.get("host")
        port = data.get("port")
        world_size = data.get("world_size", 3)
        rank = data.get("rank")
        group = data.get("group", "full")
        _log.info(f"Initializing broadcast: rank={rank}, world_size={world_size}, group={group}")
        await engine_client.collective_rpc("init_broadcast", args=(host, port, world_size, rank, group))
        return {"status": "ok"}

    @app.get("/load_weights")
    async def _load_weights(request: Request):
        """Load new weights from trainer via NCCL broadcast."""
        group = request.query_params.get("group", "full")
        await engine_client.collective_rpc("load_weights", args=(group,))
        return {"status": "ok"}

    @app.get("/torch_memory")
    async def _torch_memory():
        """
        Collect torch allocator metrics from all vLLM worker processes.

        Uses worker-extension RPC so metrics come from the actual inference
        workers holding model tensors (not just the API process).
        """
        try:
            payloads = await engine_client.collective_rpc("collect_torch_memory_metrics")
        except Exception:
            return {"samples": []}

        merged: list[dict] = []
        if isinstance(payloads, list):
            for payload in payloads:
                if isinstance(payload, list):
                    for sample in payload:
                        if isinstance(sample, dict):
                            merged.append(sample)
                elif isinstance(payload, dict):
                    merged.append(payload)
        return {"samples": merged}


@asynccontextmanager
async def custom_build_async_engine_client(args):
    """Build async engine client with custom configuration."""
    engine_args = AsyncEngineArgs.from_cli_args(args)
    async with build_async_engine_client_from_engine_args(engine_args) as engine:
        yield engine


async def custom_run_server_worker(listen_address, sock, args, **uvicorn_kwargs):
    """Run the server with custom routes."""
    async with custom_build_async_engine_client(args) as engine_client:
        app = build_app(args)
        attach_control_routes(app, engine_client)
        await init_app_state(engine_client, app.state, args)

        shutdown_task = await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
        )

        try:
            await shutdown_task
        finally:
            sock.close()


async def custom_run_server(args, **uvicorn_kwargs):
    """Run the inference server."""
    listen_address, sock = setup_server(args)
    await custom_run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


def run_server():
    """Server entry point."""
    # Initialize logging system
    setup_logging()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=config.cfg.inference_host)
    parser.add_argument("--port", type=int, default=config.cfg.inference_base_port)
    parser.add_argument("--model", type=str, default=None, help="Model path override")
    cli_args = parser.parse_args()

    _log.info(f"Starting inference server on {cli_args.host}:{cli_args.port}")
    args = build_base_args(host=cli_args.host, port=cli_args.port, model=cli_args.model)
    uvloop.run(custom_run_server(args))


if __name__ == "__main__":
    run_server()

