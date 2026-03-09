"""
Lightweight OTLP HTTP receiver for capturing per-request vLLM trace spans.

When ENABLE_VLLM_TRACING is True, vLLM is configured to export OpenTelemetry
traces via HTTP/protobuf to this receiver. Each completed request produces a
span with timing attributes (queue time, prefill time, decode time, etc.).

The receiver stores span data in a dict keyed by gen_ai.request.id, and the
orchestrator looks up timing data when logging inference events.

Protocol: HTTP/protobuf (OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf)
Endpoint: POST / with Content-Type: application/x-protobuf
"""
from __future__ import annotations

import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

from telescope.utils.tlog import get_logger

_log = get_logger("orchestrator")

# Span attribute keys from vLLM's SpanAttributes (vllm/tracing.py)
_ATTR_REQUEST_ID = "gen_ai.request.id"
_ATTR_QUEUE_TIME = "gen_ai.latency.time_in_queue"
_ATTR_TTFT = "gen_ai.latency.time_to_first_token"
_ATTR_PREFILL_TIME = "gen_ai.latency.time_in_model_prefill"
_ATTR_DECODE_TIME = "gen_ai.latency.time_in_model_decode"
_ATTR_INFERENCE_TIME = "gen_ai.latency.time_in_model_inference"
_ATTR_E2E = "gen_ai.latency.e2e"
_ATTR_MAX_TOKENS = "gen_ai.request.max_tokens"

# Max age for stored spans (seconds) — older spans are pruned
_SPAN_TTL_SECONDS = 120
# Max number of spans to store before pruning oldest
_MAX_STORED_SPANS = 10000


def _decode_any_value(value) -> object:
    """Decode an opentelemetry AnyValue protobuf to a Python object."""
    field = value.WhichOneof("value")
    if field == "string_value":
        return value.string_value
    elif field == "int_value":
        return value.int_value
    elif field == "double_value":
        return value.double_value
    elif field == "bool_value":
        return value.bool_value
    elif field == "array_value":
        return [_decode_any_value(v) for v in value.array_value.values]
    return None


def _decode_attributes(attributes) -> dict[str, object]:
    """Decode span attributes from protobuf KeyValue list to dict."""
    return {kv.key: _decode_any_value(kv.value) for kv in attributes}


class _SpanEntry:
    """A stored span with its timing data and arrival timestamp."""
    __slots__ = ("timing", "stored_at")

    def __init__(self, timing: dict, stored_at: float):
        self.timing = timing
        self.stored_at = stored_at


class OtlpReceiver:
    """
    Threaded OTLP HTTP receiver that captures vLLM per-request timing spans.

    Runs a lightweight HTTP server in a daemon thread. vLLM's BatchSpanProcessor
    POSTs serialized ExportTraceServiceRequest protobuf messages every ~5 seconds.
    The receiver parses them and stores timing data keyed by gen_ai.request.id.

    Usage:
        receiver = OtlpReceiver(port=4318)
        receiver.start()
        ...
        timing = receiver.get_and_remove("cmpl-abc123-0")
        # timing = {"queue_time": 0.002, "prefill_time": 0.05, ...} or None
        ...
        receiver.stop()
    """

    def __init__(self, port: int = 4318, host: str = "127.0.0.1", advertised_host: str | None = None):
        self.port = port
        self.host = host
        self.advertised_host = advertised_host or host
        self._spans: dict[str, _SpanEntry] = {}
        self._lock = threading.Lock()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._total_received = 0

    @property
    def endpoint_url(self) -> str:
        """The URL that vLLM should export traces to."""
        return f"http://{self.advertised_host}:{self.port}"

    def start(self):
        """Start the OTLP HTTP receiver in a daemon thread."""
        receiver = self  # Capture for the handler class

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                # vLLM passes our URL directly to OTLPSpanExporter(endpoint=...)
                # which uses it as-is, so it POSTs to "/".
                if self.path == "/":
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length)
                    response_body = receiver._handle_traces(body)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-protobuf")
                    self.send_header("Content-Length", str(len(response_body)))
                    self.end_headers()
                    self.wfile.write(response_body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                # Suppress default HTTP server logging
                pass

        self._server = HTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="otlp-receiver",
        )
        self._thread.start()
        _log.debug(f"OTLP receiver started on port {self.port}")

    def stop(self):
        """Stop the OTLP HTTP receiver."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        _log.info(f"OTLP receiver stopped (received {self._total_received} spans total)")

    def get_and_remove(
        self,
        request_id: str,
        timeout: float = 0,
        poll_interval: float = 0.05,
    ) -> dict | None:
        """
        Retrieve and remove timing data for a vLLM request.

        vLLM's BatchSpanProcessor batches spans and flushes them periodically
        (controlled by OTEL_BSP_SCHEDULE_DELAY, default 5s). This means the
        span data may not be available immediately after the HTTP response
        arrives. Use ``timeout`` to poll briefly for the span to arrive.

        Args:
            request_id: The gen_ai.request.id from the span.
                        For completions API, this is f"{CompletionResponse.id}-0"
                        (vLLM appends "-{prompt_index}" to the external request ID).
            timeout: Maximum seconds to wait for the span to arrive.
                     0 means no waiting (immediate lookup only).
            poll_interval: Seconds between poll attempts when waiting.

        Returns:
            Dict with timing fields or None if span hasn't arrived within timeout.
        """
        deadline = time.time() + timeout
        while True:
            with self._lock:
                entry = self._spans.pop(request_id, None)
            if entry is not None:
                return entry.timing
            if time.time() >= deadline:
                return None
            time.sleep(poll_interval)

    def _handle_traces(self, body: bytes) -> bytes:
        """Parse an ExportTraceServiceRequest and store span data."""
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
            ExportTraceServiceResponse,
        )

        try:
            request = ExportTraceServiceRequest()
            request.ParseFromString(body)
        except Exception as e:
            _log.warning(f"Failed to parse OTLP trace request: {e}")
            return ExportTraceServiceResponse().SerializeToString()

        now = time.time()
        count = 0

        for resource_span in request.resource_spans:
            for scope_span in resource_span.scope_spans:
                for span in scope_span.spans:
                    attrs = _decode_attributes(span.attributes)
                    request_id = attrs.get(_ATTR_REQUEST_ID)
                    if not request_id or not isinstance(request_id, str):
                        continue

                    timing = {
                        "queue_time": attrs.get(_ATTR_QUEUE_TIME),
                        "time_to_first_token": attrs.get(_ATTR_TTFT),
                        "prefill_time": attrs.get(_ATTR_PREFILL_TIME),
                        "decode_time": attrs.get(_ATTR_DECODE_TIME),
                        "inference_time": attrs.get(_ATTR_INFERENCE_TIME),
                        "e2e_latency": attrs.get(_ATTR_E2E),
                        "max_tokens": attrs.get(_ATTR_MAX_TOKENS),
                    }

                    with self._lock:
                        self._spans[request_id] = _SpanEntry(timing, now)
                        count += 1

        self._total_received += count

        # Periodically prune old spans
        self._prune_old_spans(now)

        return ExportTraceServiceResponse().SerializeToString()

    def _prune_old_spans(self, now: float):
        """Remove spans older than TTL or if storage exceeds max size."""
        cutoff = now - _SPAN_TTL_SECONDS
        with self._lock:
            # Remove expired spans
            expired = [k for k, v in self._spans.items() if v.stored_at < cutoff]
            for k in expired:
                del self._spans[k]

            # If still too many, remove oldest
            if len(self._spans) > _MAX_STORED_SPANS:
                sorted_keys = sorted(
                    self._spans.keys(),
                    key=lambda k: self._spans[k].stored_at,
                )
                to_remove = sorted_keys[: len(self._spans) - _MAX_STORED_SPANS]
                for k in to_remove:
                    del self._spans[k]

