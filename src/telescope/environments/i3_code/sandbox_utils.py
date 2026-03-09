# Unified sandbox utilities — works with any SandboxProvider via the
# _sandbox abstraction layer.
import asyncio
import io
import logging
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

from telescope.environments._sandbox import SandboxHandle, SandboxProvider

logger = logging.getLogger(__name__)

# Dedicated thread pool for CPU-bound tar building
_TAR_EXECUTOR = ThreadPoolExecutor(max_workers=128, thread_name_prefix="tar-builder")

# Limit concurrent uploads per event loop.
# With 100 sandboxes and unlimited concurrency, the sandbox API gets overwhelmed
# and all uploads start timing out.  100 keeps throughput high while preventing overload.
_UPLOAD_SEMAPHORE_SIZE = 100
_UPLOAD_SEMAPHORES: dict[int, asyncio.Semaphore] = {}


def _get_upload_semaphore() -> asyncio.Semaphore:
    """Get or create an upload semaphore for the current event loop.

    asyncio.Semaphore has loop affinity — a semaphore created in one loop
    cannot be used in another.  Since sandbox operations may run in
    different event loops (main loop vs. worker thread loops), we keep
    one semaphore per loop.
    """
    loop_id = id(asyncio.get_running_loop())
    sem = _UPLOAD_SEMAPHORES.get(loop_id)
    if sem is None:
        sem = asyncio.Semaphore(_UPLOAD_SEMAPHORE_SIZE)
        _UPLOAD_SEMAPHORES[loop_id] = sem
    return sem


class FileTooLarge(Exception):
    """Exception raised when a file size exceeds the allowed limit for sandbox upload."""

    pass


def build_tar_gz(file_map: dict[str, str]) -> bytes:
    """Build a tar.gz from a mapping of relative paths to UTF-8 string contents."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel_path, content in file_map.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=rel_path)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    data = buf.getvalue()
    if len(data) > 30 * 1024 * 1024:
        raise FileTooLarge("Bundle exceeds 30MB limit")
    logger.debug(f"Built tar.gz bundle: {len(file_map)} files, {len(data)} bytes")
    return data


async def upload_and_extract_bundle(
    provider: SandboxProvider,
    handle: SandboxHandle,
    file_map: dict[str, str],
    archive_remote: str,
    extract_dir: str,
) -> dict[str, float]:
    """Upload a single tar.gz archive to the sandbox and extract it to extract_dir.

    Works with any SandboxProvider.

    Returns:
        Dict with timing breakdown (all values in seconds):
            bundle_build_time, upload_semaphore_wait_time, bundle_upload_time,
            bundle_extract_time, bundle_total_time, bundle_size_bytes.
    """
    start_time = time.perf_counter()
    sandbox_id = handle.id
    logger.debug(f"[{sandbox_id}] Building bundle with {len(file_map)} files...")

    # Build tar.gz in dedicated thread pool to avoid blocking event loop at high concurrency
    build_start = time.perf_counter()
    loop = asyncio.get_running_loop()
    bundle_bytes = await loop.run_in_executor(_TAR_EXECUTOR, build_tar_gz, file_map)
    build_time = time.perf_counter() - build_start
    logger.debug(
        f"[{sandbox_id}] Bundle built in {build_time:.2f}s ({len(bundle_bytes)} bytes)"
    )

    semaphore = _get_upload_semaphore()
    semaphore_wait_start = time.perf_counter()
    async with semaphore:
        semaphore_wait_time = time.perf_counter() - semaphore_wait_start
        if semaphore_wait_time > 1.0:
            logger.debug(f"[{sandbox_id}] Waited {semaphore_wait_time:.2f}s for upload semaphore")

        # Upload bundle bytes
        upload_start = time.perf_counter()
        await provider.upload_bytes(handle, archive_remote, bundle_bytes)
        upload_time = time.perf_counter() - upload_start
    logger.debug(f"[{sandbox_id}] Bundle uploaded in {upload_time:.2f}s")

    # Extract using system tar (available in virtually every container image)
    extract_start = time.perf_counter()
    extract_cmd = f"mkdir -p {extract_dir} && tar xzf {archive_remote} -C {extract_dir}"
    resp = await provider.execute(handle, extract_cmd, timeout=30)
    if resp.exit_code != 0:
        raise RuntimeError(
            f"Failed to extract bundle in sandbox {sandbox_id} (exit={resp.exit_code}). "
            f"stdout={resp.stdout} stderr={resp.stderr}"
        )
    extract_time = time.perf_counter() - extract_start
    logger.debug(f"[{sandbox_id}] Bundle extracted in {extract_time:.2f}s")

    total_elapsed = time.perf_counter() - start_time
    logger.debug(
        f"[{sandbox_id}] Bundle deployed to {extract_dir} in {total_elapsed:.2f}s ({len(file_map)} files)"
    )

    return {
        "bundle_build_time": build_time,
        "upload_semaphore_wait_time": semaphore_wait_time,
        "bundle_upload_time": upload_time,
        "bundle_extract_time": extract_time,
        "bundle_total_time": total_elapsed,
        "bundle_size_bytes": float(len(bundle_bytes)),
    }
