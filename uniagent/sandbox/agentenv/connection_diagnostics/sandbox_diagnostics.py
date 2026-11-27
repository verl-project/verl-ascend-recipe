"""Persist guest service logs before the task releases its sandbox."""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("uni_agent.sandbox.diagnostics")


async def save_envd_logs(sandbox, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = {
        "sandbox_id": getattr(sandbox, "sandbox_id", None),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "source": "/var/log/agentenv",
        "status": "collecting",
    }
    archive = destination / "envd.tar.gz"
    remote = "/tmp/agentenv-envd-" + uuid.uuid4().hex + ".tar.gz"

    async def collect():
        # Logs remain active during collection; tar exit 1 means files changed.
        result = await sandbox.exec(
            ["tar", "--exclude=agentenv/envd/lock", "-czf", remote, "-C", "/var/log", "agentenv"], timeout=20
        )
        if result.exit_code not in (0, 1):
            raise RuntimeError(f"envd archive command exited {result.exit_code}")
        manifest["tar_exit_code"] = result.exit_code
        archive.touch(mode=0o600)
        await sandbox.download_file(remote, archive)
        archive.chmod(0o600)
        manifest["bytes"] = archive.stat().st_size

    try:
        await asyncio.wait_for(collect(), timeout=45)
        manifest["status"] = "saved"
        logger.info("envd logs saved sandbox_id=%s path=%s", manifest["sandbox_id"], archive)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        logger.warning(
            "envd log collection failed sandbox_id=%s error_type=%s; proceeding with teardown",
            manifest["sandbox_id"],
            type(exc).__name__,
        )
    finally:
        # The temporary guest archive is removed with the sandbox itself.
        path = destination / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        path.chmod(0o600)


@asynccontextmanager
async def sandbox_with_diagnostics(sandbox, fallback_dir):
    from uni_agent.logging.context import get_current_log_context

    async with sandbox:
        context = get_current_log_context()
        parent = Path(context.log_path).parent if context and context.log_path else Path(fallback_dir)
        sandbox_id = getattr(sandbox, "sandbox_id", None) or uuid.uuid4().hex
        destination = parent / "sandbox_logs" / sandbox_id
        logger.info("sandbox acquired sandbox_id=%s diagnostics=%s", sandbox_id, destination)
        try:
            yield sandbox
        finally:
            # Shield collection from task cancellation, and finish before __aexit__ deletes the VM.
            collector = asyncio.create_task(save_envd_logs(sandbox, destination))
            cancelled = False
            while not collector.done():
                try:
                    await asyncio.shield(collector)
                except asyncio.CancelledError:
                    cancelled = True
                except Exception:
                    break
            try:
                collector.result()
            except Exception as exc:
                logger.warning("Cannot persist envd logs sandbox_id=%s error_type=%s", sandbox_id, type(exc).__name__)
            if cancelled:
                raise asyncio.CancelledError
