"""Expose SSH loopback forwards to sandbox guests until bridge stdin closes."""

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Thread

from connection_diagnostics.callback_http_trace import HTTPTrace


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bind", required=True)
    p.add_argument("--port", type=int, default=39193)
    p.add_argument("--upstream-port", type=int, default=39293)
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--log-file", default=str(Path(__file__).resolve().parent / "artifacts" / "callback_proxy.log"))
    args = p.parse_args()
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("callback_proxy")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S +0800")
    formatter.converter = lambda timestamp: time.gmtime(timestamp + 8 * 3600)
    for handler in (
        logging.StreamHandler(),
        RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.info(
        "listeners starting bind=%s ports=%s-%s upstream=%s-%s log_file=%s",
        args.bind,
        args.port,
        args.port + args.count - 1,
        args.upstream_port,
        args.upstream_port + args.count - 1,
        log_path,
    )

    async def forward(reader, writer, port, host="127.0.0.1"):
        remote = None
        started = time.monotonic()
        connection_id = uuid.uuid4().hex[:12]
        peer = writer.get_extra_info("peername")
        local = writer.get_extra_info("sockname")
        counts = {"client_to_upstream": 0, "upstream_to_client": 0}
        last_io = dict.fromkeys(counts)
        first_close = None

        def emit(event, **fields):
            logger.info(
                "%s",
                json.dumps(
                    dict(
                        event=event,
                        connection_id=connection_id,
                        peer=peer,
                        listener=local,
                        upstream=[host, port],
                        duration_s=round(time.monotonic() - started, 3),
                        **fields,
                    )
                ),
            )

        trace = HTTPTrace(emit)
        emit("connection_open")

        def log_error(exc, phase):
            logger.warning(
                "connection_error id=%s peer=%s listener=%s upstream=%s:%s "
                "phase=%s duration_s=%.3f client_to_upstream_bytes=%s "
                "upstream_to_client_bytes=%s error_type=%s errno=%s error=%s",
                connection_id,
                peer,
                local,
                host,
                port,
                phase,
                time.monotonic() - started,
                counts["client_to_upstream"],
                counts["upstream_to_client"],
                type(exc).__name__,
                getattr(exc, "errno", None),
                exc,
                exc_info=True,
            )

        try:
            try:
                remote_reader, remote = await asyncio.open_connection(host, port)
            except (OSError, ConnectionError) as exc:
                log_error(exc, "connect_upstream")
                raise

            async def copy(source, target, direction):
                nonlocal first_close
                try:
                    while data := await source.read(65536):
                        last_io[direction] = time.monotonic()
                        trace.feed(direction, data)
                        target.write(data)
                        await target.drain()
                        counts[direction] += len(data)
                    if first_close is None:
                        first_close = direction
                    trace.eof(direction)
                    if target.can_write_eof():
                        target.write_eof()
                except (OSError, ConnectionError) as exc:
                    if first_close is None:
                        first_close = direction + ":error"
                    log_error(exc, direction)
                    raise

            await asyncio.gather(
                copy(reader, remote, "client_to_upstream"), copy(remote_reader, writer, "upstream_to_client")
            )
        except (OSError, ConnectionError):
            pass
        finally:
            emit(
                "connection_close",
                first_close=first_close,
                bytes=counts,
                idle_s={k: round(time.monotonic() - v, 3) if v is not None else None for k, v in last_io.items()},
            )
            writer.close()
            if remote is not None:
                remote.close()

    servers = []
    assets = None
    try:
        for i in range(args.count):
            servers.append(
                await asyncio.start_server(
                    lambda r, w, port=args.upstream_port + i: forward(r, w, port), args.bind, args.port + i
                )
            )
        assets = ThreadingHTTPServer(
            (args.bind, args.port - 1),
            partial(SimpleHTTPRequestHandler, directory=str(Path(__file__).resolve().parent / "assets")),
        )
        Thread(target=assets.serve_forever, daemon=True).start()
        print("AgentENV callback listeners ready", flush=True)
        while await asyncio.wait_for(asyncio.to_thread(sys.stdin.readline), timeout=60):
            pass
    finally:
        if assets:
            assets.shutdown()
            assets.server_close()
        for server in servers:
            server.close()
        await asyncio.gather(*(s.wait_closed() for s in servers))


if __name__ == "__main__":
    asyncio.run(main())
