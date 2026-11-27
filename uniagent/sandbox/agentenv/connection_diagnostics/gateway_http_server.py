"""Gateway-only HTTP server settings, independent of diagnostic logging."""

import asyncio
import socket

import uvicorn

KEEP_ALIVE_SECONDS = 60


async def run_gateway_http(app, server_args, server_address):
    """Return an OS-assigned port and a cancellable server task."""
    app.server_args = server_args
    sock = socket.socket(socket.AF_INET6 if ":" in server_address else socket.AF_INET)
    try:
        sock.bind((server_address, 0))
        port = sock.getsockname()[1]
        config = uvicorn.Config(
            app,
            host=server_address,
            port=port,
            log_level="warning",
            timeout_keep_alive=KEEP_ALIVE_SECONDS,
        )
        server = uvicorn.Server(config)

        async def serve():
            try:
                await server.serve(sockets=[sock])
            finally:
                # GatewayActor.shutdown() cancels this task. Close listeners and
                # HTTP connections on cancellation as well as on normal exit.
                try:
                    if server.started and not server.should_exit:
                        server.should_exit = True
                        await server.shutdown(sockets=[sock])
                finally:
                    sock.close()

        task = asyncio.create_task(serve())
        try:
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("Gateway HTTP server exited before startup")
                await asyncio.sleep(0.01)
        except BaseException:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
    except BaseException:
        sock.close()
        raise
    return port, task
