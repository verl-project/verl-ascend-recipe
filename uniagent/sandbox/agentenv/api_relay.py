"""Distribute TCP API connections over independent Windows SSH forwarding lanes.

This is a byte-transparent proxy, including streamed envd RPCs. It never
replays requests after forwarding bytes. Stdin EOF/heartbeat timeout stops it.
"""

import argparse
import asyncio
import json
import sys


class Relay:
    def __init__(self, ports, max_per_lane=96):
        self.ports = list(ports)
        self.active = [0] * len(self.ports)
        self.accepted = [0] * len(self.ports)
        self.max_per_lane = max_per_lane
        self.condition = asyncio.Condition()
        self.tasks = set()
        self.connect_errors = 0

    async def release(self, index):
        async with self.condition:
            self.active[index] -= 1
            self.condition.notify_all()

    async def connect(self):
        tried = set()
        for _ in self.ports:
            async with self.condition:
                await self.condition.wait_for(
                    lambda: any(i not in tried and count < self.max_per_lane for i, count in enumerate(self.active))
                )
                index = min(
                    (i for i in range(len(self.ports)) if i not in tried and self.active[i] < self.max_per_lane),
                    key=lambda i: (self.active[i], self.accepted[i]),
                )
                self.active[index] += 1
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", self.ports[index]), timeout=5
                )
            except BaseException as exc:
                await self.release(index)
                if not isinstance(exc, (OSError, asyncio.TimeoutError)):
                    raise
                self.connect_errors += 1
                tried.add(index)
            else:
                self.accepted[index] += 1
                return index, reader, writer
        raise ConnectionError("All API SSH lanes refused the connection")

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        index = None
        upstream = None
        pumps = []
        try:
            index, remote_reader, upstream = await self.connect()

            async def copy(source, target):
                while data := await source.read(65536):
                    target.write(data)
                    await target.drain()
                if target.can_write_eof():
                    target.write_eof()

            pumps = [asyncio.create_task(copy(reader, upstream)), asyncio.create_task(copy(remote_reader, writer))]
            await asyncio.gather(*pumps)
        except (OSError, asyncio.TimeoutError):
            pass
        finally:
            for pump in pumps:
                pump.cancel()
            if pumps:
                await asyncio.gather(*pumps, return_exceptions=True)
            writer.close()
            if upstream:
                upstream.close()
            if index is not None:
                await self.release(index)
            self.tasks.discard(task)

    async def close(self):
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=40191)
    parser.add_argument("--upstream-start", type=int, default=31090)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--max-per-lane", type=int, default=96)
    args = parser.parse_args()
    if not (
        1 <= args.count <= 32
        and 1 <= args.max_per_lane <= 128
        and 1 <= args.port <= 65535
        and 1 <= args.upstream_start <= 65536 - args.count
    ):
        parser.error("Invalid port range or capacity")
    relay = Relay(range(args.upstream_start, args.upstream_start + args.count), args.max_per_lane)
    server = await asyncio.start_server(relay.handle, "127.0.0.1", args.port, backlog=1024)
    print(
        f"API relay listening on {args.port}: {args.count} SSH lanes, {args.max_per_lane} API connections/lane",
        flush=True,
    )

    async def stats():
        while True:
            await asyncio.sleep(30)
            print(
                json.dumps(
                    {
                        "active": relay.active,
                        "accepted": relay.accepted,
                        "upstream_connect_errors": relay.connect_errors,
                    }
                ),
                flush=True,
            )

    reporter = asyncio.create_task(stats())
    try:
        while await asyncio.wait_for(asyncio.to_thread(sys.stdin.readline), timeout=60):
            pass
    finally:
        server.close()
        await server.wait_closed()
        reporter.cancel()
        await asyncio.gather(reporter, return_exceptions=True)
        await relay.close()


if __name__ == "__main__":
    asyncio.run(main())
