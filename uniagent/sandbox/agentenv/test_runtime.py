"""CPU-only regressions: secret handling at the Ray boundary and relay bytes."""

import asyncio
import os
import unittest
from unittest.mock import patch

from agentenv_driver import REQUIRED_ENV, runtime_environment
from api_relay import Relay
from migration_adapter import normalize_tools_kwargs


class RuntimeTests(unittest.TestCase):
    def test_migrated_dataset_binding_is_normalized_without_mutation(self):
        original = {
            "task": {
                "name": "swe_rebench",
                "metadata": {},
                "sandbox": {
                    "provider": "agentenv",
                    "template_id": "template-example",
                    "image": "template-example",
                    "workdir": "/testbed",
                    "architecture": "arm64",
                },
            }
        }
        normalized = normalize_tools_kwargs(original)
        self.assertEqual(normalized["task"]["name"], "agentenv_swe_v2")
        self.assertEqual(normalized["task"]["sandbox"]["sandbox_kwargs"]["template"], "template-example")
        self.assertEqual(original["task"]["name"], "swe_rebench")

    def environment(self):
        values = dict.fromkeys(REQUIRED_ENV, "configured")
        values.update(E2B_API_URL="http://localhost:40190", E2B_SANDBOX_URL="http://localhost:40190")
        return values

    def test_merges_without_mutating_and_forwards_key(self):
        original = {"env_vars": {"KEEP": "yes"}, "excludes": ["private"]}
        with patch.dict(os.environ, self.environment(), clear=True):
            result = runtime_environment(original)
        self.assertEqual(original["env_vars"], {"KEEP": "yes"})
        self.assertEqual(result["env_vars"]["E2B_API_KEY"], "configured")
        self.assertEqual(result["env_vars"]["KEEP"], "yes")

    def test_missing_credentials_and_unresolved_values_fail(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "Missing E2B_API_KEY"):
            runtime_environment()
        env = self.environment()
        env["E2B_API_KEY"] = "${oc.env:E2B_API_KEY}"
        with patch.dict(os.environ, env, clear=True), self.assertRaisesRegex(ValueError, "unresolved"):
            runtime_environment()


class RelayTests(unittest.IsolatedAsyncioTestCase):
    async def test_forwards_binary_data_and_half_close(self):
        payload = bytes(range(256)) * 1024

        async def echo(reader, writer):
            received = await reader.read()
            writer.write(received)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        upstream = await asyncio.start_server(echo, "127.0.0.1", 0)
        relay = Relay([upstream.sockets[0].getsockname()[1]])
        server = await asyncio.start_server(relay.handle, "127.0.0.1", 0)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
            writer.write(payload)
            await writer.drain()
            writer.write_eof()
            self.assertEqual(await asyncio.wait_for(reader.read(), timeout=5), payload)
            writer.close()
            await writer.wait_closed()
            if relay.tasks:
                await asyncio.wait_for(asyncio.gather(*relay.tasks), timeout=5)
            self.assertEqual(relay.active, [0])
        finally:
            server.close()
            upstream.close()
            await server.wait_closed()
            await upstream.wait_closed()


if __name__ == "__main__":
    unittest.main()
