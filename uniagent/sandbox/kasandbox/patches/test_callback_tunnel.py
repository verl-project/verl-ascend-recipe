"""CPU tests for routing and supervisor ownership; no live SSH endpoint is used."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uni_agent.gateway.callback_tunnel import CallbackRelayConfig, CallbackTunnelPool, forwarding_spec


class CallbackTests(unittest.TestCase):
    def test_independent_import_needs_no_ray(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import uni_agent.gateway.callback_tunnel; import sys; assert 'ray' not in sys.modules",
            ],
            check=True,
        )

    def test_routes_and_listener_address(self):
        config = CallbackRelayConfig("user@relay.example", 22, 29193, "http://pc.example:40193")
        routes = config.routes(2)
        self.assertEqual(routes[1]["advertised_base_url"], "http://pc.example:40194")
        route = {**routes[1], "host": "worker.example", "port": 54321}
        self.assertEqual(forwarding_spec(route), "127.0.0.1:29194:worker.example:54321")
        route["host"] = "2001:db8::1"
        self.assertIn("[2001:db8::1]", forwarding_spec(route))

    def test_invalid_endpoint_and_port_overflow(self):
        with self.assertRaises(ValueError):
            CallbackRelayConfig("-oProxyCommand=bad", 22, 2000, "http://pc.example")
        with self.assertRaises(ValueError):
            CallbackRelayConfig("relay.example", 22, 65535, "http://pc.example").routes(2)
        with self.assertRaises(ValueError):
            forwarding_spec({"host": "host -o bad", "port": 80, "relay_port": 8080})

    @unittest.skipUnless(os.name == "posix", "Supervisor uses POSIX SSH control sockets")
    def test_supervisor_ready_and_owned_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="cb-test-") as directory:
            root = Path(directory)
            ssh = root / "ssh"
            log = root / "calls"
            ssh.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$SSH_TEST_LOG"\nexit 0\n')
            ssh.chmod(0o700)
            config = CallbackRelayConfig("relay.example", 22, 29193, "http://pc.example:40193")
            routes = [{**config.routes(1)[0], "host": "worker.example", "port": 54321}]
            with patch.dict(
                os.environ,
                {
                    "PATH": directory + os.pathsep + os.environ["PATH"],
                    "SSH_TEST_LOG": str(log),
                    "CALLBACK_CONTROL_ROOT": directory,
                    "CALLBACK_SSH_PASSWORD": "",
                },
            ):
                pool = CallbackTunnelPool(config, routes)
                try:
                    pool.start()
                    process = pool.process
                    pool.check()
                finally:
                    pool.close()
                self.assertIsNotNone(process.poll())
                pool.close()
            calls = log.read_text()
            self.assertIn("127.0.0.1:29193:worker.example:54321", calls)
            self.assertIn("StrictHostKeyChecking=yes", calls)
            self.assertIn("-O exit relay.example", calls)


if __name__ == "__main__":
    unittest.main()
