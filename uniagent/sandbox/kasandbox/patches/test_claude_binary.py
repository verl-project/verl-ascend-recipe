"""Run with PYTHONPATH pointing to Uni-Agent after applying claude-binary.patch."""

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from uni_agent.agents.claude_code.agent import ClaudeCodeAgent, ClaudeCodeConfig


class BinaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.sandbox = SimpleNamespace(
            exec_shell=AsyncMock(return_value=SimpleNamespace(exit_code=0, stdout="ok", stderr="")),
            exec=AsyncMock(return_value=SimpleNamespace(exit_code=0)),
            upload_file=AsyncMock(),
        )

    async def test_existing_binary_keeps_upstream_behavior(self):
        agent = ClaudeCodeAgent(ClaudeCodeConfig())
        await agent._ensure_claude(self.sandbox)
        self.sandbox.upload_file.assert_not_awaited()
        self.assertEqual(self.sandbox.exec_shell.await_count, 1)

    async def test_upload_checks_hash_before_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "claude"
            binary.write_bytes(b"test binary")
            config = ClaudeCodeConfig(local_binary_path=str(binary), local_binary_sha256="0" * 64)
            agent = ClaudeCodeAgent(config)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                await agent._ensure_claude(self.sandbox)
            self.sandbox.upload_file.assert_not_awaited()
            config.local_binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
            await agent._ensure_claude(self.sandbox)
            self.sandbox.upload_file.assert_awaited_once()
            commands = [call.args[0] for call in self.sandbox.exec_shell.await_args_list]
            self.assertTrue(any("sha256sum -c" in command and "mv -f" in command for command in commands))
            self.assertTrue(commands[-1].startswith("rm -f"))

    async def test_failed_upload_verification_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "claude"
            binary.write_bytes(b"binary")
            self.sandbox.exec_shell.side_effect = [
                SimpleNamespace(exit_code=0),
                SimpleNamespace(exit_code=1, stderr="checksum failed", stdout=""),
                SimpleNamespace(exit_code=0),
            ]
            agent = ClaudeCodeAgent(ClaudeCodeConfig(local_binary_path=str(binary)))
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                await agent._ensure_claude(self.sandbox)
            self.assertTrue(self.sandbox.exec_shell.await_args.args[0].startswith("rm -f"))

    async def test_nfs_verification_and_private_home(self):
        config = ClaudeCodeConfig(
            nfs_source="nfs.example:/export",
            nfs_relative_binary="arm64/claude",
            nfs_binary_sha256="a" * 64,
            model={"model_name": "test"},
        )
        agent = ClaudeCodeAgent(config)
        await agent._ensure_claude(self.sandbox)
        commands = [call.args[0] for call in self.sandbox.exec_shell.await_args_list]
        self.assertIn("mount -t nfs -o ro", commands[0])
        self.assertIn("findmnt -rn -o SOURCE", commands[0])
        self.assertIn("sha256sum -c", commands[1])
        self.sandbox.upload_file.assert_not_awaited()
        self.assertEqual(agent._claude_argv("test")[0], "/opt/claude-shared/arm64/claude")
        self.assertTrue(agent._claude_env("http://localhost")["HOME"].startswith("/tmp/uni-agent-claude-"))

    def test_conflicting_modes_and_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            ClaudeCodeConfig(nfs_source="nfs.example:/export", local_binary_path="/tmp/claude")
        with self.assertRaises(ValueError):
            ClaudeCodeConfig(nfs_relative_binary="../claude")


if __name__ == "__main__":
    unittest.main()
