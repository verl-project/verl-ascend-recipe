"""CPU-only regression for the explicit worker environment allowlist."""

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "sandbox_environment", Path(__file__).parent / "scripts/sandbox/environment.py"
)
environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(environment)


class EnvironmentTests(unittest.TestCase):
    def test_forwards_sandbox_settings_but_not_unrelated_credentials(self):
        values = {
            "E2B_API_KEY": "test-only",
            "E2B_API_URL": "http://localhost:9191",
            "CALLBACK_RELAY_SSH_TARGET": "user@relay.example.com",
            "UNRELATED_SECRET": "do-not-forward",
        }
        with patch.dict(os.environ, values, clear=True):
            result = environment.worker_environment()
        self.assertEqual(result["E2B_API_KEY"], "test-only")
        self.assertEqual(result["CALLBACK_RELAY_SSH_TARGET"], values["CALLBACK_RELAY_SSH_TARGET"])
        self.assertNotIn("UNRELATED_SECRET", result)

    def test_missing_api_key_fails(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(SystemExit):
            environment.worker_environment()

    @unittest.skipIf(os.name == "nt", "POSIX runtime file permissions")
    def test_runtime_file_is_private_and_uses_shared_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "verl"
            entry = source / "verl/trainer/main_ppo.py"
            entry.parent.mkdir(parents=True)
            entry.touch()
            output = root / "runtime.json"
            env = {
                **os.environ,
                "E2B_API_KEY": "test-only",
                "E2B_API_URL": "http://localhost:9191",
                "UNI_AGENT_ROOT": str(root / "uni-agent"),
            }
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / "scripts/training/prepare_runtime.py"),
                    "--verl-source",
                    str(source),
                    "--output",
                    str(output),
                ],
                env=env,
                check=True,
                capture_output=True,
            )
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            runtime = json.loads(output.read_text())
            self.assertNotIn("working_dir", runtime)
            self.assertEqual(runtime["env_vars"]["E2B_API_KEY"], "test-only")
            self.assertTrue(runtime["env_vars"]["PYTHONPATH"].startswith(env["UNI_AGENT_ROOT"] + os.pathsep))


if __name__ == "__main__":
    unittest.main()
