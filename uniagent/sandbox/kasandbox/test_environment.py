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


class UpstreamSchemaTests(unittest.TestCase):
    def test_task_defaults_and_prepared_data_use_upstream_e2b(self):
        import importlib.util

        import pyarrow.parquet as pq
        import yaml
        from uni_agent.sandbox.e2b import E2BSandbox
        from uni_agent.tasks.swe_bench.task import SWEBenchTaskConfig

        root = Path(__file__).resolve().parent
        path = root / "examples/e2b_swe_grpo/prepare_data.py"
        spec = importlib.util.spec_from_file_location("prepare_data", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        row = {key: "example" for key in module.REQUIRED}
        row.update(instance_id="example__issue-1", FAIL_TO_PASS=[], PASS_TO_PASS=[])
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "rows.json").write_text(json.dumps([row]))
            manifest = {
                "schema_version": 1,
                "instances": {
                    row["instance_id"]: {"template": "example-template", "split": "train"},
                },
            }
            (work / "manifest.json").write_text(json.dumps(manifest))
            module.prepare(work / "rows.json", work / "manifest.json", work / "output")
            task = pq.read_table(work / "output/train.parquet").to_pylist()[0]["extra_info"]["tools_kwargs"]["task"]
            defaults = yaml.safe_load((root / "examples/e2b_swe_grpo/task_config.yaml").read_text())[0]
            defaults["metadata"] = task["metadata"]
            defaults["sandbox"]["sandbox_kwargs"].update(task["sandbox"]["sandbox_kwargs"])
            config = SWEBenchTaskConfig(**defaults)
            sandbox = E2BSandbox.from_config(config.sandbox)
            self.assertEqual(sandbox.template, "example-template")
            self.assertIsNone(sandbox.user)


if __name__ == "__main__":
    unittest.main()
