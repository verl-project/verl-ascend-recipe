"""Check relocated launcher paths and the 16-device preset without hardware."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

FAPO = Path(__file__).resolve().parent


class LauncherTest(unittest.TestCase):
    def run_launcher(self, mode="colocate"):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "bin"
            binary.mkdir()
            capture = root / "args.json"
            fake_python = binary / "python3"
            # Use the current interpreter directly to avoid resolving the stub again.
            fake_python.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['CAPTURE'], 'w') as f:\n"
                "    json.dump({'args': sys.argv[1:], 'cwd': os.getcwd()}, f)\n"
            )
            fake_python.chmod(0o755)
            env = dict(os.environ)
            env.update(
                VERL_ROOT=str(root),
                RAY_DATA_HOME=str(root),
                CAPTURE=str(capture),
                REWARD_MODE=mode,
                HF_HOME=str(root / "hf"),
                HF_DATASETS_CACHE=str(root / "hf/datasets"),
                PATH=str(binary) + os.pathsep + os.environ["PATH"],
            )
            result = subprocess.run(
                ["bash", str(FAPO / "run_16card_100steps.sh")],
                env=env,
                cwd="/",
                capture_output=True,
                text=True,
            )
            return result, json.loads(capture.read_text()) if capture.exists() else None, str(root)

    def test_relocated_16_device_entry(self):
        result, captured, root = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(captured["cwd"]).resolve(), Path(root).resolve())
        args = captured["args"]
        for option in (
            "trainer.n_gpus_per_node=12",
            "reward.reward_model.n_gpus_per_node=4",
            "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4",
            "reward.reward_model.rollout.tensor_model_parallel_size=2",
            "trainer.total_training_steps=100",
            "data.train_max_samples=6400",
            "trainer.save_freq=-1",
            "trainer.test_freq=-1",
            f"reward.custom_reward_function.path={FAPO}/reward_fn_reasoning.py",
        ):
            self.assertIn(option, args)

    def test_invalid_reward_mode_fails_before_training(self):
        result, captured, _ = self.run_launcher("colocated")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REWARD_MODE must be one of", result.stderr)
        self.assertIsNone(captured)


if __name__ == "__main__":
    unittest.main()
