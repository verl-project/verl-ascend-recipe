import hashlib
import json
import unittest
from pathlib import Path

from check_validation import ROLLOUT_KEYS, check_validation


def trajectory() -> str:
    lines = ["step:0 - val-core/openai/gsm8k/acc/mean@1:np.float64(0.25)"]
    for step in range(1, 101):
        values = {
            "training/global_step": step,
            "timing_s/update_actor": 1,
            "actor/loss": 0.1,
            "actor/grad_norm": 0.1,
            "critic/rewards/mean": step / 100,
            "perf/total_num_tokens": 8000,
            "perf/time_per_step": 10,
            "perf/throughput": 200,
            **dict.fromkeys(ROLLOUT_KEYS, 0.1),
        }
        if step == 100:
            values["val-core/openai/gsm8k/acc/mean@1"] = 0.8
        lines.append(f"step:{step} - " + " - ".join(f"{key}:{value}" for key, value in values.items()))
    return "\n".join(lines)


class ValidationTest(unittest.TestCase):
    def test_bundled_training_log_reproduces_the_published_summary(self):
        evidence = Path(__file__).resolve().parents[1] / "evidence/910b3-100step"
        raw = (evidence / "training_100step.log").read_bytes()
        result = check_validation(raw.decode(), 100, 4)
        result["log_sha256"] = hashlib.sha256(raw).hexdigest()
        self.assertEqual(result, json.loads((evidence / "summary.json").read_text()))

    def test_complete_log_and_numpy_validation(self):
        result = check_validation(trajectory(), 100, 4)
        self.assertEqual(result["training_steps"], 100)
        self.assertEqual(result["weighted_tokens_per_second_per_npu"], 200)
        self.assertEqual(result["validation"][0]["gsm8k_accuracy"], 0.25)

    def test_incomplete_duplicate_and_validation_only_updates_fail(self):
        lines = trajectory().splitlines()
        for text in ("\n".join(lines[:59]), "\n".join(lines + lines[-1:]), "\n".join(lines[:1] + lines[2:])):
            with self.subTest(text=text[:40]), self.assertRaisesRegex(ValueError, "contiguous"):
                check_validation(text, 100, 4)

    def test_missing_and_nonfinite_diagnostics_fail(self):
        for key in ROLLOUT_KEYS:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "Missing"):
                    check_validation(trajectory().replace(f" - {key}:0.1", "", 1), 100, 4)
                with self.assertRaisesRegex(ValueError, "Non-finite"):
                    check_validation(trajectory().replace(f"{key}:0.1", f"{key}:nan", 1), 100, 4)

    def test_zero_gradient_and_no_reward_improvement_fail(self):
        with self.assertRaisesRegex(ValueError, "gradient"):
            check_validation(trajectory().replace("actor/grad_norm:0.1", "actor/grad_norm:0", 1), 100, 4)
        with self.assertRaisesRegex(ValueError, "Reward did not increase"):
            check_validation(
                trajectory()
                .replace("critic/rewards/mean:", "other:")
                .replace("actor/loss:", "critic/rewards/mean:0.1 - actor/loss:"),
                100,
                4,
            )

    def test_wrong_device_count_and_low_throughput_fail(self):
        with self.assertRaisesRegex(ValueError, "denominator"):
            check_validation(trajectory(), 100, 2)
        with self.assertRaisesRegex(ValueError, "does not exceed"):
            check_validation(
                trajectory().replace("tokens:8000", "tokens:4000").replace("throughput:200", "throughput:100"), 100, 4
            )

    def test_missing_final_validation_fails(self):
        with self.assertRaisesRegex(ValueError, "final GSM8K"):
            check_validation(trajectory().replace(" - val-core/openai/gsm8k/acc/mean@1:0.8", ""), 100, 4)


if __name__ == "__main__":
    unittest.main()
