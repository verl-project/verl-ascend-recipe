import tempfile
import unittest
from pathlib import Path

from summarize_training import summarize


def training_lines(overrides=None):
    overrides = overrides or {}
    lines = []
    for step in range(1, 11):
        values = {
            "training/global_step": step,
            "timing_s/update_actor": 1,
            "actor/loss": 0.5,
            "actor/grad_norm": 1,
            "critic/rewards/mean": step / 10,
            "perf/total_num_tokens": 400,
            "perf/time_per_step": 1,
            "perf/throughput": 100,
        }
        values.update(overrides.get(step, {}))
        metrics = " - ".join(f"{key}:{value}" for key, value in values.items())
        lines.append(f"step:{step} - {metrics}")
    return "\n".join(lines)


class SummarizeTrainingTest(unittest.TestCase):
    def summarize_text(self, text):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "training.log"
            log.write_text(text)
            return summarize([(log, 1, 10)], expected_steps=10, devices=4)

    def test_complete_log(self):
        result = self.summarize_text(training_lines())
        self.assertTrue(result["training_length_complete"])
        self.assertTrue(result["actor_metrics_finite"])
        self.assertEqual(result["zero_gradient_steps"], [])
        self.assertAlmostEqual(result["reward"]["first_10_mean"], 0.55)
        self.assertAlmostEqual(result["throughput"]["weighted_tokens_per_second_per_device"], 100)

    def test_missing_required_metric(self):
        line = training_lines().splitlines()[4].replace(" - actor/loss:0.5", "")
        lines = training_lines().splitlines()
        lines[4] = line
        with self.assertRaisesRegex(ValueError, "Missing actor/loss at step 5"):
            self.summarize_text("\n".join(lines))

    def test_nonfinite_actor_metric_is_reported(self):
        result = self.summarize_text(training_lines({3: {"actor/loss": "nan"}}))
        self.assertFalse(result["actor_metrics_finite"])
        self.assertEqual(result["nonfinite_metric_step_counts"], {"actor/loss": 1})

    def test_incorrect_throughput_denominator(self):
        with self.assertRaisesRegex(ValueError, "different device denominator at step 7"):
            self.summarize_text(training_lines({7: {"perf/throughput": 400}}))


if __name__ == "__main__":
    unittest.main()
