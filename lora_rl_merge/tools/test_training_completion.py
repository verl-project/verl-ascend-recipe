import unittest

from check_training_completion import check_completion


def training_lines(steps):
    return "\n".join(
        f"(TaskRunnerV1 pid=1) step:{step} - training/global_step:{step} - timing_s/update_actor:1.2" for step in steps
    )


class TrainingCompletionTest(unittest.TestCase):
    def test_full_run_and_resume(self):
        for start in (1, 51):
            with self.subTest(start=start):
                self.assertEqual(check_completion(training_lines(range(start, 101)), 100), list(range(start, 101)))

    def test_numpy_global_step_and_ansi_log_prefix(self):
        text = "\x1b[32mstep:100 - training/global_step:np.int64(100) - timing_s/update_actor:np.float64(1.2)\x1b[0m"
        self.assertEqual(check_completion(text, 100), [100])

    def test_epoch_limit_does_not_count_as_completion(self):
        with self.assertRaisesRegex(ValueError, "observed 58"):
            check_completion(training_lines(range(1, 59)), 100)

    def test_validation_does_not_count_as_training(self):
        with self.assertRaisesRegex(ValueError, "observed none"):
            check_completion("step:100 - val/reward:0.9", 100)

    def test_missing_duplicate_and_inconsistent_steps(self):
        for text in (
            training_lines([51, 53, 100]),
            training_lines([99, 99, 100]),
            "step:100 - training/global_step:99 - timing_s/update_actor:1.2",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                check_completion(text, 100)


if __name__ == "__main__":
    unittest.main()
