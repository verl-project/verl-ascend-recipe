import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "dapo_predictor" / "predictor_dapo_trainer.py"
WORKER = REPO_ROOT / "dapo_predictor" / "predictor_worker.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_source(path: Path, name: str) -> str:
    source = _source(path)
    module = ast.parse(source)
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{name} not found in {path}")


class PredictorReviewRegressionTests(unittest.TestCase):
    def test_predictor_reorders_repeated_prompts_before_union_with_rollouts(self):
        fit_source = _function_source(TRAINER, "fit")

        statements = ast.walk(ast.parse(fit_source))
        operations = {}
        for node in statements:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "new_batch" for t in node.targets
            ):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                    if node.value.func.attr == "repeat":
                        operations["repeat"] = node.lineno
                    elif (
                        node.value.func.attr == "union"
                        and node.value.args
                        and isinstance(node.value.args[0], ast.Name)
                        and node.value.args[0].id == "gen_batch_output"
                    ):
                        operations["union"] = node.lineno
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "_apply_predictor_order" and isinstance(node.args[0], ast.Name):
                    if node.args[0].id == "new_batch":
                        operations["reorder"] = node.lineno
        repeat_pos, reorder_pos, union_pos = (operations[key] for key in ("repeat", "reorder", "union"))

        self.assertGreater(reorder_pos, repeat_pos)
        self.assertLess(reorder_pos, union_pos)

    def test_predictor_worker_training_guards_small_response_length_and_uses_stable_shuffle(
        self,
    ):
        update_source = _function_source(WORKER, "update_predictor")

        self.assertIn("max(1,", update_source)
        self.assertIn("torch.Generator()", update_source)
        self.assertIn("generator=", update_source)
        self.assertIn("listmle_generator", update_source)
        self.assertIn('cfg.get("epochs", 10)', update_source)

    def test_predictor_worker_avoids_global_seed_and_debug_prints(self):
        source = _source(WORKER)
        tree = ast.parse(source)

        manual_seed_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "manual_seed"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
        ]
        print_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
        ]

        self.assertEqual(manual_seed_calls, [])
        self.assertEqual(print_calls, [])

    def test_kendall_tau_is_optional_when_scipy_is_missing(self):
        update_source = _function_source(WORKER, "update_predictor")

        self.assertIn("try:", update_source)
        self.assertIn("ImportError", update_source)
        self.assertIn("kendalltau", update_source)

    def test_cold_d1_skips_features_but_online_d2_collects_before_activation(self):
        from types import SimpleNamespace

        source = _function_source(TRAINER, "_needs_predictor_features")
        namespace = {}
        exec(source, namespace)
        needs_features = namespace["_needs_predictor_features"]
        for active, online, expected in (
            (False, False, False),
            (False, True, True),
            (True, False, True),
        ):
            value = SimpleNamespace(
                _predictor_active=lambda epoch, active=active: active,
                _d2_online_enabled=lambda online=online: online,
            )
            self.assertEqual(needs_features(value, 0), expected)


if __name__ == "__main__":
    unittest.main()
