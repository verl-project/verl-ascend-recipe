import json

import numpy as np
import pytest

from dapo_predictor.rollout_logging import RolloutJSONMixin, json_values


def test_metadata_conversion_preserves_reward_values_and_does_not_mutate_input():
    source = {"acc": [np.bool_(True)], "length": np.array([3]), "nested": {"score": np.float32(0.5)}}
    converted = json_values(source)
    assert json.loads(json.dumps(converted)) == {"acc": [True], "length": [3], "nested": {"score": 0.5}}
    assert isinstance(source["acc"][0], np.bool_)
    assert isinstance(source["length"], np.ndarray)


def test_pinned_upstream_jsonl_writer_accepts_numpy_reward_metadata(tmp_path):
    pytest.importorskip("verl")
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    class Trainer(RolloutJSONMixin, RayPPOTrainer):
        pass

    trainer = object.__new__(Trainer)
    trainer.global_steps = 1
    trainer._dump_generations(
        inputs=["question"],
        outputs=["answer"],
        gts=[np.str_("42")],
        scores=[np.float32(1)],
        reward_extra_infos_dict={"acc": [np.bool_(True)]},
        dump_path=str(tmp_path),
    )
    row = json.loads((tmp_path / "1.jsonl").read_text())
    assert row == {"input": "question", "output": "answer", "gts": "42", "score": 1.0, "step": 1, "acc": True}
