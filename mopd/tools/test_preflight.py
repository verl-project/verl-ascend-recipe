import copy
import unittest
from types import SimpleNamespace

from preflight import EXPECTED_TEACHER_KEYS, validate_profile


def validated_config():
    actor = SimpleNamespace(strategy="fsdp", ppo_mini_batch_size=256, ppo_epochs=1)
    rollout = SimpleNamespace(name="vllm", n=1)
    return SimpleNamespace(
        data=SimpleNamespace(train_batch_size=256),
        actor_rollout_ref=SimpleNamespace(actor=actor, rollout=rollout),
        trainer=SimpleNamespace(
            device="npu",
            total_training_steps=100,
            save_freq=10,
            test_freq=20,
            val_before_train=True,
            resume_mode="auto",
        ),
    )


def validated_distillation():
    loss = SimpleNamespace(loss_mode="k1", use_policy_gradient=True, use_task_rewards=False)
    return SimpleNamespace(
        enabled=True,
        teacher_models=dict.fromkeys(EXPECTED_TEACHER_KEYS),
        teacher_key="data_source",
        distillation_loss=loss,
    )


class ValidateProfileTest(unittest.TestCase):
    def test_validated_profile(self):
        validate_profile(validated_config(), validated_distillation(), {"enable_thinking": False})

    def test_rejects_historical_batch128_profile(self):
        config = copy.deepcopy(validated_config())
        config.data.train_batch_size = 128
        with self.assertRaisesRegex(ValueError, "train batch size is 256"):
            validate_profile(config, validated_distillation(), {"enable_thinking": False})

    def test_rejects_thinking_template(self):
        with self.assertRaisesRegex(ValueError, "disables thinking"):
            validate_profile(validated_config(), validated_distillation(), {"enable_thinking": True})


if __name__ == "__main__":
    unittest.main()
