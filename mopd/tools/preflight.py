#!/usr/bin/env python3
"""Check resolved MOPD configuration, dataset routes and tokenizer compatibility without NPUs."""

import argparse
import json
from collections import Counter
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


EXPECTED_TEACHER_KEYS = {"openai/gsm8k", "DigitalLearningGmbH/MATH-lighteval"}


def validate_profile(config, distillation, template_kwargs) -> None:
    loss = distillation.distillation_loss
    require(distillation.enabled and len(distillation.teacher_models) == 2, "Exactly two teachers are required")
    require(distillation.teacher_key == "data_source", "Teachers must be routed by data_source")
    require(
        set(distillation.teacher_models) == EXPECTED_TEACHER_KEYS,
        "Teacher routes do not match the two validated datasets",
    )
    require(config.actor_rollout_ref.actor.strategy == "fsdp", "The validated student strategy is FSDP")
    require(
        config.actor_rollout_ref.rollout.name == "vllm" and config.trainer.device == "npu",
        "The validated rollout and device are vLLM and NPU",
    )
    require(config.data.train_batch_size == 256, "The validated train batch size is 256")
    require(config.actor_rollout_ref.actor.ppo_mini_batch_size == 256, "The validated PPO mini-batch size is 256")
    require(config.actor_rollout_ref.actor.ppo_epochs == 1, "The validated configuration uses one PPO epoch")
    require(config.actor_rollout_ref.rollout.n == 1, "The validated configuration uses one rollout per prompt")
    require(loss.loss_mode == "k1", "The validated distillation loss is k1")
    require(loss.use_policy_gradient is True, "The validated k1 path uses the policy-gradient estimator")
    require(loss.use_task_rewards is False, "The validated loss excludes task rewards")
    require(config.trainer.total_training_steps == 100, "The validated training length is 100 steps")
    require(
        config.trainer.save_freq == 10
        and config.trainer.test_freq == 20
        and config.trainer.val_before_train is True
        and config.trainer.resume_mode == "auto",
        "The validated save, validation and resume settings are save=10, test=20, val_before_train=true, resume=auto",
    )
    require(template_kwargs.get("enable_thinking") is False, "The validated data template disables thinking")


def main():
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from transformers import AutoConfig, AutoTokenizer

    from verl.utils.config import omega_conf_to_dataclass, validate_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Hydra --cfg job --resolve output")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    validate_config(config, use_reference_policy=False, use_critic=False)
    distillation = omega_conf_to_dataclass(config.distillation)
    template_kwargs = (
        OmegaConf.to_container(config.data.apply_chat_template_kwargs, resolve=True)
        if "apply_chat_template_kwargs" in config.data
        else {}
    )
    validate_profile(config, distillation, template_kwargs)

    student = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path, local_files_only=True)
    vocab = student.get_vocab()
    student_config = AutoConfig.from_pretrained(config.actor_rollout_ref.model.path, local_files_only=True)
    models = {"student": {"path": config.actor_rollout_ref.model.path, "hidden_size": student_config.hidden_size}}
    for key, teacher in distillation.teacher_models.items():
        tokenizer = AutoTokenizer.from_pretrained(teacher.model_path, local_files_only=True)
        require(tokenizer.get_vocab() == vocab, f"Token-to-ID mapping differs for {key}")
        require(tokenizer.all_special_ids == student.all_special_ids, f"Special token IDs differ for {key}")
        model_config = AutoConfig.from_pretrained(teacher.model_path, local_files_only=True)
        require(model_config.vocab_size == student_config.vocab_size, f"Vocabulary size differs for {key}")
        models[key] = {"path": teacher.model_path, "hidden_size": model_config.hidden_size, "npus": teacher.world_size}

    datasets = {}
    for split, paths in (("train", config.data.train_files), ("test", config.data.val_files)):
        counts = Counter()
        retained = Counter()
        for path in paths:
            rows = pq.read_table(path, columns=["data_source", "prompt", "reward_model"]).to_pylist()
            for row in rows:
                key = row["data_source"]
                require(key in EXPECTED_TEACHER_KEYS, f"No teacher route for {key!r} in {path}")
                require(row["reward_model"]["ground_truth"] is not None, f"Missing ground truth in {path}")
                counts[key] += 1
                tokens = student.apply_chat_template(
                    row["prompt"],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    **template_kwargs,
                )
                require(
                    isinstance(tokens, list) and all(isinstance(token, int) for token in tokens),
                    f"Tokenizer returned invalid token IDs for {path}",
                )
                if len(tokens) <= config.data.max_prompt_length:
                    retained[key] += 1
        require(set(retained) == EXPECTED_TEACHER_KEYS, f"A teacher has no retained {split} samples")
        datasets[split] = {"raw": dict(counts), "within_prompt_limit": dict(retained)}
    steps_per_epoch = sum(datasets["train"]["within_prompt_limit"].values()) // config.data.train_batch_size
    require(steps_per_epoch > 0, "No complete training batch remains after prompt filtering")
    require(
        steps_per_epoch * config.trainer.total_epochs >= config.trainer.total_training_steps,
        "The epoch limit cannot reach the requested training length",
    )
    student_npus = config.trainer.n_gpus_per_node * config.trainer.nnodes
    teacher_npus = distillation.n_gpus_per_node * distillation.nnodes
    print(
        json.dumps(
            {
                "models": models,
                "datasets": datasets,
                "steps_per_epoch": steps_per_epoch,
                "student_npus": student_npus,
                "teacher_npus": teacher_npus,
                "total_npus": student_npus + teacher_npus,
                "limitation": "Configuration and data checks do not prove teacher inference or student updates.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
