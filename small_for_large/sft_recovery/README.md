# SFT Behavior Recovery for Expert-Pruned MoE Models (small_for_large, stage 2)

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt). The data-build and evaluation
tools in this recipe are inference/trainer agnostic (they talk to an
OpenAI-compatible vLLM server and produce HF-format datasets); the RL-parity
stage (planned, see roadmap) is where `verl` enters.

## Background

The `small_for_large` recipe prunes a large MoE model to its top-k
high-frequency experts for fast RL accuracy verification. As documented in the
parent README, bare pruning is lossy: on gsm8k-500, DeepSeek-V4-Flash drops
from **92.60 to 39.58** after 256e → 64e pruning. The training-side adaptation
is still a TODO there.

This recipe is our answer to that gap: **recover the pruned model's behavior
with a short SFT pass on teacher (full model) outputs**, then measure how much
behavior came back. It is the "training-side" half of small-for-large:
pruning shrinks the model, SFT teaches it to behave like the original again.

## Core idea

1. Serve the **full model** once, generate chain-of-thought answers for the
   target training set (the teacher pass; the same server also produces the
   activation statistics for pruning, so one sweep serves both).
2. Score the teacher outputs, keep only samples that finish cleanly and fit
   the training length budget.
3. Build a small SFT dataset (a few hundred samples) from the survivors,
   re-joining multimodal inputs (images) from the source dataset.
4. SFT the **pruned model** on this dataset — visual towers frozen for VL
   models.
5. Evaluate checkpoints on a fixed, seeded subset of the training set
   ("absorption": how much of the teacher's behavior did the small model
   absorb) and on a held-out split ("generalization").
6. Optionally continue into RL (GRPO) from the recovered checkpoint —
   the RL-parity stage.

## Validated result (pilot)

Pilot model: **Step-5-Preview** (352-expert multimodal MoE, 92 text layers +
47 vision layers), pruned to top-16 experts (1.21 TB → 75 GB, single 8-NPU
node trainable), 200-sample SFT set, 8×Ascend 950DT.

| checkpoints | base | 100 steps | 200 steps | 300 steps | 400 steps |
|---|---|---|---|---|---|
| absorption accuracy (n=100) | 0% | 35% | 55% | 72% | 74% |
| truncated outputs (per 100) | 100 | 7 | 1 | 0 | 0 |

Readings:

- The pruned model recovers most of the teacher's behavior within a few
  hundred SFT steps and saturates (400 steps ≈ 74%); the curve, not a single
  number, is the evidence that "SFT recovers behavior" works.
- The base pruned model scores 0% not for lack of capability but because its
  output format is untamed (runs to the token limit on every prompt);
  truncations go 100 → 0 as behavior alignment lands.
- RL (GRPO) started from recovered checkpoints keeps improving reward /
  format-pass on a fixed 100-problem validation set — the small model remains
  trainable, which is the point of small-for-large.

## Pipeline layout (planned)

```
sft_recovery/
├── README.md                     # this file
├── REQUIRED_VERL.txt             # verl pin (RL-parity stage)
├── data_build/
│   ├── build_sft_dataset.py      # teacher pool -> filtered, seeded, render-verified jsonl  [in this skeleton]
│   ├── collect_teacher_outputs.py  # resume-capable teacher generation client               [TODO]
│   └── score_filter_eval_samples.py # score + blacklist teacher outputs                     [TODO]
├── eval/
│   ├── materialize_eval_split.py   # same-schema parquet + seed42 fixed index + GT rejoin   [TODO]
│   ├── eval_client.py              # fixed-decoding eval client, atomic writes, resume      [TODO]
│   └── score_eval.py               # math-equivalence scoring (pluggable backend)           [TODO]
└── docs/
    └── absorption_curve.md         # methodology: why the learning curve is the result       [TODO]
```

## Data build details (the part this skeleton ships)

`data_build/build_sft_dataset.py` is the core of the method. Given a directory
of per-sample teacher outputs (JSON: question / content / metrics /
blacklisted) and the source dataset (parquet with embedded images), it:

1. filters: not blacklisted, `finish_reason == stop`, total teacher tokens
   under a budget;
2. shuffles with a **fixed seed** and samples N;
3. re-joins images from the source parquet by sample index;
4. (optional, strongly recommended) renders every candidate through the
   **same processor + chat template the trainer will use** and rejects any
   sample whose rendered length exceeds the cap — this kills the "fine in the
   jsonl, exploding at train time" class of bugs;
5. writes `sft<N>.jsonl` (messages format + `source_id` for ground-truth
   rejoin later) and `selection_stats.json` with the full audit trail of the
   filter chain (pool size, per-reason skips, rejected-during-verify, final
   length percentiles).

The render verification is pluggable (`--render-module`), so the tool stays
free of trainer-specific imports while still supporting exact-pipeline
verification.

## Roadmap

- [ ] ship `collect_teacher_outputs.py` + `score_filter_eval_samples.py`
- [ ] ship the eval chain (materialize + client + scorer)
- [ ] vision-freeze zero-drift audit tool (frozen-module weight comparison
      between base and every checkpoint)
- [ ] RL-parity stage on verl GRPO: reward adapter with prefilled-think
      handling, full-model vs pruned-model same-config comparison (an ongoing
      MiniMax-M3.1-Flash top16e single-node pilot feeds this stage)
