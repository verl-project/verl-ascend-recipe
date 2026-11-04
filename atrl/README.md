# ATRL recipe

ATRL adds three things on top of an existing uni-agent checkout:

- a five-tool ReAct loop (`triton_react`)
- structured context compaction
- mixed segment rewards, plus verl's `grpo_segment` advantage

This directory does not vendor uni-agent or verl. Clone the versions below, then apply the patches.

## Versions recorded in the ATRL tree

The local ATRL tree does not store a uni-agent git SHA. It does store a verl version file:

- `verl/verl/version/version` = `0.9.0.dev`
- the snapshot commit is named `uniagent_0724` and is dated 2026-07-24

That version string matches upstream verl on the same morning:

- verl `2b47a68b66d6` (2026-07-24, `0.9.0.dev`)

Upstream uni-agent that same morning is `38c655f360f6`. Its tree already uses `uni_agent/gateway/`. The ATRL tree still uses `uni_agent/trainer/gateway/`, so it is not that checkout and a patch against `38c655f` would not describe the increment.

The increment below is therefore taken against the uni-agent release that has the current framework, which is also the version previously identified as the base:

| Component | Checkout | Notes |
| --- | --- | --- |
| uni-agent | `release/v0.1.0` at `5c5663c30a5e5d847707f49e7595471cd1e6315a` | patch applies cleanly here |
| verl | `2bcc98a9b64bdb3af5b88ac870ba0709ed709aef` (`main` when the segment-advantage patch was cut) | patch applies cleanly here |

`2b47a68` and `2bcc98a9` are not the same tree. Use the patch bases in the table. The July 24 pin is only the version recorded inside the ATRL snapshot.

Tag `v0.1.0` (`f353c58f744b`, 2026-09-22) vendors verl `v0.9.1` (`1876b06d0a3e`). The uni-agent patch was checked against `release/v0.1.0` (`5c5663c`), not against that tag.

## Apply

```bash
git clone https://github.com/verl-project/uni-agent.git
cd uni-agent
git checkout 5c5663c30a5e5d847707f49e7595471cd1e6315a
git apply /path/to/verl-ascend-recipe/atrl/patches/uni-agent-v0.1.0-atrl.patch

git clone https://github.com/verl-project/verl.git
cd verl
git checkout 2bcc98a9b64bdb3af5b88ac870ba0709ed709aef
git apply /path/to/verl-ascend-recipe/atrl/patches/verl-main-grpo-segment.patch
```

Training then uses `agent.name: triton_react`, `algorithm.adv_estimator=grpo_segment`, and `actor.loss_agg_mode=seg-mean-token-mean`. See `examples/triton_react/task_config.yaml` inside the patched uni-agent tree.

`TRITON_COMPACTION_REWARD_ALPHA` defaults to `0.5`. `TRITON_COMPACTION_MIN_REMAINING_TURNS` defaults to `3`.
