# Fully async dummy-load patch (A3, verl v0.9.0)

## Merge prerequisite

**Merge the DeepSeek-V4-Flash A3 / verl v0.9.0 upgrade before this patch.**
The prerequisite baseline is [psyloy/verl-ascend-recipe@8bf91a4](https://github.com/psyloy/verl-ascend-recipe/commit/8bf91a4c53ef77970121a6326e7ffaf6aef95462)
(`update verl v0.9.0`), including its `DeepSeek-V4-Flash/patch/A3/verl/01-*`
through `10-*` patches. That A3 upgrade is not included in this change.

Application was checked against
[verl@91c297d](https://github.com/verl-project/verl/commit/91c297dc8d9991b79c90cad5b50683a5d6d8c02b)
from `release/v0.9.0`, after applying those ten patches in order. Recheck
applicability if the prerequisite upgrade changes during review.

## Behavior

`12-fully-async-dummy-load.patch` allows the fully async runner to retain
`actor_rollout_ref.rollout.load_format=dummy` for standalone vLLM rollout.
The runner already awaits an actor-to-rollout weight update before validation
and training generation. The patch records that initial-sync contract and
rejects generation until the rollout server receives a weight version;
version `0` is valid.

The contract is enabled internally by the fully async runner. Keep
`initial_weight_sync` at its default (`false`) in user configuration.
Other standalone uses retain the existing `dummy` to `auto` fallback.
The patch rejects naive checkpoint transfer, LoRA, and enabled rollout MTP
for this dummy-loading path. It covers full-model weight synchronization.

## Application

After installing the prerequisite A3 v0.9.0 baseline and applying patches
01 through 10, run the following from the verl checkout. Replace
`/path/to/verl-ascend-recipe` with the recipe checkout path:

```bash
git apply --check /path/to/verl-ascend-recipe/DeepSeek-V4-Flash/patch/A3/verl/12-fully-async-dummy-load.patch
git apply /path/to/verl-ascend-recipe/DeepSeek-V4-Flash/patch/A3/verl/12-fully-async-dummy-load.patch
```

The existing A3 installer in the prerequisite baseline applies numbered
patches in this directory. Use the manual commands only if patch 12 has
not already been applied by that installer.

## Validation

- Applied the ten prerequisite patches and this patch to the pinned verl sources.
- Parsed all three changed Python modules successfully.
- Checked that this patch can be reversed cleanly.
- No NPU end-to-end run has been performed; dummy initialization, transferred
  weights, and first-generation correctness still require hardware validation.

The patch changes three Python files and one YAML configuration file.
No shell scripts or test files are changed or added. The patch and this
explanation were prepared with AI assistance.
