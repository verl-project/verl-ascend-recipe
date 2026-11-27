# KV-cache release during weight synchronization (A3, verl v0.9.0)

## Merge prerequisite

**Merge the DeepSeek-V4-Flash A3 / verl v0.9.0 upgrade before this patch.**
The prerequisite baseline is
[psyloy/verl-ascend-recipe@8bf91a4](https://github.com/psyloy/verl-ascend-recipe/commit/8bf91a4c53ef77970121a6326e7ffaf6aef95462),
including its A3 verl patches `01-*` through `10-*`. This change does not
include that upgrade.

Application was checked against
[verl@91c297d](https://github.com/verl-project/verl/commit/91c297dc8d9991b79c90cad5b50683a5d6d8c02b)
from `release/v0.9.0`, after applying those ten patches in order. It also
applies after the optional dummy-load patch from
[PR #159](https://github.com/verl-project/verl-ascend-recipe/pull/159);
that PR is not a prerequisite. Recheck applicability if the baseline changes.

## Behavior

In this baseline, the vLLM server's `release_kv_cache()` and
`resume_kv_cache()` hooks are placeholders. KV memory therefore remains
resident while the non-naive checkpoint engine allocates transfer buffers.

`13-release-kv-cache-before-weight-sync.patch` backports the sleep/selective-wake
implementation from
[verl v0.9.1](https://github.com/verl-project/verl/blob/1876b06d0a3e4e71e06230be10af14492ca8a75b/verl/workers/rollout/vllm_rollout/vllm_async_server.py).
It changes only three methods in `vllm_async_server.py`:

- `release_kv_cache()` sleeps the engine, then wakes only `weights`, before
  checkpoint buffer preparation and weight transfer.
- `resume_kv_cache()` wakes `kv_cache` after checkpoint finalization, then
  resets prefix caching with `reset_connector=True` before generation resumes.
- `_resolve_sleep_level()` selects level 1 for NPU, LoRA adapters, or enabled
  rollout MTP; otherwise it selects level 2.

The release and restore hooks both skip headless servers, disabled
`free_cache_engine`, and `COLOCATED` rollout. The existing manager/replica
call order and `_sleep_hybrid()` implementation remain unchanged, including
the latter's `VLLM_SLEEP_LEVEL` handling from A3 patch 02.

Use `actor_rollout_ref.rollout.free_cache_engine=true` and
`actor_rollout_ref.rollout.enable_sleep_mode=true` with a compatible vLLM/Ascend
sleep allocator. On NPU, level 1 temporarily backs up weights to CPU, releases
the sleep-managed pools, and restores weights before transfer. KV memory stays
released for the transfer window; the weight backup/restore adds overhead.
This is the upstream sleep/wake workaround, not a new KV-only allocator API.

Weight synchronization must remain serialized with request submission.
Some vLLM versions resume scheduling on `wake_up(tags=["weights"])`, while KV
is still released. This patch does not add request-admission or drain handling;
verify the deployment's existing pause/retry behavior before enabling it.

## Application

After installing the prerequisite baseline and applying patches 01 through 10,
run from the verl checkout, replacing the recipe checkout path below:

```bash
git apply --check /path/to/verl-ascend-recipe/DeepSeek-V4-Flash/patch/A3/verl/13-release-kv-cache-before-weight-sync.patch
git apply /path/to/verl-ascend-recipe/DeepSeek-V4-Flash/patch/A3/verl/13-release-kv-cache-before-weight-sync.patch
```

The prerequisite A3 installer applies numbered patches in this directory.
Use the manual commands only if this patch has not already been applied.
Restart the relevant workers after updating their installed source.

To revert, stop the job and use `git apply -R --check` followed by
`git apply -R` with the same patch path, then restart the workers.

## Validation

- Applied the ten prerequisite patches, checked/applied this patch, and checked
  its reversal. Also checked compatibility after PR #159's patch 12.
- Parsed the modified Python module and verified that only the three stated
  methods change.
- Executed the changed methods with mocked engine calls to check sleep-level
  selection, release/restore ordering, skipped branches, and error propagation.
- NPU end-to-end validation remains pending: check physical HBM headroom before
  transfer-buffer allocation, pause/retry behavior, synchronization overhead,
  and generation correctness after KV restoration. Allocator counters alone
  need not reflect physical memory unmapped by the sleep allocator.

The patch and these notes were prepared with AI assistance. Keep the PR in
draft until the prerequisite is available upstream and hardware validation is
complete.
