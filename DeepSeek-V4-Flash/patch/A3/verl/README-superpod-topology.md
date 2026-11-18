# Ascend SuperPod topology scheduling

**Merge prerequisite: the DeepSeek-V4 A3 recipe must first adopt the VERL
v0.9.0 upgrade represented by
[psyloy/verl-ascend-recipe@8bf91a4](https://github.com/psyloy/verl-ascend-recipe/commit/8bf91a4c53ef77970121a6326e7ffaf6aef95462).**
This patch is not compatible with the older `809f2d8f` recipe baseline.

The validated combination is:

1. VERL `91c297dc8d9991b79c90cad5b50683a5d6d8c02b` from `release/v0.9.0`.
2. The prerequisite recipe's `patch/A3/verl/01-*.patch` through `10-*.patch`,
   applied in filename order.
3. `11-ascend-superpod-topology.patch` from this directory.

The release branch can advance; the SHA above identifies the tested source.
The prerequisite's `install_A3.sh` discovers the numbered patches automatically.
This change does not update the older installation script or include the
prerequisite upgrade itself. Keep the PR in draft until that upgrade is merged
and the combined patch sequence has been rechecked.

## Purpose and behavior

Ray placement alone does not guarantee that every node of a multi-node rollout
replica belongs to the same physical SuperPod. This opt-in patch queries physical
SP IDs, constructs the complete resource demand of one training job, and reserves
node-pinned placement groups before model workers start.

The planner first satisfies all resource, replica-boundary, and explicit quota
constraints. In automatic mode it then minimizes the number of selected SPs and,
for equal SP counts, prefers suitable smaller free blocks to preserve larger
blocks. Feasibility and optimization share a time budget. If optimization times
out after finding a feasible solution, the planner uses that solution; it does
not relax constraints or claim a globally optimal placement.

The patch includes:

- V0, V1 `sync`, `colocate_async`, `separate_async`, and fully async entry points.
- Joint planning for training, hybrid/standalone rollout, reward, and teacher
  pools, with consistent replica numbering and nested pool offsets.
- Whole-replica SP constraints based on rollout `TP * DP * PP`.
- Isolated DCMI calls with deadlines and cleanup after timeout or caller death.
- Job-scoped placement-group reservation, repeatable pool claims, and rollback
  before retry when reservation fails.
- A targeted filter for `invalid escape sequence` SyntaxWarnings from
  `torch_npu/op_plugin_docs.py`; other warnings and DCMI errors remain visible.

It is a complete topology patch. Do not stack older topology or sorting patches
before it. The underlying recipe's ten model/backend patches are still required.

## Configuration

Keep the existing training mode, node counts, and model parallelism settings.
With `trainer.device=npu`, automatic placement requires just:

```bash
trainer.ascend_topology.enabled=true
```

To require exact physical SP quotas, pass the entire override as one shell
argument containing a list of mappings:

```bash
"trainer.ascend_topology.sp_plan=[{sp:2, nodes:8}, {sp:6, nodes:8}]"
```

The quotas count independent physical Ray nodes across the entire job, including
standalone rollout, reward, and teacher pools. Hybrid replicas reuse training
nodes and are not counted twice. Different independent pools do not share a
physical node, even when a small standalone replica uses only part of its NPUs.
For example, sixteen independent 8-NPU replicas require sixteen physical nodes,
even if each node has sixteen NPUs.

`sp_plan=[]` selects automatic mode. `sp_layout` is removed; stale use is rejected.
There are no required custom SP resource labels or extra environment variables.

All options below are under `trainer.ascend_topology`:

| Option | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Enable topology planning; otherwise retain normal scheduling |
| `sp_plan` | `[]` | Exact SP/node quotas, or automatic selection |
| `planning_timeout_s` | `30` | Shared feasibility/optimization budget |
| `allocation_retries` | `1` | Retry after failed reservation is fully rolled back |
| `dcmi_library` | `null` | Optional absolute DCMI library path |
| `probe_timeout_s` | `60` | Isolated DCMI query deadline |
| `placement_timeout_s` | `1800` | Placement-group readiness deadline |

V1 separate-async demand uses `actor_rollout_ref.rollout.nnodes/n_gpus_per_node`;
fully async uses the existing top-level `rollout.nnodes/n_gpus_per_node` settings.
For fully async, either training-side validation or dynamic resource scheduling
adds hybrid replica constraints on the training pool. Zero standalone rollout
nodes require fully async dynamic scheduling. Separate-async and fully async
reward models require their existing dedicated resource-pool setting.

## Applying and reverting manually

First prepare the validated VERL source and the ten prerequisite patches. Use
absolute paths, install the same version on every participating Ray node, and
start a new training job after installation:

```bash
RECIPE_ROOT=/absolute/path/to/verl-ascend-recipe
VERL_ROOT=/absolute/path/to/verl
TOPOLOGY_PATCH="$RECIPE_ROOT/DeepSeek-V4-Flash/patch/A3/verl/11-ascend-superpod-topology.patch"

git -C "$VERL_ROOT" apply --check "$TOPOLOGY_PATCH"
git -C "$VERL_ROOT" apply "$TOPOLOGY_PATCH"

# Revert only the topology patch after ending jobs that use it.
git -C "$VERL_ROOT" apply --reverse --check "$TOPOLOGY_PATCH"
git -C "$VERL_ROOT" apply --reverse "$TOPOLOGY_PATCH"
```

## Validation

Tests are in `DeepSeek-V4-Flash/tests/superpod_topology`. On Linux with GCC:

```bash
uv venv --python 3.12 /tmp/verl-superpod-tests
uv pip install --python /tmp/verl-superpod-tests/bin/python \
  'ray[default]==2.48.0' hydra-core pytest numpy

TEST_DIR="$RECIPE_ROOT/DeepSeek-V4-Flash/tests/superpod_topology"
/tmp/verl-superpod-tests/bin/python -m pytest "$TEST_DIR" \
  --source "$VERL_ROOT" -q -c /dev/null --rootdir "$TEST_DIR"
/tmp/verl-superpod-tests/bin/python "$TEST_DIR/ray_smoke.py" --source "$VERL_ROOT"
```

`VERL_ROOT` must contain the patched code. The CPU suite checks planning against
small exhaustive oracles, Hydra configuration, entry-point and pool method
bodies, rollback, warning filtering, and native-process cleanup. It substitutes
heavy model dependencies and uses a compiled fake DCMI library. The separate Ray
smoke test registers simulated NPU resources and checks actual placement groups,
bundle placement, cancellation, and cleanup; it does not load real NPU backends.

Validation on 2026-10-28: all ten prerequisite patches applied; the ten existing
files touched by this patch matched the validated baseline; all four new modules
were absent as expected; topology patch application/reversal restored source
bytes exactly; **70 CPU tests passed**. Local Ray 2.48 smoke validation also
passed during patch development. No real Ascend training, HCCL communication,
model loading, or weight-transfer run has been verified by these tests.

## Scope

- One allocator coordinates one Ray job. It does not provide cluster-wide
  fairness, cross-job arbitration, or Kubernetes scheduling.
- Reservations are fixed after worker startup. Dynamic scheduling activates
  existing hybrid replicas; it does not migrate physical nodes.
- Placement groups have job lifetime. End the training job to release resources
  after an allocator failure; killing only the allocator actor is insufficient.
- vLLM/SGLang backend limitations still apply. NPU prefill/decode disaggregation
  is rejected before reservation. This patch adds no training EP-domain rule.
- The patch does not include unrelated changes to model execution, TransferQueue,
  checkpoint engines, or standalone dummy-weight initialization.
