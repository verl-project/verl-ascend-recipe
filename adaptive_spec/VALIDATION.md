# Validation

## PR follow-up validation (2026-10-19)

This follow-up fixes four issues found in public revision `a0fe8a4`:

- Use the pinned verl model MTP configuration, including both `enable` and `enable_rollout`.
- Terminate the probe's process group on timeout or nonzero exit, including workers
  that survive the leader. Preserve the original failure when the group already exited.
- Make the TP leader the sole controller/evidence owner and broadcast each final K.
- Support the audited public Ascend source revision and provide a source-only preflight.

The policy, score, fallback and stabilization defaults are unchanged.

### CPU and packaging

Windows Python 3.12.14 and upstream-pinned Ruff 0.14.10:

- `PYTHONPATH=adaptive_spec python -m pytest tests/ adaptive_spec/tests/ -q --tb=short`:
  **408 passed** (331 upstream + 77 recipe), no skips.
- Repository-wide Ruff lint and format checks passed (193 Python files).
- Python 3.10 syntax parsing passed; this is not Python 3.10 runtime execution.
- Wheel/sdist build, fresh isolated wheel installation, disabled-plugin import and both
  plugin entry-point loads passed without loading torch, Ray or the inference backend.

The regressions cover evidence arriving between TP rank calls under the default
64-decision polling/two-confirmation settings, leader-only publication, propagated
leader errors, ordered decision packets, probe cleanup and modified-source rejection.

A separate Linux/Python 3.11 test used actual two-process PyTorch Gloo communication
and the pinned vLLM `GroupCoordinator.broadcast_object` method. All 129 decisions
agreed across ranks, including K5→K9 after shared evidence polling. An injected leader
policy error reached both ranks. This is a communication regression, not model generation.

### Native TP2 generation on public backend source

The follow-up wheel was tested with Qwen3.5-9B, Python 3.11.10, PyTorch 2.10.0,
torch-npu 2.10.0.post2, vLLM 0.23.0 and vLLM-Ascend
0.23.0rc2.dev1+g5ed283a6c.d20260719. The public Ascend runner at
`5ed283a6c0ff5508f060d6203bc7a34e9d38c7bb` was mounted read-only over the installed
runner in an isolated validation container; existing compiled extensions were retained.
The public runner and core vLLM source passed preflight. This did not rebuild or
certify a complete backend wheel from scratch.

Two new requests each completed 128 tokens with eager asynchronous serial MTP,
TP2, B2 and startup capacity K7. Both workers agreed on all **47** decisions;
physical depth changed **K5→K7→K5** and returned draft shapes passed the adapter's
width checks. Only rank zero owned a controller. Test-only worker readback used a
named extension method; no arbitrary-callable RPC permission was enabled.

Synthetic phase fallbacks and shorter dwell were used to force both transitions.
This tests transport and control correctness; it is not a calibrated performance
comparison or output-quality equivalence test. Test instrumentation is outside the
recipe. Logs, source manifests and restoration receipts are retained separately.

### Remaining integration coverage

Full supported-stack verl/Ray training, optional Elastic device integration, output
quality and matched package throughput remain pending. The native vLLM test above
does not establish training convergence or an Elastic × dynamic-K interaction gain.
Historical results below retain their original measurement boundary.

## Historical provenance (2026-10-09)

The following checks were recorded for the private delivery at commit
`6ef655072dedb79d26bcb4782ad29eebcbe96518`, extracted from community base
`1028677a58172f26714640d2ce994eac4f20744a`. They are not new device
measurements of this public PR. Raw receipts remain outside the package.

## CPU and packaging validation

Local checks used Python 3.12.14 and Ruff 0.14.10:

- 331 upstream tests and 67 recipe tests passed (398 total).
- Six additional, out-of-repository historical-oracle tests passed: 32,000 stabilization
  steps and 7,680 posterior/scoring queries. The aggregate is 404 tests. These compare
  normal-lifecycle components, not an entire historical runner replay.
- Ruff lint/format checks and Python 3.10 syntax parsing passed.
- Wheel and sdist builds passed. A separate wheel installation exercised the documented
  context-builder, plan-only and replay commands, then loaded the generated map into the
  actual controller and made an Engine decision.
- Replay records are CPU fixtures, explicitly marked as non-device measurements; their
  emitted runtime is disabled. They are not performance evidence.

Public verl contracts were inspected at the REQUIRED_VERL pin. Acquire/release methods
were exercised with a CPU protocol simulator, not a real Ray actor. The private CI runs
all 67 recipe tests on Python 3.10 and 3.12; check the workflow for the commit being reviewed.

## Bounded device validation

Offline collection completed on a source-verified Ascend 0.23 environment with Qwen3.5-9B,
TP2, K5/K7, two states (B1/L256 and B2/L512), one repeat and 64-token continuations.
Both depths completed all states (six responses total). Every state/depth retained more
than the required two clean decode steps. The collector generated a directly loadable
phase map and matching enabled runtime configuration. Plan-only was also exercised on
that environment. The backend was vLLM 0.23.0 and vLLM-Ascend
0.23.0rc2.dev1+g5ed283a6c.d20260719, with exact supported runner fingerprints.

The generated map was then used for a bounded native vLLM online-plugin test: two sibling
responses, 32 tokens each, eager async scheduling, TP2 and startup MTP capacity K7.
This is a short integration check, not a complete verl/Ray rollout, quality evaluation,
representative calibration grid or throughput comparison. Read-only worker RPC on the final wheel confirmed the adapter was installed on
both TP workers, physical/drafted K was 5, each controller made 12 decisions and
recorded 18 evidence blocks. The trusted callable RPC permission was limited to
this isolated validation process; production configuration does not enable it.
Receipts and raw logs are retained outside the distributed package.

Collection used the calibration-r3 wheel. The final wheel differs in the collector only
by streaming file hashing/JSONL reading; the observer and device probe code are unchanged.
Final CLI/artifact tests exercise those I/O changes. Failed environment/script setup
attempts and their logs remain in private local/server validation directories; they are
not counted as passed device experiments and are not shipped.

## Evidence limits

Full supported-stack verl/Ray execution, optional Elastic device integration and matched
package throughput measurements remain pending. Existing CPU router regressions do not
establish end-to-end device performance. Historical gains came from a customized
rollout-only experiment stack without actor weight updates; some historical runs used
K11, which this package excludes. Do not attribute those gains to the extracted recipe.
