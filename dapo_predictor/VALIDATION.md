# Verification

## Checked configuration

CPU verification uses Python 3.11.14, PyTorch 2.9.0+cpu and the verl source pinned by [REQUIRED_VERL.txt](REQUIRED_VERL.txt). Ruff uses the repository version, 0.14.10.

| Check | Result |
| --- | --- |
| Repository and predictor tests with pinned verl sources | 431 passed; no skipped tests |
| Launcher Hydra composition | D1, reduced-gate D1, standard-DAPO baseline and D2 candidate-report modes compose |
| Python 3.10 syntax | Checked for recipe Python files |
| Ruff check / format | Checked with 0.14.10 |
| Shell syntax | run_ews.sh checked with bash -n |
| Device end-to-end run of the revised path | Pending an allocated NPU window |

The CPU suite covers historical-only calibration, whole-batch FCFS fallback, bounded refill/cancellation, restored output order, D1 bf16 features with an fp32 head, single-prompt updates, response-mask length labels, raw-token hydration, opt-in manager configuration, D2 candidate validation and checkpoint/resume. The manager integration uses the pinned DataProto and manager class with test workers; it does not start a Ray/NPU training cluster.

## Run the checks

From the recipe repository, with CPU PyTorch available:

```bash
python -m pip install pytest numpy hydra-core codetiming tensordict ruff==0.14.10
python -m ruff check dapo_predictor
python -m ruff format --check dapo_predictor
python -m pytest tests/ dapo_predictor/tests/ -q -rs
bash -n dapo_predictor/run_ews.sh
git diff --check
```

The CI predictor job installs CPU PyTorch 2.8.0 and runs the device-free cases. The pinned-verl manager integration also requires the core source checkout and its dependencies. Skipped dependency-gated tests are listed by pytest rather than counted as passing.

To inspect configuration without a device job, use the [launcher dry run](README.md#run-d1--ews). Combine its printed overrides with Hydra's dapo_trainer configuration, then check that the manager is unchanged when disabled and registered when enabled.

## Device acceptance workload

Run these checks on devices assigned to the job, from a fresh isolated environment following the README. Use another independent environment for the step-by-step reader check; do not count one installation as both paths.

A small integration workload is 16 DAPO-Math training prompts, 8 validation prompts, 2 responses per prompt, batches of 8, 512 prompt tokens, 1024 response tokens and 3 RL passes. Freeze the selected prompt IDs, model revision and sampling seed. This checks wiring and state transitions; it is not the 64K performance workload in [PERFORMANCE.md](PERFORMANCE.md).

1. Install the documented runtime, pinned verl and both recipe overlays, download the model/data, and record their identities.
2. Run the matched baseline with `PREDICTOR_ENABLE=false`; verify the upstream manager/worker path.
3. Run D1 with `MIN_EPOCH=0 MIN_SAMPLES=0`; verify FCFS while calibration is absent, then prediction, EWS admission, rewards and actor updates. Keep outputs in their original training-row order.
4. Save a checkpoint after an intermediate update, exit normally, resume with the same configuration and check predictor weights/calibration/counters. Do not count outputs twice across resume.
5. For D2, use an externally fitted compatible checkpoint and held-out report. Check candidate admission, frozen inference, a monitoring boundary, fallback and hash-bound resume. To shorten an integration check only, set the documented interval/gates explicitly and record them.
6. Record generated tokens, generation wall time, predictor forward/update time, full training-step time, in-flight limits, cap hits and exit status. Stop the job normally and check that its distributed processes have exited.

For representative performance, follow the complete workload and repeated-order comparison in PERFORMANCE.md rather than deriving a speedup from this small check. Final acceptance requires the actual public training path, not only configuration composition or CPU callbacks.

GitHub Actions and local results are separate checks. Inspect runs for the exact pushed head SHA; an approval-required run has not executed the tests.
