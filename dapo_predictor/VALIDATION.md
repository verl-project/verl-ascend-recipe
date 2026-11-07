# Verification

## Checked configuration

CPU verification uses Python 3.11.14, PyTorch 2.9.0+cpu and the verl source pinned by [REQUIRED_VERL.txt](REQUIRED_VERL.txt). Ruff uses the repository version, 0.14.10.

| Check | Result |
| --- | --- |
| Repository and predictor tests with pinned verl sources | 465 passed; no skipped tests; 6 dependency/device-discovery warnings in the CPU-only container |
| Launcher Hydra composition | D1, reduced-gate D1, standard-DAPO baseline and D2 candidate-report modes compose |
| Python 3.10 syntax | Checked for recipe Python files |
| Ruff check / format | Checked with 0.14.10 |
| Shell syntax | run_ews.sh checked with bash -n |
| Online D2 device training | 20 epochs, 40 actor updates, 5120 responses; first predictive admission at epoch 11; exit 0 |
| Matched native FCFS device training | 20 epochs, 40 actor updates, 5120 responses; exit 0 |
| Independent-install AI-style smoke | 12 updates, 192 responses; exit 0 |
| Predictor checkpoint resume | Steps 13–16 after the 12-step smoke; 64 new responses, restored head and cumulative counters |
| Cleanup and telemetry repair | 134 predictor CPU tests passed; separate 12-update device smoke exited 0 with no resource_tracker, KeyError or traceback |

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

To inspect configuration without a device job, use the [launcher dry run](README.md#run-d2--ews). Combine its printed overrides with Hydra's dapo_trainer configuration, then check that the manager is unchanged when disabled and registered when enabled.

## Device acceptance workload

The completed reader-style runs used Qwen3-1.7B and the image documented in the README, with Python 3.11.14, PyTorch/torch-npu 2.9.0, vLLM 0.19.1, vLLM-Ascend 0.19.1rc1 and Transformers 5.5.3. Both full runs used one fixed 32-prompt DAPO-Math population, batches of 16, 8 responses per prompt, an 8192 response-token cap and 20 epochs. Results and timing boundaries are in [PERFORMANCE.md](PERFORMANCE.md).

The two installation paths were simulated by the same agent in separately installed environments, sharing public read-only downloads. The reader-style path ran the full comparison; the AI-style path ran the agreed short smoke. Capped 256-token smoke responses gave zero GRPO gradients, so the smoke establishes execution and state transitions rather than learning quality.

Device testing exposed cold-start feature loss, a TensorDict/DataProto worker mismatch and NumPy reward metadata that could not be written as JSON. These were fixed before the completed full runs. The D2 full-run exit also logged shared-memory resource-tracker errors despite returning zero. Explicit server shutdown was subsequently tested in a separate 12-step run, which exited normally, cleared its NPU processes and produced none of those errors. The completed training measurements remain unchanged.

Run these checks on devices assigned to the job, from a fresh isolated environment following the README. The full reader-style acceptance uses one fixed question population for 20 RL epochs and a matched FCFS baseline. An independent AI-style environment performs a short deployment smoke, not a second full experiment.

A small integration workload uses the same frozen 32 DAPO-Math prompts, 2 responses per prompt, batches of 8, a 1024 prompt-token cap, a 256 response-token cap and 3 RL passes. Validation is disabled; the loader's validation file points to this same population and is not an independent evaluation set. Freeze the selected prompt IDs, model revision and sampling seed. This checks wiring and state transitions; it is not the 64K performance workload in [PERFORMANCE.md](PERFORMANCE.md).

1. Install the documented runtime, pinned verl and both recipe overlays, download the model/data, and record their identities.
2. Run the matched baseline with `PREDICTOR_ENABLE=false`; verify the upstream manager/worker path.
3. Run D1 with `MIN_EPOCH=0 MIN_SAMPLES=0`; verify FCFS while calibration is absent, then prediction, EWS admission, rewards and actor updates. Keep outputs in their original training-row order.
4. Save a checkpoint after an intermediate update, exit normally, resume with the same configuration and check predictor weights/calibration/counters. Do not count outputs twice across resume.
5. For default D2, start without external weights. Check pre-response feature collection, asynchronous CPU fitting, no overlapping fitting jobs, FCFS during warm-up, epoch/sample activation, all-rank publication, incremental updates, pre-response prediction quality and resume. External frozen-D2 compatibility uses a separate explicitly selected checkpoint/report.
6. Record generated tokens, generation wall time, predictor forward/update time, full training-step time, in-flight limits, cap hits and exit status. Stop the job normally and check that its distributed processes have exited.

For representative performance, follow the complete workload and repeated-order comparison in PERFORMANCE.md rather than deriving a speedup from this small check. Final acceptance requires the actual public training path, not only configuration composition or CPU callbacks.

GitHub Actions and local results are separate checks. Inspect runs for the exact pushed head SHA; an approval-required run has not executed the tests.
