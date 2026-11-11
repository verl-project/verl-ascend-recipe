# Functional validation

This page records functional checks for online D2 + EWS.

## 1. Tests and code checks

| Check | Result |
| --- | --- |
| Repository and predictor CPU tests | 468 passed, 6 warnings, no skipped tests |
| Ruff code checks | Passed with repository-pinned Ruff 0.14.10 |
| Recipe formatting | 38 Python files pass Ruff format checks |
| EWS naming compatibility | New and legacy configuration/import paths tested; conflicting capacity settings rejected |
| Manager integration | Long-first admission, FCFS fallback, bounded dispatch and original output order verified |

CPU tests use Python 3.11.14, PyTorch 2.9.0+cpu and the verl source pinned by [REQUIRED_VERL.txt](REQUIRED_VERL.txt). Warnings concern dependency deprecations and device discovery in the CPU-only environment.

Both checks configured by pre-commit (Ruff checking and formatting) were run directly. This is separate from running the pre-commit wrapper or GitHub Actions; inspect the workflows for the current PR head to determine their status.

To run the checks from this recipe repository in an environment with CPU PyTorch, the pinned verl source and its dependencies:

```bash
python -m pip install pytest numpy hydra-core codetiming tensordict ruff==0.14.10
python -m ruff check dapo_predictor
python -m ruff format --check dapo_predictor
python -m pytest tests/ dapo_predictor/tests/ -q -rs
bash -n dapo_predictor/run_ews.sh
git diff --check
```

## 2. Device workload

The full functional run used Qwen3-1.7B and one fixed set of 32 DAPO-Math questions for 20 RL epochs: 40 training updates and 5120 generated responses.

The environment used the runtime image listed in the [README](README.md), Python 3.11.14, PyTorch/torch-npu 2.9.0, vLLM 0.19.1, vLLM-Ascend 0.19.1rc1 and Transformers 5.5.3.

The full run checks cold start, background fitting and activation. Separate shorter runs check checkpoint recovery, independent installation and shutdown.

## 3. Verified behavior

| Area | Result |
| --- | --- |
| Cold start | FCFS admission for the first 10 epochs; 2560 responses collected |
| Background fitting | Heads trained and published during cold start; no recorded fitting failures in the completed full run |
| Activation | First predictive admission in epoch 11, at training update 21 |
| Continued training | Observations and predictor updates continued through all 20 epochs |
| Checkpoint saving | Actor, optimizer, data progress and predictor state saved |
| Resume | A separate run restored training/predictor state and continued with additional updates |
| Shutdown | After the cleanup fix, a separate device smoke exited normally without the related shared-memory KeyError, resource_tracker warnings or Traceback |
| Independent installation | Installation and short smoke completed in a separately installed software environment |

The full run and shorter checks cover different scopes: the full run verifies the training lifecycle; separate checks verify installation, recovery and the corrected shutdown path. The two installation paths were exercised by the same agent, not independent human reviewers.

The subsequent EWS naming migration was checked by the CPU and manager integration tests above; no additional device experiment was run for that naming-only change.
