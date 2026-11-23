# Saved verification evidence

This directory contains historical results, not a completed acceptance of the
reorganized code. No new server run was started to publish these artifacts.

## Full historical run

`full-opt20-30steps-001/` contains the full 40-layer Xing4.0-29B-A4B acceptance:

- 30 actual optimizer updates and 31 resident weight versions.
- 155 raw comparison groups and 15,872 FP32 elements; all finite, zero bit mismatches.
- `raw/`: all 62 original PyTorch files, two per version, totaling 208,419 bytes.
- `continuous-audit.json`: every comparison, tensor hash, and optimizer/IPC audit receipt.
- `steps.csv`: all 31 versions in a readable table.
- `summary.json`, `model-geometry.json`, `source.json`: acceptance summary,
  40-layer/topology verification, historical source revisions and file hashes.
- `performance.csv`, `performance-report.json`: historical performance comparisons
  and their limitations. The timing runs were not interleaved A/B measurements;
  they do not establish zero overhead against the ordinary baseline.
- `SHA256SUMS.json`: byte hashes of the published artifacts, excluding the manifest itself.

Original evidence was preserved byte-for-byte. Source paths and dates inside the
reports describe the historical machine/run. The `.gitattributes` file disables
newline conversion for evidence so the published hashes survive checkout on Windows.
Model weights, large parameter snapshots, server credentials and runtime caches
are excluded.

## Offline raw-logprob verification

In an environment with CPU PyTorch and NumPy installed:

```bash
python examples/zero_mismatch/xing4/verify_saved_evidence.py
```

The verifier loads tensors on CPU with `weights_only=True`, checks all published
file hashes, and recomputes all 155 raw comparisons and their recorded hashes.
This independently verifies the included log-probability tensors. The saved audit
records optimizer/IPC checks, but their full parameter snapshots are not included;
the offline verifier does not independently rerun those checks.

## Reproduction launchers

```bash
# NPU regressions: requires an available machine, not run during this publication.
bash examples/zero_mismatch/xing4/run_unit_tests.sh /path/to/vllm-ascend-tree

# One-update smoke or full 30-update acceptance, when the machine is available.
bash examples/zero_mismatch/xing4/run_acceptance.sh fresh-smoke 1
bash examples/zero_mismatch/xing4/run_acceptance.sh fresh-full 30
```

The general setup, dependency versions, model path and work-directory configuration
are described in the parent README. The newly added launchers and offline verifier
were published without executing them, honoring the user's instruction to stop
script execution. `integration-20261123.json` records the earlier observed unit-test
summary and interrupted integration state; it is not a substitute for the raw test log.
