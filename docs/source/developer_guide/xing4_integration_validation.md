# Xing4 integration validation — 2026-11-23

The new NPU test module passed all 16 parameterized cases (12 Sinkhorn cases and
four mHC forward/backward cases). The companion verl GEMM and vocabulary tests
also passed, for 18 passing tests in the combined invocation.

Ruff and `git diff --check` passed. `bash format.sh ci` could not start on the
local machine because `pre-commit` was not installed.

The user requested stopping server execution because the machine was in use.
The full-model smoke test and queued follow-up were stopped; they do not establish
a completed optimizer/IPC acceptance for the relocated code. Historical perf20
30-step evidence remains applicable only to the historical implementation.

The generic Qwen numerical source already present on this branch was checked
against the recovered server source and matched. This integration adds only the
explicit Xing adapter package, its regression test and documentation. It excludes
the later, incompletely validated performance experiments.

The explicit NPU test launcher is `bash scripts/run_xing4_zero_mismatch_tests.sh`.
It was added without executing it after the user's stop instruction.

The companion verl branch publishes the full historical audit, performance/step
CSV tables and all 62 raw log-probability files, with a SHA256 manifest and a CPU
verification entry point:
[saved evidence](https://github.com/verl-project/verl-ascend-recipe/tree/rl-zero-mismatch-verl/examples/zero_mismatch/xing4/evidence).
The companion recipe's `run_unit_tests.sh` combines the 16 Xing cases with the two
verl regression cases. These artifacts describe the historical accepted run and
do not claim completion of the interrupted integration smoke test.
