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
