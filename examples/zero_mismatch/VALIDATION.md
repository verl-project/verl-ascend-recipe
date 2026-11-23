# Integration validation — 2026-11-23

The source was recovered from the accepted isolated Xing `perf20` implementation
and the production verl / vllm-ascend Qwen numerical alignment changes.

Completed during this integration:

- Ruff checks and formatting checks on the changed Python source.
- `git diff --check` on the two integration worktrees.
- 18 NPU regression tests passed: Xing Sinkhorn / mHC forward and gradients,
  native GEMM dispatch, and vocabulary reduction.
- Dependency checkout inspection: no local changes in vLLM, MindSpeed,
  MindSpeed-Bridge or Megatron-LM. Megatron-Bridge has the separately supplied
  Qwen3-ASR registration compatibility patch.

The one-update full-model smoke test was interrupted at the user's request because
the machine was in use. The integration launchers and their process trees were
stopped. The queued validation of the final launcher was stopped as well. No
completed real optimizer update, resident IPC acceptance or new 30-step acceptance
is claimed for this reorganized source. The launcher was corrected to put its own
repository first on the import path and to use its repository as the working
directory; this final correction has static validation only.

`bash format.sh ci` in the vllm-ascend worktree could not start because `pre-commit`
was not installed on the local machine. The separate Ruff checks passed. The
historical 30-update evidence described in the Xing README remains historical.

AI assistance was used to organize the code. No upstream pull request was opened.

Historical reports, CSV tables and all 62 raw log-probability files are now
published under `xing4/evidence/`; see its README for provenance and the offline
verifier. Publication did not start any server job or execute the new launchers.
