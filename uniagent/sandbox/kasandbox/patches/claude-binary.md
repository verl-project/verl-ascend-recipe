# Claude binary upload and read-only NFS

Base: public Uni-Agent `04d38d47b9085709c2e3700bd913020d601a82bc`.

Only `uni_agent/agents/claude_code/agent.py` is changed. This patch does not
depend on the callback-tunnel patch and can be applied before or after it.

Set `CLAUDE_CODE_LOCAL_BINARY` and `CLAUDE_CODE_LOCAL_BINARY_SHA256` to upload a
worker-local binary, or set `CLAUDE_CODE_NFS_SOURCE`, `CLAUDE_CODE_NFS_RELATIVE_BINARY`
and `CLAUDE_CODE_NFS_BINARY_SHA256` for a read-only guest NFS mount. The modes are
mutually exclusive. NFS guests need sudo/mount tools and access to the export.
If neither mode is configured, the original upstream installation behavior is retained.

Apply from a clean checkout of that revision:

```bash
git apply --check /path/to/kasandbox/patches/claude-binary.patch
git apply /path/to/kasandbox/patches/claude-binary.patch
PYTHONPATH="$PWD" python /path/to/kasandbox/patches/test_claude_binary.py
```

Tests run without NPU hardware or live sandbox credentials.
