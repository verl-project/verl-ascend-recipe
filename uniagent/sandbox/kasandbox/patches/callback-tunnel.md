# Per-Gateway callback tunnels

Base: public Uni-Agent `04d38d47b9085709c2e3700bd913020d601a82bc`.

The callback supervisor, Gateway configuration/address publication, manager
startup/cleanup and lazy package imports form one extension and must be applied
together. No Claude, E2B, task or framework implementation is modified.
This patch is independent of claude-binary.patch; either application order works.

Enable with `CALLBACK_TUNNEL_ENABLED=true`, and configure `CALLBACK_RELAY_SSH_TARGET`,
`CALLBACK_RELAY_SSH_PORT`, `CALLBACK_RELAY_PORT` and `CALLBACK_ADVERTISED_BASE_URL`.
Reserve one consecutive relay and advertised port per Gateway. SSH host keys must
be verified in advance. Tunnels target the actual listener on each Gateway node.
Leave the feature disabled for direct Gateway reachability.

Apply from a clean checkout of that revision:

```bash
git apply --check /path/to/kasandbox/patches/callback-tunnel.patch
git apply /path/to/kasandbox/patches/callback-tunnel.patch
PYTHONPATH="$PWD" python /path/to/kasandbox/patches/test_callback_tunnel.py
```

Tests run without NPU hardware or live sandbox credentials.
