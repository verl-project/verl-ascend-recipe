"""Environment forwarded to sandbox/agent workers, independent of the policy."""

import os


def worker_environment():
    required = ("E2B_API_KEY", "E2B_API_URL")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise SystemExit(f"Missing environment variables: {missing}")
    env = {
        key: os.environ[key]
        for key in (
            *required,
            "E2B_DOMAIN",
            "E2B_SANDBOX_URL",
            "CLAUDE_CODE_LOCAL_BINARY",
            "CLAUDE_CODE_LOCAL_BINARY_SHA256",
            "CLAUDE_CODE_NFS_SOURCE",
            "CLAUDE_CODE_NFS_MOUNT_PATH",
            "CLAUDE_CODE_NFS_RELATIVE_BINARY",
            "CLAUDE_CODE_NFS_BINARY_SHA256",
            "CALLBACK_TUNNEL_ENABLED",
            "CALLBACK_RELAY_SSH_TARGET",
            "CALLBACK_RELAY_SSH_PORT",
            "CALLBACK_RELAY_PORT",
            "CALLBACK_ADVERTISED_BASE_URL",
            "CALLBACK_SSH_PASSWORD",
            "CALLBACK_CONTROL_ROOT",
        )
        if key in os.environ
    }
    env["E2B_DEBUG"] = "false"
    return env
