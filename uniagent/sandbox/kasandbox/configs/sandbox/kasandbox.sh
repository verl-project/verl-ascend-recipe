#!/usr/bin/env bash
# Addresses and credentials are supplied by the deployment, never baked in.
export KASANDBOX_TUNNEL_MODE="${KASANDBOX_TUNNEL_MODE:-ssh}"
export KASANDBOX_SSH_PORT="${KASANDBOX_SSH_PORT:-22}"
export KASANDBOX_API_PORT="${KASANDBOX_API_PORT:-9191}"
export KASANDBOX_PROXY_PORT="${KASANDBOX_PROXY_PORT:-9192}"
export KASANDBOX_REMOTE_API_PORT="${KASANDBOX_REMOTE_API_PORT:-29191}"
export KASANDBOX_REMOTE_PROXY_PORT="${KASANDBOX_REMOTE_PROXY_PORT:-29192}"
case "$KASANDBOX_TUNNEL_MODE" in
    ssh)
        : "${KASANDBOX_SSH_TARGET:?Set user@relay-host}"
        export E2B_API_URL="http://127.0.0.1:$KASANDBOX_API_PORT"
        export E2B_SANDBOX_URL="http://127.0.0.1:$KASANDBOX_PROXY_PORT"
        ;;
    direct) : "${E2B_API_URL:?Set API URL}" "${E2B_SANDBOX_URL:?Set client-proxy URL}" ;;
    *) echo 'KASANDBOX_TUNNEL_MODE must be ssh or direct' >&2; exit 2 ;;
esac
export E2B_DOMAIN="${E2B_DOMAIN:-e2b.app}"
export CALLBACK_TUNNEL_ENABLED="${CALLBACK_TUNNEL_ENABLED:-true}"
export CALLBACK_RELAY_PORT="${CALLBACK_RELAY_PORT:-29193}"
export CALLBACK_RELAY_SSH_TARGET="${CALLBACK_RELAY_SSH_TARGET:-${KASANDBOX_SSH_TARGET:-}}"
export CALLBACK_RELAY_SSH_PORT="${CALLBACK_RELAY_SSH_PORT:-$KASANDBOX_SSH_PORT}"
if [[ "$CALLBACK_TUNNEL_ENABLED" == true ]]; then
    : "${CALLBACK_ADVERTISED_BASE_URL:?Set sandbox-reachable callback URL}"
    : "${CALLBACK_RELAY_SSH_TARGET:?Set callback SSH relay target}"
fi
export CLAUDE_CODE_BINARY_MODE="${CLAUDE_CODE_BINARY_MODE:-installed}"
case "$CLAUDE_CODE_BINARY_MODE" in
    nfs)
        : "${CLAUDE_CODE_NFS_SOURCE:?Set NFS host:/export}"
        : "${CLAUDE_CODE_NFS_RELATIVE_BINARY:?Set version/architecture/claude path}"
        : "${CLAUDE_CODE_NFS_BINARY_SHA256:?Set SHA256 of binary}"
        export CLAUDE_CODE_NFS_MOUNT_PATH="${CLAUDE_CODE_NFS_MOUNT_PATH:-/opt/claude-shared}"
        export CLAUDE_CODE_LOCAL_BINARY="" CLAUDE_CODE_LOCAL_BINARY_SHA256=""
        ;;
    upload)
        : "${CLAUDE_CODE_LOCAL_BINARY:?Set absolute path to ARM64 Claude binary}"
        : "${CLAUDE_CODE_LOCAL_BINARY_SHA256:?Set SHA256 of binary}"
        [[ "$CLAUDE_CODE_LOCAL_BINARY" = /* && -r "$CLAUDE_CODE_LOCAL_BINARY" ]] || { echo 'Unreadable Claude binary' >&2; exit 2; }
        export CLAUDE_CODE_NFS_SOURCE=""
        ;;
    installed) export CLAUDE_CODE_NFS_SOURCE="" CLAUDE_CODE_LOCAL_BINARY="" CLAUDE_CODE_LOCAL_BINARY_SHA256="" ;;
    *) echo 'CLAUDE_CODE_BINARY_MODE must be nfs, upload or installed' >&2; exit 2 ;;
esac
