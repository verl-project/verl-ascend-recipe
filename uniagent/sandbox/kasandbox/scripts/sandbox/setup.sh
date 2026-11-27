#!/usr/bin/env bash
# Set up API/proxy on each node; prepare callback configuration on the head.
# GatewayManager owns callback tunnels; the caller owns API/proxy cleanup.
_SANDBOX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$_SANDBOX_ROOT/scripts/sandbox/callback_tunnel.sh"

setup_sandbox() {
    local traced=false
    [[ $- != *x* ]] || { traced=true; set +x; }
    local config="${SANDBOX_CONFIG:-$_SANDBOX_ROOT/configs/sandbox/kasandbox.sh}"
    local secrets="${SANDBOX_ENV_FILE-$_SANDBOX_ROOT/.runtime/kasandbox.env}"
    if [[ -n "$secrets" ]]; then
        if [[ -f "$secrets" ]]; then
            source "$secrets"
        elif [[ -v SANDBOX_ENV_FILE ]]; then
            echo "SANDBOX_ENV_FILE does not exist: $secrets" >&2
            return 2
        fi
    fi
    source "$config"
    : "${E2B_API_KEY:?Inject E2B_API_KEY or provide SANDBOX_ENV_FILE}"
    : "${E2B_API_URL:?Set E2B_API_URL}"
    export E2B_API_KEY E2B_API_URL E2B_DEBUG=false
    case "$KASANDBOX_TUNNEL_MODE" in
        ssh)
            : "${KASANDBOX_SSH_TARGET:?Set the remote SSH endpoint explicitly}"
            local -a auth=()
            [[ -z "${KASANDBOX_SSH_PASSWORD:-}" ]] || auth=(sshpass -e)
            SANDBOX_CONTROL_DIR=$(mktemp -d /tmp/uniagent-sandbox.XXXXXX)
            SANDBOX_CONTROL_SOCKET="$SANDBOX_CONTROL_DIR/ssh"
            SSHPASS="${KASANDBOX_SSH_PASSWORD:-}" "${auth[@]}" ssh \
                -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes \
                -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
                -o ConnectTimeout=15 -o ControlPersist=no \
                -M -S "$SANDBOX_CONTROL_SOCKET" -f -N -T \
                -p "$KASANDBOX_SSH_PORT" \
                -L "127.0.0.1:${KASANDBOX_API_PORT}:127.0.0.1:${KASANDBOX_REMOTE_API_PORT}" \
                -L "127.0.0.1:${KASANDBOX_PROXY_PORT}:127.0.0.1:${KASANDBOX_REMOTE_PROXY_PORT}" \
                "$KASANDBOX_SSH_TARGET"
            ;;
        direct) ;;
        *) echo "KASANDBOX_TUNNEL_MODE must be ssh or direct" >&2; return 2 ;;
    esac
    curl --noproxy '*' --fail --show-error --max-time 10 "${E2B_API_URL%/}/health"
    [[ "$traced" != true ]] || set -x
}

prepare_sandbox_callback() {
    local traced=false
    [[ $- != *x* ]] || { traced=true; set +x; }
    prepare_callback_tunnels
    [[ "$traced" != true ]] || set -x
}

stop_sandbox() {
    if [[ -n "${SANDBOX_CONTROL_SOCKET:-}" ]]; then
        ssh -S "$SANDBOX_CONTROL_SOCKET" -O exit "$KASANDBOX_SSH_TARGET" 2>/dev/null || true
        rm -f -- "$SANDBOX_CONTROL_SOCKET"
        rmdir -- "$SANDBOX_CONTROL_DIR" 2>/dev/null || true
    fi
}
