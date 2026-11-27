#!/usr/bin/env bash
# Source from run_reproduce.sh. The shell owns only its API connection.
start_api_tunnel() {
    local -a auth=()
    [[ -z "${SSHPASS:-}" ]] || auth=(sshpass -e)
    API_CONTROL_DIR=$(mktemp -d "$AGENTENV_CONTROL_DIR/api.XXXXXX")
    "${auth[@]}" ssh -M -S "$API_CONTROL_DIR/s" -f -N -T \
        -o StrictHostKeyChecking=yes -o UserKnownHostsFile="${SSH_KNOWN_HOSTS:-$HOME/.ssh/known_hosts}" \
        -o ExitOnForwardFailure=yes -o ConnectTimeout=15 -o ControlPersist=no \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        -p "${AGENTENV_RELAY_SSH_PORT:-22}" \
        -L "127.0.0.1:${AGENTENV_API_PORT:-40190}:127.0.0.1:${AGENTENV_RELAY_API_PORT:-40191}" \
        "${AGENTENV_RELAY_TARGET:?Set AGENTENV_RELAY_TARGET}"
    curl --noproxy '*' --fail --silent --show-error --max-time 15 "$E2B_API_URL/health"
}

stop_api_tunnel() {
    if [[ -n "${API_CONTROL_DIR:-}" ]]; then
        ssh -S "$API_CONTROL_DIR/s" -O exit "${AGENTENV_RELAY_TARGET:?Set AGENTENV_RELAY_TARGET}" 2>/dev/null || true
        rmdir "$API_CONTROL_DIR" 2>/dev/null || true
    fi
}
