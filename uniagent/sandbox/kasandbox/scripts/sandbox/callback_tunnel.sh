#!/usr/bin/env bash
# Callback SSH masters now belong to GatewayManager after dynamic discovery.
# This shell helper only exports configuration; it never forwards to HEAD_HOST.
_CALLBACK_USER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
prepare_callback_tunnels() {
    case "$CALLBACK_TUNNEL_ENABLED" in
        true)
            : "${CALLBACK_ADVERTISED_BASE_URL:?Set the reachable callback relay URL explicitly}"
            : "${CALLBACK_RELAY_SSH_TARGET:?Set CALLBACK_RELAY_SSH_TARGET}"
            export CALLBACK_TUNNEL_ENABLED CALLBACK_ADVERTISED_BASE_URL
            export CALLBACK_CONTROL_ROOT="${CALLBACK_CONTROL_ROOT:-/tmp}"
            export CALLBACK_RELAY_SSH_TARGET CALLBACK_RELAY_SSH_PORT CALLBACK_RELAY_PORT
            export CALLBACK_SSH_PASSWORD="${CALLBACK_SSH_PASSWORD-${KASANDBOX_SSH_PASSWORD:-}}"
            ;;
        false) export CALLBACK_TUNNEL_ENABLED ;;
        *) echo "CALLBACK_TUNNEL_ENABLED must be true or false" >&2; return 2 ;;
    esac
}
