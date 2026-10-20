#!/usr/bin/env bash
set -euo pipefail

: "${VERL_ROOT:?Set VERL_ROOT to a clean verl checkout}"
FAPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE=fa856c3d97fbf93c059322298cc24bec3a944bfe
if [[ "$(git -C "$VERL_ROOT" rev-parse HEAD)" != "$BASE" ]]; then
    echo "Expected verl commit $BASE; use a separate checkout at that commit." >&2
    exit 1
fi
if git -C "$VERL_ROOT" apply --reverse --check "$FAPO_DIR/patches/verl.patch" 2>/dev/null; then
    echo "FAPO verl patch already applied."
    exit 0
fi
git -C "$VERL_ROOT" apply --check "$FAPO_DIR/patches/verl.patch"
git -C "$VERL_ROOT" apply "$FAPO_DIR/patches/verl.patch"
