#!/usr/bin/env bash
# Prepare upstream Uni-Agent and independently selectable recipe extensions.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target="${UNI_AGENT_ROOT:-$ROOT/.runtime/uni-agent}"
revision=04d38d47b9085709c2e3700bd913020d601a82bc
if [[ ! -e "$target" ]]; then
    git clone --no-checkout https://github.com/verl-project/uni-agent.git "$target"
    git -C "$target" checkout --detach "$revision"
fi
[[ "$(git -C "$target" rev-parse HEAD)" == "$revision" ]] || {
    echo "Use a dedicated Uni-Agent checkout at $revision: $target" >&2; exit 2;
}
# Either extension can be applied alone or in either order. An empty value skips both.
read -ra patches <<< "${UNI_AGENT_PATCHES-claude-binary callback-tunnel}"
for name in "${patches[@]}"; do
    case "$name" in claude-binary|callback-tunnel) ;; *) echo "Unknown patch: $name" >&2; exit 2 ;; esac
    patch="$ROOT/patches/$name.patch"
    if git -C "$target" apply --reverse --check "$patch" 2>/dev/null; then
        echo "$name already applied"
    else
        git -C "$target" apply --check "$patch"
        git -C "$target" apply "$patch"
    fi
done
printf 'Prepared upstream Uni-Agent: %s\n' "$target"
printf 'Use this UNI_AGENT_ROOT on both workers and install the upstream dependencies.\n'
