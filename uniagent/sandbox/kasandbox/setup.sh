#!/usr/bin/env bash
# Creates a dedicated dependency checkout; never modifies an installed package.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target="${UNI_AGENT_ROOT:-$ROOT/.runtime/uni-agent}"
revision=1622670bf156c0c11bb1a7317e4eb1f3c3fc7680
if [[ ! -e "$target" ]]; then
    git clone --no-checkout https://github.com/verl-project/uni-agent.git "$target"
    git -C "$target" checkout --detach "$revision"
fi
[[ "$(git -C "$target" rev-parse HEAD)" == "$revision" ]] || {
    echo "Use a dedicated Uni-Agent checkout at $revision: $target" >&2; exit 2;
}
patch="$ROOT/patches/uni-agent.patch"
if git -C "$target" apply --reverse --check "$patch" 2>/dev/null; then
    echo 'Recipe patch already applied'
else
    [[ -z "$(git -C "$target" status --porcelain)" ]] || { echo 'Dependency checkout has unrelated changes' >&2; exit 2; }
    git -C "$target" apply --check "$patch"
    git -C "$target" apply "$patch"
fi
printf 'Prepared Uni-Agent: %s\n' "$target"
printf 'Use this path as UNI_AGENT_ROOT on both workers. Install dependencies in your training environment.\n'
