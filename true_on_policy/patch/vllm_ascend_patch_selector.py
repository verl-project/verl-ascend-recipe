"""Select vLLM-Ascend source patches based on detected version and upstream state."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

try:
    from packaging.version import parse as parse_version
except ImportError:  # pragma: no cover - packaging ships with verl
    parse_version = None

_PATCH_DIR_NAME = "vllm_ascend_patches"

_BATCH_INVARIANT_REL = Path("vllm_ascend/batch_invariant.py")
_FA3_REL = Path("vllm_ascend/attention/fa3_v1.py")
_PLATFORM_REL = Path("vllm_ascend/platform.py")
_VERSION_REL = Path("vllm_ascend/_version.py")

# Markers left by the true_on_policy patch itself (any variant).
_BATCH_INVARIANT_MARKER = "BatchInvariantSumFunction"
_FA3_MARKER = "AscendFABackend"
# vllm-ascend >= 0.23.0 ships the FA3 platform routing upstream; the v0.18.0
# patch still adds it via platform.py hunks.
_FA3_ROUTING_MARKER = "_validate_fa3_backend"
# vllm-ascend >= 0.23.0 rewrote batch_invariant.py (envs.VLLM_BATCH_INVARIANT,
# ascend_config-based overrides, deterministic algorithms). Each patch variant
# rebases onto its own batch_invariant.py baseline, so the file style must
# match the selected variant.
_NEW_STYLE_BATCH_INVARIANT_MARKER = "use_deterministic_algorithms"

TRUE_ON_POLICY_PATCH_V0_18_0 = "vllm_ascend_true_on_policy_v0.18.0.patch"
TRUE_ON_POLICY_PATCH_V0_23_0 = "vllm_ascend_true_on_policy_v0.23.0.patch"


@dataclass(frozen=True, slots=True)
class VllmAscendPatchPlan:
    vllm_ascend_version: str
    vllm_ascend_branch: str
    patch_files: tuple[Path, ...]
    skip_reasons: tuple[str, ...]


def _read_vllm_ascend_version(vllm_ascend_root: Path) -> str:
    # setuptools-scm only generates _version.py at build/install time, so source
    # checkouts report "unknown" and selection falls back to feature detection.
    version_file = vllm_ascend_root / _VERSION_REL
    if version_file.is_file():
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', version_file.read_text(encoding="utf-8"))
        if match:
            return match.group(1)
    return "unknown"


def _is_vllm_ascend_0_18_x(version: str) -> bool:
    if parse_version is not None:
        try:
            parsed = parse_version(version)
            return parsed.major == 0 and parsed.minor == 18
        except Exception:
            pass
    return version.startswith("0.18")


def _is_vllm_ascend_0_23_x(version: str) -> bool:
    if parse_version is not None:
        try:
            parsed = parse_version(version)
            return parsed.major == 0 and parsed.minor == 23
        except Exception:
            pass
    return version.startswith("0.23")


def _vllm_ascend_branch_label(version: str, has_fa3_routing: bool, has_new_style_batch_invariant: bool) -> str:
    if _is_vllm_ascend_0_18_x(version):
        return "releases/v0.18.x"
    if _is_vllm_ascend_0_23_x(version):
        return "releases/v0.23.x"
    if has_fa3_routing and has_new_style_batch_invariant:
        return "v0.23.x-style (feature detect)"
    if not has_fa3_routing and not has_new_style_batch_invariant:
        return "v0.18.x-style (feature detect)"
    return "hybrid (feature detect)"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _has_true_on_policy_patch(vllm_ascend_root: Path) -> bool:
    batch_invariant = vllm_ascend_root / _BATCH_INVARIANT_REL
    if not batch_invariant.is_file() or _BATCH_INVARIANT_MARKER not in _read_text(batch_invariant):
        return False

    fa3 = vllm_ascend_root / _FA3_REL
    return fa3.is_file() and _FA3_MARKER in _read_text(fa3)


def _has_upstream_fa3_routing(vllm_ascend_root: Path) -> bool:
    platform_py = vllm_ascend_root / _PLATFORM_REL
    return platform_py.is_file() and _FA3_ROUTING_MARKER in _read_text(platform_py)


def _has_new_style_batch_invariant(vllm_ascend_root: Path) -> bool:
    batch_invariant = vllm_ascend_root / _BATCH_INVARIANT_REL
    return batch_invariant.is_file() and _NEW_STYLE_BATCH_INVARIANT_MARKER in _read_text(batch_invariant)


def _true_on_policy_patch_name(has_fa3_routing: bool) -> str:
    return TRUE_ON_POLICY_PATCH_V0_23_0 if has_fa3_routing else TRUE_ON_POLICY_PATCH_V0_18_0


def select_vllm_ascend_source_patches(recipe_dir: Path, vllm_ascend_root: Path) -> VllmAscendPatchPlan:
    """Return the patch file to apply; empty when the tree already contains all changes.

    Feature detection takes precedence over the version number: vllm-ascend
    0.23.0 merged the FA3 backend and its platform.py routing upstream, so the
    v0.23.0 patch only rewrites batch_invariant.py and adds accept_output_buffer
    to fa3_v1.py, while the v0.18.0 patch still creates fa3_v1.py and patches
    platform.py. Trees mixing the two baselines (e.g. v0.18.0 with only the FA3
    PR merged) match neither variant and raise instead of mis-applying.
    """
    patch_dir = recipe_dir / "patch" / _PATCH_DIR_NAME
    version = _read_vllm_ascend_version(vllm_ascend_root)
    has_fa3_routing = _has_upstream_fa3_routing(vllm_ascend_root)
    has_new_style_bi = _has_new_style_batch_invariant(vllm_ascend_root)
    branch = _vllm_ascend_branch_label(version, has_fa3_routing, has_new_style_bi)

    patch_files: list[Path] = []
    skip_reasons: list[str] = []

    if _has_true_on_policy_patch(vllm_ascend_root):
        skip_reasons.append(
            f"skip true_on_policy patch: tree already contains {_BATCH_INVARIANT_MARKER} and {_FA3_MARKER}"
        )
    elif has_fa3_routing != has_new_style_bi:
        raise RuntimeError(
            f"[true_on_policy] unrecognized vllm-ascend tree state at {vllm_ascend_root}: "
            f"upstream FA3 routing {'present' if has_fa3_routing else 'absent'} but "
            f"batch_invariant.py is {'new' if has_new_style_bi else 'old'}-style. "
            "This looks like a hybrid tree (e.g. v0.18.0 with only the FA3 PR merged); "
            "neither patch variant applies. Point VLLM_ASCEND_ROOT at a vanilla "
            "v0.18.0 or v0.23.0 tree."
        )
    else:
        patch_files.append(patch_dir / _true_on_policy_patch_name(has_fa3_routing))

    for patch_file in patch_files:
        if not patch_file.is_file():
            raise FileNotFoundError(f"[true_on_policy] patch file not found: {patch_file}")

    return VllmAscendPatchPlan(
        vllm_ascend_version=version,
        vllm_ascend_branch=branch,
        patch_files=tuple(patch_files),
        skip_reasons=tuple(skip_reasons),
    )
