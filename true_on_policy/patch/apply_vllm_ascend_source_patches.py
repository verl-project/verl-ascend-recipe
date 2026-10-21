"""Apply vLLM-Ascend source patches required by true_on_policy (idempotent, version-aware)."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from .vllm_ascend_patch_selector import VllmAscendPatchPlan, select_vllm_ascend_source_patches

logger = logging.getLogger(__name__)

_BATCH_INVARIANT_REL = Path("vllm_ascend/batch_invariant.py")


def _recipe_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _verl_root(recipe_dir: Path) -> Path:
    return recipe_dir.parent.parent


def _resolve_vllm_ascend_root(verl_root: Path) -> Path | None:
    env_root = os.environ.get("VLLM_ASCEND_ROOT", "").strip()
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(verl_root.parent / "vllm-ascend")

    try:
        import vllm_ascend

        candidates.insert(0, Path(vllm_ascend.__file__).resolve().parent.parent)
    except ImportError:
        pass

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / _BATCH_INVARIANT_REL).is_file():
            return candidate
    return None


def _git_apply_check(repo_root: Path, patch_file: Path, *, reverse: bool = False) -> bool:
    cmd = ["git", "-C", str(repo_root), "apply"]
    if reverse:
        cmd.append("-R")
    cmd.extend(["--check", str(patch_file)])
    return subprocess.run(cmd, capture_output=True, check=False).returncode == 0


def _apply_patch_file(repo_root: Path, patch_file: Path) -> None:
    if _git_apply_check(repo_root, patch_file, reverse=True):
        logger.info("[true_on_policy] vllm-ascend patch already applied: %s", patch_file.name)
        return

    if not _git_apply_check(repo_root, patch_file):
        raise RuntimeError(
            f"[true_on_policy] vllm-ascend patch does not apply cleanly: {patch_file.name} (tree at {repo_root})."
        )

    subprocess.run(
        ["git", "-C", str(repo_root), "apply", str(patch_file)],
        check=True,
    )
    logger.warning("[true_on_policy] applied vllm-ascend patch %s", patch_file.name)


def apply_vllm_ascend_source_patches() -> None:
    """Apply version-selected vLLM-Ascend source patches (batch_invariant + FA3 for train-infer consistency)."""
    recipe_dir = _recipe_dir()
    verl_root = _verl_root(recipe_dir)
    vllm_ascend_root = _resolve_vllm_ascend_root(verl_root)

    if vllm_ascend_root is None:
        logger.warning(
            "[true_on_policy] vllm-ascend source tree not found; "
            "set VLLM_ASCEND_ROOT or place vllm-ascend next to verl. Skipping vllm-ascend patches."
        )
        return

    plan = select_vllm_ascend_source_patches(recipe_dir, vllm_ascend_root)
    logger.info(
        "[true_on_policy] detected vllm-ascend %s (%s) at %s",
        plan.vllm_ascend_version,
        plan.vllm_ascend_branch,
        vllm_ascend_root,
    )
    for reason in plan.skip_reasons:
        logger.info("[true_on_policy] %s", reason)

    if not plan.patch_files:
        logger.info("[true_on_policy] no vllm-ascend source patches needed")
        return

    for patch_file in plan.patch_files:
        _apply_patch_file(vllm_ascend_root, patch_file)


def get_vllm_ascend_patch_plan() -> VllmAscendPatchPlan | None:
    """Return the patch plan for the resolved vllm-ascend tree (for diagnostics/tests)."""
    recipe_dir = _recipe_dir()
    verl_root = _verl_root(recipe_dir)
    vllm_ascend_root = _resolve_vllm_ascend_root(verl_root)
    if vllm_ascend_root is None:
        return None
    return select_vllm_ascend_source_patches(recipe_dir, vllm_ascend_root)
