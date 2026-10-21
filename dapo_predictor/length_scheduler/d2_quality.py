"""Coarse, rank-first admission for externally trained frozen D2 candidates.

No torch, actor training, or filesystem mutation is performed here. Epochs are
completed RL data passes, not the number of optimizer iterations in a head fit.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import combinations
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from .checkpoint import file_sha256


@dataclass(frozen=True)
class D2QualityConfig:
    warmup_epochs: int = 10
    interval_epochs: int = 5
    min_prompts: int = 32
    min_pairs: int = 100
    max_rank_pairs: int = 20000
    min_rank_accuracy: float = 0.60
    rank_relative_gap: float = 0.10
    max_normalized_mae: float = 1.0
    min_scale_ratio: float = 0.25
    max_scale_ratio: float = 4.0
    min_p90_coverage: float = 0.70
    max_p90_ratio: float = 4.0
    bad_windows_before_fallback: int = 2
    risk_weight: float = 0.5

    def __post_init__(self):
        for name in (
            "warmup_epochs",
            "interval_epochs",
            "min_prompts",
            "min_pairs",
            "max_rank_pairs",
            "bad_windows_before_fallback",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "warmup_epochs" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0.5 < self.min_rank_accuracy <= 1:
            raise ValueError("min_rank_accuracy must be above chance and <= 1")
        if not 0 <= self.min_p90_coverage <= 0.9:
            raise ValueError("min_p90_coverage must be between 0 and nominal 0.9")
        if not 0 <= self.rank_relative_gap < 1 or self.risk_weight < 0:
            raise ValueError("invalid relative gap or risk weight")
        if not 0 < self.min_scale_ratio <= 1 <= self.max_scale_ratio:
            raise ValueError("scale guards must bracket 1")
        if self.max_normalized_mae <= 0 or self.max_p90_ratio < 1:
            raise ValueError("invalid MAE or p90 inflation guard")
        if self.max_rank_pairs < self.min_pairs:
            raise ValueError("max_rank_pairs must be >= min_pairs")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None = None, **overrides):
        values = dict(values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown D2 quality options: {sorted(unknown)}")
        return cls(**(values | overrides))


class QualityWindow:
    """Bounded-by-identity sufficient statistics, never count siblings as prompts."""

    def __init__(self, state=None):
        if state is not None and not isinstance(state, dict):
            raise ValueError("quality window must be a mapping")
        self.groups = deepcopy(state or {})
        for key, group in self.groups.items():
            if (
                not isinstance(group, dict)
                or not isinstance(group.get("prompt_id"), str)
                or not group["prompt_id"].strip()
            ):
                raise ValueError("invalid validation identity")
            for name in ("epoch", "n", "censored", "covered", "unknown"):
                value = group.get(name)
                minimum = 1 if name in ("epoch", "n") else 0
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError("invalid quality window count")
            if key != json.dumps([group["epoch"], group["prompt_id"]], separators=(",", ":")):
                raise ValueError("quality window key/identity mismatch")
            for name in ("observed", "expected", "median", "p90"):
                value = group.get(name)
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid quality window length")
            if (
                group["censored"] > group["n"]
                or group["covered"] > group["n"] - group["censored"]
                or group["unknown"] > group["censored"]
                or group["p90"] < group["median"]
            ):
                raise ValueError("quality window coverage/quantiles contradict counts")

    def add(self, *, prompt_id, epoch, observed, expected, median, p90, censored=False):
        values = (observed, expected, median, p90)
        if (
            not isinstance(prompt_id, str)
            or not prompt_id.strip()
            or any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) or x < 0 for x in values)
        ):
            raise ValueError("quality observations require identity and finite non-negative lengths")
        if not isinstance(censored, bool):
            raise ValueError("censored must be a boolean")
        if float(p90) < float(median):
            raise ValueError("p90 must not be below median")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ValueError("observation epoch must be a one-based RL epoch")
        # A prompt may recur in multiple batches; pool all of its siblings.
        key = json.dumps([epoch, str(prompt_id)], separators=(",", ":"))
        group = self.groups.get(
            key,
            {
                "epoch": epoch,
                "prompt_id": str(prompt_id),
                "n": 0,
                "observed": 0.0,
                "expected": 0.0,
                "median": 0.0,
                "p90": 0.0,
                "censored": 0,
                "covered": 0,
                "unknown": 0,
            },
        )
        updates = {
            name: group[name] + float(value)
            for name, value in zip(("observed", "expected", "median", "p90"), values, strict=True)
        }
        if not all(math.isfinite(value) for value in updates.values()):
            raise ValueError("quality statistics overflow")
        group["n"] += 1
        for name, value in zip(("observed", "expected", "median", "p90"), values, strict=True):
            group[name] = updates[name]
        group["censored"] += int(bool(censored))
        if censored and observed <= p90:
            group["unknown"] += 1
        elif not censored and observed <= p90:
            group["covered"] += 1
        self.groups[key] = group

    def metrics(self, config: D2QualityConfig):
        # Aggregate repeated prompt identities too: they are not independent new prompts.
        pooled = {}
        for group in self.groups.values():
            target = pooled.setdefault(
                group["prompt_id"],
                {key: 0 for key in ("n", "observed", "expected", "median", "p90", "censored", "covered", "unknown")},
            )
            for key in target:
                target[key] += group[key]
            if not all(math.isfinite(target[key]) for key in ("observed", "expected", "median", "p90")):
                raise ValueError("pooled quality statistics overflow")
        exact = [g for _, g in sorted(pooled.items()) if not g["censored"]]
        means = [{key: g[key] / g["n"] for key in ("observed", "expected", "median", "p90")} for g in exact]
        correct = 0.0
        pairs = 0
        total_pairs = len(means) * (len(means) - 1) // 2
        if total_pairs <= config.max_rank_pairs:
            selected_pairs = combinations(range(len(means)), 2)
        else:
            # Bound CPU work for full datasets; never change global RNG state.
            generator = random.Random(0)
            selected = set()
            while len(selected) < config.max_rank_pairs:
                i, j = sorted(generator.sample(range(len(means)), 2))
                selected.add((i, j))
            selected_pairs = sorted(selected)
        for i, j in selected_pairs:
            a, b = means[i], means[j]
            actual = a["observed"] - b["observed"]
            if abs(actual) <= config.rank_relative_gap * max(a["observed"], b["observed"], 1):
                continue
            score_a = a["median"] + config.risk_weight * (a["p90"] - a["median"])
            score_b = b["median"] + config.risk_weight * (b["p90"] - b["median"])
            if not math.isfinite(score_a) or not math.isfinite(score_b):
                raise ValueError("non-finite risk score in quality window")
            predicted = score_a - score_b
            correct += 0.5 if predicted == 0 else float(actual * predicted > 0)
            pairs += 1
        observed = sum(g["observed"] for g in means)
        expected = sum(g["expected"] for g in means)
        p90_total = sum(g["p90"] for g in means)
        absolute_error = sum(abs(g["expected"] - g["observed"]) for g in means)
        if not all(math.isfinite(value) for value in (observed, expected, p90_total, absolute_error)):
            raise ValueError("aggregate quality statistics overflow")
        count = sum(g["n"] for g in pooled.values())
        known_covered = sum(g["covered"] for g in pooled.values())
        unknown = sum(g["unknown"] for g in pooled.values())
        # Coverage is prompt-balanced; right-censored outcomes contribute bounds.
        coverage_lower = sum(g["covered"] / g["n"] for g in pooled.values()) / max(len(pooled), 1)
        coverage_upper = sum((g["covered"] + g["unknown"]) / g["n"] for g in pooled.values()) / max(len(pooled), 1)
        return {
            "distinct_prompts": len(pooled),
            "exact_prompts": len(exact),
            "rollouts": count,
            "comparable_pairs": pairs,
            "rank_accuracy": correct / pairs if pairs else None,
            "normalized_mae": absolute_error / observed if observed else None,
            "scale_ratio": expected / observed if observed else None,
            "p90_ratio": p90_total / observed if observed else None,
            "p90_coverage_lower": coverage_lower,
            "p90_coverage_upper": coverage_upper,
            "coverage_known_count": known_covered,
            "coverage_unknown_count": unknown,
            "censored_rollouts": sum(g["censored"] for g in pooled.values()),
        }


def assess(metrics, config: D2QualityConfig):
    """Rank is the primary gate; only gross scale/tail errors are safety vetoes."""
    reasons = []
    sufficient = metrics["exact_prompts"] >= config.min_prompts and metrics["comparable_pairs"] >= config.min_pairs
    if not sufficient:
        reasons.append("insufficient_independent_prompts_or_pairs")
    rank = metrics["rank_accuracy"]
    if rank is None or rank < config.min_rank_accuracy:
        reasons.append("ranking_below_threshold")
    severe = []
    mae, scale, p90 = (metrics[key] for key in ("normalized_mae", "scale_ratio", "p90_ratio"))
    if mae is None or scale is None or p90 is None:
        reasons.append("missing_exact_scale_evidence")
    else:
        if mae > config.max_normalized_mae:
            severe.append("gross_length_error")
        if not config.min_scale_ratio <= scale <= config.max_scale_ratio:
            severe.append("gross_scale_bias")
        if p90 > config.max_p90_ratio:
            severe.append("inflated_p90")
    if metrics["p90_coverage_upper"] < config.min_p90_coverage:
        severe.append("tail_undercoverage")
    elif metrics["p90_coverage_lower"] < config.min_p90_coverage:
        reasons.append("censoring_prevents_tail_verification")
    return {
        "passed": not reasons and not severe,
        "sufficient": sufficient,
        "severe": bool(severe) and sufficient,
        "reasons": reasons + severe,
        "metrics": metrics,
    }


def make_validation_report(rows, *, checkpoint, trained_through_epoch, validation_epoch, risk_weight=0.5):
    """Build a candidate report from held-out, pre-rollout forecasts (not fitted labels)."""
    if not isinstance(trained_through_epoch, int) or isinstance(trained_through_epoch, bool):
        raise ValueError("trained_through_epoch must be an integer")
    if not isinstance(validation_epoch, int) or isinstance(validation_epoch, bool):
        raise ValueError("validation_epoch must be an integer")
    if not 0 <= trained_through_epoch < validation_epoch:
        raise ValueError("validation must be strictly after the candidate training cutoff")
    if (
        isinstance(risk_weight, bool)
        or not isinstance(risk_weight, Real)
        or not math.isfinite(risk_weight)
        or risk_weight < 0
    ):
        raise ValueError("risk_weight must be finite and non-negative")
    window = QualityWindow()
    for row in rows:
        if not trained_through_epoch < row["epoch"] <= validation_epoch:
            raise ValueError("validation row overlaps candidate training or lies in the future")
        window.add(**row)
    return {
        "schema_version": 1,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "trained_through_epoch": trained_through_epoch,
        "validation_epoch": validation_epoch,
        "risk_weight": risk_weight,
        "groups": window.groups,
    }


class D2QualityLifecycle:
    """Candidate changes only at coarse boundaries; an admitted head stays frozen."""

    def __init__(self, config=None):
        self.config = config or D2QualityConfig()
        self.status = "waiting"
        self.last_boundary = -1
        self.bad_windows = 0
        self.checkpoint = None
        self.checkpoint_sha256 = None
        self.report_sha256 = None
        self.last_decision = {}
        self.window = QualityWindow()

    @property
    def active(self):
        return self.status == "active"

    def due(self, completed_epochs):
        return (
            completed_epochs >= self.config.warmup_epochs
            and (completed_epochs - self.config.warmup_epochs) % self.config.interval_epochs == 0
            and completed_epochs > self.last_boundary
        )

    def boundary(self, completed_epochs, report_path=None):
        if not self.due(completed_epochs):
            return None
        self.last_boundary = completed_epochs
        if self.active:
            try:
                decision = assess(self.window.metrics(self.config), self.config)
            except (ValueError, OverflowError) as error:
                decision = {
                    "passed": False,
                    "sufficient": False,
                    "severe": True,
                    "reasons": [f"invalid_monitor: {error}"],
                }
            if decision["sufficient"]:
                self.bad_windows = 0 if decision["passed"] else self.bad_windows + 1
                if decision["severe"] or self.bad_windows >= self.config.bad_windows_before_fallback:
                    self.status = "suspended"
            else:
                # Never silently keep scheduling forever with no monitoring evidence.
                self.status = "suspended"
            self.window = QualityWindow()
        else:
            decision = {"passed": False, "reasons": ["external_candidate_required"]}
            if report_path and Path(report_path).is_file():
                try:
                    raw = Path(report_path).read_bytes()
                    report = json.loads(raw)
                    if not isinstance(report, dict):
                        raise ValueError("candidate report must be a JSON object")
                    if report.get("schema_version") != 1 or report["risk_weight"] != self.config.risk_weight:
                        raise ValueError("candidate report schema or risk_weight mismatch")
                    start, end = report["trained_through_epoch"], report["validation_epoch"]
                    if any(isinstance(x, bool) or not isinstance(x, int) for x in (start, end)):
                        raise ValueError("candidate epochs must be integers")
                    historical_start = completed_epochs == 0 and self.config.warmup_epochs == 0
                    if not 0 <= start < end or (
                        not historical_start
                        and not completed_epochs - self.config.interval_epochs < end <= completed_epochs
                    ):
                        raise ValueError("candidate validation must be recent, completed, and held out")
                    if self.status == "suspended" and report["checkpoint_sha256"] == self.checkpoint_sha256:
                        raise ValueError("suspended checkpoint needs a newly trained candidate")
                    checkpoint = Path(report["checkpoint"])
                    if not checkpoint.is_absolute() or file_sha256(checkpoint) != report["checkpoint_sha256"]:
                        raise ValueError("candidate checkpoint SHA-256 mismatch")
                    # Validate grouped state centrally; never trust supplied metrics.
                    window = QualityWindow(report["groups"])
                    for group in window.groups.values():
                        if not start < group["epoch"] <= end:
                            raise ValueError("validation group overlaps training cutoff")
                    if not window.groups or max(g["epoch"] for g in window.groups.values()) != end:
                        raise ValueError("validation report has no rows from its declared final epoch")
                    decision = assess(window.metrics(self.config), self.config)
                    if decision["passed"]:
                        self.checkpoint = str(checkpoint)
                        self.checkpoint_sha256 = report["checkpoint_sha256"]
                        self.report_sha256 = hashlib.sha256(raw).hexdigest()
                        self.bad_windows = 0
                        self.status = "active"
                        self.window = QualityWindow()
                except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
                    decision = {"passed": False, "reasons": [f"invalid_candidate: {error}"]}
        self.last_decision = {"completed_epochs": completed_epochs, "status": self.status, **decision}
        return self.last_decision

    def state_dict(self):
        return deepcopy(
            {
                "schema_version": 1,
                "config": asdict(self.config),
                "status": self.status,
                "last_boundary": self.last_boundary,
                "bad_windows": self.bad_windows,
                "checkpoint": self.checkpoint,
                "checkpoint_sha256": self.checkpoint_sha256,
                "report_sha256": self.report_sha256,
                "last_decision": self.last_decision,
                "window": self.window.groups,
            }
        )

    def load_state_dict(self, state):
        if (
            not isinstance(state, dict)
            or state.get("schema_version") != 1
            or state.get("config") != asdict(self.config)
        ):
            raise ValueError("D2 quality checkpoint configuration mismatch")
        required = {
            "status",
            "last_boundary",
            "bad_windows",
            "checkpoint",
            "checkpoint_sha256",
            "report_sha256",
            "last_decision",
            "window",
        }
        if not required.issubset(state) or isinstance(state["schema_version"], bool):
            raise ValueError("incomplete D2 quality checkpoint state")
        if state.get("status") not in ("waiting", "active", "suspended"):
            raise ValueError("invalid D2 quality status")
        for name in ("last_boundary", "bad_windows"):
            value = state.get(name)
            minimum = -1 if name == "last_boundary" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError("invalid D2 quality checkpoint counter")
        boundary = state["last_boundary"]
        if boundary != -1 and (
            boundary < self.config.warmup_epochs or (boundary - self.config.warmup_epochs) % self.config.interval_epochs
        ):
            raise ValueError("D2 checkpoint boundary is not on the configured epoch grid")
        window = QualityWindow(state.get("window"))
        if any(g["epoch"] <= boundary for g in window.groups.values()):
            raise ValueError("D2 checkpoint monitoring window predates its boundary")
        if state.get("checkpoint") is not None and not isinstance(state["checkpoint"], str):
            raise ValueError("invalid D2 checkpoint path")
        if state.get("checkpoint"):
            for name in ("checkpoint_sha256", "report_sha256"):
                digest = state.get(name)
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("invalid D2 checkpoint SHA-256")
        if state["checkpoint"] and file_sha256(state["checkpoint"]) != state["checkpoint_sha256"]:
            raise ValueError("frozen D2 checkpoint SHA-256 mismatch on resume")
        if state["status"] == "active" and not state["checkpoint"]:
            raise ValueError("active D2 state requires a checkpoint")
        if state["status"] == "active" and boundary == -1:
            raise ValueError("active D2 state requires an admission boundary")
        if not isinstance(state["last_decision"], dict):
            raise ValueError("invalid D2 quality decision state")
        if state["checkpoint"] is None and (
            state["checkpoint_sha256"] is not None or state["report_sha256"] is not None
        ):
            raise ValueError("D2 checkpoint identity is incomplete")
        for key in (
            "status",
            "last_boundary",
            "bad_windows",
            "checkpoint",
            "checkpoint_sha256",
            "report_sha256",
            "last_decision",
        ):
            setattr(self, key, deepcopy(state[key]))
        self.window = window
