"""Bounded CPU replay and nonblocking fitting for the online D2 backend."""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F

from .distribution import GatedDistributionHead, PrefillDistributionPrior


def spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
    """Spearman correlation with average ranks for ties."""

    def ranks(values):
        order = values.argsort()
        _, counts = torch.unique_consecutive(values[order], return_counts=True)
        average = counts.cumsum(0).float() - (counts.float() + 1) / 2
        result = torch.empty_like(values, dtype=torch.float32)
        result[order] = average.repeat_interleave(counts)
        return result - result.mean()

    if x.numel() < 2 or x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return None
    a, b = ranks(x), ranks(y)
    denominator = a.norm() * b.norm()
    return None if denominator == 0 else float(((a * b).sum() / denominator).clamp(-1.0, 1.0))


def atomic_save(value, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fit_candidate(payload: dict) -> dict:
    """Fit one immutable snapshot; previous projection and optimizer stay aligned."""
    torch.set_num_threads(1)
    started = time.monotonic()
    hidden = payload["hidden"].float()
    lengths = payload["lengths"].float()
    censored = payload["censored"].bool()
    if hidden.ndim != 3 or lengths.ndim != 2 or hidden.shape[0] != lengths.shape[0]:
        raise ValueError("expected prompt/tap/hidden features and prompt/sibling lengths")
    if censored.shape != lengths.shape or lengths.numel() == 0:
        raise ValueError("invalid censor mask or empty replay")
    if not torch.isfinite(hidden).all() or not torch.isfinite(lengths).all() or (lengths < 0).any():
        raise ValueError("invalid online feature or length")
    keys = tuple(payload["feature_keys"])
    if len(keys) != hidden.shape[1] or len(set(keys)) != len(keys):
        raise ValueError("feature schema does not match replay")
    cfg = payload["config"]
    previous = payload.get("previous")
    torch.manual_seed(int(cfg.get("seed", 1)))
    y = torch.log1p(lengths).reshape(-1)
    if previous is None:
        dimension = min(int(cfg.get("projection_dim", 64)), hidden.shape[0], hidden.shape[-1])
        if dimension < 1:
            raise ValueError("projection_dim must be positive")
        bank = {}
        for index, key in enumerate(keys):
            rows = F.normalize(hidden[:, index], dim=-1)
            center = rows.mean(0)
            _, singular, vh = torch.linalg.svd(rows - center, full_matrices=False)
            scale = (singular[:dimension].mean() / max(1, len(rows)) ** 0.5).clamp_min(1e-6)
            bank[key] = {
                "center": center,
                "components": vh[:dimension].T.contiguous(),
                "scale": scale,
            }
        head = GatedDistributionHead(dimension, len(keys), 0.03)
        checkpoint = {
            "schema": "online-censored-lognormal-v1",
            "feature_keys": keys,
            "projection_dim": dimension,
            "projection_bank": bank,
            "standardized_sigma_floor": 0.03,
            "head_state_dict": head.state_dict(),
            "log_length_center": float(y.mean()),
            "log_length_scale": float(y.std(unbiased=False).clamp_min(0.25)),
        }
    else:
        checkpoint = previous
        if tuple(checkpoint["feature_keys"]) != keys:
            raise ValueError("online feature schema changed")
    prior = PrefillDistributionPrior(checkpoint)
    head = prior.head
    head.train()
    projected = prior.project({key: hidden[:, index] for index, key in enumerate(keys)})
    projected = projected.repeat_interleave(lengths.shape[1], dim=0)
    standardized_y = (y - prior.center) / prior.scale
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=1e-5, foreach=False, fused=False
    )
    if previous is not None and "optimizer_state" in previous:
        optimizer.load_state_dict(previous["optimizer_state"])
    steps = int(cfg.get("fit_steps", 50))
    batch_size = int(cfg.get("batch_size", 256))
    if steps < 1 or batch_size < 1:
        raise ValueError("fit_steps and batch_size must be positive")
    for _ in range(steps):
        indices = torch.randperm(len(y))[:batch_size]
        mu, sigma = head(projected[indices])
        z = (standardized_y[indices] - mu) / sigma
        exact = sigma.log() + 0.5 * z.square() + 0.9189385332046727
        tail = -torch.special.log_ndtr(-z)
        loss = torch.where(censored.reshape(-1)[indices], tail, exact).mean()
        if not torch.isfinite(loss):
            raise ValueError("non-finite online D2 loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in head.parameters()):
        raise ValueError("non-finite online head parameters")
    checkpoint = dict(checkpoint)
    checkpoint.update(
        head_state_dict={key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
        optimizer_state=optimizer.state_dict(),
        online_version=int(payload["version"]),
        trained_through_step=int(payload["step"]),
        fit_seconds=time.monotonic() - started,
        fit_loss=float(loss.detach()),
        fit_prompt_observations=len(hidden),
    )
    check = PrefillDistributionPrior(checkpoint).predict({key: hidden[:2, index] for index, key in enumerate(keys)})
    if not all(torch.isfinite(value).all() for value in check.values()):
        raise ValueError("candidate produces invalid summaries")
    return checkpoint


class OnlineD2:
    """One CPU fitter, latest-data coalescing and a separately published head."""

    def __init__(self, directory: str | Path, config=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = dict(config or {})
        self.max_prompts = int(self.config.get("max_prompt_observations", 512))
        self.min_prompts = int(self.config.get("min_fit_prompts", 32))
        if self.min_prompts < 2 or self.max_prompts < self.min_prompts:
            raise ValueError("require 2 <= min_fit_prompts <= max_prompt_observations")
        self.hidden = self.lengths = self.censored = None
        self.feature_keys = ()
        self.process = None
        self.log = None
        self.checkpoint = None
        self.version = 0
        self.last_submitted_step = 0
        self.step = 0
        self.observed_rollouts = 0
        self.failures = 0
        self.failure = ""

    @property
    def path(self) -> Path:
        return self.directory / "current.pt"

    def observe(self, hidden, lengths, censored, feature_keys, step: int) -> None:
        hidden, lengths, censored = (value.detach().cpu().clone() for value in (hidden, lengths, censored))
        if hidden.ndim != 3 or lengths.ndim != 2 or len(hidden) != len(lengths):
            raise ValueError("unaligned online D2 observation")
        if censored.shape != lengths.shape or not torch.isfinite(hidden).all():
            raise ValueError("invalid online feature/censor observation")
        if not torch.isfinite(lengths).all() or (lengths < 0).any():
            raise ValueError("invalid online response lengths")
        keys = tuple(feature_keys)
        if len(keys) != hidden.shape[1] or (self.feature_keys and keys != self.feature_keys):
            raise ValueError("online feature schema changed")
        if step <= self.step:
            raise ValueError("online observation steps must increase")
        if self.hidden is not None and (
            hidden.shape[1:] != self.hidden.shape[1:] or lengths.shape[1:] != self.lengths.shape[1:]
        ):
            raise ValueError("online hidden/sibling dimensions changed")
        self.feature_keys = keys
        for name, value in (
            ("hidden", hidden),
            ("lengths", lengths),
            ("censored", censored),
        ):
            old = getattr(self, name)
            setattr(
                self,
                name,
                (value if old is None else torch.cat((old, value)))[-self.max_prompts :],
            )
        self.step = int(step)
        self.observed_rollouts += lengths.numel()

    def poll(self) -> bool:
        if self.process is None or self.process.poll() is None:
            return False
        code = self.process.returncode
        self.process = None
        self.log.close()
        self.log = None
        output = self.directory / "fit-output.pt"
        try:
            if code != 0:
                raise RuntimeError(f"CPU fit exited {code}; see fit-{self.last_submitted_step}.log")
            candidate = torch.load(output, map_location="cpu", weights_only=True)
            if candidate["trained_through_step"] != self.last_submitted_step:
                raise ValueError("stale candidate step")
            if candidate["online_version"] != self.version + 1:
                raise ValueError("out-of-order candidate version")
            PrefillDistributionPrior(candidate)
            atomic_save(candidate, self.path)
            self.checkpoint = candidate
            self.version += 1
            self.failure = ""
            return True
        except Exception as error:
            self.failures += 1
            self.failure = str(error)
            return False

    def submit(self) -> bool:
        if self.process is not None or self.hidden is None or len(self.hidden) < self.min_prompts:
            return False
        if self.step == self.last_submitted_step:
            return False
        payload = dict(
            hidden=self.hidden,
            lengths=self.lengths,
            censored=self.censored,
            feature_keys=self.feature_keys,
            config=self.config,
            previous=self.checkpoint,
            version=self.version + 1,
            step=self.step,
        )
        atomic_save(payload, self.directory / "fit-input.pt")
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        self.log = (self.directory / f"fit-{self.step}.log").open("ab")
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    __name__,
                    "--input",
                    str(self.directory / "fit-input.pt"),
                    "--output",
                    str(self.directory / "fit-output.pt"),
                ],
                env=env,
                stdout=self.log,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            self.log.close()
            self.log = None
            raise
        self.last_submitted_step = self.step
        return True

    def state_dict(self) -> dict:
        return copy.deepcopy(
            {
                "schema": "online-d2-state-v1",
                "config": self.config,
                "hidden": self.hidden,
                "lengths": self.lengths,
                "censored": self.censored,
                "feature_keys": self.feature_keys,
                "checkpoint": self.checkpoint,
                "version": self.version,
                "step": self.step,
                "observed_rollouts": self.observed_rollouts,
                "failures": self.failures,
            }
        )

    def load_state_dict(self, state: dict) -> None:
        if state.get("schema") != "online-d2-state-v1" or state["config"] != self.config:
            raise ValueError("online D2 checkpoint/config mismatch")
        if self.process is not None:
            raise RuntimeError("cannot restore over a running fit")
        for key in (
            "hidden",
            "lengths",
            "censored",
            "feature_keys",
            "checkpoint",
            "version",
            "step",
            "observed_rollouts",
            "failures",
        ):
            setattr(self, key, state[key])
        if self.version < 0 or self.step < 0 or self.observed_rollouts < 0:
            raise ValueError("invalid online D2 counters")
        if self.hidden is None:
            if self.lengths is not None or self.censored is not None or self.observed_rollouts != 0:
                raise ValueError("missing online replay with nonempty observations")
        else:
            if (
                self.hidden.ndim != 3
                or self.lengths.ndim != 2
                or self.censored.shape != self.lengths.shape
                or len(self.hidden) != len(self.lengths)
                or len(self.hidden) > self.max_prompts
                or len(self.feature_keys) != self.hidden.shape[1]
                or self.observed_rollouts < self.lengths.numel()
                or not torch.isfinite(self.hidden).all()
                or not torch.isfinite(self.lengths).all()
                or (self.lengths < 0).any()
            ):
                raise ValueError("invalid online replay state")
        if (self.checkpoint is None) != (self.version == 0):
            raise ValueError("online D2 version disagrees with head")
        if self.checkpoint is not None:
            if self.checkpoint["trained_through_step"] > self.step or self.checkpoint["online_version"] != self.version:
                raise ValueError("online D2 head comes from a future step")
            PrefillDistributionPrior(self.checkpoint)
            if self.hidden is not None:
                if tuple(self.checkpoint["feature_keys"]) != tuple(self.feature_keys):
                    raise ValueError("online replay/head feature schemas disagree")
                summary = PrefillDistributionPrior(self.checkpoint).predict(
                    {key: self.hidden[:1, index] for index, key in enumerate(self.feature_keys)}
                )
                if not all(torch.isfinite(value).all() for value in summary.values()):
                    raise ValueError("invalid online head predictions on resume")
            atomic_save(self.checkpoint, self.path)

    def close(self) -> None:
        if self.process is not None:
            try:
                self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=15)
            self.poll()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_save(
        fit_candidate(torch.load(args.input, map_location="cpu", weights_only=True)),
        args.output,
    )


if __name__ == "__main__":
    main()
