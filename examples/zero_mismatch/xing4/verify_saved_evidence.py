"""Verify the published historical log-probability data entirely on CPU."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def comparison(reference, actual):
    require(reference.dtype == actual.dtype == torch.float32, "Expected raw FP32 tensors")
    require(reference.shape == actual.shape and reference.numel() > 0, "Tensor shape mismatch")
    left = reference.contiguous().numpy()
    right = actual.contiguous().numpy()
    return {
        "count": left.size,
        "mismatch": int(np.count_nonzero(left != right)),
        "bit_mismatch": int(np.count_nonzero(left.view(np.uint32) != right.view(np.uint32))),
        "max_abs": float(np.max(np.abs(left - right))),
        "finite": bool(np.isfinite(left).all() and np.isfinite(right).all()),
        "reference_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
        "actual_sha256": hashlib.sha256(right.tobytes()).hexdigest(),
    }


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8-sig"))
    for name, expected_hash in manifest.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), f"Manifest path escapes evidence directory: {name}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash, f"File hash mismatch: {name}")
    audit = json.loads((root / "continuous-audit.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    require(summary["passed"] is True and audit["passed"] is True, "Historical report did not pass")
    steps = summary["real_updates"]
    require(summary["weight_versions"] == steps + 1, "Invalid version count")
    records = {(item["version"], item["path"]): item for item in audit["raw_tensor_checks"]}
    require(
        len(records) == len(audit["raw_tensor_checks"]) == (steps + 1) * 5,
        "Invalid audit record count",
    )
    groups = 0
    elements = 0
    for version in range(steps + 1):
        inference = torch.load(
            root / f"raw/inference-theta{version}.pt", map_location="cpu", weights_only=True
        )
        native = torch.load(
            root / f"raw/dual-theta{version}.pt", map_location="cpu", weights_only=True
        )
        require(
            torch.equal(native["rollout"], inference["rollout"].repeat(2, 1)),
            f"Target differs at version {version}",
        )
        pairs = [
            ("pd", inference["rollout"], inference["batched_prefill"], 64),
            ("mixed_batch", inference["batched_prefill"], inference["single_prefill"], 64),
            ("recompute", native["rollout"], native["recompute"], 128),
            ("grad_forward", native["rollout"], native["grad_forward"], 128),
            ("grad_vs_recompute", native["recompute"], native["grad_forward"], 128),
        ]
        for name, reference, actual, count in pairs:
            result = comparison(reference, actual)
            require(result["count"] == count and result["finite"], f"Invalid tensor at {version}/{name}")
            require(result["bit_mismatch"] == result["mismatch"] == 0, f"Mismatch at {version}/{name}")
            require(result["max_abs"] == 0.0, f"Nonzero difference at {version}/{name}")
            require(
                all(result[key] == records[version, name][key] for key in result),
                f"Audit disagreement at {version}/{name}",
            )
            groups += 1
            elements += count
    require(groups == summary["comparison_groups"], "Comparison count disagrees with summary")
    require(elements == summary["compared_elements"], "Element count disagrees with summary")
    return {
        "raw_logprobs_passed": True,
        "groups": groups,
        "elements": elements,
        "optimizer_and_ipc_reaudited": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path(__file__).parent / "evidence/full-opt20-30steps-001",
    )
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.evidence), indent=2))
