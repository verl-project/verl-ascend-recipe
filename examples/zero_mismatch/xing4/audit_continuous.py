"""Independent CPU audit of every raw FP32 path and all native updates."""

import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from settings import RUNS_DIR

root = Path(sys.argv[1]).resolve()
assert root.is_relative_to(RUNS_DIR)
torch.set_num_threads(4)
report = json.loads((root / "result.json").read_text())
steps = report["requested_steps"]
checks = []
violations = []


def require(ok, reason):
    if not ok:
        violations.append(reason)


require(report.get("completed") is True and report["optimizer_steps"] == steps, "Incomplete optimizer step count")
require(len(report["phases"]) == steps + 1, "Incomplete weight versions")
initial = report["resident_initial"]
require(len(initial) == 8 and len({x["pid"] for x in initial}) == 8, "Invalid resident identities")
for version, phase in enumerate(report["phases"]):
    require(phase["version"] == version and phase.get("resident") == initial, f"Resident changed at theta{version}")
    dp = root / f"dual-theta{version}.pt"
    ip = root / f"inference-theta{version}.pt"
    if not dp.exists() or not ip.exists():
        require(False, f"Missing tensors theta{version}")
        continue
    d = torch.load(dp, map_location="cpu", weights_only=True)
    i = torch.load(ip, map_location="cpu", weights_only=True)
    for name, a, b, count in [
        ("pd", i["rollout"], i["batched_prefill"], 64),
        ("mixed_batch", i["batched_prefill"], i["single_prefill"], 64),
        ("recompute", d["rollout"], d["recompute"], 128),
        ("grad_forward", d["rollout"], d["grad_forward"], 128),
        ("grad_vs_recompute", d["recompute"], d["grad_forward"], 128),
    ]:
        a = a.contiguous().numpy()
        b = b.contiguous().numpy()
        assert a.dtype == b.dtype == np.float32 and a.shape == b.shape
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        mismatch = int(np.count_nonzero(a != b))
        bits = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
        maximum = float(np.max(np.abs(a - b)))
        record = {
            "version": version,
            "path": name,
            "count": a.size,
            "mismatch": mismatch,
            "bit_mismatch": bits,
            "max_abs": maximum,
            "finite": finite,
            "reference_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "actual_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
        }
        checks.append(record)
        require(
            finite and a.size == count and mismatch == bits == 0 and maximum == 0,
            f"theta{version}/{name} nonzero or invalid",
        )
        for field in ["count", "mismatch", "bit_mismatch", "max_abs", "finite"]:
            require(record[field] == phase[name][field], f"Report/raw disagreement theta{version}/{name}/{field}")
    require(torch.equal(d["rollout"], i["rollout"].repeat(2, 1)), f"Inference/native target mismatch theta{version}")
pre = json.loads((root / "rollout-weights-before.json").read_text())
post = json.loads((root / "rollout-weights-after.json").read_text())
require(pre == post, "Initial synchronization changed inference weights")
updates = []
for version in range(report["optimizer_steps"]):
    before = json.loads((root / f"parameters-theta{version}.json").read_text())
    after_path = root / f"parameters-theta{version + 1}.json"
    if not after_path.exists():
        require(False, f"Missing parameter audit update{version + 1}")
        continue
    after = json.loads(after_path.read_text())
    require(len(before) == len(after) == 16, f"Wrong native rank count update{version + 1}")
    changed = []
    for a, b in zip(before, after, strict=False):
        require(
            a["rank"] == b["rank"] and a["parameters"].keys() == b["parameters"].keys(),
            f"Parameter key/rank mismatch update{version + 1}",
        )
        changed.append(sum(v["sha256"] != b["parameters"][n]["sha256"] for n, v in a["parameters"].items()))
    phase = report["phases"][version]
    require(
        all(x > 0 for x in changed) and changed == phase.get("changed_parameters"),
        f"No verified change update{version + 1}",
    )
    norm = np.asarray(phase["optimizer"]["grad_norm"])
    require(bool(np.isfinite(norm).all() and (norm > 0).all()), f"Invalid gradient update{version + 1}")
    sync = phase.get("weight_sync", [])
    require(
        len(sync) == 16
        and sum(bool(x["sent"]) for x in sync) == 8
        and all(x["tensor_count"] > 0 for x in sync)
        and len({x["name_digest"] for x in sync}) == 1,
        f"Invalid IPC receipts update{version + 1}",
    )
    updates.append(
        {
            "step": version + 1,
            "changed_parameters_per_rank": changed,
            "grad_norm": norm.tolist(),
            "device_ipc_receipts": len(sync),
        }
    )
receipt = {
    "passed": not violations,
    "requested_steps": steps,
    "completed_steps": report["optimizer_steps"],
    "audited_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "run": str(root),
    "raw_tensor_checks": checks,
    "updates": updates,
    "violations": violations,
    "artifact_sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.iterdir()
        if p.suffix in [".pt", ".json"] and p.name != "continuous-audit.json"
    },
}
(root / "continuous-audit.json").write_text(json.dumps(receipt, indent=2))
print(
    json.dumps(
        {"passed": receipt["passed"], "steps": len(updates), "raw_comparisons": len(checks), "violations": violations},
        indent=2,
    )
)
sys.exit(0 if receipt["passed"] else 1)
