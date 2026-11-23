import hashlib
import json
import re
import sys
from pathlib import Path

from settings import MODEL_PATH, RUNS_DIR

root = Path(sys.argv[1]).resolve()
assert root.is_relative_to(RUNS_DIR)
config_path = MODEL_PATH / "config.json"
config = json.loads(config_path.read_text())
text = config.get("text_config", config)
assert text["num_hidden_layers"] == 40 and text["hidden_size"] == 3584 and text["n_routed_experts"] == 64
cfg = json.loads((root / "resolved-training-config.json").read_text())
engine = cfg["engine_config"]
assert (
    Path(cfg["model_config"]["path"]).resolve() == config_path.parent and cfg["model_config"]["override_config"] == {}
)
expected = {
    "tensor_model_parallel_size": 2,
    "pipeline_model_parallel_size": 2,
    "expert_model_parallel_size": 8,
    "expert_tensor_parallel_size": 1,
    "context_parallel_size": 1,
}
assert all(engine[k] == v for k, v in expected.items())
native = json.loads((root / "parameters-theta0.json").read_text())
assert len(native) == 16
nr = []
for rank in native:
    layers = sorted(
        {int(m.group(1)) for name in rank["parameters"] if (m := re.search(r"decoder\.layers\.(\d+)\.", name))}
    )
    assert layers == list(range(20)), (rank["rank"], layers)
    nr.append({"rank": rank["rank"], "local_language_layers": 20})
rollout = json.loads((root / "rollout-weights-before.json").read_text())
assert len(rollout) == 8
vr = []
for rank in rollout:
    layers = sorted(
        {int(m.group(1)) for name in rank["parameters"] if (m := re.search(r"model\.layers\.(\d+)\.", name))}
    )
    assert layers == list(range(40)), (rank["rank"], layers)
    vr.append({"rank": rank["rank"], "language_layers": 40})
result = {
    "passed": True,
    "weight_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "num_hidden_layers": 40,
    "hidden_size": 3584,
    "num_experts": 64,
    "training_topology": expected,
    "native": nr,
    "rollout": vr,
}
(root / "model-geometry.json").write_text(json.dumps(result, indent=2))
print("Full40 geometry verified on 16 training ranks and 8 inference ranks")
