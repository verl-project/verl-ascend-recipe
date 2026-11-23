"""Record installed source provenance without copying models or credentials."""

import datetime
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def repository(path):
    try:
        root = subprocess.check_output(["git", "-C", str(path), "rev-parse", "--show-toplevel"], text=True).strip()
        commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        return {"root": root, "commit": commit}
    except subprocess.CalledProcessError:
        return {"root": str(path), "commit": None}


def main():
    run_root = Path(sys.argv[1]).resolve()
    recipe = Path(__file__).resolve().parent
    files = list(recipe.glob("*.py")) + list(recipe.glob("*.sh")) + [recipe / "case_manifest.json"]
    repos = {"verl": repository(recipe)}
    for module in ["verl", "vllm", "vllm_ascend", "vllm_xing4_0_plugin"]:
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Required package not installed: {module}")
        path = Path(spec.origin).parent
        repos[module] = repository(path)
        if module == "vllm_ascend":
            files.extend((path / "patch/worker/xing4").glob("*.py"))
    receipt = {
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repositories": repos,
        "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
    (run_root / "source.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
