#!/usr/bin/env python3
"""Read Docker image metadata and compare it with the validated offline image."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

RUNTIME_FORMAT = (
    "{{json .Config.Env}}{{json .Config.Cmd}}{{json .Config.Entrypoint}}{{.Config.WorkingDir}}{{.Config.User}}"
)


def verify(image: str, reference: dict) -> None:
    def inspect(fmt: str) -> str:
        return subprocess.check_output(["docker", "image", "inspect", image, "--format", fmt], text=True)

    layers = inspect("{{json .RootFS.Layers}}")
    runtime = inspect(RUNTIME_FORMAT)
    metadata = inspect("{{.Created}}|{{.Architecture}}|{{.Os}}|{{len .RootFS.Layers}}").strip()
    expected_metadata = "|".join(str(reference[k]) for k in ("created", "architecture", "os", "layer_count"))
    if hashlib.sha256(layers.encode()).hexdigest() != reference["rootfs_layers_sha256"]:
        raise ValueError("RootFS layers differ from the validated image")
    if hashlib.sha256(runtime.encode()).hexdigest() != reference["runtime_config_fields_sha256"]:
        raise ValueError("Runtime configuration differs from the validated image")
    if metadata != expected_metadata:
        raise ValueError("Creation time, architecture, OS or layer count differs from the validated image")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument(
        "--reference", type=Path, default=Path(__file__).resolve().parents[1] / "evidence/910b3-100step/image.json"
    )
    args = parser.parse_args()
    try:
        verify(args.image, json.loads(args.reference.read_text()))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Image verification failed: {exc}\n")
    print("RootFS layers, runtime configuration and platform metadata match the validated image.")


if __name__ == "__main__":
    main()
