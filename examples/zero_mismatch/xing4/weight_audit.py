"""Read-only resident inference weight fingerprints before/after initial sync."""


def capture_weight_state(worker, root, phase):
    import hashlib
    from pathlib import Path

    import torch

    path = Path(root) / phase / f"rank{worker.rank}"
    path.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, value in worker.model_runner.model.named_parameters():
        cpu = value.detach().cpu().contiguous()
        result[name] = {
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "sha256": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
        if cpu.numel() <= 8192 or (".layers.0." in name and cpu.numel() <= 10000000):
            torch.save(cpu, path / f"{name}.pt")
    return {"rank": worker.rank, "parameters": result}
