"""Diagnostic orchestration around native optimizer and production IPC sync."""

import asyncio
import hashlib
import os

import torch
import torch.distributed as dist

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.workers.engine_workers_tinker import TinkerTrainingWorker


class OnlineWorker(TinkerTrainingWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        print(
            "NPU_ENV_AUDIT",
            {
                key: os.environ.get(key)
                for key in ("RANK", "LOCAL_RANK", "ASCEND_RT_VISIBLE_DEVICES", "VLLM_BATCH_INVARIANT")
            },
            flush=True,
        )
        return super().reset()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def parameter_audit(self):
        with self.engine.eval_mode():
            records = {}
            for chunk, model in enumerate(self.engine.module):
                for name, value in model.named_parameters():
                    if not value.requires_grad:
                        continue
                    data = value.detach().cpu().contiguous()
                    records[f"{chunk}.{name}"] = {
                        "sha256": hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest(),
                        "dtype": str(data.dtype),
                        "shape": list(data.shape),
                    }
            return {"rank": dist.get_rank(), "pid": os.getpid(), "parameters": records}

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_and_step(self, data):
        import math

        from verl.utils import tensordict_utils as tu

        # Keep native gradients alive until the native optimizer consumes them.
        # This is the same lifetime as BaseEngine.train_batch, while retaining
        # Tinker's pre-update logprobs for the independent strict comparison.
        with self.engine.train_mode(zero_grad_on_exit=True):
            self.engine.optimizer_zero_grad()
            tu.assign_non_tensor(data, disable_auto_offload=True)
            output = super().forward_backward(data)
            norm = self.engine.optimizer_step()
            if norm is not None:
                assert math.isfinite(float(norm)) and float(norm) > 0, norm
            if output is not None:
                tu.assign_non_tensor(output, native_optimizer_grad_norm=float(norm))
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def export_and_send(self, job_id, ipc_dir):
        from verl.utils.device import is_support_ipc
        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
            BucketedWeightSender,
            weight_sync_zmq_handle,
        )

        assert is_support_ipc(), "This diagnostic requires device IPC; no filesystem shared-memory fallback"
        os.environ["VERL_WEIGHT_SYNC_IPC_DIR"] = ipc_dir
        rank = dist.get_rank()
        params, peft = self.engine.get_per_tensor_param()
        assert peft is None
        count = [0]
        digest = hashlib.sha256()

        def audited_weights():
            for name, tensor in params:
                count[0] += 1
                digest.update(name.encode())
                yield name, tensor

        try:
            if rank < 8:
                sender = BucketedWeightSender(
                    weight_sync_zmq_handle(job_id, 0, rank), bucket_size_mb=256, use_shm=False
                )
                asyncio.run(sender.async_send_weights(audited_weights()))
            else:
                for _ in audited_weights():
                    pass
            dist.barrier()
            return {"rank": rank, "sent": rank < 8, "tensor_count": count[0], "name_digest": digest.hexdigest()}
        finally:
            self.engine.to(device="cpu", model=True, optimizer=False, grad=False)
