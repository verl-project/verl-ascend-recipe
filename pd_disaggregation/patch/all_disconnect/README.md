# Mooncake all-peer disconnect variant

This variant keeps Ascend Mooncake peer connections alive during rollout and
disconnects every tracked peer immediately before vLLM-Ascend unregisters KV
buffers for sleep mode. Do not enable `ASCEND_USE_SHORT_CONNECTION` with this
variant.

The vLLM-Ascend patch also converts a pending remote KV receive into an empty
receive when partial rollout aborts the decode-side request before worker
dispatch. The worker therefore still sends `DONE_RECVING_MSG`, allowing the
prefill node to release its delayed KV blocks without waiting for the fallback
abort timeout.

`DONE_RECVING_MSG` delivery is retried with a fresh ZMQ REQ socket when the
send, reply, or ACK exchange fails. This keeps a transient side-channel failure
from leaving prefill KV pinned until `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`.

Apply the patches in this order:

```bash
cd Mooncake
git checkout v0.3.9
git apply ../verl-ascend-recipe/pd_disaggregation/patch/all_disconnect/mooncake.patch

cd ../vllm-ascend
git checkout releases/v0.23.0
git apply ../verl-ascend-recipe/pd_disaggregation/patch/vllm-ascend.patch
git apply ../verl-ascend-recipe/pd_disaggregation/patch/all_disconnect/vllm-ascend.patch

cd ../verl
git checkout v0.9.0
git apply ../verl-ascend-recipe/pd_disaggregation/patch/verl.patch
git apply ../verl-ascend-recipe/pd_disaggregation/patch/all_disconnect/verl.patch
```

Ascend peer tracking still requires:

```bash
export ASCEND_AUTO_CONNECT=0
unset ASCEND_USE_SHORT_CONNECTION
```
