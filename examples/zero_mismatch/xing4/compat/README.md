# Megatron-Bridge compatibility patch

The inspected `/Megatron-Bridge` checkout was at
`fcbb6031103d0ca845c1a54d4fee55ecfcca17b6` with one modified file:
`src/megatron/bridge/models/qwen3_asr/hf_qwen3_asr/__init__.py`.

The patch prevents duplicate Transformers auto-class registration when a newer
Transformers installation already knows `qwen3_asr`. It was present in the
historically verified environment and is independent of Xing arithmetic.

Inspect and check the patch against that dependency checkout before applying:

```bash
git -C /path/to/Megatron-Bridge apply --check /path/to/megatron-bridge-qwen3-asr-registration.patch
git -C /path/to/Megatron-Bridge apply /path/to/megatron-bridge-qwen3-asr-registration.patch
```

The recipe never applies it automatically. No changes were found in the inspected
vLLM, MindSpeed, MindSpeed-Bridge, or Megatron-LM checkouts. The original Xing model
plugin's local registration wrappers are replaced by `plugin/` in this recipe.
