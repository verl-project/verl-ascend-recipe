# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Preserve the native GDN norm's effective FP32 gamma during HF export."""

from dataclasses import replace


class GDNFP32NormExportMapping:
    """Retain Bridge's distribution, but add one with native forward precision.

    The trainer stores a BF16 zero-centered delta and evaluates delta.float()+1.
    Rounding this effective gamma back to BF16 on export changes the function,
    even without an optimizer update. Wrap only the export conversion task;
    neither the source parameter nor Bridge's global mapping class is changed.
    """

    def __init__(self, base_mapping):
        self._base_mapping = base_mapping

    def __getattr__(self, name):
        return getattr(self._base_mapping, name)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        from megatron.bridge.models.conversion.param_mapping import AutoMapping

        # The parent mapping performs the original TP gather and PP broadcast.
        # Its RMSNorm subclass adds one in the stored dtype; do that last step
        # here in FP32 instead, matching the native forward's gamma exactly.
        values = AutoMapping.megatron_to_hf(self._base_mapping, megatron_weights, megatron_module)
        return {name: value.float() + 1.0 for name, value in values.items()}


def preserve_gdn_norm_export_fp32(tasks):
    from megatron.bridge.models.conversion.param_mapping import RMSNorm2ZeroCenteredRMSNormMapping

    return [
        replace(task, mapping=GDNFP32NormExportMapping(task.mapping))
        if isinstance(task.mapping, RMSNorm2ZeroCenteredRMSNormMapping)
        and task.global_param_name.endswith(".self_attention.out_norm.weight")
        else task
        for task in tasks
    ]
