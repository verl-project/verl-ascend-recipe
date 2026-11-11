"""Normalize reward metadata at the JSONL serialization boundary."""

from collections.abc import Mapping

import numpy as np


def json_values(value):
    if isinstance(value, np.generic):
        return json_values(value.item())
    if isinstance(value, np.ndarray):
        return json_values(value.tolist())
    if isinstance(value, Mapping):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(item) for item in value]
    return value


class RolloutJSONMixin:
    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        return super()._dump_generations(
            inputs=json_values(inputs),
            outputs=json_values(outputs),
            gts=json_values(gts),
            scores=json_values(scores),
            reward_extra_infos_dict=json_values(reward_extra_infos_dict),
            dump_path=dump_path,
        )
