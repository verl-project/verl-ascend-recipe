# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os

import aiohttp
from transformers import PreTrainedTokenizer

from verl.utils.ray_utils import get_event_loop
from verl.utils.reward_score.math_dapo import last_boxed_only_string, normalize_final_answer, remove_boxed

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def verify(
    solution_str: str,
    gt: str,
) -> tuple[bool, str]:
    solution_str = solution_str[-300:]
    boxed_answer = last_boxed_only_string(solution_str)
    if boxed_answer is not None:
        extracted_answer = remove_boxed(boxed_answer)
    else:
        extracted_answer = "[INVALID]"

    pred = normalize_final_answer(extracted_answer)
    gt = normalize_final_answer(gt)
    return (pred == gt), pred


async def compute_score_baseline(
    solution_str: str,
    ground_truth: str,
    **kwargs,
):
    loop = get_event_loop()
    """Compute the reward score for Baseline."""
    correct, pred = await loop.run_in_executor(None, lambda: verify(solution_str, ground_truth))
    reward_score = 1.0 if correct else -1.0
    return {"score": reward_score, "acc": correct, "pred": pred}


# FAPO Hyper-parameters
FAPO_GENRM_TEMPLATE = (
    "The following is a math problem with its ground truth answer, along with an AI solution (split into steps):\n\n"
    "[Math Problem]\n\n"
    "{problem}\n\n"
    "[Ground Truth]\n\n"
    "{ground_truth}\n\n"
    "[AI Solution]\n\n"
    "{solution}\n\n"
    "Your task is to review and critique the solution step by step. "
    "Once you identify an error in a step, return the index of the step where the earliest error occurs. "
    "Otherwise, return the index of -1 (which typically denotes 'not found').\n\n"
    "Please reason step by step, put your final answer (i.e., the index) in \\boxed{{}}."
)
GRM_SAMPLING_PARAMS = {
    "max_new_tokens": 4096,
}
FLAWED_REWARD_PENALTY = 1.0


async def generate_aiohttp(router_address: str, prompt_ids: list[int], sampling_params: dict):
    """Ask the GRM router to continue ``prompt_ids`` and return the generated text.

    Two server flavours answer this call and they do not share a wire format:

    * SGLang serves a native ``/generate`` that takes ``input_ids`` and answers
      with ``output_ids``.
    * vLLM is served through ``vllm.entrypoints.openai.api_server``, which only
      exposes the OpenAI-compatible routes. Posting to ``/generate`` there gets a
      404, the reward silently degrades to the rule-based score, and the GRM
      never runs.

    ``/v1/completions`` is the portable choice: its ``prompt`` field accepts a
    list of token ids, so the caller does not have to detokenize, and SGLang
    serves the same route. The response carries text rather than ids, hence the
    return type.
    """
    payload = {
        "prompt": prompt_ids,
        "max_tokens": sampling_params.get("max_new_tokens", 16384),
        "temperature": sampling_params.get("temperature", 0.0),
    }
    # "model" is optional: the server resolves the served model itself, and the
    # router fans out to a single-model worker anyway.
    if sampling_params.get("model"):
        payload["model"] = sampling_params["model"]
    url = f"http://{router_address}/v1/completions"
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
    try:
        async with session.post(url, json=payload) as resp:
            output = await resp.text()
            try:
                output = json.loads(output)
            except Exception:
                logger.error(f"Failed to parse JSON response: {output}")
                return None
            choices = output.get("choices") or []
            if not choices:
                logger.error(f"GRM response carried no choices: {output}")
                return None
            return choices[0].get("text")
    finally:
        await session.close()


async def compute_score_fapo(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer,
):
    """Compute the reward score for FAPO."""
    loop = get_event_loop()

    question, split = extra_info["question"], extra_info["split"]
    correct, pred = await loop.run_in_executor(None, lambda: verify(solution_str, ground_truth))
    reward_score = 1.0 if correct else -1.0
    is_flawed_positive = False

    # for test set or incorrect solution, directly return the reward score
    if split == "test" or not correct:
        return {"score": reward_score, "acc": correct, "pred": pred, "is_flawed_positive": is_flawed_positive}

    grm_prompt = FAPO_GENRM_TEMPLATE.format(
        problem=question,
        ground_truth=ground_truth,
        solution=solution_str,
    )
    grm_prompt_ids = await loop.run_in_executor(
        None,
        lambda: reward_model_tokenizer.apply_chat_template(
            [{"role": "user", "content": grm_prompt}],
            tokenize=True,
            add_generation_prompt=True,
        ),
    )
    # transformers >= 5 returns a BatchEncoding here instead of a plain list of
    # ids. It is not JSON serializable, so posting it to the router fails with
    # "Object of type BatchEncoding is not JSON serializable".
    if not isinstance(grm_prompt_ids, list):
        grm_prompt_ids = grm_prompt_ids["input_ids"]
    grm_response = await generate_aiohttp(
        router_address=reward_router_address,
        prompt_ids=grm_prompt_ids,
        sampling_params=GRM_SAMPLING_PARAMS,
    )
    if grm_response is not None:
        try:
            err_location = remove_boxed(last_boxed_only_string(grm_response))
            is_flawed_positive = int(err_location) != -1
        except Exception:
            is_flawed_positive = False

        # WARN, not INFO: this module pins its logger to WARN, so INFO never
        # surfaces and a silent GRM fallback would be indistinguishable from
        # "GRM never called" in the logs.
        logger.warning(
            "FAPO_GRM_CALL sample=%s -> response_len=%d is_flawed_positive=%s",
            extra_info.get("index"),
            len(grm_response),
            is_flawed_positive,
        )

        if is_flawed_positive:
            reward_score -= FLAWED_REWARD_PENALTY
    else:
        logger.warning("FAPO_GRM_CALL sample=%s -> NO_RESPONSE", extra_info.get("index"))

    return {"score": reward_score, "acc": correct, "pred": pred, "is_flawed_positive": is_flawed_positive}
