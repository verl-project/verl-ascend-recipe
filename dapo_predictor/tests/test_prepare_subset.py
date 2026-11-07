from unittest.mock import Mock

import pytest

from dapo_predictor.prepare_subset import prompt_token_count


@pytest.mark.parametrize("output", [[1, 2, 3], {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}])
def test_prompt_cap_counts_token_ids_not_batch_encoding_keys(output):
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = output
    messages = [{"role": "user", "content": "Example question"}]
    assert prompt_token_count(tokenizer, messages) == 3
    tokenizer.apply_chat_template.assert_called_once_with(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
