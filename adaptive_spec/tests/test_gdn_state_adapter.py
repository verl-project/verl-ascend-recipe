# Copyright 2026
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

from types import SimpleNamespace

from adaptive_spec.vllm_plugin import gdn_state_v0_23 as m


def test_method_override_does_not_modify_upstream_torch():
    calls = []

    def native(**kw):
        calls.append(("native", kw))
        return "native"

    def adapted(**kw):
        calls.append(("adapted", kw))
        return "adapted"

    original_torch = SimpleNamespace(
        ops=SimpleNamespace(_C_ascend=SimpleNamespace(npu_recurrent_gated_delta_rule=native, other=lambda: "other")),
        sentinel=7,
    )
    namespace = {"torch": original_torch}
    exec(
        "def core(self, value=9, *, flag=True):\n"
        " return (torch.ops._C_ascend.npu_recurrent_gated_delta_rule(value=value,flag=flag),\n"
        "         torch.ops._C_ascend.other(),torch.sentinel)",
        namespace,
    )
    original = namespace["core"]
    patched = m.scoped_recurrent_adapter(original, adapted)
    assert patched(None) == ("adapted", "other", 7)
    assert original(None) == ("native", "other", 7)
    assert original_torch.ops._C_ascend.npu_recurrent_gated_delta_rule is native
    assert patched.__defaults__ == original.__defaults__ and patched.__kwdefaults__ == original.__kwdefaults__
    assert patched.__wrapped__ is original


def test_non_speculative_call_keeps_original_path():
    calls = []

    def native(**kw):
        calls.append(kw)
        return 73

    adapted = m.make_recurrent_adapter(native)
    assert adapted(query="q", state="s") == 73
    assert calls == [{"query": "q", "state": "s"}]


def test_empty_speculative_batch_keeps_original_path():
    accepted = SimpleNamespace(numel=lambda: 0)
    calls = []

    def native(**kw):
        calls.append(kw)
        return 74

    assert m.make_recurrent_adapter(native)(num_accepted_tokens=accepted) == 74
    assert calls[0]["num_accepted_tokens"] is accepted


def test_invalid_state_table_shape_fails_before_state_mutation():
    accepted = SimpleNamespace(numel=lambda: 3)
    indices = SimpleNamespace(numel=lambda: 8)
    import pytest

    with pytest.raises(ValueError, match="rectangular"):
        m.make_recurrent_adapter(lambda **kw: None)(num_accepted_tokens=accepted, ssm_state_indices=indices)


def test_adapter_updates_method_copied_to_qwen_before_install():
    namespace = {
        "torch": SimpleNamespace(
            ops=SimpleNamespace(_C_ascend=SimpleNamespace(npu_recurrent_gated_delta_rule=lambda **kw: "native"))
        )
    }
    exec("def core(self): return torch.ops._C_ascend.npu_recurrent_gated_delta_rule()", namespace)
    source = type("Ascend", (), {"_forward_core": namespace["core"]})
    qwen = type("Qwen", (), {"_forward_core": source._forward_core})
    assert m.bind_recurrent_adapter(source, (qwen,), lambda **kw: "adapted")
    assert qwen()._forward_core() == "adapted"
    assert source()._forward_core() == "adapted"
    assert not m.bind_recurrent_adapter(source, (qwen,), lambda **kw: "unused")


def test_adapter_repairs_late_stale_alias_without_replacing_other_methods():
    namespace = {
        "torch": SimpleNamespace(
            ops=SimpleNamespace(_C_ascend=SimpleNamespace(npu_recurrent_gated_delta_rule=lambda **kw: "native"))
        )
    }
    exec("def core(self): return torch.ops._C_ascend.npu_recurrent_gated_delta_rule()", namespace)
    source = type("Ascend", (), {"_forward_core": namespace["core"]})
    unrelated = type("Other", (), {"_forward_core": lambda self: "other"})
    m.bind_recurrent_adapter(source, (unrelated,), lambda **kw: "adapted")
    qwen = type("Qwen", (), {"_forward_core": namespace["core"]})
    assert m.bind_recurrent_adapter(source, (qwen,), lambda **kw: "unused")
    assert qwen()._forward_core() == "adapted"
    assert unrelated()._forward_core() == "other"
