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

"""Pinned 0.23 positive-depth MTP CPU draft transport; no NPU imports at install."""

import hashlib
import inspect
from functools import wraps
from pathlib import Path

from ..backend import ASCEND_RUNNER_SOURCES, VLLM_RUNNER_SHA256

_MARKER = "_verl_mtp_variable_copy_v023"
_SNAPSHOT = "_verl_mtp_cpu_draft_snapshot_v023"
_COPY_SHAS = frozenset(ASCEND_RUNNER_SOURCES)
_READ_SHA = VLLM_RUNNER_SHA256


def _guard(method, expected):
    method = inspect.unwrap(method)
    path = inspect.getsourcefile(method)
    allowed = {expected} if isinstance(expected, str) else expected
    if path is None or hashlib.sha256(Path(path).read_bytes()).hexdigest() not in allowed:
        raise RuntimeError("0.23 draft transport source fingerprint mismatch")
    return method


def patch_copy_and_read(runner_class):
    copy = runner_class._copy_draft_token_ids_to_cpu
    read = runner_class._get_draft_token_ids_cpu
    marked = (getattr(copy, _MARKER, False), getattr(read, _MARKER, False))
    if any(marked):
        if not all(marked):
            raise RuntimeError("Partial 0.23 draft transport installation")
        return False
    copy_source = _guard(copy, _COPY_SHAS)
    _guard(read, _READ_SHA)
    torch = copy_source.__globals__["torch"]

    @wraps(copy)
    def copy_drafts(self, scheduler_output, zeros_only=False):
        # A skipped/failed new copy must not publish the preceding batch's IDs.
        setattr(self, _SNAPSHOT, None)
        self._draft_token_req_ids = None
        if not self.num_spec_tokens:
            return
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests or self.input_batch.sampling_metadata.output_token_ids
        ):
            return
        drafts = self._draft_token_ids
        if drafts is None:
            return
        if not torch.is_tensor(drafts):
            return copy(self, scheduler_output, zeros_only=zeros_only)
        if drafts.ndim != 2:
            raise RuntimeError("Expected two-dimensional physical MTP drafts")
        rows, width = map(int, drafts.shape)
        ids = tuple(self.input_batch.req_ids)
        if rows != len(ids) or len(set(ids)) != len(ids):
            raise RuntimeError("Draft rows and unique request IDs disagree")
        buffer = self.draft_token_ids_cpu
        event, stream = self.draft_token_ids_event, self.draft_token_ids_copy_stream
        if buffer is None or event is None or stream is None:
            raise RuntimeError("Draft CPU transport buffers unavailable")
        if width < 1 or rows > buffer.shape[0] or width > buffer.shape[1]:
            raise RuntimeError("Physical MTP draft exceeds positive-depth buffer contract")
        default_stream = torch.npu.current_stream()
        with torch.npu.stream(stream):
            if zeros_only:
                buffer[:rows, :width] = 0
            else:
                stream.wait_stream(default_stream)
                buffer[:rows, :width].copy_(drafts, non_blocking=True)
            event.record()
        setattr(self, _SNAPSHOT, (ids, width))
        self._draft_token_req_ids = list(ids)

    @wraps(read)
    def read_drafts(self):
        if isinstance(self._draft_token_ids, list):
            return read(self)
        snapshot = getattr(self, _SNAPSHOT, None)
        if snapshot is None:
            if self._draft_token_req_ids is not None:
                raise RuntimeError("Draft IDs published without a completed copy submission")
            return [], []
        ids, width = snapshot
        if tuple(self._draft_token_req_ids or ()) != ids:
            raise RuntimeError("Draft request IDs changed after copy submission")
        buffer, event = self.draft_token_ids_cpu, self.draft_token_ids_event
        if buffer is None or event is None or len(ids) > buffer.shape[0] or width > buffer.shape[1]:
            raise RuntimeError("Draft CPU buffer changed after copy submission")
        event.synchronize()
        return buffer[: len(ids), :width].tolist(), list(ids)

    setattr(copy_drafts, _MARKER, True)
    setattr(read_drafts, _MARKER, True)
    runner_class._copy_draft_token_ids_to_cpu = copy_drafts
    runner_class._get_draft_token_ids_cpu = read_drafts
    return True
