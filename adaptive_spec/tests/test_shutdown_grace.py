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

from adaptive_spec.vllm_plugin import shutdown_v0_23 as m


def test_workers_share_one_cleanup_deadline_and_native_fallback(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(m.time, "monotonic", lambda: now[0])
    waits, native = [], []

    class Worker:
        def join(self, timeout):
            waits.append(timeout)
            now[0] += 21

    workers = [Worker(), Worker()]

    class cls:
        _ensure_worker_termination = staticmethod(lambda procs: native.append(procs))

    m.patch_executor(cls)
    first = cls._ensure_worker_termination
    m.patch_executor(cls)
    assert cls._ensure_worker_termination is first
    cls._ensure_worker_termination(workers)
    assert waits == [30, 9]
    assert native == [workers]


def test_real_worker_exits_before_native_termination():
    import multiprocessing

    worker = multiprocessing.Process(target=print, args=("cleanup test",))
    worker.start()
    seen = []

    class Executor:
        @staticmethod
        def _ensure_worker_termination(workers):
            seen.extend((p.is_alive(), p.exitcode) for p in workers)

    m.patch_executor(Executor)
    Executor()._ensure_worker_termination([worker])
    assert seen == [(False, 0)]
