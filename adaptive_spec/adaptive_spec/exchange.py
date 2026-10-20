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

"""Low-frequency exchange of compact completed-response evidence."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import uuid
from pathlib import Path

from .evidence import AcceptanceEvidence, ResponseSummary


class CrossEngineEvidenceExchange:
    """Share counts, never prompt text, through one append-only file per process."""

    def __init__(
        self,
        directory: str | Path,
        *,
        poll_decisions: int = 64,
        source_id: str | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.poll_decisions = max(1, int(poll_decisions))
        self.source_id = source_id or (f"{socket.gethostname()}-pid{os.getpid()}-{uuid.uuid4().hex}")
        file_id = hashlib.sha256(self.source_id.encode("utf-8")).hexdigest()[:24]
        self.path = self.directory / f"source-{file_id}.jsonl"
        self._offsets: dict[str, int] = {}
        self._decision_count = 0

    def publish(
        self,
        key: str,
        response_id: str,
        summary: ResponseSummary,
    ) -> None:
        record = {
            "schema_version": 1,
            "source_id": self.source_id,
            "key": str(key),
            "response_id": str(response_id),
            "exposures": list(summary.exposures),
            "successes": list(summary.successes),
            "block_count": int(summary.block_count),
        }
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")) + chr(10)
        with self.path.open("ab") as stream:
            stream.write(encoded.encode("utf-8"))

    def maybe_poll(self, evidence: AcceptanceEvidence) -> None:
        self._decision_count += 1
        if self._decision_count != 1 and self._decision_count % self.poll_decisions != 0:
            return
        for source in sorted(self.directory.glob("source-*.jsonl")):
            if source == self.path:
                continue
            source_key = str(source)
            offset = self._offsets.get(source_key, 0)
            size = source.stat().st_size
            if size < offset:
                offset = 0
            with source.open("rb") as stream:
                stream.seek(offset)
                while True:
                    position = stream.tell()
                    raw = stream.readline()
                    if not raw:
                        break
                    if raw[-1:] != bytes((10,)):
                        stream.seek(position)
                        break
                    record = json.loads(raw)
                    if int(record.get("schema_version", 0)) != 1:
                        raise ValueError("unsupported evidence exchange schema")
                    evidence.commit_summary(
                        str(record["key"]),
                        str(record["response_id"]),
                        exposures=record["exposures"],
                        successes=record["successes"],
                        block_count=int(record["block_count"]),
                    )
                self._offsets[source_key] = stream.tell()
