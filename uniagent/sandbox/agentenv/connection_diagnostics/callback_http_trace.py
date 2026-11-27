"""Bounded, passive HTTP/1 framing observation; never changes forwarded bytes."""

import re
from collections import deque


class Framing:
    def __init__(self, headers, body, end):
        self.headers, self.body, self.end = headers, body, end
        self.buffer = bytearray()
        self.state = "headers"
        self.remaining = 0

    def feed(self, data):
        if self.state == "unknown":
            return
        self.buffer.extend(data)
        while True:
            if self.state in ("headers", "chunk", "trailers"):
                marker = b"\r\n\r\n" if self.state == "headers" else b"\r\n"
                i = self.buffer.find(marker)
                if i < 0:
                    if len(self.buffer) > 65536:
                        raise ValueError("header or framing line exceeds limit")
                    return
                line = bytes(self.buffer[:i])
                del self.buffer[: i + len(marker)]
                if self.state == "headers":
                    lines = line.split(b"\r\n")
                    fields = {}
                    for item in lines[1:]:
                        k, sep, v = item.partition(b":")
                        if sep:
                            fields[k.lower()] = v.strip().lower()
                    response = lines[0].startswith(b"HTTP/")
                    status = int(lines[0].split()[1]) if response else None
                    self.headers(lines[0], fields)
                    if response and 100 <= status < 200:
                        continue
                    if fields.get(b"transfer-encoding", b"").endswith(b"chunked"):
                        self.state = "chunk"
                    elif b"content-length" in fields:
                        self.remaining = int(fields[b"content-length"])
                        self.state = "fixed"
                    elif not response or status in (204, 304):
                        self.remaining = 0
                        self.state = "fixed"
                    else:
                        self.state = "close"
                elif self.state == "chunk":
                    self.remaining = int(line.split(b";")[0], 16)
                    self.state = "chunk_body" if self.remaining else "trailers"
                elif not line:
                    self.end()
                    self.state = "headers"
                continue
            if self.state in ("fixed", "chunk_body"):
                n = min(self.remaining, len(self.buffer))
                if n:
                    self.body(bytes(self.buffer[:n]))
                    del self.buffer[:n]
                    self.remaining -= n
                if self.remaining:
                    return
                if self.state == "fixed":
                    self.end()
                    self.state = "headers"
                else:
                    self.state = "chunk_crlf"
                continue
            if self.state == "chunk_crlf":
                if len(self.buffer) < 2:
                    return
                if self.buffer[:2] != b"\r\n":
                    raise ValueError("chunk framing")
                del self.buffer[:2]
                self.state = "chunk"
                continue
            if self.state == "close":
                self.body(bytes(self.buffer))
                self.buffer.clear()
                return


class HTTPTrace:
    def __init__(self, emit):
        self.emit = emit
        self.pending = deque()
        self.sequence = 0
        self.active = None
        self.tail = b""
        self.last = None
        self.failed = False
        self.request = Framing(self.request_headers, lambda b: None, lambda: None)
        self.response = Framing(self.response_headers, self.response_body, self.response_end)

    def request_headers(self, line, fields):
        self.sequence += 1
        parts = line.split()
        path = parts[1].split(b"?")[0] if len(parts) > 1 else b""
        match = re.fullmatch(
            rb"/sessions/([A-Za-z0-9_-]+)/v1/(messages(?:/count_tokens)?|chat/completions|responses)", path
        )
        request = dict(
            request_seq=self.sequence,
            session_id=match[1].decode() if match else None,
            endpoint=match[2].decode() if match else "other",
            status=None,
            response_complete=False,
            message_stop=False,
        )
        if len(self.pending) >= 128:
            raise ValueError("too many pending requests")
        self.pending.append(request)
        self.emit("http_request", **request)

    def response_headers(self, line, fields):
        status = int(line.split()[1])
        if 100 <= status < 200:
            return
        self.active = self.pending[0] if self.pending else dict(request_seq=None, session_id=None)
        self.active.update(status=status, response_complete=False, message_stop=False)
        self.tail = b""
        self.emit("http_response_start", **self.active)

    def response_body(self, data):
        # Detect SSE control events only, never retain or log generated content.
        data = self.tail + data
        if self.active is not None and re.search(rb"(?:^|\n)event:\s*message_stop\s*\r?\n", data):
            self.active["message_stop"] = True
        self.tail = data[-64:]

    def response_end(self):
        if self.active is None:
            return
        self.active["response_complete"] = True
        self.emit("http_response_end", **self.active)
        self.last = dict(self.active)
        if self.pending:
            self.pending.popleft()
        self.active = None
        self.tail = b""

    def feed(self, direction, data):
        if self.failed:
            return
        try:
            (self.request if direction == "client_to_upstream" else self.response).feed(data)
        except Exception as exc:
            self.failed = True
            self.request.buffer.clear()
            self.response.buffer.clear()
            self.emit("http_trace_unavailable", error_type=type(exc).__name__)

    def eof(self, direction):
        if direction == "upstream_to_client" and self.response.state == "close" and not self.failed:
            self.response_end()
        self.emit(
            "http_eof",
            direction=direction,
            pending_requests=len(self.pending),
            trace_available=not self.failed,
            request=self.active or (self.pending[0] if self.pending else self.last),
        )
