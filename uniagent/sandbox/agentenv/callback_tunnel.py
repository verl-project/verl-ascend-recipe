"""Own one reverse SSH forward per gateway, scoped to GatewayManager lifetime."""

import asyncio
import atexit
import logging
import os
import socket as socket_module
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger("uni_agent.gateway.ssh")


class CallbackTunnels:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.routes = {}
        self.processes = {}
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        self.log_dir = (
            Path(__file__).resolve().parent
            / "artifacts"
            / "ssh"
            / (stamp + "-" + socket_module.gethostname() + "-" + str(os.getpid()))
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_files = {}
        self.log_threads = {}
        self.directory = tempfile.TemporaryDirectory(prefix="cb.", dir=os.environ["AGENTENV_CONTROL_DIR"])
        atexit.register(self.close)

    def ssh(self, socket):
        auth = ["sshpass", "-e"] if os.environ.get("SSHPASS") else []
        return auth + [
            "ssh",
            "-v",
            "-S",
            str(socket),
            "-p",
            os.environ.get("AGENTENV_RELAY_SSH_PORT", "22"),
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "UserKnownHostsFile=" + os.environ.get("SSH_KNOWN_HOSTS", str(Path.home() / ".ssh/known_hosts")),
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
        ]

    def establish(self, index, upstream):
        target = os.environ["AGENTENV_RELAY_TARGET"]
        socket = Path(self.directory.name) / str(index)
        if index in self.routes:
            if self.routes[index] != upstream:
                raise RuntimeError("Gateway address changed during training")
            subprocess.run(self.ssh(socket) + ["-O", "check", target], check=True, capture_output=True, timeout=20)
            return
        port = int(os.environ.get("AGENTENV_RELAY_CALLBACK_PORT", "30193")) + index
        log_file = (self.log_dir / f"gateway-{port}.control.log").open("ab", buffering=0)
        self.log_files[index] = log_file
        # Hold remote cat's stdin open in this actor. If the actor dies, EOF
        # terminates the SSH session and releases its forwards (unlike -f -N).
        process = subprocess.Popen(
            self.ssh(socket) + ["-M", "-T", "-o", "ControlPersist=no", target, "cat >/dev/null"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        def save_stderr():
            with (self.log_dir / f"gateway-{port}.ssh.log").open("a", encoding="utf-8", buffering=1) as output:
                for line in iter(process.stderr.readline, b""):
                    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")
                    output.write(stamp + " " + line.decode("utf-8", errors="replace"))
                output.write(datetime.now(timezone(timedelta(hours=8))).isoformat() + " stderr EOF\n")

        thread = threading.Thread(target=save_stderr, name=f"callback-ssh-log-{index}", daemon=True)
        self.log_threads[index] = thread
        thread.start()
        logger.info(
            "callback SSH started index=%s port=%s upstream=%s pid=%s logs=%s",
            index,
            port,
            upstream,
            process.pid,
            self.log_dir,
        )
        self.processes[index] = process
        try:
            deadline = time.monotonic() + 25
            while not socket.exists():
                if process.poll() is not None:
                    raise RuntimeError(f"Callback SSH connection exited with status {process.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Callback SSH control socket did not become ready")
                time.sleep(0.1)
            # A successful control command confirms the listener was bound.
            subprocess.run(
                self.ssh(socket) + ["-O", "forward", "-R", f"127.0.0.1:{port}:{upstream}", target],
                check=True,
                timeout=20,
                stderr=log_file,
            )
            self.routes[index] = upstream
        except BaseException:
            self._stop_process(index)
            raise

    def _stop_process(self, index):
        process = self.processes.pop(index, None)
        if process is None:
            return
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        logger.info("callback SSH stopped index=%s pid=%s exit_code=%s", index, process.pid, process.returncode)
        thread = self.log_threads.pop(index, None)
        if thread:
            thread.join(timeout=3)
        log_file = self.log_files.pop(index, None)
        if log_file:
            log_file.close()

    async def route(self, index, handle):
        url = urlsplit(handle.base_url)
        async with self.lock:
            await asyncio.to_thread(self.establish, index, url.netloc)
        port = int(os.environ.get("AGENTENV_CALLBACK_PORT", "40193")) + index
        handle.base_url = urlunsplit(
            ("http", f"{os.environ['AGENTENV_HOST']}:{port}", url.path, url.query, url.fragment)
        )
        return handle

    def close(self):
        target = os.environ["AGENTENV_RELAY_TARGET"]
        for index in list(self.routes):
            try:
                subprocess.run(
                    self.ssh(Path(self.directory.name) / str(index)) + ["-O", "exit", target],
                    capture_output=True,
                    timeout=20,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
            self.routes.pop(index, None)
        for index in list(self.processes):
            self._stop_process(index)
        self.directory.cleanup()


async def route_session(manager, index, handle):
    if os.environ.get("AGENTENV_CALLBACK_ENABLED") != "1":
        return handle
    if not hasattr(manager, "_agentenv_tunnels"):
        manager._agentenv_tunnels = CallbackTunnels()
    return await manager._agentenv_tunnels.route(index, handle)


async def close_tunnels(manager):
    if hasattr(manager, "_agentenv_tunnels"):
        await asyncio.to_thread(manager._agentenv_tunnels.close)
