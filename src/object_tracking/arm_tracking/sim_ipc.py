"""Nonblocking newest-only process pipes for the simulator planner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import threading
import time
from typing import Sequence

from .sim_closed_loop import SimCommand, SimState, decode_message, encode_message


@dataclass(frozen=True)
class PlannerClientMetrics:
    states_submitted: int
    pending_states_replaced: int
    commands_received: int
    stale_commands: int
    process_starts: int
    last_error: str | None


class LatestPlannerProcess:
    """Run one isolated planner worker per episode without blocking PhysX."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
    ) -> None:
        self.command = tuple(str(value) for value in command)
        if not self.command:
            raise ValueError("planner worker command is required")
        self.cwd = None if cwd is None else str(cwd)
        self._condition = threading.Condition()
        self._pending: SimState | None = None
        self._latest: SimCommand | None = None
        self._stop = False
        self._process: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._states_submitted = 0
        self._pending_states_replaced = 0
        self._commands_received = 0
        self._stale_commands = 0
        self._process_starts = 0
        self._last_error: str | None = None

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._stop = False
            self._process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
            )
            self._process_starts += 1
            self._thread = threading.Thread(
                target=self._run,
                name="sim-planner-process",
                daemon=True,
            )
            self._thread.start()

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
            process = self._process
            thread = self._thread
        if process is not None and process.stdin is not None:
            process.stdin.close()
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        if thread is not None:
            thread.join(timeout=1.0)
        with self._condition:
            self._process = None
            self._thread = None

    def submit(self, state: SimState) -> SimCommand | None:
        """Queue one state and return the newest response without blocking."""

        with self._condition:
            if self._thread is None:
                raise RuntimeError("planner process is not started")
            if self._pending is not None:
                self._pending_states_replaced += 1
            self._pending = state
            self._states_submitted += 1
            self._condition.notify()
            return self._latest

    def submit_and_wait(self, state: SimState, *, timeout_s: float) -> SimCommand:
        """Submit one state and wait for its matching response.

        This is intended for episode startup, while physics is paused. Runtime
        updates should continue to use :meth:`submit` so PhysX never blocks on
        the planner.
        """

        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + timeout_s
        self.submit(state)
        with self._condition:
            while True:
                if self._latest is not None and self._latest.state_sequence == state.sequence:
                    return self._latest
                if self._last_error is not None:
                    raise RuntimeError(f"planner startup failed: {self._last_error}")
                process = self._process
                if process is None or (process.poll() is not None and not self._stop):
                    return_code = None if process is None else process.returncode
                    raise RuntimeError(f"planner exited before responding: {return_code}")
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    raise TimeoutError(
                        f"planner did not respond to state {state.sequence} "
                        f"within {timeout_s:.3f}s"
                    )
                self._condition.wait(timeout=remaining_s)

    def latest_command(self) -> SimCommand | None:
        with self._condition:
            return self._latest

    def metrics(self) -> PlannerClientMetrics:
        with self._condition:
            return PlannerClientMetrics(
                states_submitted=self._states_submitted,
                pending_states_replaced=self._pending_states_replaced,
                commands_received=self._commands_received,
                stale_commands=self._stale_commands,
                process_starts=self._process_starts,
                last_error=self._last_error,
            )

    def _next_state(self) -> SimState | None:
        with self._condition:
            self._condition.wait_for(lambda: self._stop or self._pending is not None)
            if self._stop:
                return None
            state = self._pending
            self._pending = None
            return state

    def _run(self) -> None:
        process = self._process
        assert process is not None and process.stdin is not None and process.stdout is not None
        while True:
            state = self._next_state()
            if state is None:
                break
            try:
                process.stdin.write(encode_message(state))
                process.stdin.flush()
                payload = process.stdout.readline(64 * 1024 + 1)
                if not payload or len(payload) > 64 * 1024:
                    raise ConnectionError("planner returned an empty or oversized response")
                decoded = decode_message(payload)
                if not isinstance(decoded, SimCommand):
                    raise ValueError("planner response is not a command")
                with self._condition:
                    if decoded.state_sequence != state.sequence:
                        self._stale_commands += 1
                    else:
                        self._latest = decoded
                        self._commands_received += 1
                    self._last_error = None
                    self._condition.notify_all()
            except (BrokenPipeError, OSError, ValueError) as exc:
                with self._condition:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    self._condition.notify_all()
                break
