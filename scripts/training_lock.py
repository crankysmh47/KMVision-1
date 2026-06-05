"""Prevent multiple Phase C training jobs on the same GPU."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "logs" / "week_queue" / "training.lock"


def _pid_alive(pid: int) -> bool:
    """Windows-safe process liveness check (os.kill(pid, 0) is unreliable on Windows)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_lock() -> tuple[int | None, str | None]:
    if not LOCK_PATH.is_file():
        return None, None
    try:
        text = LOCK_PATH.read_text(encoding="utf-8").strip().splitlines()
        pid = int(text[0].split(":", 1)[1].strip()) if text else 0
        cmd = text[1].split(":", 1)[1].strip() if len(text) > 1 else ""
        return pid, cmd
    except (ValueError, IndexError):
        return None, None


def clear_stale_lock() -> bool:
    """Remove lock file if the owning process is gone. Returns True if cleared."""
    pid, _ = read_lock()
    if pid is None:
        return False
    if _pid_alive(pid):
        return False
    LOCK_PATH.unlink(missing_ok=True)
    return True


def acquire_lock(*, pid: int, label: str) -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    clear_stale_lock()
    existing_pid, existing_cmd = read_lock()
    if existing_pid and _pid_alive(existing_pid) and existing_pid != pid:
        raise RuntimeError(
            f"Training already running (pid={existing_pid}, {existing_cmd}). "
            f"Stop it before starting another job."
        )
    LOCK_PATH.write_text(f"pid: {pid}\ncmd: {label}\n", encoding="utf-8")


def release_lock(*, pid: int) -> None:
    if not LOCK_PATH.is_file():
        return
    locked_pid, _ = read_lock()
    if locked_pid in (None, pid):
        LOCK_PATH.unlink(missing_ok=True)


def stale_lock_message() -> str | None:
    pid, cmd = read_lock()
    if pid is None:
        return None
    if _pid_alive(pid):
        return f"Training lock held by pid={pid} ({cmd})"
    return f"Stale training lock (pid={pid} dead). Removing {LOCK_PATH}"
