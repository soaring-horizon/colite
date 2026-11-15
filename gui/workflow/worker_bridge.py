"""Explicit WSL adapter. Own one CLI process group; never invoke a shell.

The desktop sends a heartbeat and abort request using a unique task control
file. Loss of the desktop heartbeat terminates the group instead of leaving
orphan reconstruction workers. Pause is handled between CLI steps by Qt.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing CLI program")
    control = Path(args.control)
    aborted = False

    def request_abort(*_):
        nonlocal aborted
        aborted = True

    signal.signal(signal.SIGTERM, request_abort)
    signal.signal(signal.SIGINT, request_abort)
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        while child.poll() is None:
            try:
                stale = time.time() - control.stat().st_mtime > 8
                aborted |= stale or control.read_text(encoding="utf-8") == "abort"
            except OSError:
                aborted = True
            if aborted:
                break
            time.sleep(0.1)
        if aborted:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            return 130
        return child.wait()
    finally:
        # Reap remaining descendants of this worker, including on bridge errors.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
