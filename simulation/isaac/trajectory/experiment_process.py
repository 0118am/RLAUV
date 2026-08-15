"""Subprocess execution and display helpers for experiment tooling."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

def configure_plots() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.figsize": (10, 5), "axes.grid": True, "grid.alpha": 0.25})


def shell_join(command: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _display_path(path: str, cwd: Path) -> str:
    """Render an absolute path relative to the subprocess working directory."""

    candidate = Path(path)
    if not candidate.is_absolute():
        return path
    try:
        return os.path.relpath(candidate, start=cwd)
    except ValueError:
        # Different Windows drives cannot be relativized. Keep the executable
        # command valid while limiting this fallback to that platform edge case.
        return path


def display_command(command: Sequence[object], *, cwd: Path) -> str:
    """Format a command for notebooks without leaking long absolute paths.

    The returned value is presentation-only. ``run_command`` always receives
    the original absolute values, which keeps profile/checkpoint resolution
    independent of the notebook's working directory.
    """

    display_parts: list[str] = []
    for value in command:
        part = str(value)
        name, separator, assigned_value = part.partition("=")
        if separator and assigned_value.startswith(os.path.sep):
            display_parts.append(f"{name}={_display_path(assigned_value, cwd)}")
        else:
            display_parts.append(_display_path(part, cwd))
    return shell_join(display_parts)


def run_command(
    command: Sequence[object],
    *,
    cwd: Path,
    execute: bool = False,
    label: str | None = None,
    extra_env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> int | None:
    """Preview or run a subprocess, optionally keeping verbose output out of a notebook."""

    normalized = [str(part) for part in command]
    displayed_command = display_command(normalized, cwd=cwd)
    if label:
        print(f"[{label}]")
    print(displayed_command)
    if not execute:
        return None

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("TERM", "xterm")
    if extra_env:
        env.update(extra_env)

    def terminate_process_group(process: subprocess.Popen) -> None:
        """Stop a command and all of its children after an interrupted caller."""

        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        except ProcessLookupError:
            pass

    started = time.time()
    process: subprocess.Popen | None = None
    try:
        if log_path is None:
            process = subprocess.Popen(
                normalized,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
            return_code = process.wait()
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"\n[{label or 'COMMAND'}]\n{shell_join(normalized)}\n")
                process = subprocess.Popen(
                    normalized,
                    cwd=str(cwd),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                return_code = process.wait()
    except KeyboardInterrupt:
        if process is not None:
            terminate_process_group(process)
        print("\n[interrupted] terminated command process group")
        raise
    print(f"\n[exit={return_code}] elapsed={(time.time() - started) / 60.0:.1f} min")
    if return_code != 0:
        if log_path is not None and log_path.exists():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-8_000:]
            print(f"\n[last output from {_display_path(str(log_path), cwd)}]\n{tail}")
        raise RuntimeError(f"Command failed with exit code {return_code}: {displayed_command}")
    return return_code

