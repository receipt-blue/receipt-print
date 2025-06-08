import fcntl
import os
import pty
import select
import struct
import subprocess
import termios
from typing import Iterable

from .printer import sanitize_output


def run_command_standard(cmd: str) -> str:
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        out = res.stdout + (("\n" + res.stderr) if res.stderr else "")
        return sanitize_output(out).rstrip()
    except Exception as e:
        return f"Error running '{cmd}': {e}"


def run_command_in_wrapped_tty(cmd: str, columns: int) -> str:
    """run `cmd` in a pseudo-tty sized to `columns`"""
    master_fd, slave_fd = pty.openpty()
    rows = 999
    fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        text=True,
        close_fds=True,
    )
    chunks = []
    while proc.poll() is None:
        rlist, _, _ = select.select([master_fd], [], [], 0.1)
        if master_fd in rlist:
            data = os.read(master_fd, 1024)
            if not data:
                break
            chunks.append(data.decode("utf-8", "replace"))
    while True:
        rlist, _, _ = select.select([master_fd], [], [], 0)
        if not rlist:
            break
        data = os.read(master_fd, 1024)
        if not data:
            break
        chunks.append(data.decode("utf-8", "replace"))
    os.close(master_fd)
    try:
        os.close(slave_fd)
    except OSError:
        pass
    return sanitize_output("".join(chunks)).rstrip()


def run_shell_commands(
    commands: Iterable[str], wrap_tty: bool = True, columns: int = 80
) -> str:
    pairs = []
    for cmd in commands:
        if wrap_tty:
            out = run_command_in_wrapped_tty(cmd, columns)
        else:
            out = run_command_standard(cmd)
        pairs.append(f"$ {cmd}\n{out}")
    return "\n\n\n".join(pairs)
