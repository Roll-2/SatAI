"""Launch a project command fully detached (double fork, new session).

Usage::

    .venv/bin/python Tools/detached_runner.py <logfile> <module> [args...]

The child escapes the calling process group (os.setsid + second fork), so it
survives the parent terminal closing. Logs go to <logfile>.
"""

import os
import sys


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: detached_runner.py <logfile> <module> [args...]")
        sys.exit(2)

    logfile, module, extra = sys.argv[1], sys.argv[2], sys.argv[3:]

    if os.fork() > 0:
        sys.exit(0)          # parent exits immediately
    os.setsid()              # child becomes session leader
    if os.fork() > 0:
        sys.exit(0)          # intermediate exits -> grandchild reparented

    log = open(logfile, "ab", buffering=0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)

    python = sys.executable
    os.execv(python, [python, "-u", "-m", module, *extra])


if __name__ == "__main__":
    main()
