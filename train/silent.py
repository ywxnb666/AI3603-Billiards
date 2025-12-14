import contextlib
import os
import sys


@contextlib.contextmanager
def suppress_output(enabled: bool = True):
    """Redirect stdout/stderr to nul to silence noisy training loops."""
    if not enabled:
        yield
        return

    devnull_path = os.devnull
    with open(devnull_path, "w", encoding="utf-8") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
