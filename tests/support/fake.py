"""Small harnesses for protocol and process-lifecycle tests."""

import os
import time

from kcoral.errors import ExecutionError


def opaque(*args):
    return object()


def structural(*args):
    return {"ok": True, "values": [None, False, 7, 1.5, "text"]}


def binary(*args):
    return b"binary-result"


def sleep(seconds=0.0, *args):
    time.sleep(float(seconds))
    return {"slept": float(seconds)}


cpu_sleep = sleep


def crash(*args):
    os._exit(1)


def crash_after_output(*args):
    os.write(2, b"fatal: simulated device-side assert\n")
    os._exit(1)


def nope():
    raise ExecutionError("runtime", "deliberate test failure")


missing = nope
