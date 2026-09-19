"""Private entry point launched by the bubblewrap worker backend."""

from __future__ import annotations

import os
from multiprocessing.connection import Connection

from . import sandbox


def main() -> None:
    conn = Connection(os.dup(0))
    with open(os.devnull, "rb") as null:
        os.dup2(null.fileno(), 0)
    sandbox.activate()
    # Both sides are trusted KCoral processes. This channel is not an
    # untrusted-code boundary and intentionally uses the existing protocol.
    device, factory, max_requests = conn.recv()
    from .worker import worker_main

    worker_main(
        device,
        conn,
        factory,
        max_requests,
        capture_dir=f"{sandbox.WORKSPACE}/{sandbox.PRIVATE}/output",
        isolated=True,
    )


if __name__ == "__main__":
    main()
