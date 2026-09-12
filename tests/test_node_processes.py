"""Real node-manager cleanup across process sessions and parent crashes."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SERVER = r'''
import json, os, signal, subprocess, sys, time
from pathlib import Path
root = Path(sys.argv[1])
mode = sys.argv[2]
history = root / 'generations.jsonl'
if history.exists():
    previous = json.loads(history.read_text().splitlines()[-1])
    remaining = [pid for pid in previous if Path(f'/proc/{pid}').exists()]
    (root / 'replacement-check.json').write_text(json.dumps(remaining))

worker = r"""
import json, os, signal, subprocess, sys, time
from pathlib import Path
os.setsid()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
grandchild = subprocess.Popen([sys.executable, '-c',
    'import os,signal,time; os.setsid(); '
    'signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)'])
Path(sys.argv[1]).write_text(json.dumps([os.getpid(), grandchild.pid]))
while True: time.sleep(1)
"""
worker_file = root / f'worker-{os.getpid()}.json'
child = subprocess.Popen([sys.executable, '-c', worker, str(worker_file)])
while not worker_file.exists(): time.sleep(.01)
with history.open('a') as stream:
    stream.write(json.dumps([os.getpid(), *json.loads(worker_file.read_text())]) + '\n')
if mode == 'crash' and len(history.read_text().splitlines()) == 1:
    os._exit(7)
if mode == 'ignore': signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True: time.sleep(1)
'''


def wait_for(check, timeout=8):
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() >= deadline:
            pytest.fail("node lifecycle condition timed out")
        time.sleep(0.02)


@pytest.mark.parametrize("mode", ["normal", "ignore", "crash", "health-failure"])
def test_node_reaps_descendants_that_escape_the_server_session(tmp_path, mode):
    binary = Path(
        os.environ.get(
            "KCORAL_NODE_BIN", Path(__file__).resolve().parents[1] / "target/debug/kcoral-node"
        )
    )
    if not binary.is_file():
        if os.environ.get("KCORAL_REQUIRE_GATEWAY_TESTS") == "1":
            pytest.fail(f"required node binary is missing: {binary}")
        pytest.skip("build kcoral-node to test process ownership")
    server = tmp_path / "server.py"
    server.write_text(SERVER)
    history = tmp_path / "generations.jsonl"
    command = [
        str(binary),
        "--router-endpoint",
        "http://127.0.0.1:1",
        "--node-id",
        "cleanup",
        "--server-url",
        "http://127.0.0.1:1",
        "--termination-grace-seconds",
        "0.1",
        "--health-interval-seconds",
        "0.2",
        "--startup-grace-seconds",
        "0.3" if mode == "health-failure" else "600",
        "--failure-threshold",
        "1",
        "--restart-min-delay-seconds",
        "0.05",
        "--restart-max-delay-seconds",
        "0.05",
        "--restart-jitter",
        "0",
        "--",
        sys.executable,
        str(server),
        str(tmp_path),
        mode,
    ]
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    with (tmp_path / "node.log").open("w") as log:
        node = subprocess.Popen(command, stdout=log, stderr=log)
        try:

            def ready():
                assert node.poll() is None, (tmp_path / "node.log").read_text()
                return history.exists() and bool(history.read_text().strip())

            wait_for(ready)
            if mode in {"crash", "health-failure"}:
                check = tmp_path / "replacement-check.json"
                wait_for(lambda: check.exists())
                assert json.loads(check.read_text()) == []
            node.terminate()
            assert node.wait(timeout=8) == 0, (tmp_path / "node.log").read_text()
            descendants = [
                pid for line in history.read_text().splitlines() for pid in json.loads(line)
            ]
            wait_for(lambda: all(not Path(f"/proc/{pid}").exists() for pid in descendants))
            assert unrelated.poll() is None
        finally:
            if node.poll() is None:
                node.kill()
                node.wait()
            # Test-local fallback cleanup if the implementation regresses.
            if history.exists():
                for line in history.read_text().splitlines():
                    for pid in json.loads(line):
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            unrelated.terminate()
            unrelated.wait()
