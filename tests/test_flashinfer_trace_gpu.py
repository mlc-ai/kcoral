"""Real HTTP and GPU integration coverage for FlashInfer Trace evaluation."""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

from flashinfer_trace import (
    BenchmarkConfig,
    BuildSpec,
    Definition,
    EvaluationStatus,
    FlashInferTraceClient,
    RandomInput,
    SafetensorsInput,
    Solution,
    SourceFile,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("BENCH_GPU_TEST") != "1",
    reason="real FlashInfer Trace integration requires BENCH_GPU_TEST=1",
)


def _available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.fixture(scope="module")
def gpu_server(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Path]:
    port = _available_port()
    worktree = Path(__file__).parents[1]
    log_directory = tmp_path_factory.mktemp("flashinfer-trace-server-logs")
    environment = os.environ.copy()
    source_path = str(worktree / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else f"{source_path}{os.pathsep}{environment['PYTHONPATH']}"
    )
    visible_gpu = environment.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    gpu_id = visible_gpu if visible_gpu.isdigit() else "0"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "benchmark_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--gpus",
            gpu_id,
            "--workers-per-gpu",
            "1",
            "--log-dir",
            str(log_directory),
        ],
        cwd=worktree,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"benchmark server exited during startup: {process.stderr.read()}")
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    else:
        process.terminate()
        pytest.fail("benchmark server did not become healthy")

    try:
        yield url, log_directory
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _safetensors_bytes(values: np.ndarray) -> bytes:
    payload = np.ascontiguousarray(values, dtype=np.float32).tobytes()
    header = json.dumps(
        {
            "stored": {
                "dtype": "F32",
                "shape": list(values.shape),
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode()
    return struct.pack("<Q", len(header)) + header + payload


def _solutions() -> list[Solution]:
    cuda_source = """
__global__ void add_kernel(const float* random, const float* stored, float* output, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) output[index] = random[index] + stored[index];
}

void run(tvm::ffi::TensorView random, tvm::ffi::TensorView stored,
         tvm::ffi::TensorView output) {
  int n = static_cast<int>(output.numel());
  add_kernel<<<(n + 255) / 256, 256>>>(static_cast<const float*>(random.data_ptr()),
      static_cast<const float*>(stored.data_ptr()), static_cast<float*>(output.data_ptr()), n);
}
"""
    triton_source = """
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(random, stored, output, length: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.arange(0, block_size)
    mask = offsets < length
    tl.store(output + offsets, tl.load(random + offsets, mask=mask)
             + tl.load(stored + offsets, mask=mask), mask=mask)

def run(random, stored):
    output = torch.empty_like(random)
    add_kernel[(1,)](random, stored, output, length=random.numel(), block_size=256)
    return output
"""
    return [
        Solution(
            name="python_add",
            definition="vector_add",
            author="tests",
            spec=BuildSpec(
                language="python",
                target_hardware=["cuda"],
                entry_point="solution.py::run",
                destination_passing_style=False,
            ),
            sources=[
                SourceFile(
                    path="solution.py",
                    content="def run(random, stored):\n    return random + stored\n",
                )
            ],
        ),
        Solution(
            name="triton_add",
            definition="vector_add",
            author="tests",
            spec=BuildSpec(
                language="triton",
                target_hardware=["cuda"],
                entry_point="solution.py::run",
                destination_passing_style=False,
            ),
            sources=[SourceFile(path="solution.py", content=triton_source)],
        ),
        Solution(
            name="cuda_add",
            definition="vector_add",
            author="tests",
            spec=BuildSpec(
                language="cuda",
                target_hardware=["cuda"],
                entry_point="solution.cu::run",
                destination_passing_style=True,
                binding="tvm-ffi",
            ),
            sources=[SourceFile(path="solution.cu", content=cuda_source)],
        ),
        Solution(
            name="incorrect_add",
            definition="vector_add",
            author="tests",
            spec=BuildSpec(
                language="python",
                target_hardware=["cuda"],
                entry_point="solution.py::run",
                destination_passing_style=False,
            ),
            sources=[
                SourceFile(
                    path="solution.py",
                    content="def run(random, stored):\n    return random - stored\n",
                )
            ],
        ),
    ]


def _write_trace_set(root: Path) -> TraceSet:
    definition = Definition(
        name="vector_add",
        op_type="elementwise",
        axes={"length": {"type": "var"}},
        inputs={
            "random": TensorSpec(shape=["length"], dtype="float32"),
            "stored": TensorSpec(shape=["length"], dtype="float32"),
        },
        outputs={"output": TensorSpec(shape=["length"], dtype="float32")},
        reference="def run(random, stored):\n    return random + stored\n",
        constraints=["length == 256"],
    )
    workloads = [
        Workload(
            axes={"length": 256},
            inputs={
                "random": RandomInput(),
                "stored": SafetensorsInput(
                    path="resources/inputs.safetensors",
                    tensor_key="stored",
                ),
            },
            uuid=f"gpu-workload-{index}",
        )
        for index in range(2)
    ]
    definition_path = root / "definitions" / "vector_add.json"
    workload_path = root / "workloads" / "vector_add.jsonl"
    resource_path = root / "resources" / "inputs.safetensors"
    definition_path.parent.mkdir(parents=True)
    workload_path.parent.mkdir(parents=True)
    resource_path.parent.mkdir(parents=True)
    definition_path.write_text(definition.model_dump_json(), encoding="utf-8")
    workload_path.write_text(
        "".join(
            Trace(definition=definition.name, workload=workload).model_dump_json() + "\n"
            for workload in workloads
        ),
        encoding="utf-8",
    )
    resource_path.write_bytes(_safetensors_bytes(np.arange(256, dtype=np.float32)))
    for solution in _solutions():
        solution_path = root / "solutions" / f"{solution.name}.json"
        solution_path.parent.mkdir(parents=True, exist_ok=True)
        solution_path.write_text(solution.model_dump_json(), encoding="utf-8")
    return TraceSet(root)


def _request_statuses(log_directory: Path) -> list[str | None]:
    run_directories = list((log_directory / "runs").iterdir())
    assert len(run_directories) == 1
    events = [
        json.loads(line)
        for line in (run_directories[0] / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return [
        event.get("status")
        for event in events
        if event["event"] == "request_finished" and event["http_status"] == 200
    ]


def test_real_python_cuda_safetensors_and_incorrect_solution(
    tmp_path: Path, gpu_server: tuple[str, Path]
) -> None:
    gpu_server_url, log_directory = gpu_server
    trace_set = _write_trace_set(tmp_path)
    config = BenchmarkConfig(
        warmup_runs=0,
        iterations=2,
        num_trials=2,
        profile_baseline=True,
        timeout_seconds=300,
    )
    definition = trace_set.definitions["vector_add"]
    workload_traces = trace_set.workloads["vector_add"]

    def evaluate_solution(client: FlashInferTraceClient, solution_name: str) -> list[Trace]:
        solution = trace_set.solutions[solution_name]
        return client.evaluate_many(
            definition,
            solution,
            workload_traces,
            resource_root=trace_set.path,
        )

    with FlashInferTraceClient(gpu_server_url, config=config, max_workers=2) as client:
        python_solution = trace_set.solutions["python_add"]
        single_python_trace = client.evaluate(
            definition,
            python_solution,
            workload_traces[0],
            resource_root=trace_set.path,
        )
        python_traces = evaluate_solution(client, "python_add")
        assert [trace.workload.uuid for trace in python_traces] == [
            "gpu-workload-0",
            "gpu-workload-1",
        ]
        statuses_after_first_solution = _request_statuses(log_directory)
        triton_traces = evaluate_solution(client, "triton_add")
        cuda_traces = evaluate_solution(client, "cuda_add")
        incorrect_traces = evaluate_solution(client, "incorrect_add")

    for trace in (single_python_trace, *python_traces, *triton_traces, *cuda_traces):
        assert trace.evaluation is not None
        assert trace.evaluation.status == EvaluationStatus.PASSED, trace.evaluation.log
        assert trace.evaluation.performance is not None
        assert trace.evaluation.performance.latency_ms > 0
        assert trace.evaluation.performance.reference_latency_ms > 0

    for trace in incorrect_traces:
        assert trace.evaluation is not None
        assert trace.evaluation.status == EvaluationStatus.INCORRECT_NUMERICAL
        assert trace.evaluation.performance is None

    assert statuses_after_first_solution.count("CACHE_MISS") == 1
    assert statuses_after_first_solution.count("COMPLETED") == 3
    all_statuses = _request_statuses(log_directory)
    assert all_statuses.count("CACHE_MISS") == 1
    assert all_statuses.count("COMPLETED") == 9
