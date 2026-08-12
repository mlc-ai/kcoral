"""CPU-side tests for FlashInfer Trace program construction."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from flashinfer_trace import (
    BenchmarkConfig,
    BuildSpec,
    Definition,
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


def _safetensors_bytes(values: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    payload = contiguous.tobytes()
    header = json.dumps(
        {
            "stored": {
                "dtype": "F32",
                "shape": list(contiguous.shape),
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode()
    return struct.pack("<Q", len(header)) + header + payload


def _definition() -> Definition:
    return Definition(
        name="vector_add",
        op_type="elementwise",
        axes={"length": {"type": "var"}},
        inputs={
            "random": TensorSpec(shape=["length"], dtype="float32"),
            "stored": TensorSpec(shape=["length"], dtype="float32"),
        },
        outputs={"output": TensorSpec(shape=["length"], dtype="float32")},
        reference="def run(random, stored):\n    return random + stored\n",
        constraints=["length > 0"],
    )


def _solution(language: str = "python") -> Solution:
    if language == "cuda":
        content = """
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
        binding = "tvm-ffi"
        destination_passing_style = True
        path = "solution.cu"
    else:
        content = "def run(random, stored):\n    return random + stored\n"
        binding = None
        destination_passing_style = False
        path = "solution.py"
    return Solution(
        name=f"{language}_vector_add",
        definition="vector_add",
        author="tests",
        spec=BuildSpec(
            language=language,
            target_hardware=["cuda"],
            entry_point=f"{path}::run",
            destination_passing_style=destination_passing_style,
            binding=binding,
        ),
        sources=[SourceFile(path=path, content=content)],
    )


def _workload(path: str = "resources/inputs.safetensors") -> Workload:
    return Workload(
        axes={"length": 4},
        inputs={
            "random": RandomInput(),
            "stored": SafetensorsInput(path=path, tensor_key="stored"),
        },
        uuid="workload-1",
    )


def _write_trace_set(root: Path, solution: Solution, workload: Workload) -> bytes:
    definition = _definition()
    definition_path = root / "definitions" / "vector_add.json"
    solution_path = root / "solutions" / f"{solution.name}.json"
    workload_path = root / "workloads" / "vector_add.jsonl"
    resource_path = root / "resources" / "inputs.safetensors"
    for path in (definition_path, solution_path, workload_path, resource_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    definition_path.write_text(definition.model_dump_json(), encoding="utf-8")
    solution_path.write_text(solution.model_dump_json(), encoding="utf-8")
    workload_path.write_text(
        Trace(definition=definition.name, workload=workload).model_dump_json() + "\n",
        encoding="utf-8",
    )
    resource_bytes = _safetensors_bytes(np.arange(4, dtype=np.float32))
    resource_path.write_bytes(resource_bytes)
    return resource_bytes


def _build_program(root: Path, language: str = "python"):
    solution = _solution(language)
    resource_bytes = _write_trace_set(root, solution, _workload())
    trace_set = TraceSet.from_path(root)
    client = FlashInferTraceClient(
        config=BenchmarkConfig(warmup_runs=1, iterations=1, num_trials=1)
    )
    try:
        program = client.build_program(
            trace_set.definitions["vector_add"],
            solution,
            trace_set.workloads["vector_add"][0].workload,
            client.config.resolve_eval_config(trace_set.definitions["vector_add"]),
            resource_root=root,
        )
    finally:
        client.close()
    return program, resource_bytes


def test_program_uploads_remote_modules_and_real_safetensors_bytes(tmp_path: Path) -> None:
    first_program, resource_bytes = _build_program(tmp_path)
    instructions = first_program.instructions

    modules = {
        instruction["id"]: instruction
        for instruction in instructions
        if instruction["op"] == "upload" and instruction["kind"] == "module"
    }
    assert {"trace_schema", "trace_data", "trace_compile", "trace_benchmark"} <= set(modules)
    assert all(module.get("entry") for module in modules.values())

    bytes_upload = next(
        instruction
        for instruction in instructions
        if instruction["op"] == "upload" and instruction["kind"] == "bytes"
    )
    assert bytes_upload["blob"] == hashlib.sha256(resource_bytes).hexdigest()

    second_program, _ = _build_program(tmp_path / "second")
    second_bytes_upload = next(
        instruction
        for instruction in second_program.instructions
        if instruction["op"] == "upload" and instruction["kind"] == "bytes"
    )
    assert second_bytes_upload["blob"] == bytes_upload["blob"]

    run_ids = {instruction["id"] for instruction in instructions if instruction["op"] == "run"}
    assert {
        "normalized",
        "input_generator",
        "reference_callable",
        "solution_callable",
        "evaluation",
    } <= run_ids


def test_cuda_program_uses_direct_compile_instruction(tmp_path: Path) -> None:
    program, _ = _build_program(tmp_path, "cuda")
    instructions = program.instructions

    cuda_upload = next(
        instruction
        for instruction in instructions
        if instruction["op"] == "upload" and instruction.get("language") == "cuda"
    )
    compile_instruction = next(
        instruction
        for instruction in instructions
        if instruction["op"] == "run" and instruction["id"] == "compiled_cuda"
    )
    assert cuda_upload["entry"] == "run"
    assert compile_instruction["fn"] == "builtin.compile_cuda"
    assert compile_instruction["args"][0] == {"$ref": cuda_upload["id"]}


def test_program_reads_changed_safetensors_content(tmp_path: Path) -> None:
    solution = _solution()
    workload = _workload()
    _write_trace_set(tmp_path, solution, workload)
    definition = _definition()
    client = FlashInferTraceClient(
        config=BenchmarkConfig(warmup_runs=1, iterations=1, num_trials=1)
    )
    try:
        resolved_config = client.config.resolve_eval_config(definition)
        first_program = client.build_program(
            definition,
            solution,
            workload,
            resolved_config,
            resource_root=tmp_path,
        )
        changed_bytes = _safetensors_bytes(np.arange(4, dtype=np.float32) + 10)
        (tmp_path / "resources" / "inputs.safetensors").write_bytes(changed_bytes)
        second_program = client.build_program(
            definition,
            solution,
            workload,
            resolved_config,
            resource_root=tmp_path,
        )
    finally:
        client.close()

    first_blob = next(
        instruction["blob"]
        for instruction in first_program.instructions
        if instruction["op"] == "upload" and instruction["kind"] == "bytes"
    )
    second_blob = next(
        instruction["blob"]
        for instruction in second_program.instructions
        if instruction["op"] == "upload" and instruction["kind"] == "bytes"
    )
    assert first_blob != second_blob
    assert second_blob == hashlib.sha256(changed_bytes).hexdigest()


@pytest.mark.parametrize("resource_path", ["../outside.safetensors", "/tmp/outside.safetensors"])
def test_program_rejects_resource_paths_outside_trace_set(
    tmp_path: Path, resource_path: str
) -> None:
    solution = _solution()
    workload = _workload(resource_path)
    _write_trace_set(tmp_path, solution, workload)
    client = FlashInferTraceClient()
    try:
        with pytest.raises(ValueError, match="resource path"):
            client.build_program(
                _definition(),
                solution,
                workload,
                client.config.resolve_eval_config(_definition()),
                resource_root=tmp_path,
            )
    finally:
        client.close()
