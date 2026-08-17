import argparse
import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).parents[1] / "scripts" / "stress_b200.py"
_SPEC = importlib.util.spec_from_file_location("stress_b200", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
stress_b200 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stress_b200
_SPEC.loader.exec_module(stress_b200)


CUDA_SOURCE = """
#include <tvm/ffi/tvm_ffi.h>
void run(tvm::ffi::TensorView a, tvm::ffi::TensorView b, tvm::ffi::TensorView c) {}
"""

TRITON_SOURCE = """\
import triton
import triton.language as tl

@triton.jit
def kernel(a, b, c):
    pass

def run(a, b, c):
    kernel[(1,)](a, b, c)
"""


def _write_kernel(root: Path, workload: str, name: str, source: str = CUDA_SOURCE) -> Path:
    path = root / workload / "run" / "success" / "exp_000" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def test_discover_kernels_deduplicates_and_interleaves(tmp_path):
    _write_kernel(tmp_path, "gemm_n7168_k5120", "kernel_t0.cu", "gemm zero")
    _write_kernel(tmp_path, "gemm_n7168_k5120", "kernel_t1.cu", "gemm one")
    _write_kernel(tmp_path, "mha_with_lse_d128", "kernel_t0.cu", "mha zero")
    _write_kernel(tmp_path, "mha_with_lse_d128", "kernel_t1.cu", "mha zero")

    kernels = stress_b200.discover_kernels(
        tmp_path,
        ["gemm_n7168_k5120", "mha_with_lse_d128"],
        kernels_per_workload=0,
    )

    assert [kernel.workload for kernel in kernels] == [
        "gemm_n7168_k5120",
        "mha_with_lse_d128",
        "gemm_n7168_k5120",
    ]
    assert len({kernel.sha256 for kernel in kernels}) == 3


def test_discover_kernels_can_keep_duplicates(tmp_path):
    _write_kernel(tmp_path, "gemm_n7168_k5120", "kernel_t0.cu")
    _write_kernel(tmp_path, "gemm_n7168_k5120", "kernel_t1.cu")

    kernels = stress_b200.discover_kernels(
        tmp_path,
        ["gemm_n7168_k5120"],
        kernels_per_workload=0,
        keep_duplicates=True,
    )

    assert len(kernels) == 2
    assert kernels[0].sha256 == kernels[1].sha256


def test_discover_kernels_interleaves_cuda_and_triton_without_cross_dedup(tmp_path):
    cuda_root = tmp_path / "cuda"
    triton_root = tmp_path / "triton"
    _write_kernel(cuda_root, "gemm_n7168_k5120", "kernel_t0.cu", "same source")
    _write_kernel(cuda_root, "mha_with_lse_d128", "kernel_t0.cu", "cuda mha")
    _write_kernel(triton_root, "gemm_n7168_k5120", "kernel_t0.py", "same source")
    _write_kernel(triton_root, "mha_with_lse_d128", "kernel_t0.py", "triton mha")

    kernels = stress_b200.discover_kernels(
        {"cuda": cuda_root, "triton": triton_root},
        ["gemm_n7168_k5120", "mha_with_lse_d128"],
        kernels_per_workload=0,
    )

    assert [(kernel.workload, kernel.language) for kernel in kernels] == [
        ("gemm_n7168_k5120", "cuda"),
        ("gemm_n7168_k5120", "triton"),
        ("mha_with_lse_d128", "cuda"),
        ("mha_with_lse_d128", "triton"),
    ]
    assert kernels[0].sha256 == kernels[1].sha256


def test_triton_only_default_skips_workloads_missing_from_that_corpus(tmp_path):
    triton_root = tmp_path / "triton"
    _write_kernel(triton_root, "gemm_n7168_k5120", "kernel_t0.py", TRITON_SOURCE)

    kernels = stress_b200.discover_kernels({"triton": triton_root}, selected_languages=["triton"])

    assert [(kernel.language, kernel.workload) for kernel in kernels] == [
        ("triton", "gemm_n7168_k5120")
    ]


def test_bounded_triton_selection_prefers_success_sources(tmp_path):
    triton_root = tmp_path / "triton"
    _write_kernel(
        triton_root,
        "gemm_n7168_k5120",
        "trajectory/kernel_t0.py",
        "trajectory",
    )
    success = _write_kernel(triton_root, "gemm_n7168_k5120", "success/kernel_v0.py", "success")

    kernels = stress_b200.discover_kernels(
        {"triton": triton_root},
        selected_workloads=["gemm_n7168_k5120"],
        selected_languages=["triton"],
    )

    assert [kernel.path for kernel in kernels] == [success]


def test_bounded_triton_selection_prefers_blackwell_manifest_source(tmp_path):
    triton_root = tmp_path / "triton"
    hopper = _write_kernel(triton_root, "gemm_n7168_k5120", "success/kernel_v0.py", "hopper")
    blackwell = _write_kernel(
        triton_root, "gemm_n7168_k5120", "other-success/kernel_v0.py", "blackwell"
    )
    records = [
        {
            "path": hopper.relative_to(triton_root).as_posix(),
            "source_arch": "hopper",
            "source_kind": "success",
        },
        {
            "path": blackwell.relative_to(triton_root).as_posix(),
            "source_arch": "blackwell",
            "source_kind": "success",
        },
    ]
    (triton_root / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )

    kernels = stress_b200.discover_kernels(
        {"triton": triton_root},
        selected_workloads=["gemm_n7168_k5120"],
        selected_languages=["triton"],
    )

    assert [kernel.path for kernel in kernels] == [blackwell]


def test_checked_in_default_corpora_select_both_languages():
    kernels = stress_b200.discover_kernels(
        {
            "cuda": stress_b200.DEFAULT_CUDA_CORPUS,
            "triton": stress_b200.DEFAULT_TRITON_CORPUS,
        }
    )

    assert {kernel.language for kernel in kernels} == {"cuda", "triton"}
    assert sum(kernel.language == "cuda" for kernel in kernels) == 6
    assert sum(kernel.language == "triton" for kernel in kernels) == 5


def test_build_gemm_program_compiles_allocates_and_benchmarks(tmp_path):
    path = _write_kernel(tmp_path, "gemm_n7168_k5120", "kernel_t0.cu")
    kernel = stress_b200.Kernel("gemm_n7168_k5120", path, "kernel_t0.cu", "digest")

    instructions = stress_b200.build_program(
        kernel, seed=7, warmup=2, repeat=9, flush_l2=True
    ).instructions

    assert instructions[0] == {
        "op": "upload",
        "id": "source",
        "kind": "module",
        "source": CUDA_SOURCE,
        "language": "cuda",
        "entry": "run",
    }
    assert instructions[1]["fn"] == "builtin.compile_cuda"
    assert [instruction["fn"] for instruction in instructions[2:5]] == [
        "builtin.randn",
        "builtin.randn",
        "builtin.empty",
    ]
    benchmark = instructions[5]
    assert benchmark["fn"] == "builtin.benchmark"
    assert benchmark["args"][-1] == {"warmup": 2, "repeat": 9, "flush_l2": True}
    assert instructions[-1]["op"] == "return"


def test_build_triton_program_uploads_python_without_cuda_compilation(tmp_path):
    path = _write_kernel(tmp_path, "gemm_n7168_k5120", "kernel_t0.py", TRITON_SOURCE)
    kernel = stress_b200.Kernel(
        "gemm_n7168_k5120",
        path,
        "kernel_t0.py",
        "digest",
        language="triton",
    )

    instructions = stress_b200.build_program(
        kernel, seed=7, warmup=2, repeat=9, flush_l2=True
    ).instructions

    assert instructions[0] == {
        "op": "upload",
        "id": "source",
        "kind": "module",
        "source": TRITON_SOURCE,
        "entry": "run",
    }
    # Python is the protocol default and is therefore omitted from serialized JSON.
    assert instructions[0].get("language", "python") == "python"
    assert all(instruction.get("fn") != "builtin.compile_cuda" for instruction in instructions)
    benchmark = next(
        instruction for instruction in instructions if instruction.get("fn") == "builtin.benchmark"
    )
    assert benchmark["args"][0] == {"$ref": "source"}


def test_backward_program_zeroes_accumulation_outputs(tmp_path):
    path = _write_kernel(tmp_path, "mha_bwd_d128", "kernel_t0.cu")
    kernel = stress_b200.Kernel("mha_bwd_d128", path, "kernel_t0.cu", "digest")

    instructions = stress_b200.build_program(
        kernel, seed=0, warmup=1, repeat=1, flush_l2=False
    ).instructions

    by_id = {instruction.get("id"): instruction for instruction in instructions}
    assert [by_id[name]["fn"] for name in ("dq", "dk", "dv")] == [
        "builtin.zeros",
        "builtin.zeros",
        "builtin.zeros",
    ]


def test_request_allocator_honors_fixed_request_count():
    allocator = stress_b200.RequestAllocator(
        started=100.0, duration_seconds=None, requests=3, rate=0.0
    )

    assert [allocator.claim()[0] for _ in range(3)] == [0, 1, 2]
    assert allocator.claim() is None


def test_summarize_reports_distributions_and_failures():
    records = [
        {
            "status": "COMPLETED",
            "language": "cuda",
            "workload": "gemm",
            "client_elapsed_ms": 10,
            "queue_ms": 1,
            "lease_wait_ms": 2,
            "lease_held_ms": 3,
            "benchmark_latency_ms": 4,
            "activities_stable": True,
        },
        {
            "status": "HTTP_503",
            "language": "triton",
            "workload": "gemm",
            "client_elapsed_ms": 20,
            "error_kind": "busy",
        },
    ]

    summary = stress_b200.summarize(records, elapsed_seconds=2)

    assert summary["requests"] == 2
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["request_rate"] == 1
    assert summary["statuses"] == {"COMPLETED": 1, "HTTP_503": 1}
    assert summary["errors"] == {"busy": 1}
    assert summary["by_language"] == {"cuda": 1, "triton": 1}
    assert summary["client_elapsed_ms"]["p50"] == 15
    assert summary["lease_held_ms"]["p99"] == 3


def test_validate_args_supplies_default_duration():
    args = argparse.Namespace(
        duration_seconds=None,
        requests=None,
        timeout_seconds=300.0,
        connect_timeout_seconds=10.0,
        concurrency=8,
        rate=0.0,
        warmup=1,
        repeat=5,
        kernels_per_workload=1,
    )

    stress_b200.validate_args(args)

    assert args.duration_seconds == 60.0


def test_parser_selects_both_languages_by_default():
    args = stress_b200.parser().parse_args([])

    assert args.language is None
    assert args.cuda_corpus == stress_b200.DEFAULT_CUDA_CORPUS
    assert args.triton_corpus == stress_b200.DEFAULT_TRITON_CORPUS
