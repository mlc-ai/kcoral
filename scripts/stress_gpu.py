#!/usr/bin/env python3
"""Drive benchmark-server with CUDA and Triton kernels from AccRL corpora."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_server import Client, Program
from benchmark_server.client import BenchmarkServerError, ProtocolError, TransportError

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUDA_CORPUS = REPO_ROOT / "cuda_kernels"
DEFAULT_TRITON_CORPUS = REPO_ROOT / "triton_kernels"
LANGUAGES = ("cuda", "triton")
SOURCE_ARCHITECTURES = ("b200", "h100")
TARGET_TO_SOURCE_ARCH = {"sm_100a": "b200", "sm_90a": "h100"}
KERNEL_GLOBS = {"cuda": "kernel_t*.cu", "triton": "*.py"}


def normalize_source_arch(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    aliases = {
        "b200": "b200",
        "blackwell": "b200",
        "sm_100a": "b200",
        "compute_100a": "b200",
        "h100": "h100",
        "hopper": "h100",
        "sm_90a": "h100",
        "compute_90a": "h100",
    }
    return aliases.get(normalized)


def corpus_metadata(corpus: Path, language: str) -> dict[str, dict[str, Any]]:
    manifest_path = corpus / "manifest.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"missing architecture manifest: {manifest_path}")
    metadata: dict[str, dict[str, Any]] = {}
    with manifest_path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = record.get("path")
            if isinstance(path, str):
                source_arch = normalize_source_arch(record.get("source_arch"))
                if source_arch is None:
                    raise ValueError(f"{manifest_path}: {path!r} has no supported source_arch")
                record["source_arch"] = source_arch
                metadata[path] = record

    if language == "cuda":
        correct_turns: set[tuple[str, str, int]] = set()
        correctness_dir = corpus / "turn_correctness_arch"
        for path in correctness_dir.glob("*.csv"):
            with path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    try:
                        turn = int(row.get("turn", ""))
                    except ValueError:
                        continue
                    if row.get("correctness") == "Correct":
                        correct_turns.add((path.stem, row.get("trajectory_id", ""), turn))
        for record in metadata.values():
            record["known_success"] = (
                record.get("run"),
                record.get("experiment"),
                record.get("turn"),
            ) in correct_turns
    return metadata


def kernel_path_key(
    path: Path,
    language: str,
    corpus: Path,
    metadata: dict[str, dict[str, Any]],
) -> tuple[int, int, str]:
    """Prefer known-correct, directly runnable sources for bounded selections."""
    relative = path.relative_to(corpus).as_posix()
    record = metadata.get(relative, {})
    if language != "triton":
        return (0 if record.get("known_success") else 1, 0, relative)
    source_kind = record.get("source_kind", path.parent.name)
    if source_kind == "success":
        source_rank = 0
    elif source_kind == "workspace":
        source_rank = 1
    else:
        source_rank = 2
    shape_rank = 0 if record.get("valid_triton_shape", True) else 1
    return (source_rank, shape_rank, relative)


@dataclass(frozen=True)
class WorkloadSpec:
    random_tensors: tuple[tuple[str, tuple[int, ...], str], ...]
    empty_tensors: tuple[tuple[str, tuple[int, ...], str, bool], ...]
    argument_names: tuple[str, ...]


_MHA_SHAPE = (4, 48, 4096, 128)
_LSE_SHAPE = (4, 48, 4096)

WORKLOADS: dict[str, WorkloadSpec] = {
    "gemm_n7168_k5120": WorkloadSpec(
        random_tensors=(
            ("a", (8192, 5120), "bfloat16"),
            ("b", (7168, 5120), "bfloat16"),
        ),
        empty_tensors=(("c", (8192, 7168), "bfloat16", False),),
        argument_names=("a", "b", "c"),
    ),
    "mha_with_lse_d128": WorkloadSpec(
        random_tensors=tuple((name, _MHA_SHAPE, "bfloat16") for name in ("q", "k", "v")),
        empty_tensors=(
            ("o", _MHA_SHAPE, "bfloat16", False),
            ("lse", _LSE_SHAPE, "float32", False),
        ),
        argument_names=("q", "k", "v", "o", "lse"),
    ),
    "mha_with_lse_d128_causal": WorkloadSpec(
        random_tensors=tuple((name, _MHA_SHAPE, "bfloat16") for name in ("q", "k", "v")),
        empty_tensors=(
            ("o", _MHA_SHAPE, "bfloat16", False),
            ("lse", _LSE_SHAPE, "float32", False),
        ),
        argument_names=("q", "k", "v", "o", "lse"),
    ),
    "mha_with_lse_h48_d128": WorkloadSpec(
        random_tensors=tuple((name, _MHA_SHAPE, "bfloat16") for name in ("q", "k", "v")),
        empty_tensors=(
            ("o", _MHA_SHAPE, "bfloat16", False),
            ("lse", _LSE_SHAPE, "float32", False),
        ),
        argument_names=("q", "k", "v", "o", "lse"),
    ),
    "mha_bwd_d128": WorkloadSpec(
        random_tensors=(
            *((name, _MHA_SHAPE, "bfloat16") for name in ("q", "k", "v", "o", "do")),
            ("lse", _LSE_SHAPE, "float32"),
        ),
        # Some backward kernels accumulate into dK/dV, so initialize every output.
        empty_tensors=tuple((name, _MHA_SHAPE, "bfloat16", True) for name in ("dq", "dk", "dv")),
        argument_names=("q", "k", "v", "o", "do", "lse", "dq", "dk", "dv"),
    ),
    "mha_bwd_d128_causal": WorkloadSpec(
        random_tensors=(
            *((name, _MHA_SHAPE, "bfloat16") for name in ("q", "k", "v", "o", "do")),
            ("lse", _LSE_SHAPE, "float32"),
        ),
        empty_tensors=tuple((name, _MHA_SHAPE, "bfloat16", True) for name in ("dq", "dk", "dv")),
        argument_names=("q", "k", "v", "o", "do", "lse", "dq", "dk", "dv"),
    ),
}


@dataclass(frozen=True)
class Kernel:
    workload: str
    path: Path
    relative_path: str
    sha256: str
    language: str = "cuda"
    source_arch: str = "unknown"


def discover_kernels(
    corpus: Path | dict[str, Path],
    selected_workloads: list[str] | None = None,
    kernels_per_workload: int = 1,
    keep_duplicates: bool = False,
    selected_languages: list[str] | None = None,
    selected_source_arches: list[str] | None = None,
) -> list[Kernel]:
    """Return a deterministic selection interleaved by workload, language, and arch."""
    names = selected_workloads or sorted(WORKLOADS)
    unknown = sorted(set(names) - set(WORKLOADS))
    if unknown:
        raise ValueError(f"unsupported workload(s): {', '.join(unknown)}")
    if kernels_per_workload < 0:
        raise ValueError("kernels_per_workload must be non-negative")

    if isinstance(corpus, Path):
        corpora = {"cuda": corpus}
        languages = selected_languages or ["cuda"]
    else:
        corpora = corpus
        languages = selected_languages or list(LANGUAGES)
    unknown_languages = sorted(set(languages) - set(LANGUAGES))
    if unknown_languages:
        raise ValueError(f"unsupported language(s): {', '.join(unknown_languages)}")
    source_arches = selected_source_arches or list(SOURCE_ARCHITECTURES)
    unknown_arches = sorted(set(source_arches) - set(SOURCE_ARCHITECTURES))
    if unknown_arches:
        raise ValueError(f"unsupported source architecture(s): {', '.join(unknown_arches)}")

    by_group: dict[tuple[str, str, str], list[Kernel]] = {}
    for language in languages:
        language_corpus = corpora.get(language)
        if language_corpus is None:
            raise ValueError(f"no corpus configured for language {language!r}")
        if not language_corpus.is_dir():
            raise ValueError(f"missing {language} corpus directory: {language_corpus}")
        metadata = corpus_metadata(language_corpus, language)
        for source_arch in source_arches:
            for workload in names:
                root = language_corpus / workload
                if not root.is_dir():
                    continue
                kernels: list[Kernel] = []
                seen: set[str] = set()
                paths = sorted(
                    root.rglob(KERNEL_GLOBS[language]),
                    key=lambda path: kernel_path_key(path, language, language_corpus, metadata),
                )
                for path in paths:
                    relative = path.relative_to(language_corpus).as_posix()
                    record = metadata.get(relative)
                    path_source_arch = (
                        record.get("source_arch") if record is not None else "unknown"
                    )
                    if path_source_arch != source_arch:
                        continue
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    if not keep_duplicates and digest in seen:
                        continue
                    seen.add(digest)
                    kernels.append(
                        Kernel(
                            workload=workload,
                            path=path,
                            relative_path=relative,
                            sha256=digest,
                            language=language,
                            source_arch=source_arch,
                        )
                    )
                    if kernels_per_workload and len(kernels) >= kernels_per_workload:
                        break
                if kernels:
                    by_group[(language, source_arch, workload)] = kernels

    if selected_languages is not None:
        for language in languages:
            if not any(group_language == language for group_language, _, _ in by_group):
                raise ValueError(f"no {language} kernels found in selected workloads")
    if selected_workloads is not None:
        for workload in names:
            if not any(group_workload == workload for _, _, group_workload in by_group):
                raise ValueError(f"no kernels found for workload {workload!r}")
    if not by_group:
        raise ValueError("no kernels found for the selected languages and source architectures")

    # Avoid exhausting one language/architecture/workload before touching the next.
    groups = [
        (language, source_arch, workload)
        for workload in names
        for language in languages
        for source_arch in source_arches
        if (language, source_arch, workload) in by_group
    ]
    interleaved: list[Kernel] = []
    for index in range(max(len(items) for items in by_group.values())):
        for group in groups:
            items = by_group[group]
            if index < len(items):
                interleaved.append(items[index])
    return interleaved


def build_program(
    kernel: Kernel,
    *,
    seed: int,
    warmup: int,
    repeat: int,
    flush_l2: bool,
) -> Program:
    spec = WORKLOADS[kernel.workload]
    program = Program()
    source = program.upload(
        id="source",
        kind="module",
        source=kernel.path.read_text(encoding="utf-8"),
        language="cuda" if kernel.language == "cuda" else "python",
        entry="run",
    )
    callable_kernel = source
    if kernel.language == "cuda":
        callable_kernel = program.run(id="compiled", fn="builtin.compile_cuda", args=[source])

    tensors: dict[str, Any] = {}
    for offset, (name, shape, dtype) in enumerate(spec.random_tensors):
        tensors[name] = program.run(
            id=name,
            fn="builtin.randn",
            args=[{"shape": list(shape), "dtype": dtype, "seed": seed + offset}],
        )
    for name, shape, dtype, zero in spec.empty_tensors:
        tensors[name] = program.run(
            id=name,
            fn="builtin.zeros" if zero else "builtin.empty",
            args=[{"shape": list(shape), "dtype": dtype}],
        )

    benchmark_args = [callable_kernel]
    benchmark_args.extend(tensors[name] for name in spec.argument_names)
    benchmark_args.append({"warmup": warmup, "repeat": repeat, "flush_l2": flush_l2})
    timing = program.run(id="timing", fn="builtin.benchmark", args=benchmark_args)
    program.return_(key="timing", value=timing)
    return program


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(records: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    completed = [record for record in records if record["status"] == "COMPLETED"]
    client_ms = [float(record["client_elapsed_ms"]) for record in records]
    queue_ms = [float(record["queue_ms"]) for record in completed]
    lease_wait_ms = [float(record["lease_wait_ms"]) for record in completed]
    lease_held_ms = [float(record["lease_held_ms"]) for record in completed]
    benchmark_ms = [
        float(record["benchmark_latency_ms"])
        for record in completed
        if record.get("benchmark_latency_ms") is not None
    ]

    def distribution(values: list[float]) -> dict[str, float | None]:
        return {
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values) if values else None,
        }

    return {
        "requests": len(records),
        "completed": len(completed),
        "failed": len(records) - len(completed),
        "elapsed_seconds": elapsed_seconds,
        "request_rate": len(records) / elapsed_seconds if elapsed_seconds else 0.0,
        "completed_rate": len(completed) / elapsed_seconds if elapsed_seconds else 0.0,
        "statuses": dict(sorted(Counter(record["status"] for record in records).items())),
        "errors": dict(
            sorted(
                Counter(
                    str(record.get("error_kind")) for record in records if record.get("error_kind")
                ).items()
            )
        ),
        "by_workload": dict(sorted(Counter(record["workload"] for record in records).items())),
        "by_language": dict(sorted(Counter(record["language"] for record in records).items())),
        "by_source_arch": dict(
            sorted(Counter(record["source_arch"] for record in records).items())
        ),
        "by_target_arch": dict(
            sorted(Counter(record["target_arch"] for record in records).items())
        ),
        "by_language_source_arch": dict(
            sorted(
                Counter(
                    f"{record['language']}/{record['source_arch']}" for record in records
                ).items()
            )
        ),
        "by_source_target_arch": dict(
            sorted(
                Counter(
                    f"{record['source_arch']}->{record['target_arch']}" for record in records
                ).items()
            )
        ),
        "client_elapsed_ms": distribution(client_ms),
        "queue_ms": distribution(queue_ms),
        "lease_wait_ms": distribution(lease_wait_ms),
        "lease_held_ms": distribution(lease_held_ms),
        "benchmark_latency_ms": distribution(benchmark_ms),
        "unstable_activity_requests": sum(
            record.get("activities_stable") is False for record in completed
        ),
    }


class JsonlSink:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._stream = path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def execute_one(
    client: Client,
    kernel: Kernel,
    sequence: int,
    worker_slot: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.monotonic()
    base = {
        "type": "request",
        "timestamp": utc_now(),
        "sequence": sequence,
        "worker_slot": worker_slot,
        "language": kernel.language,
        "source_arch": kernel.source_arch,
        "target_arch": args.target_arch,
        "target_sm": args.target_sm,
        "workload": kernel.workload,
        "kernel": kernel.relative_path,
        "sha256": kernel.sha256,
    }
    try:
        program = build_program(
            kernel,
            seed=args.seed + sequence * 32,
            warmup=args.warmup,
            repeat=args.repeat,
            flush_l2=args.flush_l2,
        )
        result = client.execute(
            program,
            timeout_seconds=args.timeout_seconds,
            output_limit_bytes=65536,
        )
        timing = result.results.get("timing", {})
        return {
            **base,
            "status": result.status,
            "request_id": result.request_id,
            "client_elapsed_ms": (time.monotonic() - started) * 1000,
            "queue_ms": result.queue_ms,
            "elapsed_ms": result.elapsed_ms,
            "lease_wait_ms": result.lease_wait_ms,
            "lease_held_ms": result.lease_held_ms,
            "benchmark_latency_ms": timing.get("latency_ms_median"),
            "activities_stable": timing.get("activities_stable"),
            "error_kind": result.error.get("kind") if result.error else None,
            "error_message": result.error.get("message") if result.error else None,
            "stderr": result.stderr[-4096:],
        }
    except BenchmarkServerError as exc:
        return {
            **base,
            "status": f"HTTP_{exc.status_code}",
            "request_id": exc.request_id,
            "client_elapsed_ms": (time.monotonic() - started) * 1000,
            "error_kind": exc.kind or "http",
            "error_message": exc.message,
        }
    except (TransportError, ProtocolError) as exc:
        return {
            **base,
            "status": "CLIENT_ERROR",
            "request_id": None,
            "client_elapsed_ms": (time.monotonic() - started) * 1000,
            "error_kind": type(exc).__name__,
            "error_message": str(exc),
        }
    except Exception as exc:  # keep the load running and make the local failure visible
        return {
            **base,
            "status": "DRIVER_ERROR",
            "request_id": None,
            "client_elapsed_ms": (time.monotonic() - started) * 1000,
            "error_kind": type(exc).__name__,
            "error_message": str(exc),
        }


class RequestAllocator:
    def __init__(
        self,
        *,
        started: float,
        duration_seconds: float | None,
        requests: int | None,
        rate: float,
    ) -> None:
        self._lock = threading.Lock()
        self._started = started
        self._deadline = started + duration_seconds if duration_seconds is not None else None
        self._requests = requests
        self._rate = rate
        self._next = 0

    def claim(self) -> tuple[int, float] | None:
        with self._lock:
            sequence = self._next
            if self._requests is not None and sequence >= self._requests:
                return None
            scheduled = self._started + sequence / self._rate if self._rate else time.monotonic()
            if self._deadline is not None and scheduled >= self._deadline:
                return None
            self._next += 1
            return sequence, scheduled


def run_load(
    args: argparse.Namespace,
    kernels: list[Kernel],
    sink: JsonlSink,
) -> tuple[list[dict[str, Any]], float]:
    started = time.monotonic()
    allocator = RequestAllocator(
        started=started,
        duration_seconds=args.duration_seconds,
        requests=args.requests,
        rate=args.rate,
    )
    records: list[dict[str, Any]] = []
    records_lock = threading.Lock()

    def worker(slot: int) -> None:
        with Client(args.url, connect_timeout_seconds=args.connect_timeout_seconds) as client:
            while True:
                claimed = allocator.claim()
                if claimed is None:
                    return
                sequence, scheduled = claimed
                delay = scheduled - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                record = execute_one(client, kernels[sequence % len(kernels)], sequence, slot, args)
                sink.write(record)
                with records_lock:
                    records.append(record)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker, slot) for slot in range(args.concurrency)]
        for future in futures:
            future.result()
    return records, time.monotonic() - started


def prewarm(args: argparse.Namespace, kernels: list[Kernel]) -> None:
    with Client(args.url, connect_timeout_seconds=args.connect_timeout_seconds) as client:
        for sequence, kernel in enumerate(kernels):
            record = execute_one(client, kernel, sequence, 0, args)
            if record["status"] != "COMPLETED":
                raise RuntimeError(
                    f"prewarm failed for {kernel.relative_path}: "
                    f"{record.get('error_kind')}: {record.get('error_message')}"
                )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--url", default="http://127.0.0.1:8000")
    result.add_argument(
        "--cuda-corpus",
        "--corpus",
        dest="cuda_corpus",
        type=Path,
        default=DEFAULT_CUDA_CORPUS,
        help="CUDA corpus root (--corpus is retained as an alias)",
    )
    result.add_argument("--triton-corpus", type=Path, default=DEFAULT_TRITON_CORPUS)
    result.add_argument(
        "--language",
        action="append",
        choices=LANGUAGES,
        help="kernel language to include; repeat the flag (default: both)",
    )
    result.add_argument(
        "--source-arch",
        action="append",
        choices=SOURCE_ARCHITECTURES,
        help=(
            "source GPU architecture; repeat to include both "
            "(default: match server; --list shows both)"
        ),
    )
    result.add_argument(
        "--workload",
        action="append",
        choices=sorted(WORKLOADS),
        help="workload to include; repeat the flag (default: all)",
    )
    result.add_argument(
        "--kernels-per-workload",
        type=int,
        default=1,
        help=(
            "number selected per language/source-arch/workload; 0 uses every kernel (default: 1)"
        ),
    )
    result.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="retain byte-identical sources when selecting the full corpus",
    )
    limit = result.add_mutually_exclusive_group()
    limit.add_argument("--duration-seconds", type=float)
    limit.add_argument("--requests", type=int)
    result.add_argument("--concurrency", type=int, default=8)
    result.add_argument(
        "--rate", type=float, default=0.0, help="open-loop requests/sec; 0 is closed-loop"
    )
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--repeat", type=int, default=5)
    result.add_argument("--flush-l2", action="store_true")
    result.add_argument("--prewarm", action="store_true")
    result.add_argument("--timeout-seconds", type=float, default=300.0)
    result.add_argument("--connect-timeout-seconds", type=float, default=10.0)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument(
        "--allow-unsupported-target",
        "--allow-non-b200",
        dest="allow_unsupported_target",
        action="store_true",
        help="allow a target other than H100 (sm_90a) or B200 (sm_100a)",
    )
    result.add_argument("--allow-errors", action="store_true")
    result.add_argument("--list", action="store_true", help="list selected kernels and exit")
    result.add_argument("--output", type=Path)
    return result


def validate_args(args: argparse.Namespace) -> None:
    if args.duration_seconds is None and args.requests is None:
        args.duration_seconds = 60.0
    for name in ("duration_seconds", "timeout_seconds", "connect_timeout_seconds"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.requests is not None and args.requests <= 0:
        raise ValueError("requests must be positive")
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if args.rate < 0:
        raise ValueError("rate must be non-negative")
    if args.warmup <= 0 or args.repeat <= 0:
        raise ValueError("warmup and repeat must be positive")
    if args.kernels_per_workload < 0:
        raise ValueError("kernels-per-workload must be non-negative")


def default_output(target_arch: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "stress-results" / f"{target_arch}-{stamp}.jsonl"


def target_architecture(target_sm: Any) -> str | None:
    return TARGET_TO_SOURCE_ARCH.get(target_sm) if isinstance(target_sm, str) else None


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        validate_args(args)
        corpora = {
            "cuda": args.cuda_corpus.resolve(),
            "triton": args.triton_corpus.resolve(),
        }
        if args.list:
            kernels = discover_kernels(
                corpora,
                args.workload,
                args.kernels_per_workload,
                args.keep_duplicates,
                args.language,
                args.source_arch,
            )
            for kernel in kernels:
                print(
                    f"{kernel.language}\t{kernel.source_arch}\t{kernel.workload}\t"
                    f"{kernel.sha256}\t{kernel.relative_path}"
                )
            return 0

        with Client(args.url, connect_timeout_seconds=args.connect_timeout_seconds) as client:
            health = client.health()
        target_sm = (health.get("target") or {}).get("arch")
        target_arch = target_architecture(target_sm)
        if target_arch is None and not args.allow_unsupported_target:
            raise RuntimeError(
                f"server target is {target_sm!r}; expected H100 'sm_90a' or B200 'sm_100a'"
            )
        args.target_arch = target_arch or "unknown"
        args.target_sm = target_sm
        source_arches = args.source_arch or ([target_arch] if target_arch is not None else [])
        if not source_arches:
            raise ValueError("--source-arch is required for an unsupported target")
        kernels = discover_kernels(
            corpora,
            args.workload,
            args.kernels_per_workload,
            args.keep_duplicates,
            args.language,
            source_arches,
        )

        if args.prewarm:
            print(f"Prewarming {len(kernels)} kernels...", file=sys.stderr)
            prewarm(args, kernels)

        output = (args.output or default_output(args.target_arch)).resolve()
        sink = JsonlSink(output)
        sink.write(
            {
                "type": "run_started",
                "timestamp": utc_now(),
                "url": args.url,
                "health": health,
                "target_arch": args.target_arch,
                "target_sm": args.target_sm,
                "source_arches": source_arches,
                "kernel_count": len(kernels),
                "kernel_count_by_language": dict(
                    sorted(Counter(kernel.language for kernel in kernels).items())
                ),
                "kernel_count_by_source_arch": dict(
                    sorted(Counter(kernel.source_arch for kernel in kernels).items())
                ),
                "kernel_count_by_language_source_arch": dict(
                    sorted(
                        Counter(
                            f"{kernel.language}/{kernel.source_arch}" for kernel in kernels
                        ).items()
                    )
                ),
                "concurrency": args.concurrency,
                "rate": args.rate,
                "duration_seconds": args.duration_seconds,
                "requests": args.requests,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "flush_l2": args.flush_l2,
            }
        )
        try:
            records, elapsed = run_load(args, kernels, sink)
            summary = summarize(records, elapsed)
            sink.write({"type": "run_finished", "timestamp": utc_now(), "summary": summary})
        finally:
            sink.close()
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        print(f"JSONL: {output}", file=sys.stderr)
        return 0 if args.allow_errors or summary["failed"] == 0 else 1
    except (
        BenchmarkServerError,
        ProtocolError,
        TransportError,
        OSError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"stress_gpu: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
