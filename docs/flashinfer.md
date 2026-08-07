# FlashInfer Trace client

FlashInfer Trace is a data interchange format for kernel definitions,
implementations, workloads, and evaluation results. The `flashinfer_bench`
Python package included in this distribution owns the self-contained `TraceSet`,
`Definition`, `Solution`, `Workload`, and `Trace` models plus all remote benchmark
orchestration. It does not import or install the external `flashinfer-bench`
project.

The server remains generic. It caches tensor bytes, assigns requests to graphics
processing unit (GPU) workers, executes instructions, and returns structured
results. The client owns the remaining responsibilities:

1. `TraceSet` loads definitions, solutions, workloads, and existing evaluations.
2. `FlashInferBenchmark` prepares each workload once and shares those inputs
   across its solutions.
3. `FlashInferProgramBuilder` creates one `/execute` program for each solution
   and workload pair. The program contains the reference, candidate solution,
   output checks, and repeated timing.
4. The client submits programs concurrently. The benchmark-server worker pool
   handles GPU assignment and queueing.
5. `FlashInferResultMapper` converts correctness and timing results into bundled
   `Evaluation` and `Trace` objects.

## Install and run

Install the ordinary package in the client and server environments:

```bash
pip install benchmark-server
```

Start benchmark-server, load a Trace directory, and run the remote benchmark:

```python
from flashinfer_bench import (
    BenchmarkConfig,
    FlashInferBenchmark,
    TraceSet,
)

trace_set = TraceSet.from_path("Example-FlashInfer-Trace")
config = BenchmarkConfig(iterations=50, num_trials=3)

with FlashInferBenchmark(
    trace_set,
    "http://127.0.0.1:8000",
    config,
) as benchmark:
    result = benchmark.run_all(dump_traces=True, resume=True)
```

`run_all` returns a `TraceSet` containing status, correctness, performance,
environment, and log records. With `dump_traces=True`, `TraceSet.add_traces`
also appends each result to the established directory layout:
`traces/<author>/<op_type>/<definition>.jsonl`.

## Trace data model

The `flashinfer_bench` API covers the remote benchmark workflow:

- axes and tensor specifications: `AxisConst`, `AxisVar`, and `TensorSpec`;
- kernels: `Definition`, `BuildSpec`, `SourceFile`, and `Solution`;
- workload inputs: `RandomInput`, `ScalarInput`, and `SafetensorsInput`;
- results: `Correctness`, `Performance`, `Environment`, `Evaluation`, and
  `Trace`;
- execution settings: `BenchmarkConfig`, `EvalConfig`, and
  `ResolvedEvalConfig`;
- directory loading, JSONL result persistence, resume state, and summary counts
  through `TraceSet`.

Unknown JSON fields are ignored so datasets may contain metadata outside this
minimum viable implementation. Unsupported evaluation or build behavior still
produces an explicit failure rather than silently changing semantics.

## One request per solution and workload

Each request contains all trials for one solution and workload pair. The
client-generated program performs these steps:

1. upload the reference, solution, and a small Python validation harness;
2. upload tensor inputs and pass scalar inputs as ordinary values;
3. run the reference and candidate for every correctness trial;
4. return output shape and data type checks, then stop on a mismatch;
5. return numerical errors, then stop if they exceed the configured tolerance;
6. time the candidate and optionally the reference with configured warmups and
   repeats;
7. return timing statistics for conversion into an evaluation trace.

The validation harness is ordinary Python source uploaded by the client.
benchmark-server adds no FlashInfer-specific instruction, built-in function, or
scheduler.

By default the client queries `/health` and uses the reported GPU worker count
as its request concurrency. `max_workers` can override this value. The server
continues to own GPU assignment, queue time, timeouts, process isolation, and
worker restarts.

## Tensor cache

Every non-scalar input uses `Program.upload(kind="tensor")`. The program
computes a SHA-256 content key over the raw tensor bytes. `Client.execute` first
sends the program and keys without binary data:

- a cached key proceeds directly to execution;
- a missing key produces `CACHE_MISS` with the exact missing-key set;
- the client retries once with only those binary objects.

The client uses stable random seeds and prepares each workload only once.
Different solution requests for the same workload therefore share content keys.
Safetensors inputs use the same byte cache. The client reads the documented
safetensors layout directly and needs no additional package for that format.

## Supported scope and extension points

The default implementation supports:

- numerical evaluation with strict shape and data type checks and configurable
  relative and absolute tolerances;
- multiple correctness trials and repeated timing;
- deterministic random, scalar, and safetensors inputs;
- return-value and destination-passing solutions;
- single-file Python solutions;
- single-file Triton solutions, where Triton kernels are embedded in Python;
- single-file NVIDIA CUDA solutions that export a TVM Foreign Function
  Interface (TVM FFI) function and require no extra linked libraries.

The default implementation explicitly rejects multi-file solutions, additional
CUDA link dependencies, standalone C++, TileLang, match-ratio correctness, and
specialized statistical evaluators.

All FlashInfer-specific logic and extension points live under the
`flashinfer_bench` package. These client interfaces can be replaced without
changing the protocol or server:

- `WorkloadInputProvider` for other generators or external data stores;
- `SolutionAdapter` for other languages, packages, or prebuilt libraries;
- `FlashInferProgramBuilder` for specialized correctness and timing flows;
- `FlashInferResultMapper` for additional metrics or result fields.
