# Builtin Tools

A built-in is a function supplied by the server. Call it by its `builtin.*` name
in a `Program.run` instruction, or import `kcoral.builtin` inside uploaded Python.
The [client guide](../client_guide.md) explains how to choose and combine them.
GPU means graphics processing unit; CPU means central processing unit.

## Function overview

| `fn` | Arguments | Returns |
|---|---|---|
| `builtin.randn` | `spec = {shape, dtype, seed?}` | a random tensor |
| `builtin.empty` | `spec = {shape, dtype}` | an uninitialized tensor |
| `builtin.zeros` | `spec = {shape, dtype}` | a zero tensor |
| `builtin.compile_tirx` | `(kernel, bindings?)` — `bindings` binds `T.constexpr` dimensions | a compiled module |
| `builtin.compile_cuda` | `(source, cfg?)` — `cfg = {extra_cuda_cflags?}` | the module's exported function |
| `builtin.compile_cuda_binary` | `(source, cfg)` — `cfg = {arch, extra_cuda_cflags?}` | shared-object bytes compiled for `arch`, such as `sm_100a` |
| `builtin.compile_cutedsl` | `(kernel, *tensors, cfg?)` — the tensors it specializes on; `cfg = {options?}` | a compiled kernel |
| `builtin.compile_triton` | `(kernel, *args, cfg)` — the args it specializes on; `cfg = {grid, **launch keywords}` | a callable bound to that grid |
| `builtin.benchmark` | `(mod, *tensors, cfg?)` — `cfg = {warmup_ms?, repeat_ms?, warmup?, repeat?, flush_l2?}` | timing statistics |
| `builtin.check_close` | `(actual, expected, cfg?)` — `cfg = {atol?, rtol?}` | comparison statistics |
| `builtin.assert_close` | same as `check_close` | comparison statistics; fails on mismatch |

`benchmark` returns `latency_ms_median`, `latency_ms_mean`, `latency_ms_min`,
`latency_ms_max`, `activities_stable`, `flush_l2`, `warmup`, and `repeat`.
Each latency is the CUPTI span from the earliest to the latest GPU activity of
one call — kernels, copies, and memsets, plus host time between them; the flush
and host work outside those endpoints are excluded. `activities_stable` is
`false` when the iterations did not all launch the same activities.

`check_close` and `assert_close` compare on `actual`'s device, so `expected` may
be a CPU tensor, and return `passed`, `max_abs_err`, `max_rel_err`,
`rtol`, and `atol`.

The compilation builtins are registered `cpu_only`, so a worker drops its GPU
lease while their host compilation runs and another worker measures meanwhile.
CUDA C compilation is split at that boundary: nvcc and linking run without the
lease, then the worker reacquires it before loading the shared object and
registering its CUDA module. A new builtin should declare `cpu_only` only when its
off-lease phase touches no GPU at all. Any driver/module-loading finalization must
be deferred until the engine reacquires the lease; otherwise it can perturb a
neighbouring worker's kernel or timing. A `get_function` handle declared
`cpu_only` runs off the lease the same way, and is checked.

## Tensor creation

`randn`, `zeros` and `empty` accept a specification with `shape` and optional
`dtype` (default `float16`). `randn` also accepts `seed` for reproducible input
and requires a floating-point type. `zeros` initializes every element to zero;
`empty` leaves values uninitialized. The returned tensor lives on the worker's
GPU until explicitly returned or the request ends. Supported element types and
upload choices are listed under [tensors](../client_guide.md#tensors).

## Compilation

| Function | Configuration | Required worker environment |
| --- | --- | --- |
| `compile_tirx` | Optional `bindings` supplies compile-time parameters; concrete kernels reject bindings | TVM, a tensor compiler, and its TIRx kernel language |
| `compile_cuda` | Optional `extra_cuda_cflags` adds compiler flags; the server chooses the GPU architecture | CUDA toolkit, a host C++ compiler, `ninja`, and TVM FFI |
| `compile_cuda_binary` | Required `arch` comes from the GPU server's `Client.target()`; optional `extra_cuda_cflags` | CPU compilation worker with the CUDA toolchain and TVM FFI |
| `compile_cutedsl` | Specializes on supplied tensors; optional `options` configures compilation | NVIDIA CuTeDSL, the Python language for CuTe GPU kernels |
| `compile_triton` | Required `grid`; other configuration keys are launch options | Triton, a GPU kernel programming language and compiler |

TVM FFI is TVM's foreign-function interface for exchanging tensors and calling
compiled code. `compile_cuda_binary` returns shared-library bytes and does not
load or execute them; upload those bytes in a later GPU request. The other
compilation functions return request-local callable objects.

Missing dependencies produce an `unavailable` failure. Invalid configuration
produces `parse`; compiler failures produce `compile`. See the
[language guidance](../tutorials/benchmark-kernel.md#languages-supported-by-remote-compilation)
and [library protocol](../protocol.md#library) for source conventions and exports.

## Measurement configuration

CUPTI is NVIDIA's CUDA Profiling Tools Interface. `benchmark` uses its activity
timestamps to measure each invocation's GPU activity span. L2 is the GPU's
second-level cache; flushing it avoids measuring a previously cached input.

| `cfg` key | Default | Meaning |
|---|---|---|
| `warmup_ms` | `25` | How long to warm up. The server times 5 calls, then runs as many iterations as fit the budget |
| `repeat_ms` | `100` | How long to spend on timed iterations, converted to a count the same way |
| `warmup` | — | An explicit warmup iteration count, used instead of the budget |
| `repeat` | — | An explicit timed iteration count, used instead of the budget |
| `flush_l2` | `true` | Zero a buffer twice the size of L2 before every call, outside the timed span, so each call starts with a cold cache |

Explicit iteration counts override their respective millisecond budgets. Setting
both counts also skips the five-call estimate. Keep `activities_stable` with
the latency result: a false value means iterations launched different activities.
See [measurement guidance](../tutorials/benchmark-kernel.md#measuring) for interpreting the span.

## Correctness configuration

`check_close` and `assert_close` accept `rtol` (relative tolerance, default
`1e-2`) and `atol` (absolute tolerance, default `1e-3`). They return `passed`,
`max_abs_err`, `max_rel_err`, `rtol` and `atol`. Comparison runs on the actual
tensor's device, so the expected value may be a CPU tensor. `check_close`
reports a mismatch as data; `assert_close` raises a `correctness` failure and
stops subsequent instructions.

## Scheduling and failures

`cpu_only` marks a function that must not access the GPU. When invoked as its own
`run` instruction, it releases the worker's exclusive GPU lease while it runs.
Compiling from inside uploaded Python does not release that lease. Keep compile
operations at instruction level when they should overlap another request's
measurement. The [protocol error table](../protocol.md#errors) defines failures.
