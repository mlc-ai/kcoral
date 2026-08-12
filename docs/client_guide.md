# Writing a benchmark program

This guide shows how to get a kernel timed on the server, and which choices
matter along the way. [`protocol.md`](protocol.md) is the field-level
specification — every instruction, every error code — but you should not need it
to write a first program.

## The shape of a program

Every benchmark follows the same six steps: upload the kernel, make the tensors,
compile, check correctness, time it, return the results.

```python
import numpy as np
from benchmark_server import Client, Program

KERNEL = r"""
from __future__ import annotations
from tvm.script import tirx as T

@T.jit
def main(A: T.Buffer((N,), "float32"), B: T.Buffer((N,), "float32"), *, N: T.constexpr):
    T.device_entry()
    i = T.cta_id([N])
    t = T.thread_id([1])
    B[i] = A[i] + 1.0
"""
REFERENCE = "def main(a):\n    return a + 1.0\n"

program = Program()
kernel = program.upload(id="kernel", kind="module", source=KERNEL)
reference = program.upload(id="reference", kind="module", source=REFERENCE)

src = program.upload(id="src", kind="tensor", value=np.arange(256, dtype=np.float32))
dst = program.run(id="dst", fn="builtin.empty", args=[{"shape": [256], "dtype": "float32"}])

compiled = program.run(id="compiled", fn="builtin.compile_tirx", args=[kernel, {"N": 256}])
program.run(id="invoke", fn=compiled, args=[src, dst])

expected = program.run(id="expected", fn=reference, args=[src])
check = program.run(id="check", fn="builtin.assert_close", args=[dst, expected])
timing = program.run(id="timing", fn="builtin.benchmark", args=[compiled, src, dst])

program.return_(key="check", value=check)
program.return_(key="timing", value=timing)

with Client("http://localhost:8000") as client:
    result = client.execute(program, timeout_seconds=120)
print(result.results["timing"]["latency_ms_median"])
```

Four rules the example relies on:

- **Handles live for one request.** `kernel`, `compiled` and the rest are names
  inside this one program. A second `client.execute()` call shares nothing with
  the first, so every program has to upload everything it needs.
- **`run` computes a value but does not send it back.** It binds the result to a
  handle that later instructions can use; the response carries only what
  `return_` selects. Above, `invoke` runs the kernel and the client never sees
  its result, while `timing` comes back because a `return_` names it.
- **The entry object is found by name.** A module upload binds one object out of
  its source: the one named by `entry` if the upload sets it, otherwise `main`,
  otherwise the source's single top-level `def` or `class`. A source with two
  top-level definitions, no `main`, and no `entry` is ambiguous — the upload
  fails, and the error lists the candidates it found.
- **An uploaded module need not be a kernel.** `REFERENCE` above is ordinary
  Python executed on the worker, torch included, so `def main(a, b): return a @ b`
  is a perfectly good baseline to measure against.

Runnable versions of all this live in [`../examples`](../examples):
`remote_compile_client.py` (four languages, compiled on the server) and
`library_upload_client.py` (built on the client, uploaded prebuilt).

## Where to compile

A kernel can be built by the server, from source you upload, or built by you and
uploaded as a finished shared object. The server supports both equally — it runs
whatever program you send it.

**Compiling on the server is the recommended starting point.** Upload the kernel
as source text and let one of the `builtin.compile_*` builtins build it. The
server compiles for its own GPU architecture, and all four compile builtins are
registered `cpu_only`, meaning the worker gives up the GPU while compiling — so
your compile does not consume benchmark time, and another worker measures
meanwhile.

**Upload a prebuilt library** when your kernel needs custom compilation that the
server's `compile_*` builtins cannot support. You compile the kernel and upload
the library; the server only loads the compiled object and calls `entry`.

| | Server-side compile | Prebuilt library |
|---|---|---|
| Use when | The build is a standard one | Custom compilation the `compile_*` builtins cannot support |
| Client needs | Nothing | A CUDA toolchain and the right arch |
| Build surface | `extra_cuda_cflags` (CUDA C), `options` (CuTeDSL) — nothing else | Anything you can run |
| Repeat cost | Recompiled, except CUDA C (disk-cached) and Triton (its own cache) | Cached by hash — a resubmission re-sends nothing |
| Fails as | A `compile` error naming the diagnostic | `cudaErrorNoKernelImageForDevice` at launch, if built for the wrong arch |

In practice, reach for a library when the build spans several translation units,
comes out of a code generator, needs flags the builtins do not expose, or is a
CUTLASS-heavy recipe you already have working. Anything else is cheaper to
compile on the server.

A library must be built for the GPU the server runs, not the one sitting in your
client machine:

```python
arch = client.target()["arch"]   # e.g. "sm_100a"
```

`protocol.md` covers the three producers a library may come from
(`TVM_FFI_DLL_EXPORT_TYPED_FUNC`, `tvm.Executable.export_library`, and CuTeDSL's
`--enable-tvm-ffi`) and the link flags each one needs.

## Languages supported by remote compilation

All four follow the same upload-then-compile shape; what differs is what the
compile call has to be told.

| Language | Upload | Compile call | Watch for |
|---|---|---|---|
| TIRx | `source` | `compile_tirx(kernel, bindings?)` | `bindings` supplies a `@T.jit` kernel's `T.constexpr` values; a `@T.prim_func` is already concrete and rejects them. The source must open with `from __future__ import annotations`, or a shape annotation like `T.Buffer((N,), dtype)` evaluates at `def` time and raises `NameError: N` |
| CUDA C | `source`, `language="cuda"`, `entry` | `compile_cuda(kernel, cfg?)` | `void f(tvm::ffi::TensorView, ...)`; `main` is rejected; builds are cached on disk, so recompiling the same source is much cheaper than the first build |
| CuTeDSL | `source`, `entry` | `compile_cutedsl(kernel, *tensors, cfg?)` | Specializes on the tensors, so pass the ones it will run on; nothing is cached |
| Triton | `source`, `entry` | `compile_triton(kernel, *args, cfg)` | `cfg["grid"]` is required; pass scalars and constexprs positionally; other `cfg` keys are launch keywords |

If a server lacks the toolchain a builtin needs, that builtin fails with
`unavailable` and the rest of the server keeps working. `GET /health` lists the
`versions` actually installed, so check there before assuming a language is
available.

## Measuring

`builtin.benchmark(mod, *tensors, cfg?)` reports **per-iteration GPU kernel time
measured by CUPTI** (through triton's proton profiler) rather than wall time, so
launch overhead and host-side work stay out of the number.

| `cfg` key | Default | Meaning |
|---|---|---|
| `warmup_ms` | `25` | How long to warm up. The server times 5 calls, then runs as many iterations as fit the budget |
| `repeat_ms` | `100` | How long to spend on timed iterations, converted to a count the same way |
| `warmup` | — | An explicit warmup iteration count, used instead of the budget |
| `repeat` | — | An explicit timed iteration count, used instead of the budget |
| `flush_l2` | `true` | Zero a buffer twice the size of L2 before every call, outside the timed span, so each call starts with a cold cache |

It returns `latency_ms_median`, `latency_ms_mean`, `latency_ms_min` and
`latency_ms_max`, along with the `flush_l2`, `warmup` and `repeat` it actually
used.

- **Prefer the millisecond budgets to explicit counts.** `warmup_ms` and
  `repeat_ms` adapt to the kernel, so a microsecond kernel and a millisecond
  kernel both get a sensible number of iterations. Set `warmup`/`repeat` when
  two runs have to use identical counts to be comparable; setting both of them
  also skips the 5-call estimate.
- **Leave `flush_l2` on** unless you specifically mean to measure a cache-hot
  kernel. With it off, a small kernel reads its input straight out of L2 and
  reports a latency it would never reach in a real pipeline.
- **Account for GPU time with `lease_held_ms`, not `elapsed_ms`.** Workers take
  turns on a GPU through a lease, and `lease_held_ms` is the part of the request
  that actually occupied the card. The remainder,
  `elapsed_ms - lease_wait_ms - lease_held_ms`, happened off the GPU — compiling,
  mostly.

## Checking correctness

`builtin.check_close(actual, expected, cfg?)` compares two tensors and returns
`passed`, `max_abs_err`, `max_rel_err`, `rtol` and `atol`. A mismatch is data
rather than a failure, so the program carries on. `builtin.assert_close` runs
the same comparison but raises a `correctness` failure when it does not pass,
which stops the program — use it when there is no point timing a kernel that
computes the wrong answer. Both default to `rtol=1e-2` and `atol=1e-3`.

Compare on the server against a Python reference module, the way the first
example does, rather than returning the output and comparing on the client. That
avoids shipping the output tensor back, and it keeps working for tensors too
large to return at all.

## Tensors

**Create them on the server** unless the client needs to specify their exact
values:

```python
program.run(id="x", fn="builtin.randn", args=[{"shape": [4096, 4096], "dtype": "bfloat16", "seed": 0}])
program.run(id="y", fn="builtin.zeros", args=[{"shape": [4096, 4096], "dtype": "float32"}])
```

Uploading instead pays for local generation, hashing, and the transfer, which
makes it markedly more expensive than creating the tensor on the server, and
increasingly so as the tensor grows. Re-uploading identical bytes hits the
server's blob cache and skips the transfer, but the cached bytes are still
copied to the GPU on every request, so the gap narrows rather than closes.

Upload when the values themselves matter: a reference you computed locally, a
fixed input your results have to stay reproducible against, or a tensor whose
contents the kernel's work depends on, such as the `q_indptr` and `kv_indptr`
arrays telling a paged-attention kernel where each sequence begins. `kind="tensor"`
accepts a NumPy array, a torch tensor, any object supporting DLPack, or raw
bytes together with `dtype` and `shape`.

`randn` requires a floating-point dtype and takes an optional `seed`. It,
`empty` and `zeros` all fall back to `float16` when `dtype` is omitted, so it is
worth stating explicitly.

Accepted dtypes: `bool`, `uint8`, `int8`, `int16`, `int32`, `int64`, `float16`,
`float32`, `float64`, `bfloat16`, `float8_e4m3fn`, `float8_e5m2`.

Returned tensors arrive as CPU `numpy.ndarray`, with `bfloat16` and the float8
types carried by `ml_dtypes`. To hand one to torch, reinterpret the raw bytes:

```python
torch.from_numpy(value.view(np.uint8)).view(torch.bfloat16)
```

## Handling failures

A submission ends in one of three ways, and the difference between them matters:

```python
result = client.execute(program)
if result.status == "FAILED":
    error = result.error          # kind, message, instruction_index, instruction_id, traceback
```

- **`COMPLETED`** — every instruction ran.
- **`FAILED`** — one instruction failed, and the instructions after it were
  skipped. Returns that already ran are still in `results`, so putting a
  `return_` before a risky instruction preserves the work up to that point.
  `error["kind"]` is one of `parse`, `compile`, `runtime`, `correctness`,
  `serialization`, `unavailable` or `engine`, and `error["instruction_id"]`
  names the instruction that failed.
- **An exception** — the request never produced a program outcome at all.
  `BenchmarkServerError` carries `status_code` and `kind`: `503` with a
  `Retry-After` header means no worker was free, and `504` means the program hit
  `timeout_seconds` (default 300 s, maximum 3600). `TransportError` means the
  request never reached the server, and `ProtocolError` means the response did
  not follow the protocol.
