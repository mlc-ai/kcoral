# Instruction protocol

The server exposes one synchronous endpoint, `POST /benchmark` (plus `GET /health`).
A request body is a **program**: an ordered list of instructions the server runs
on a GPU worker, returning one result per instruction. There is no session state —
every request is self-contained, and any handles it defines live only for that
request.

```
POST /benchmark          Content-Type: application/json
```

## Request envelope

```json
{
  "instructions": [ /* one or more upload / run instructions, in order */ ],
  "options": { "timeout_seconds": 120 }
}
```

| Field | Type | Notes |
|---|---|---|
| `instructions` | array | Non-empty, executed top to bottom. |
| `options` | object | Optional. See [Options](#options). |

Every instruction is an object with a unique string **`id`** and an **`op`** of
either `"upload"` or `"run"`. An instruction's result is referred to elsewhere by
its `id` (a *handle*). There is no control flow: data flows straight through
handles, so every `$ref` must point at an **earlier** instruction (the program is
a straight-line DAG).

---

## `upload`

Hands the server a typed object and binds it to a handle. Uploads are
**content-addressed**: the object is identified by `key = "sha256:" + sha256(canonical_bytes)`,
cached by the server, and can be re-sent later by `key` alone.

```json
{ "id": "kernel", "op": "upload", "kind": "function", "key": "sha256:…",
  "inline": { "source": "…" } }
```

| Field | Type | Notes |
|---|---|---|
| `id` | string | Handle name (unique within the program). |
| `op` | string | `"upload"`. |
| `kind` | string | `"function"` or `"tensor"` (see below). |
| `key` | string | `"sha256:" + hex(sha256(canonical_bytes))`. Verified server-side. |
| `inline` | object | The payload. **Optional** — omit it when the bytes are already cached under `key` (see [Caching](#caching-and-cache_miss)). |

The server recomputes the key from the payload and rejects a mismatch with `400`,
so the two sides can never disagree on identity.

### Kinds and their canonical bytes

The **key is computed over the canonical bytes**, which are fixed per kind:

| kind | `inline` payload | canonical bytes |
|---|---|---|
| `function` | `{ "source": "<python source>" }` | the UTF-8 source bytes |
| `tensor` | `{ "dtype": "<torch dtype>", "shape": [...], "data_b64": "<base64>" }` | `{"dtype":…,"shape":…}` (compact JSON) + `\0` + raw row-major bytes |

- **`function`** — Python source defining `main`. For a TIRx kernel `main` is a
  `@T.jit`; for a reference it is a plain callable invoked through its handle.
- **`tensor`** — `data_b64` is base64 of the tensor's raw row-major bytes in the
  given `dtype`; `dtype` is a torch dtype name (`float16`, `float32`, `bfloat16`, …).

> `kind: "object"` is accepted by the parser but has no canonical byte form yet,
> so object uploads are not usable at the moment.

### Caching and `CACHE_MISS`

`inline` lets the client avoid re-sending bytes the server already has:

1. Send the upload with `inline` present → the server verifies + caches it, then runs.
2. Send it with `key` only (no `inline`). If the bytes are cached, it runs. If
   **any** referenced key is absent, the server responds — before running anything —
   with:

   ```json
   { "status": "CACHE_MISS", "missing_keys": ["sha256:…", "sha256:…"] }
   ```

   The client resends the program with `inline` for exactly those keys.

This keeps the fast path (warm cache) payload-free while never assuming the cache.

---

## `run`

Calls a function over earlier results and binds the return value to a handle.

```json
{ "id": "mod",  "op": "run", "fn": "builtin.compile_tirx",
  "args": [ { "$ref": "kernel" }, { "N": 256 } ] }

{ "id": "out",  "op": "run", "fn": { "$ref": "mod" },
  "args": [ { "$ref": "x" }, { "$ref": "y" } ] }
```

| Field | Type | Notes |
|---|---|---|
| `id` | string | Handle for the return value. |
| `op` | string | `"run"`. |
| `fn` | string \| `{"$ref": id}` | A **builtin name**, or a handle that must resolve to something callable (e.g. a compiled module, or an uploaded function). |
| `args` | array | Optional (defaults to `[]`). Positional arguments; each element is either a **literal** (any JSON value) or a **handle** `{"$ref": id}`, resolved to that instruction's value before the call. |

Resolution rules:

- `fn` as a string is looked up in the builtin registry; an unknown name fails the
  instruction (`kind: "runtime"`).
- `fn` as `{"$ref": id}` must resolve to a callable handle. A **compiled module is
  callable**, so a kernel is run in place with `fn` set to the module's handle; a
  tensor handle is not callable and fails (`kind: "runtime"`).
- Each `{"$ref": id}` in `args` is replaced by that handle's value; everything else
  is passed through as a literal.

The return value becomes the instruction's result: structural JSON passes through;
an opaque GPU object (a tensor, a compiled module) is **not transmitted** — it stays
server-side and comes back as `{ "handle": "<id>" }`.

### Builtins

The functions a `run` can name. All tensor arguments are handles; a trailing plain
object is an optional config.

| `fn` | Arguments | Returns |
|---|---|---|
| `builtin.randn` | `spec = {shape, dtype, seed?}` | a random tensor (→ handle) |
| `builtin.empty` | `spec = {shape, dtype}` | an uninitialized tensor (→ handle) |
| `builtin.zeros` | `spec = {shape, dtype}` | a zero tensor (→ handle) |
| `builtin.compile_tirx` | `(kernel, bindings?)` — `bindings` binds `T.constexpr` dims, e.g. `{"N": 256}` | a compiled module (→ handle) |
| `builtin.benchmark` | `(mod, *tensors, cfg?)` — `cfg = {warmup?, repeat?}` | `{latency_ms, warmup, repeat}` |
| `builtin.check_close` | `(actual, expected, cfg?)` — `cfg = {atol?, rtol?}` | `{passed, max_abs_err, rtol, atol}` |
| `builtin.assert_close` | same as `check_close` | same on success; **fails** the instruction (`kind: "correctness"`) on mismatch |

`check_close` is a measurement (a mismatch is data, the instruction stays `OK`);
`assert_close` treats a mismatch as a failure that short-circuits the rest of the
program. Pick by intent: benchmark-a-wrong-kernel-anyway vs. abort-on-mismatch.

---

## Options

| Field | Type | Default | Notes |
|---|---|---|---|
| `timeout_seconds` | number | `300` | Per-request GPU execution deadline. Clamped to the server max (`3600`). Exceeding it kills + respawns the worker and returns `504`. |

---

## Response

For any program the server actually ran, the HTTP status is **`200`** and the body is:

```json
{
  "status": "COMPLETED",
  "results": [ { "id": "…", "op": "…", "status": "…", "value": … }, … ]
}
```

- **`status`** (body) — `"COMPLETED"` if every instruction is `OK`; `"FAILED"` if any
  instruction failed (its successors are then `SKIPPED`).
- **`results`** — one object per instruction, in program order.

### Result object

| Field | When present | Notes |
|---|---|---|
| `id`, `op` | always | Echo the instruction. |
| `status` | always | `"OK"`, `"FAILED"`, or `"SKIPPED"`. |
| `value` | `OK` runs | The structural result, or `{ "handle": "<id>" }`. Uploads and in-place runs (return `None`) have no `value`. |
| `error` | `FAILED` / `SKIPPED` | On `FAILED`: `{ "kind": …, "message": … }`. On `SKIPPED`: `{ "reason": "predecessor_failed" }`. |

`error.kind` names the failing stage — `parse`, `compile`, `runtime`,
`correctness`, `unavailable` (the instruction needs an optional server dependency,
e.g. tvm, that isn't installed), or `engine` (an unexpected server fault).
`timeout` is not a per-instruction kind; it appears only at the top level on a
`504` (see below).

### Status / error codes

| HTTP | Body | Meaning |
|---|---|---|
| `200` | `status: COMPLETED` | Ran; every instruction succeeded. |
| `200` | `status: FAILED` | Ran; an instruction failed (rest `SKIPPED`). |
| `200` | `status: CACHE_MISS` | Not run; resend the `missing_keys` uploads with `inline`. |
| `400` | `{error}` | Malformed request (bad JSON, unknown op/kind, key mismatch, forward `$ref`). |
| `503` | `{error}` | All workers busy (has `Retry-After`). |
| `504` | `status: ERROR, error.kind: timeout` | Execution exceeded the deadline. |
| `500` | `status: ERROR, error.kind: engine` | Worker crashed. |

`200` means "the server processed your request," **not** "your kernel is correct" —
a compile error or a failed `assert_close` is a well-formed `200` response whose body
reports the failure. Non-`200` is reserved for transport/infra problems.

---

## Worked example

Upload an input tensor and a TIRx kernel, compile the kernel, run it on the
tensor, assert it matches a reference, and benchmark it.

**Request**

```json
{
  "instructions": [
    { "id": "kernel", "op": "upload", "kind": "function", "key": "sha256:…",
      "inline": { "source": "from __future__ import annotations\nfrom tvm.script import tirx as T\n@T.jit\ndef main(A: T.Buffer((N,), \"float32\"), B: T.Buffer((N,), \"float32\"), *, N: T.constexpr):\n    T.device_entry()\n    i = T.cta_id([N])\n    t = T.thread_id([1])\n    B[i] = A[i] + 1.0\n" } },
    { "id": "reffn", "op": "upload", "kind": "function", "key": "sha256:…",
      "inline": { "source": "def main(a):\n    return a + 1.0\n" } },
    { "id": "a",   "op": "upload", "kind": "tensor", "key": "sha256:…",
      "inline": { "dtype": "float32", "shape": [256], "data_b64": "…" } },
    { "id": "out", "op": "run", "fn": "builtin.empty", "args": [ { "shape": [256], "dtype": "float32" } ] },
    { "id": "mod", "op": "run", "fn": "builtin.compile_tirx", "args": [ { "$ref": "kernel" }, { "N": 256 } ] },
    { "id": "run", "op": "run", "fn": { "$ref": "mod" }, "args": [ { "$ref": "a" }, { "$ref": "out" } ] },
    { "id": "ref", "op": "run", "fn": { "$ref": "reffn" }, "args": [ { "$ref": "a" } ] },
    { "id": "chk", "op": "run", "fn": "builtin.assert_close", "args": [ { "$ref": "out" }, { "$ref": "ref" } ] },
    { "id": "perf","op": "run", "fn": "builtin.benchmark", "args": [ { "$ref": "mod" }, { "$ref": "a" }, { "$ref": "out" }, { "warmup": 10, "repeat": 50 } ] }
  ],
  "options": { "timeout_seconds": 120 }
}
```

**Response (correct kernel)**

```json
{
  "status": "COMPLETED",
  "results": [
    { "id": "kernel", "op": "upload", "status": "OK" },
    { "id": "reffn",  "op": "upload", "status": "OK" },
    { "id": "a",   "op": "upload", "status": "OK" },
    { "id": "out", "op": "run", "status": "OK", "value": { "handle": "out" } },
    { "id": "mod", "op": "run", "status": "OK", "value": { "handle": "mod" } },
    { "id": "run", "op": "run", "status": "OK" },
    { "id": "ref", "op": "run", "status": "OK", "value": { "handle": "ref" } },
    { "id": "chk", "op": "run", "status": "OK", "value": { "passed": true, "max_abs_err": 0.0, "rtol": 0.01, "atol": 0.001 } },
    { "id": "perf","op": "run", "status": "OK", "value": { "latency_ms": 0.0073, "warmup": 10, "repeat": 50 } }
  ]
}
```

**Response (wrong kernel — `assert_close` fails)**

```json
{
  "status": "FAILED",
  "results": [
    "… earlier instructions OK …",
    { "id": "chk",  "op": "run", "status": "FAILED",
      "error": { "kind": "correctness", "message": "outputs differ: max_abs_err=1.0 exceeds atol=0.001, rtol=0.01" } },
    { "id": "perf", "op": "run", "status": "SKIPPED",
      "error": { "reason": "predecessor_failed" } }
  ]
}
```

See `examples/example_client.py` for a runnable client that computes the keys and
posts this program.
