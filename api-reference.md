# GPU Server v2 API Reference

## 1. Overview

GPU Server v2 lets a client send a program and input files to a remote GPU server, run them, and receive the result.

Main goals:

- **General**: the server does not care what the task does; task logic stays in the submitted program.
- **Isolated**: one GPU runs one request at a time, and runtime state is cleaned up when the request ends.
- **Simple**: one request carries everything needed for one run, with no sessions or cross-request objects.
- **Efficient**: identical files can reuse the cache without changing the files or execution behavior seen by the program.

The first version is for trusted callers and does not isolate malicious code.

---

## 2. Interfaces

### 2.1 `POST /execute`

Execute a program on one GPU, optionally uploading referenced file content inline.

The request is synchronous. The connection remains open while the request is queued and executed, and a successful response carries the entry function's return value.

#### 2.1.1 Request Format

Content type:

```text
multipart/form-data
```

`multipart/form-data` is an HTTP multipart request format that can carry JSON metadata and multiple binary files in one request body. HTTP stands for Hypertext Transfer Protocol, and JSON stands for JavaScript Object Notation, a structured text data format.

The request body supports the following parts:

| Part name | Content type | Required | Description |
|---|---|---:|---|
| `job` | `application/json` | Yes | Execution settings and file manifest |
| `blob:<sha256>` | `application/octet-stream` | No | Optional file content identified by its full SHA-256 hash |

A request must contain exactly one `job` part and zero or one blob part for each unique hash referenced by `job.files`.

A manifest is JSON data describing file paths and their content references. `job.files` maps each working-directory-relative path to an object of the form `{"blob": "<sha256>"}`, where the value is the file content's full lowercase SHA-256 hash. Uploaded content does not carry file paths in multipart part names.

Example blob part names:

```text
blob:81d4<remaining SHA-256 hexadecimal characters>
blob:97ab<remaining SHA-256 hexadecimal characters>
```

Each `blob` value must contain 64 lowercase hexadecimal characters. A multipart part name is the concatenation of `blob:` and that value. The server recomputes each supplied part's SHA-256 and verifies its name. SHA-256 is a cryptographic hash function used here for content addressing, integrity verification, and file caching.

Each hash referenced by `job.files` may be supplied by an inline multipart part or resolved from an immutable object already in the server cache. The server verifies every inline blob. It stores manifest-referenced blobs and immediately holds references that prevent eviction; it handles unreferenced blobs according to the next paragraph. The server then performs one atomic cache operation that resolves and acquires the remaining references before the request is queued or assigned a GPU. If any hash is unavailable, the server returns HTTP 404 with `error` set to `blob_not_found` and a `missing_blobs` array. It does not queue the request, assign a GPU, or execute the script.

A duplicate blob part or content that does not match the hash in its part name produces `invalid_request`. An inline blob not referenced by `job.files` is discarded after hash verification and is not stored in the cache; request processing continues, the successful response carries an `unused_blob` warning, and the server also records a WARNING log entry. Multiple paths may reference one blob, and multipart part order does not affect request semantics. A one-shot client may still include all referenced blobs inline in `POST /execute` without any preflight request.

##### `job` Object

The required `job` part has the following format:

```json
{
  "language": "python",
  "entry": {
    "file": "main.py",
    "function": "main"
  },
  "files": {
    "main.py": {
      "blob": "<main_sha256>"
    },
    "data/input.bin": {
      "blob": "<input_sha256>"
    }
  },
  "timeout_seconds": 60,
  "stdout_limit_bytes": 1048576,
  "stderr_limit_bytes": 1048576
}
```

| Field | Required | Default | Description |
|---|---:|---|---|
| `language` | No | `"python"` | The entry script's language |
| `entry` | No | See below | The entry location |
| `entry.file` | No | `"main.py"` | The entry file's path in the working directory |
| `entry.function` | No | `"main"` | The entry function name |
| `files` | Yes | None | Mapping from working-directory-relative paths to file-content references |
| `files.<path>.blob` | Yes | None | Full SHA-256 hash of the corresponding file content |
| `timeout_seconds` | No | Server configuration | A positive number specifying the execution timeout in seconds |
| `stdout_limit_bytes` | No | Server configuration | Maximum standard-output bytes included in the response |
| `stderr_limit_bytes` | No | Server configuration | Maximum standard-error bytes included in the response |

`files` is the only required field. When the other fields are omitted, the server uses the default language and entry point, equivalent to executing:

```python
from main import main

return_value = main()
```

In the first version, `language` accepts only `"python"`. Any other value returns `invalid_request`. This field determines how the server loads the entry file and invokes the entry function; a protocol revision must define the loading and invocation rules for each additional language.

`entry.file` must satisfy the file path rules in Section 2.1.1 and must be a key in `files`. `entry.function` must be a no-argument callable defined in the entry file.

The server rejects unknown fields to prevent clients from assuming that an unsupported setting has taken effect.

`job` does not declare outputs. The entry function's return value is the only application-level result.

`stdout_limit_bytes` and `stderr_limit_bytes` must be non-negative integers and cannot exceed the server-configured maximum. A value of `0` excludes that output stream from the response. These limits apply only to API responses and SDK execution results; the logs defined in Section 5 always preserve complete output.

The rest of this document describes behavior using the default entry point, `main.py` and `main()`. Unless stated otherwise, these descriptions also apply to a custom `entry`: replace `main.py` with `entry.file` and `main()` with `entry.function`.

##### File Path Rules

Every key in `files` must:

- use `/` as the path separator;
- be relative to the working directory;
- contain no empty path segments, `.` segments, or `..` segments;
- contain no null characters;
- not begin with `/`;
- not point outside the working directory through a symbolic link;
- be unique within a request;
- not overwrite internal files created by the server.

After path normalization, a path must be exactly identical to the path submitted by the client:

| `files` key | Result |
|---|---|
| `data/input.bin` | Accepted |
| `./input.bin` | Rejected |
| `data/../input.bin` | Rejected |
| `/etc/passwd` | Rejected |

The server rejects duplicate JSON keys and collisions after path normalization, and creates parent directories as needed.

##### Python Entry Conventions

When `language` is `"python"`, the entry file must define a callable with the same name as `entry.function`. With the default values, this is:

```python
def main():
    ...
    return result
```

The entry function must:

- accept no arguments;
- be able to import other Python files provided by the same manifest;
- be able to read artifacts using paths relative to the current working directory;
- see only one GPU at runtime, with that device appearing as `cuda:0`;
- return a supported value;
- finish before the request timeout.

Before importing the entry file, the server changes the current working directory to the request's working directory and places that directory at the beginning of the Python module search path for the duration of the request.

The server imports the entry file as a module. Top-level code in the file runs before the entry function is invoked. Placing the actual work inside the entry function is recommended to make failure locations and execution-time measurements clear.

The server passes no command-line arguments, injects no application object, and does not look for any function name other than `entry.function`.

##### Script Execution Environment

Each GPU has one worker slot, and requests in the same slot execute serially.

Before running a script, the server limits GPU visibility so that the assigned physical GPU appears in the script as:

```text
cuda:0
```

The script uses the server's preconfigured Python interpreter and installed dependencies. Clients cannot select another interpreter or install dependencies through this interface.

The server provides the following environment variables:

| Environment variable | Description |
|---|---|
| `GPU_SERVER_REQUEST_ID` | The unique request identifier used for logging and troubleshooting |
| `GPU_SERVER_WORK_DIR` | The absolute path of the working directory for the current request |

Programs should prefer relative paths when accessing uploaded artifacts. The absolute working-directory path is valid only for the current request and should not be retained for later use.

#### 2.1.2 Response

##### Request Identifier

After an `/execute` request enters the request handler and before the request body is parsed, the server generates a UUID v4. UUID stands for universally unique identifier, and v4 means that the identifier is randomly generated.

The request identifier uses a lowercase canonical UUID string:

```text
7f61b94e-034a-4e80-b67d-eca52bb952cc
```

This value is consistently named `request_id` throughout the protocol and is used across request parsing, queuing, worker-process execution, logging, and response generation.

Every `/execute` response returns the request identifier in an HTTP response header:

```http
X-Request-ID: 7f61b94e-034a-4e80-b67d-eca52bb952cc
```

The JSON metadata in successful and error responses also contains the same `request_id`. For byte-string and tensor return values, this field is in the `result` part of the multipart response.

Clients cannot specify or override `request_id`. The server uses the same value throughout a single HTTP request; a client retry generates a new `request_id`. If business-level duplicate submissions need to be identified in the future, an independent idempotency key should be added; `request_id` must not represent an idempotency relationship.

The value of the `GPU_SERVER_REQUEST_ID` environment variable is exactly identical to the `request_id` in the response.

##### Return Value

The entry function may return a value recursively composed of the following types:

- `None`, `bool`, `int`, finite `float`, and `str`;
- `bytes`, `bytearray`, and `memoryview`;
- tensor objects implementing the DLPack producer protocol, including `tvm_ffi.Tensor`;
- `list`;
- `tuple`;
- `dict` with string keys.

Lists, tuples, and dictionaries may contain the types above at any nesting level. A single return value may contain multiple byte strings and multiple tensors.

DLPack is a standard protocol for exchanging tensors in memory. A DLPack producer object must provide `__dlpack__()` and `__dlpack_device__()`. Common tensor types that the entry function may return directly include:

- `torch.Tensor`;
- `numpy.ndarray`;
- `cupy.ndarray`;
- `jax.Array`;
- `tvm_ffi.Tensor`;
- any other object implementing `__dlpack__()` and `__dlpack_device__()`.

The server first normalizes the tensor to `tvm_ffi.Tensor` from the `apache-tvm-ffi` package. An existing `tvm_ffi.Tensor` is used directly; other DLPack producers are converted through `tvm_ffi.from_dlpack(..., require_contiguous=True)`:

```python
import tvm_ffi


def normalize_tensor(value):
    if isinstance(value, tvm_ffi.Tensor):
        tensor = value
    elif hasattr(value, "__dlpack__") and hasattr(value, "__dlpack_device__"):
        tensor = tvm_ffi.from_dlpack(value, require_contiguous=True)
    else:
        raise UnsupportedReturnTypeError()

    if not tensor.is_contiguous():
        raise ReturnSerializationError("tensor must be C-contiguous")

    return tensor
```

The first version accepts only dense C-contiguous tensors. If a DLPack producer exports a non-contiguous tensor or cannot be imported by `tvm_ffi.from_dlpack`, the server returns `invalid_return_value`. The server does not attempt to make the tensor contiguous automatically. Other objects that do not implement the DLPack producer protocol also produce `invalid_return_value`.

See the official [`tvm_ffi.from_dlpack` documentation](https://tvm.apache.org/ffi/reference/python/generated/tvm_ffi.from_dlpack.html).

###### Return Value Description Tree

The server recursively encodes the Python return value as a description tree:

| Python value | Description node |
|---|---|
| A subtree fully representable as JSON | `{"type": "json", "value": ...}` |
| `bytes`, `bytearray`, `memoryview` | `{"type": "bytes", ...}` |
| A C-contiguous tensor implementing the DLPack producer protocol | `{"type": "tensor", ...}` |
| `list` | `{"type": "list", "items": [...]}` |
| `tuple` | `{"type": "tuple", "items": [...]}` |
| `dict` | `{"type": "dict", "items": {...}}` |

A JSON node may contain:

- `null`;
- Boolean values;
- integers;
- finite floating-point numbers;
- strings;
- arrays containing only JSON values;
- objects with string keys and JSON values.

A tuple remains a `tuple` in the description tree, allowing the client to restore it as a tuple. Non-finite floating-point numbers, including positive infinity, negative infinity, and not-a-number values, are not JSON values.

If an entire subtree can be represented as JSON, the server combines that subtree into a single `json` node to avoid generating a description node for every scalar.

###### Pure JSON Return Values

When the return value consists entirely of JSON values, the response content type is `application/json`.

Example `main.py`:

```python
def main():
    return {
        "correct": True,
        "median_ms": 0.128,
        "samples_ms": [0.127, 0.128, 0.131],
    }
```

Successful response:

```json
{
  "status": "ok",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "return": {
    "type": "json",
    "value": {
      "correct": true,
      "median_ms": 0.128,
      "samples_ms": [0.127, 0.128, 0.131]
    }
  },
  "elapsed_ms": 1245.6,
  "queue_ms": 18.2,
  "stdout": "",
  "stderr": ""
}
```

###### Return Values Containing Binary Values

When a byte string or tensor appears in the return value tree, the response content type is `multipart/form-data` and contains:

| Part name | Content type | Description |
|---|---|---|
| `result` | `application/json` | Execution metadata and the complete return value description tree |
| `return:<index>` | `application/octet-stream` | The raw bytes of one byte string or tensor |

`<index>` starts at `0`. The server assigns part identifiers to binary values in depth-first traversal order. A part identifier is represented as a string in JSON, for example:

```json
"part": "return:0"
```

The string is exactly identical to the multipart part's `name`:

```http
Content-Disposition: form-data; name="return:0"
Content-Type: application/octet-stream
```

The client must use the `part` field in the description node to locate the data and should not calculate the numbering or parse the number in the identifier. A part identifier is valid only within the current HTTP response.

Each binary node also carries:

| Field | Description |
|---|---|
| `part` | The multipart part name in the current response |
| `size` | The number of raw data bytes |
| `sha256` | The full SHA-256 hash of the raw data |

`part` locates the data, while `sha256` verifies its integrity. Multiple return value nodes may refer to the same part to reuse binary data with exactly identical content.

###### Byte String Nodes

A byte string node has the following format:

```json
{
  "type": "bytes",
  "part": "return:0",
  "size": 15,
  "sha256": "<sha256>"
}
```

The corresponding multipart part stores the raw bytes of the `bytes`, `bytearray`, or `memoryview` value.

###### Tensor Nodes

A tensor node has the following format:

```json
{
  "type": "tensor",
  "dtype": "float16",
  "shape": [32, 128],
  "part": "return:0",
  "size": 8192,
  "sha256": "<sha256>"
}
```

The server normalizes a DLPack producer to `tvm_ffi.Tensor`, verifies that it has a C-contiguous layout, synchronizes its device, and copies it to host memory. Tensor bytes are stored in contiguous row-major order, and multibyte scalar values use little-endian byte order. A non-contiguous tensor produces `invalid_return_value`.

In a tensor node:

- `dtype` indicates the element data type;
- `shape` indicates the length of each dimension;
- `size` must equal the product of all dimension lengths in the shape multiplied by the number of bytes per element.

###### Nested Return Value Example

When `output_tensor` is any supported DLPack producer object, `main()` may return:

```python
def main():
    return {
        "correct": True,
        "metrics": {
            "median_ms": 0.128,
            "samples_ms": [0.127, 0.128, 0.131],
        },
        "outputs": [
            output_tensor,
            b"binary metadata",
        ],
    }
```

The return value description tree in the `result` part is:

```json
{
  "status": "ok",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "return": {
    "type": "dict",
    "items": {
      "correct": {
        "type": "json",
        "value": true
      },
      "metrics": {
        "type": "json",
        "value": {
          "median_ms": 0.128,
          "samples_ms": [0.127, 0.128, 0.131]
        }
      },
      "outputs": {
        "type": "list",
        "items": [
          {
            "type": "tensor",
            "dtype": "float16",
            "shape": [32, 128],
            "part": "return:0",
            "size": 8192,
            "sha256": "<sha256>"
          },
          {
            "type": "bytes",
            "part": "return:1",
            "size": 15,
            "sha256": "<sha256>"
          }
        ]
      }
    }
  },
  "elapsed_ms": 1245.6,
  "queue_ms": 18.2,
  "stdout": "",
  "stderr": ""
}
```

The multipart response also contains the two binary parts `return:0` and `return:1`.

###### Type and Serialization Limits

If the return value has a type not listed in this section, the server returns `invalid_return_value`. Examples include:

- generators and iterators;
- open file objects;
- instances of other Python classes that do not implement the DLPack producer protocol;
- functions and modules;
- dictionaries whose keys are not strings.

The server should provide the path of the value that cannot be encoded in the error message, for example:

```json
{
  "status": "error",
  "error": "invalid_return_value",
  "message": "unsupported value at $.outputs[2].metadata",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc"
}
```

The server does not preserve relationships between Python object references. If the same object appears multiple times in the return value tree, the client receives multiple independent references; the corresponding binary content may share the same part.

A circular reference cannot be represented as a finite description tree. The server must proactively check for circular references while recursively constructing the description tree and cannot wait for Python to reach its recursion depth limit.

During this check, the server records the object identities of `list`, `tuple`, and `dict` values on the current recursion path:

1. Before entering a container, if its object identity is already on the current recursion path, a circular reference has been found.
2. When entering a container, add its object identity to the current recursion path.
3. After encoding the container, remove its object identity from the current recursion path.

This set records only the current recursion path, not every object already visited. Consequently, multiple locations may refer to the same non-circular object; those locations are encoded separately.

When a circular reference is found, the server returns `invalid_return_value` and includes the location where the cycle was found in the error message:

```json
{
  "status": "error",
  "error": "invalid_return_value",
  "message": "circular reference at $.outputs[1]",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc"
}
```

After constructing the description tree, the server uses Python `json.dumps` to generate `result`. The call keeps the default `check_circular=True` and sets `allow_nan=False`, providing an additional check for circular references and non-finite floating-point numbers. The custom recursive check remains necessary because a cycle may occur before construction of the description tree is complete.

The server must configure:

- a maximum nesting depth;
- a maximum number of description nodes;
- a maximum number of JSON metadata bytes;
- a maximum number of bytes for an individual binary value;
- a maximum number of bytes for the entire response.

Before sending the HTTP response headers, the server completes traversal of the return value and serializes all binary content to temporary files. This allows the server to return a complete JSON error response if traversal, synchronization, or serialization fails.

##### Successful Response Metadata

Every successful response contains:

| Field | Description |
|---|---|
| `status` | Always `"ok"` |
| `request_id` | The UUID v4 generated by the server for the current HTTP request |
| `return` | The type and representation of the entry function's return value |
| `elapsed_ms` | The time from starting to import the entry file through completion of return value serialization |
| `queue_ms` | The time spent waiting for an available GPU worker process |
| `stdout` | Standard output captured while importing the entry file and executing the entry function |
| `stderr` | Standard error captured while importing the entry file and executing the entry function |
| `warnings` | Optional array of non-fatal warnings; omitted when there are no warnings |

An inline blob that is not referenced by the manifest uses this warning:

```json
{
  "warnings": [
    {
      "code": "unused_blob",
      "blobs": [
        "<sha256>"
      ]
    }
  ]
}
```

`unused_blob` does not prevent execution. The corresponding blob has passed hash verification but is not stored in the cache; clients that intend to pre-upload it should call `POST /blobs`.

`elapsed_ms` includes:

- importing the entry file;
- executing top-level code in the file;
- executing the entry function;
- synchronizing and serializing the return value.

`elapsed_ms` does not include:

- HTTP request upload time;
- queueing time;
- HTTP response download time.

GPU kernel performance should be measured by the entry function using GPU synchronization and timing methods appropriate for the runtime environment. `elapsed_ms` describes the server's total execution time and must not be used as a kernel performance result.

The server uses `stdout_limit_bytes` and `stderr_limit_bytes` to limit standard output and standard error in the response. Limits are measured in raw bytes. The response keeps the first N bytes of each stream and decodes them as UTF-8, replacing invalid byte sequences. If truncation occurs, the response additionally contains:

```json
{
  "stdout_truncated": true,
  "stderr_truncated": false
}
```

#### 2.1.3 Errors

All error responses use `application/json`.

General format:

```json
{
  "status": "error",
  "error": "<error-type>",
  "message": "<human-readable message>",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "stdout": "",
  "stderr": ""
}
```

The `request_id` in an error response is a lowercase canonical UUID v4 and is exactly identical to the `X-Request-ID` response header. If the script has started executing, the response includes `stdout` and `stderr`.

When referenced hashes are unavailable after supplied inline blobs have been stored, the response has this form:

```json
{
  "status": "error",
  "error": "blob_not_found",
  "message": "one or more referenced blobs are missing",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "missing_blobs": [
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  ]
}
```

The SDK performs cache fallback based on `blob_not_found` and the machine-readable `missing_blobs` field.

##### Error Types

| Error | HTTP status code | Description |
|---|---:|---|
| `invalid_request` | 400 | The request format, fields, paths, or entry declaration are invalid |
| `blob_not_found` | 404 | One or more blobs referenced by `job.files` are absent from both inline parts and the server cache |
| `execution_failed` | 400 | Importing the entry file or executing the entry function raises an exception |
| `invalid_return_value` | 400 | The entry function's return value does not conform to the protocol |
| `timeout` | 408 | Execution time exceeds `timeout_seconds` |
| `unavailable` | 503 | No worker process is currently available, or a worker process crashes |
| `internal_error` | 500 | A valid request triggers an internal server error |

`invalid_return_value` covers unsupported Python types, DLPack tensors that cannot be imported or are non-contiguous, non-string dictionary keys, circular references, non-finite floating-point values, and return values that exceed resource limits. If a return value conforms to the protocol but synchronization or serialization still fails, the server returns `internal_error`.

An execution failure response contains a Python traceback:

```json
{
  "status": "error",
  "error": "execution_failed",
  "message": "main() raised RuntimeError: correctness check failed",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "traceback": "...",
  "stdout": "...",
  "stderr": "..."
}
```

A traceback may expose source code and local paths. Deployments serving untrusted callers should disable traceback details or sanitize them.

### 2.2 `POST /blobs/check`

Check which content-addressed blobs are currently absent from the server cache.

#### 2.2.1 Request Format

The request uses `application/json`:

```json
{
  "blobs": [
    "<sha256-1>",
    "<sha256-2>"
  ]
}
```

`blobs` is a batch of unique hashes. Each hash must use the canonical form of exactly 64 lowercase hexadecimal characters.

#### 2.2.2 Response

A successful response uses `application/json` and lists the requested hashes that are currently absent:

```json
{
  "missing": [
    "<sha256-2>"
  ]
}
```

The result is advisory. It does not reserve cache objects or prevent eviction, and it makes no durability promise; in particular, it creates no cache lease. Eviction may occur between this check and a later execution request.

#### 2.2.3 Errors

Error responses use `application/json` and contain `status`, `error`, and `message`. Because this endpoint does not execute an entry function, responses contain no script `stdout` or `stderr`:

| Error | HTTP status code | Description |
|---|---:|---|
| `invalid_request` | 400 | The JSON body is malformed, a hash is invalid, or a hash is duplicated |
| `internal_error` | 500 | The server cannot inspect the cache |

### 2.3 `POST /blobs`

Upload one or more content-addressed blobs without executing a job.

#### 2.3.1 Request Format

The request uses `multipart/form-data` and contains one or more `blob:<sha256>` parts with content type `application/octet-stream`. It contains no `job` part. Each hash must use the canonical 64-character lowercase hexadecimal form, and the server verifies each part's bytes against its declared hash.

#### 2.3.2 Response

A successful response uses `application/json`:

```json
{
  "status": "ok",
  "stored": [
    "<sha256-1>"
  ],
  "already_present": [
    "<sha256-2>"
  ]
}
```

The endpoint is idempotent. Each upload is written to a temporary file, hash-verified, and atomically renamed or published into the cache. Concurrent uploads of the same hash leave one immutable, complete cache file. At startup, the server removes incomplete upload temporary files while preserving complete cache objects.

#### 2.3.3 Errors

Error responses use `application/json` and contain `status`, `error`, and `message`. Because this endpoint does not execute an entry function, responses contain no script `stdout` or `stderr`:

| Error | HTTP status code | Description |
|---|---:|---|
| `invalid_request` | 400 | The multipart body or a part name is malformed, a blob part is duplicated, or content does not match its declared hash |
| `internal_error` | 500 | The server cannot store the uploaded content |

### 2.4 `GET /health`

Return service status, GPU worker-process status, and the number of queued requests.

#### 2.4.1 Request Format

`GET /health` accepts no request body, query parameters, or other execution settings:

```http
GET /health HTTP/1.1
Host: server:8000
```

#### 2.4.2 Response

A successful response has the content type `application/json`:

```json
{
  "status": "ok",
  "gpu_count": 2,
  "queue_length": 3,
  "workers": [
    {
      "gpu_id": 0,
      "status": "busy",
      "uptime_seconds": 3600
    },
    {
      "gpu_id": 1,
      "status": "idle",
      "uptime_seconds": 3580
    }
  ]
}
```

Worker-process status values:

| Status | Description |
|---|---|
| `idle` | Can accept a request |
| `busy` | Is executing a request |
| `restarting` | Is replacing a runtime process that timed out or failed |
| `unhealthy` | Cannot execute requests |

As long as at least one worker process can continue executing requests, the server returns HTTP 200. The response may also include `restarting` or `unhealthy` worker processes. A health check only reads server state and does not execute an additional GPU kernel on each call.

#### 2.4.3 Errors

When no worker process can execute requests, the server returns HTTP 503:

```json
{
  "status": "error",
  "error": "unavailable",
  "message": "no healthy GPU worker is available"
}
```

If the server cannot read its internal state, it returns HTTP 500 with the error type `internal_error`.

---

## 3. Python Client SDK

SDK stands for software development kit. The first version provides a synchronous Python client whose `execute()` method matches the synchronous `POST /execute` interface.

The SDK is a convenience wrapper around the protocol and does not form a security boundary. The server repeats all validation. If the SDK runs in an environment that an automated agent can modify, the agent can bypass or change the SDK. Preventing reward hacking—manipulating the submission path to obtain an undeserved evaluation reward—requires a trusted orchestrator outside the agent-controlled environment to control submission.

The recommended public entry point remains `client.execute(...)`. The optional `client.prepare_files(...)` and `client.execute_prepared(...)` methods expose a cache-aware workflow. The SDK provides no chained request builder, upload session, handle to a server-side object, pickle serialization, or configurable automatic retry policy. The only automatic execution retry is the single missing-blob fallback defined in Section 3.6. The first version provides no asynchronous client.

### 3.1 Client

The primary call is:

```python
from benchmark_server import Client, Entry


with Client("http://server:8000") as client:
    result = client.execute(
        files={
            "main.py": benchmark_source,
            "submission/kernel.py": kernel_source,
            "data/input.bin": input_bytes,
        },
        language="python",
        entry=Entry(file="main.py", function="main"),
        timeout_seconds=60,
    )
```

The public call signatures are:

| API | Contract |
|---|---|
| `Client(base_url, *, headers=None, connect_timeout_seconds=10)` | Creates a client for `base_url`. Optional `headers` are sent with each request. |
| `client.execute(files, *, language="python", entry=Entry(), timeout_seconds=None, stdout_limit_bytes=None, stderr_limit_bytes=None) -> ExecutionResult` | Uploads files and synchronously executes the entry function. |
| `client.prepare_files(files) -> PreparedFiles` | Hashes, deduplicates, and uploads currently missing file content without executing it. |
| `client.execute_prepared(prepared, *, language="python", entry=Entry(), timeout_seconds=None, stdout_limit_bytes=None, stderr_limit_bytes=None) -> ExecutionResult` | Synchronously executes a previously prepared file manifest. |
| `client.health() -> Health` | Synchronously reads server and worker health. |
| `client.close() -> None` | Releases the client's HTTP transport resources. |
| `client.__enter__() -> Client` | Returns the open client. |
| `client.__exit__(...) -> None` | Calls `close()` when the context exits. |

`Entry` is a frozen dataclass with the fields `file: str = "main.py"` and `function: str = "main"`.

`PreparedFiles` is a frozen dataclass with one field, `manifest: Mapping[str, str]`, an immutable mapping from each normalized remote path to its full lowercase SHA-256 hash. It intentionally retains no source bytes.

The common call can omit all execution defaults:

```python
from benchmark_server import Client


with Client("http://server:8000") as client:
    result = client.execute(
        files={"main.py": "def main():\n    return 42\n"},
    )
```

The common `client.execute(files, ...)` call remains the recommended interface. Every execution request sends the required `job`, including explicit `language` and `entry` values and a file manifest. When `timeout_seconds`, `stdout_limit_bytes`, or `stderr_limit_bytes` is `None`, the SDK omits the corresponding field so that the server-configured default applies.

`connect_timeout_seconds` covers only connection establishment. The SDK does not automatically derive a response-reading deadline from the execution timeout, because request upload time and queueing time are excluded from the server's execution timeout.

### 3.2 File Inputs

The public input type is `FileContent = str | bytes | bytearray | memoryview | pathlib.Path`.

UTF-8 means the 8-bit Unicode Transformation Format used here to encode text.

| Value type | Uploaded content |
|---|---|
| `str` | The string encoded as UTF-8 |
| `bytes` | The raw bytes |
| `bytearray` | The value converted to raw bytes |
| `memoryview` | The viewed data converted to raw bytes |
| `pathlib.Path` | The contents read from the local file |

The `files` argument maps remote relative paths to `FileContent` values. Mapping keys become keys in `job.files` and paths in the request's working directory. A plain `str` value always means UTF-8 file content and never denotes a local path; callers must use `pathlib.Path` explicitly for a local file.

Before sending file-related requests, the SDK validates path safety, normalizes all remote paths, and rejects any collision after normalization. Duplicate keys cannot occur in a Python mapping. Before execution, the SDK also verifies that `entry.file` is present in the manifest, requires a positive `timeout_seconds` when one is provided, and requires both output limits to be non-negative integers. These client-side checks improve error reporting; the server repeats validation and remains authoritative.

`client.prepare_files(files)` reads or encodes each value once, computes its full SHA-256 hash, deduplicates identical content, and sends one batch `POST /blobs/check`. If any hashes are missing, it uploads only those hashes with `POST /blobs`, then returns the path-to-hash `PreparedFiles` manifest. Multiple paths with identical content use one upload part, and every upload part has content type `application/octet-stream`.

`client.execute_prepared(prepared, ...)` sends a job-only `POST /execute`. Because `PreparedFiles` retains no source bytes, cache eviction after preparation raises `BenchmarkServerError` with `code == "blob_not_found"` and `missing_blobs` populated, and this method cannot recover automatically.

The recommended `client.execute(files, ...)` operation performs the following cache-aware sequence while retaining the encoded bytes until execution succeeds or fails:

1. Read or encode every input exactly once, hash it, and deduplicate identical content.
2. Send one batch `POST /blobs/check`.
3. If the check reports missing blobs, upload only those blobs with `POST /blobs`.
4. Send a job-only `POST /execute`.
5. Only if that response is `blob_not_found` with `missing_blobs`, retry once with the same `job` plus inline parts for only those missing hashes.

The cache check is advisory, so eviction can occur between the check and execution. The server guarantees that the failed missing-blob attempt did not queue the request, assign a GPU, or execute code. The SDK never loops the fallback and surfaces the second failure.

### 3.3 Execution Results

`client.execute()` returns a frozen `ExecutionResult` dataclass with the following fields:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `request_id` | `str` | Required | The request identifier returned by the server |
| `value` | `ReturnValue` | Required | The recursively decoded entry-function return value |
| `elapsed_ms` | `float` | Required | Total server execution time defined in Section 2.1.2 |
| `queue_ms` | `float` | Required | Worker queue time defined in Section 2.1.2 |
| `stdout` | `str` | Required | Captured standard output |
| `stderr` | `str` | Required | Captured standard error |
| `stdout_truncated` | `bool` | `False` | Whether standard output was truncated |
| `stderr_truncated` | `bool` | `False` | Whether standard error was truncated |
| `warnings` | `tuple[ExecutionWarning, ...]` | `()` | Non-fatal warnings returned by the server |

`ExecutionWarning` is a frozen dataclass with `code: str` and `blobs: tuple[str, ...]`. The only warning code defined by the first version is `unused_blob`. A warning is not raised as an exception.

Fields are accessed directly:

```python
print(result.request_id)
print(result.value)
print(result.elapsed_ms, result.queue_ms)
print(result.stdout, result.stderr)
```

`ReturnValue` is decoded recursively:

| Description node | Decoded Python value |
|---|---|
| `json` | Ordinary Python JSON values: `None`, `bool`, `int`, finite `float`, `str`, `list`, and string-keyed `dict` |
| `bytes` | `bytes` |
| `tensor` | `tvm_ffi.Tensor` |
| `list` | `list[ReturnValue]` |
| `tuple` | `tuple[ReturnValue, ...]` |
| `dict` | `dict[str, ReturnValue]` |

For a multipart response, the client looks up each binary value by the node's `part` field and verifies its `size` and full SHA-256 hash. It rejects duplicate, missing, and unreferenced binary parts. It also verifies that the `request_id` in the response body is a canonical UUID v4 and exactly matches the `X-Request-ID` response header.

The SDK never uses `pickle`. Malformed JSON metadata, multipart structure, hashes, description trees, binary-part references, or request identifiers raise `ProtocolError`.

### 3.4 Tensors

The SDK declares `apache-tvm-ffi` as a required runtime dependency. TVM FFI means the TVM Foreign Function Interface, which supplies the `tvm_ffi.Tensor` class. DLPack is a standard protocol for exchanging tensors in memory.

Regardless of which supported DLPack producer the entry function returns, a decoded tensor node returns a `tvm_ffi.Tensor`. It resides in central processing unit (CPU) memory and has a C-contiguous, row-major layout because the network transport has already copied the raw bytes to host memory. Its shape and data type exactly match the response metadata. SDK-owned backing storage remains alive for the lifetime of the `tvm_ffi.Tensor`.

Callers can convert the value through DLPack:

```python
import numpy as np
import torch
import tvm_ffi


value = result.value
assert isinstance(value, tvm_ffi.Tensor)
np_value = np.from_dlpack(value)
torch_value = torch.from_dlpack(value)
```

NumPy and PyTorch are optional caller dependencies. The SDK does not require either package merely to hold a `tvm_ffi.Tensor`.

DLPack conversion is local and zero-copy after the HTTP payload has already been downloaded. It does not make the network transport zero-copy.

Official documentation:

- [`tvm_ffi.Tensor`](https://tvm.apache.org/ffi/reference/python/generated/tvm_ffi.Tensor.html)
- [`tvm_ffi.from_dlpack`](https://tvm.apache.org/ffi/reference/python/generated/tvm_ffi.from_dlpack.html)

### 3.5 Health Checks

The health API uses exactly these frozen dataclasses:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerHealth:
    gpu_id: int
    status: str
    uptime_seconds: float


@dataclass(frozen=True)
class Health:
    status: str
    gpu_count: int
    queue_length: int
    workers: tuple[WorkerHealth, ...]
```

Example:

```python
from benchmark_server import Client


with Client("http://server:8000") as client:
    health = client.health()

for worker in health.workers:
    print(worker.gpu_id, worker.status, worker.uptime_seconds)
```

`client.health()` synchronously calls `GET /health` and decodes an HTTP 200 response into `Health`. An HTTP 503 response raises `BenchmarkServerError` with `code == "unavailable"`. Other structured health errors use the same `BenchmarkServerError` mapping.

### 3.6 Errors and Retries

`BenchmarkServerError` exposes:

| Field | Type | Meaning |
|---|---|---|
| `status_code` | `int` | HTTP status code |
| `code` | `str` | Server error name from the `error` field |
| `message` | `str` | Human-readable server message |
| `request_id` | `str` or `None` | Request identifier when one is present |
| `stdout` | `str` | Captured standard output, or an empty string when absent |
| `stderr` | `str` | Captured standard error, or an empty string when absent |
| `traceback` | `str` or `None` | Python traceback when one is present |
| `missing_blobs` | `tuple[str, ...]` | Missing content hashes, or an empty tuple when the response omits the field |

Failures map to exceptions as follows:

| Failure | Exception |
|---|---|
| A structured server error response | `BenchmarkServerError` |
| Connection establishment, upload, download, connection loss, or another transport failure | `TransportError` |
| Malformed JSON, multipart data, binary size or hash, return-value tree, or request-identifier mismatch | `ProtocolError` |
| An invalid local argument | `ValueError` |

`client.execute()` performs only the one missing-blob fallback defined in Section 3.2. It retries only when the first execution response is `blob_not_found` with `missing_blobs`, includes only those missing blobs inline, and makes at most one additional execution request. A second failure is returned to the caller. `client.execute_prepared()` exposes the same machine-readable field through `BenchmarkServerError` but does not retry because it has no source bytes.

The SDK does not automatically retry a transport failure, a `timeout`, another form of `invalid_request`, or any other error. If a connection fails after an execution request reaches the server, execution may have completed even though the client did not receive the response. A higher-level caller may retry only when its application semantics can tolerate a possible duplicate execution.

`timeout_seconds` is the server execution timeout and excludes request upload and queueing. The SDK does not derive a short response-reading timeout from it.

---

## 4. Execution Model

### 4.1 Server Startup Parameters

Command-line parameters configure the listening address, GPUs, directories, and resource limits. The first version defines no configuration-file format.

```bash
benchmark-server \
  --host 127.0.0.1 \
  --port 8000 \
  --devices 0,1 \
  --cache-dir /var/cache/benchmark-server \
  --cache-capacity-bytes 10737418240 \
  --work-dir /tmp/benchmark-server \
  --log-dir /var/log/benchmark-server \
  --default-timeout-seconds 60 \
  --max-timeout-seconds 3600 \
  --default-stdout-limit-bytes 1048576 \
  --default-stderr-limit-bytes 1048576 \
  --max-stdout-limit-bytes 16777216 \
  --max-stderr-limit-bytes 16777216 \
  --worker-termination-grace-seconds 5
```

| Parameter | Required | Default | Description |
|---|---:|---|---|
| `--host` | No | `127.0.0.1` | HTTP listening address |
| `--port` | No | `8000` | HTTP listening port |
| `--devices` | Yes | None | Comma-separated physical GPU identifiers; one worker slot is created for each identifier |
| `--cache-dir` | No | `./cache` | Directory for content-addressed blobs and cache upload temporary files |
| `--cache-capacity-bytes` | Yes | None | Target upper bound in bytes for complete cached blobs; must be a positive integer |
| `--work-dir` | No | `./work` | Root for request-private working directories |
| `--log-dir` | No | `./logs` | Log root defined in Section 5 |
| `--default-timeout-seconds` | No | `60` | Execution timeout used when `job.timeout_seconds` is omitted |
| `--max-timeout-seconds` | No | `3600` | Maximum execution timeout that a client may request |
| `--default-stdout-limit-bytes` | No | `1048576` | Response limit used when `job.stdout_limit_bytes` is omitted |
| `--default-stderr-limit-bytes` | No | `1048576` | Response limit used when `job.stderr_limit_bytes` is omitted |
| `--max-stdout-limit-bytes` | No | `16777216` | Maximum standard-output response byte count that a client may request |
| `--max-stderr-limit-bytes` | No | `16777216` | Maximum standard-error response byte count that a client may request |
| `--worker-termination-grace-seconds` | No | `5` | Grace period between terminating a timed-out request process and forcefully killing it |

The server validates every parameter before creating worker processes or binding the listening port. An invalid parameter, a directory creation failure, or an unavailable GPU identifier causes the server to write the reason to standard error and exit with a nonzero status.

One front-end process maintains the cache index, reference counts, and LRU order, so only one server instance may use a given `cache-dir` at a time. At startup, the server acquires an exclusive operating-system file lock at `<cache-dir>/.lock`; startup fails if the lock is already held. Only the front-end process holds the lock file descriptor, and worker processes do not inherit it. The operating system automatically releases the lock after a graceful or forced front-end exit. Separate server instances may use separate cache directories.

### 4.2 Processes and Scheduling

#### Process Model

The server contains:

- one front-end process that receives HTTP requests and schedules tasks;
- one worker slot for each configured GPU;
- at most one request executing on each GPU at any time.

The front-end process does not initialize the GPU runtime environment. Each worker process is bound to one physical GPU through CUDA device visibility settings. CUDA is the GPU programming and runtime platform used by the execution environment.

#### Request Lifecycle

Each `POST /execute` request proceeds through the following steps:

1. Generate the request UUID and establish the request logging context.
2. Parse and validate the multipart request body.
3. Validate the `job.files` manifest, every target path, the entry file, and the set of optional inline blob parts.
4. Ingest each supplied inline blob through a temporary file and verify its SHA-256. Discard an unreferenced blob and produce an `unused_blob` warning; atomically publish a referenced blob to the cache and immediately hold a reference that prevents eviction.
5. In one atomic cache operation, resolve the remaining hashes and confirm that every manifest reference has been acquired. If any hash is missing, return `blob_not_found` before queueing, GPU assignment, or script execution.
6. Wait for an idle GPU worker process.
7. Create a new working directory for the request.
8. Materialize files in the working directory according to the `job.files` manifest.
9. Set the current working directory and Python module search path.
10. Import the entry file.
11. Retrieve and validate the entry function.
12. Invoke the entry function without arguments.
13. Synchronize and serialize the return value.
14. Capture standard output and standard error.
15. Delete the working directory and release references to cached files.
16. Return the request UUID in the response header and response metadata.

Cleanup runs after a missing cache reference, success, script failure, serialization failure, and timeout. Before responding, the server releases every cache reference already acquired for the request.

#### Scheduling

The front end maintains a first-in, first-out request queue. When multiple GPU worker processes become idle at the same time, the front end selects the next worker process in round-robin order.

Each worker process executes requests serially, preventing multiple benchmark requests from sharing one GPU at the same time and reducing interference with performance measurements.

`queue_ms` measures the time from completion of request validation until a worker process is assigned.

#### Timeout and Recovery

The timeout covers:

- importing the entry file;
- executing top-level code in the file;
- executing the entry function;
- synchronizing and serializing the return value.

Queueing time does not count toward the execution timeout.

After a timeout, the server:

1. Terminates the process executing the uploaded script.
2. Waits for a short, configurable exit grace period.
3. Forcefully terminates the process if it has not exited.
4. Replaces the affected runtime process before that GPU accepts another request.
5. Deletes the working directory for the request.
6. Returns a `timeout` error.

The server does not attempt to recover Python code that has timed out.

### 4.3 File Cache

#### Purpose

The file cache stores immutable content-addressed upload objects to:

- avoid repeatedly uploading and storing identical file content across requests;
- construct request working directories efficiently from `job.files`;
- allow one content hash to be reused across requests or at multiple paths in one request.

The cache stores only uploaded file bytes. It does not store entry-function return values, imported Python modules, or GPU memory. File changes made by a script in its request working directory are not written back to the cache.

#### Mechanism

The cache is an LRU cache keyed by SHA-256, with reference counting and a capacity target. LRU means least recently used.

- The hash ensures that identical content corresponds to one immutable cache entry.
- The reference count is the number of active requests currently using the blob.
- LRU order tracks which unused entries have gone longest without access.
- Capacity limits the total bytes of complete blobs that the cache aims to retain.

When `POST /execute` begins using a unique hash, its reference count increases; the count decreases when the request ends. A hash appearing multiple times in one manifest still counts as one request reference. An entry with a positive reference count is never evicted.

Uploading a new blob or actually using a blob for execution marks it as most recently used. `POST /blobs/check` tests only presence and does not refresh LRU order.

When the cache exceeds capacity, the server first evicts the least-recently-used entries whose reference count is zero until it reaches the target. If all candidates are still referenced, the cache may temporarily exceed capacity and resumes eviction after references are released.

#### Configuration

Section 4.1 configures the cache through:

- `--cache-dir`: directory containing complete blobs, upload temporary files, and the exclusive lock;
- `--cache-capacity-bytes`: target upper bound for the total bytes of complete cached blobs.

Capacity accounting excludes upload temporary files, request working directories, and logs. Capacity is a cache-retention target and does not limit the total input bytes referenced by an executing request.

Cache directory layout:

```text
<cache-dir>/
├── .lock
├── objects/
│   ├── 00/
│   │   └── 00a1...<full 64-character SHA-256>
│   ├── 81/
│   │   └── 81d4...<full 64-character SHA-256>
│   └── ff/
│       └── ff92...<full 64-character SHA-256>
└── tmp/
    ├── upload-<uuid>.part
    └── upload-<uuid>.part
```

- `.lock` is the exclusive lock file defined in Section 4.1.
- `objects/<first two hash characters>/<full hash>` stores immutable blobs that have completed verification.
- `tmp/` stores upload temporary files that have not completed verification.

The first two hash characters divide objects among 256 directories, avoiding concentration of all cache files in one directory. The cache uses no separate metadata files: the file name supplies the hash, file size supplies the byte count, and file modification time supplies initial LRU order after restart. Actual use by `POST /execute` updates the modification time; `POST /blobs/check` does not.

Capacity counts only complete blobs under `objects/` and excludes `.lock` and `tmp/`. An upload is first written under `tmp/` and moved to its `objects/` path after SHA-256 verification.

At restart, the server scans completely published objects in `cache-dir`, resets every reference count to zero, and establishes the initial LRU order from file modification times. It performs capacity eviction immediately after the scan. Section 4.1 defines the single-instance restriction and exclusive file lock for one `cache-dir`.

#### Usage

Clients may pre-upload file content through `POST /blobs` and then reference it by hash in `job.files`.

`POST /blobs/check` returns the hashes currently missing. Its result only helps avoid duplicate uploads and does not guarantee that a blob remains available for a later request.

`POST /execute` uses `job.files` to reference cached content and may also provide currently missing blobs inline. Blobs used by a request are not evicted before that request ends; an absent reference produces `blob_not_found`. The HTTP API exposes no cache lease or cache lock.

### 4.4 Isolation

The following state belongs to only one request:

- the working directory;
- the uploaded file layout;
- the imported entry module;
- captured standard output and standard error;
- returned Python objects;
- GPU memory reachable only from objects in the current request.

The server provides no sessions, cross-request Python objects, function handles, registers, or application-level shared state.

The file content cache may persist across requests. The cache stores only immutable uploaded bytes and does not store Python modules, GPU tensors, or execution results.

Python and native dynamic-link libraries may create process-global state. The implementation must ensure that a request cannot observe an entry module or working directory imported by the previous request. If a long-running worker process cannot completely clear this state, the script runtime process must be replaced before executing the next request.

The working directory is private to the request. The script can read, modify, delete, or create files in it, and all changes are visible only to the current request. After the request ends, the server deletes the entire working directory.

---

## 5. Observability

The deployment setting `log_dir` specifies the server's log root. Logs are outside the HTTP API, and the Python SDK does not read the log directory.

### 5.1 Log Directory

Each server-process start creates an independent run directory:

```text
<log_dir>/
└── runs/
    ├── 20260716T235012.123456Z/
    │   ├── events.jsonl
    │   ├── server.stdout.log
    │   ├── server.stderr.log
    │   └── requests/
    │       └── <request_id>/
    │           ├── stdout.log
    │           └── stderr.log
    └── 20260716T235012.123456Z-1/
        └── ...
```

The run directory name is the server start timestamp in Coordinated Universal Time (UTC), with microsecond precision. Directory creation directly calls atomic `mkdir(..., exist_ok=False)`. If the name already exists, the server tries the numeric suffixes `-1`, `-2`, and so on. Generating a log directory requires no file lock and must not check for existence before attempting creation.

Every restart creates a new directory and never continues writing previous-run log files. Concurrently starting server processes also receive distinct directories. Preventing multiple server processes from using the same GPU or shared cache is a separate resource-management concern and may independently use file locks.

### 5.2 Structured Events

`events.jsonl` uses JSON Lines format, storing one complete JSON object per line. Worker processes send structured events to the front end, which is the file's sole writer. Each event is emitted with one write operation and immediately flushed.

The following events are defined:

- `server_started`: the server process started;
- `request_started`: request processing started;
- `request_finished`: a request succeeded or failed;
- `worker_restarted`: a worker process was replaced;
- `server_error`: the server itself encountered an error;
- `server_stopped`: the server shut down normally.

Every event contains `server_run`, the current run directory name. Request-related events also contain `request_id`.

Example `request_finished` event:

```json
{
  "timestamp": "2026-07-16T23:50:12.123Z",
  "level": "INFO",
  "event": "request_finished",
  "server_run": "20260716T235012.123456Z",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "http_status": 200,
  "error": null,
  "gpu_id": 0,
  "queue_ms": 18.2,
  "elapsed_ms": 1245.6,
  "stdout_path": "requests/7f61b94e-034a-4e80-b67d-eca52bb952cc/stdout.log",
  "stderr_path": "requests/7f61b94e-034a-4e80-b67d-eca52bb952cc/stderr.log"
}
```

On failure, `error` uses an error type defined in Section 2.1.3.

When a request contains an inline blob not referenced by the manifest, `request_finished` uses level `WARNING` and adds `"warnings": ["unused_blob"]`. The API response warning retains the complete hash list; hashes are not used as structured-log labels.

### 5.3 Standard Output and Standard Error

The files contain:

- `server.stdout.log`: standard output written directly by the server process, worker processes, and their dependencies;
- `server.stderr.log`: standard error written directly by the server process, worker processes, and their dependencies;
- `requests/<request_id>/stdout.log`: standard output from the entry script and its child processes;
- `requests/<request_id>/stderr.log`: standard error from the entry script and its child processes.

While running the entry script, the server redirects standard output and standard error by file descriptor in the script runtime process. This captures output written by Python, native dynamic-link libraries, and child processes in the corresponding request logs.

Log files always preserve complete output. They do not use `stdout_limit_bytes` or `stderr_limit_bytes` and have no truncation fields. API responses and the SDK's `ExecutionResult` return prefixes according to the request limits and report truncation through `stdout_truncated` and `stderr_truncated`.

Complete logs may consume disk space indefinitely. The deployment operator allocates space for `log_dir` and enforces retention by deleting complete old run directories; the server does not truncate individual log files.

### 5.4 Restarts and Abnormal Termination

During a normal shutdown, the server writes `server_stopped` to the current `events.jsonl`. If the server process is forcibly terminated, this event may be absent, and an executing request may have `request_started` without `request_finished`.

After abnormal termination, previously written `events.jsonl`, request output, and server output remain in place. The final JSON line may be incomplete, and readers should ignore an unparseable final line. The next start creates a new run directory and does not modify old logs.

At startup, the server removes stale working directories and incomplete upload temporary files while preserving the complete content-addressed cache and all logs. Incomplete synchronous requests from the previous run are not resumed; clients observe a transport failure and decide whether to resubmit.

The first version provides no `/logs`, `/metrics`, or distributed-tracing endpoint. `GET /health` continues to report the state of the current server instance.

---

## 6. Complete Example

### 6.1 Uploaded `main.py`

```python
import json
from pathlib import Path


def main():
    config = json.loads(Path("config.json").read_text())
    input_data = Path("data/input.bin").read_bytes()

    # Build and run the GPU workload here.
    output_size = len(input_data)

    return {
        "correct": True,
        "output_size": output_size,
        "warmup_iterations": config["warmup_iterations"],
        "median_ms": 0.128,
    }
```

### 6.2 Request

This one-shot request includes every referenced blob inline and requires no `POST /blobs/check` or `POST /blobs` preflight.

```bash
curl -X POST http://server:8000/execute \
  -F 'job={"language":"python","entry":{"file":"main.py","function":"main"},"files":{"main.py":{"blob":"<main_sha256>"},"config.json":{"blob":"<config_sha256>"},"data/input.bin":{"blob":"<input_sha256>"}},"timeout_seconds":60,"stdout_limit_bytes":1048576,"stderr_limit_bytes":1048576};type=application/json' \
  -F 'blob:<main_sha256>=@main.py;type=application/octet-stream' \
  -F 'blob:<config_sha256>=@config.json;type=application/octet-stream' \
  -F 'blob:<input_sha256>=@input.bin;type=application/octet-stream'
```

Replace `<main_sha256>`, `<config_sha256>`, and `<input_sha256>` with the corresponding file content's full lowercase SHA-256, using the same values in `job.files` and the multipart part names.

### 6.3 Response

```json
{
  "status": "ok",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "return": {
    "type": "json",
    "value": {
      "correct": true,
      "output_size": 1048576,
      "warmup_iterations": 10,
      "median_ms": 0.128
    }
  },
  "elapsed_ms": 842.7,
  "queue_ms": 0.4,
  "stdout": "",
  "stderr": ""
}
```

---

## 7. Potential Future Features

This section records possible extension directions and is not part of the current v2 protocol commitment.

### 7.1 Additional Execution Languages

A future version may support languages other than Python through `job.language`.

The first candidate is a C++ TVM FFI library:

- the client uploads a precompiled TVM FFI dynamic library as a blob;
- `entry.file` points to the dynamic library;
- `entry.function` names a TVM FFI function exported by the library;
- the server loads the dynamic library and invokes the entry function;
- the entry function's return value continues to use the existing return-value description tree and multipart encoding.

This direction requires the server to load precompiled libraries and does not require it to compile arbitrary C++ source code.

### 7.2 Client SDK Languages

Future versions may provide:

- a TypeScript client SDK;
- a Rust client SDK.

These SDKs should use the same HTTP protocol, SHA-256 blob calculation, error types, and return-value description tree as the Python SDK. Each language SDK provides language-appropriate types and calling conventions without changing the server protocol.
