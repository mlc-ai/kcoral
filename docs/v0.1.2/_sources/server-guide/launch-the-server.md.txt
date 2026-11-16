# Launch the server

KCoral servers run in two modes: GPU and CPU. A GPU server handles the usual
kernel development workflow, from compilation to correctness checks and
benchmarking. A CPU server runs jobs that do not need a GPU, such as compilation,
so you can add compilation capacity independently of your GPU machines. Install
the [server environment](../getting-started/installation.md#install-the-server)
for the jobs you plan to run before starting either mode.

## Start an instance

```{warning}
KCoral allows clients to execute arbitrary code on its workers. Only allow
trusted clients to access your KCoral server or Router. Deploy on a trusted,
isolated network and never expose these endpoints to the public internet.
Run workers in a sandbox with restricted permissions and access to host resources.
```

For a typical setup, start a GPU server:

```bash
kcoral server --host 127.0.0.1 --port 8000
```

By default, this starts eight worker processes sharing physical GPU `0` and
listens at `http://127.0.0.1:8000`. Each worker handles one request at a time.
These defaults assume no `KCORAL_SERVER_DEVICE` or `KCORAL_SERVER_GPUS`
environment override is set.

Startup logs show the workers being initialized. Wait for the pool to be ready
and the HTTP server to start listening. The final lines look like this, with
timestamps and intermediate messages omitted; the target depends on your GPU:

```text
INFO    pool_ready sandbox=bubblewrap mode=gpu target={'arch': 'sm_100a'} workers=8
INFO:   Application startup complete.
INFO:   Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

You can then check that the server is reachable:

```bash
curl http://127.0.0.1:8000/health
```

A healthy server returns JSON with `"status": "ok"`. The response also includes
its GPU architecture in `target` and installed runtime versions in `versions`.
If startup fails, see [logs](logging.md) for how to investigate. If bubblewrap
cannot start, the server warns and continues without filesystem isolation;
see [isolation](#isolate-worker-files-with-bubblewrap) below.

### Common settings

The following options let you adapt the GPU server to your machine and workload:

- `--gpus 0` selects physical GPU `0`, the default. The value is a GPU ID,
  not a GPU count. To use multiple GPUs, pass their IDs separated by commas:
  `--gpus 0,1` selects GPUs `0` and `1`. All selected GPUs must have the same
  target architecture.
- `--workers-per-gpu 8` runs eight workers per GPU by default. They take turns
  holding exclusive GPU access through a lease. Clients can mark functions with
  `cpu_only=True` to declare that they do not use the GPU, allowing them to run
  while another worker uses it. See
  [running CPU-only functions](../tutorials/benchmark-kernel.md#overlap-cpu-work-with-gpu-execution)
  for how clients select this behavior. More workers can improve utilization
  when requests spend substantial time compiling or doing other CPU work. Once
  the GPU stays busy, adding workers brings little benefit and increases pressure
  on CPU resources, host memory, and GPU memory. Start with the default of eight
  and adjust for your workload.
- `--max-requests-per-worker 1`, the default, replaces each worker after one
  request, giving the next request a fresh process and GPU context. This prevents
  a faulty kernel's process state from affecting later requests. Increasing the
  limit spreads replacement overhead across multiple requests; setting it to
  `0` removes scheduled replacements entirely. Reuse can help with short
  requests, but cleanup and error checks cannot contain every effect of invalid
  GPU code. Failed workers may still need replacement.
- `--host 127.0.0.1` and `--port 8000` are the default listening address and
  port. This address accepts connections only from the same machine. Use
  `--host 0.0.0.0` to listen on all network interfaces when clients connect over
  a trusted network; clients use the server's reachable address in their URLs.
  You can also set `KCORAL_SERVER_HOST` and `KCORAL_SERVER_PORT`; explicit
  command-line options take precedence over these environment variables.
- `--default-timeout-seconds 300` gives requests a five-minute execution
  budget by default when they omit a timeout. `--max-timeout-seconds 900` caps
  client-requested budgets at 15 minutes by default. Increase these for longer
  jobs. The budget covers execution of the request itself, excluding time spent
  waiting for server capacity or access to the GPU.
- `--log-dir logs` writes request and worker event logs under `logs` by
  default, unless `KCORAL_LOG_DIR` is set. Pass another directory to change it.
  See [logs](logging.md) for how to follow requests and diagnose failures.

For example, to use two GPUs with four workers on each:

```bash
kcoral server --gpus 0,1 --workers-per-gpu 4
```

See [configuration](#configuration) for the full option reference. Once the
server is running, [Writing a program](../client-guide/writing-a-program.md)
walks through sending your first request.

## Add a CPU server for compilation

One GPU server can handle both compilation and measurement. If compilation
becomes a bottleneck, you can move CPU-only jobs to a CPU server and scale the
two roles separately. Start these instances in separate terminals, on the
same machine or on separate hosts:

```bash
# Compilation server
kcoral server --device cpu --num-workers 16 --host 0.0.0.0 --port 8000

# GPU execution server
kcoral server --device gpu --host 0.0.0.0 --port 8001
```

In CPU mode, `--num-workers` controls the number of worker processes and defaults
to one. The example allows up to 16 requests to run concurrently. Choose a count
that fits the host's CPU and memory capacity, accounting for compilers that use
multiple threads. CPU mode ignores `--gpus` and `--workers-per-gpu`.

Both servers report `pool_ready` followed by the HTTP startup messages shown
above. For the CPU command, the pool line looks like this, with the timestamp
omitted:

```text
INFO    pool_ready sandbox=bubblewrap mode=cpu target={} workers=16
```

Check `/health` at each server's address to confirm it is reachable. The GPU
server listens on port `8001` in this example and still defaults to eight
workers on GPU `0`.

The [Remote Compilation tutorial](../tutorials/remote-compilation.md) demonstrates
compiling CUDA C on a CPU server and passing the resulting library to a GPU
server. Other compilation workflows can run there if their dependencies are
installed and they do not require GPU access. The CPU server cannot execute
GPU kernels. Its `target` is empty, so clients should read the GPU server's
target before compiling a library for it.
CPU workers accept Python modules, bytes and files; tensor and library uploads
require a GPU worker.

## Isolate worker files with bubblewrap

Code running on the server can read and write files. Giving each request a
working directory keeps its outputs together, but does not by itself
prevent that code from modifying the server's files or another worker's data.
Filesystem isolation limits the damage an accidental file operation can cause.

The KCoral server uses [bubblewrap](https://github.com/containers/bubblewrap),
a Linux sandboxing tool, to give each worker
its own view of the filesystem. Bubblewrap uses Linux namespaces to separate the
worker's environment from the host. KCoral makes runtime dependencies available
read-only and gives the worker a private writable directory at `/work`. Other
workers' files are hidden, and network access is disabled. Keep the server's
cache and logs outside the read-only runtime paths, whose contents remain
visible to workers. Uploaded code still runs on the host's CPU and, in GPU mode, its
assigned GPU. `/work/.kcoral` is reserved for runtime files and cannot receive
uploads.

This isolation is enabled by default with `--sandbox bubblewrap`. Install
bubblewrap and allow unprivileged user namespaces on the host or container;
see the [installation requirements](../getting-started/installation.md#server-system-requirements).
Before creating workers, the server checks that bubblewrap can start. If the
check fails or times out, it warns and disables isolation for that run. For
example, if bubblewrap is not installed, the warning includes:

```text
RuntimeWarning: bubblewrap could not start; filesystem isolation is disabled for this server run: bubblewrap isolation requires bwrap on PATH; install bubblewrap
```

Without isolation, uploaded code has the server process's access to host files. Fix the
reported problem and restart the server to check again.

To disable filesystem isolation explicitly and skip the startup check, use
`--sandbox none`.

When workers are reused (`--max-requests-per-worker` greater than `1`, or `0`),
the server clears their working files between requests. If cleanup fails or
code leaves resources such as background threads or child processes running,
the server replaces the worker before accepting another request on it.

For dependencies outside the
[standard runtime paths](https://github.com/mlc-ai/kcoral/blob/main/python/kcoral/sandbox.py#L180-L201),
add read-only paths:

```bash
kcoral server --sandbox-readonly-path /opt/custom-compiler
```

Repeat the option for multiple paths. Directories also enter the Python module
search path. All workers can read these paths, so exclude private data and
other workspaces.

This feature assumes **trusted programs**. It does not isolate hostile code
sharing an interpreter or provide GPU memory isolation.

## Configuration

The following options configure a standalone server. Use `--help` to display
command-line help.
Explicit command-line options take precedence over environment variables.
Defaults in this table assume none of those environment variables is set.

A worker is a process that executes one request at a time. A lease gives a
worker exclusive access to its GPU while it executes or measures GPU work.

### Binding and worker selection

| Option | Default | Environment variable | Meaning |
| --- | --- | --- | --- |
| `--host` | `127.0.0.1` | `KCORAL_SERVER_HOST` | Listening address |
| `--port` | `8000` | `KCORAL_SERVER_PORT` | Listening port |
| `--device` | `gpu` | `KCORAL_SERVER_DEVICE` | Worker mode: `gpu` or `cpu` |
| `--gpus` | `0` | `KCORAL_SERVER_GPUS` | Comma-separated physical GPU IDs |
| `--num-workers` | `1` | — | Worker count in CPU mode |
| `--workers-per-gpu` | `8` | — | Workers per GPU in GPU mode |
| `--max-requests-per-worker` | `1` | — | Requests before replacement; `0` allows unlimited reuse |
| `--worker-termination-grace-seconds` | `5` | — | Seconds between SIGTERM and SIGKILL when stopping a failed worker |
| `--sandbox` | `bubblewrap` | — | Filesystem isolation; `none` disables it |
| `--sandbox-readonly-path` | No additional paths | — | Additional read-only dependency path; repeatable |

`--gpus` takes comma-separated physical device numbers such as `0,1`. Workers
select their devices from this option, so setting `CUDA_VISIBLE_DEVICES` on the
front-end does not restrict the server. All selected GPUs must report the same
target architecture. CPU mode ignores `--gpus` and uses `--num-workers` instead.

The default replaces a worker after each request, giving the next request a
fresh process and GPU context. Reusing workers can reduce replacement overhead,
but reset and poison detection cannot contain every effect of invalid GPU code.

### Time and size limits

All options ending in `-bytes` take an integer number of bytes, not a value
with a unit suffix.

| Option | Default | Meaning |
| --- | --- | --- |
| `--worker-wait-timeout-seconds` | `1800` | Time to wait for a free worker before a 503 response |
| `--default-timeout-seconds` | `300` | Execution limit when a request omits its timeout |
| `--max-timeout-seconds` | `900` | Upper bound for a request's timeout |
| `--max-request-bytes` | `268435456` (256 MiB) | Maximum request body size |
| `--max-response-bytes` | `268435456` (256 MiB) | Maximum serialized response size |
| `--output-limit-bytes` | `1048576` (1 MiB) | Default captured stdout/stderr limit per stream |
| `--max-output-limit-bytes` | `16777216` (16 MiB) | Maximum requested captured output per stream |

Worker acquisition can wait up to 30 minutes by default. The execution budget
starts after worker assignment and excludes time waiting for another worker's
GPU lease. It defaults to 5 minutes and is capped at 15 minutes, so a long queue
wait does not give a running program a longer execution budget.

Request `timeout_seconds` and `output_limit_bytes` override their respective
defaults, up to these server maximums. See {ref}`protocol options <options>`
for clamping and [errors](../client-guide/protocol.md#errors) for request failures.

### Cache

When repeatedly evaluating a kernel or iterating on agent-generated kernels,
you often upload the same harness files, input tensors, or compiled libraries
with each request. KCoral caches uploaded content so clients can reuse unchanged
uploads without transferring their contents again. The Python client handles
cache misses automatically by resending the required content.

The server keeps two caches, both keyed by the SHA-256 hash of the uploaded
bytes:

- The **memory cache** holds tensor, byte-string, and library uploads in the
  server process's CPU memory. Its contents are lost when the server restarts.
- The **file cache** holds file uploads on disk. Its contents can survive server
  restarts, independently of the temporary files created for each request.

| Option | Default | Meaning |
| --- | --- | --- |
| `--cache-capacity-bytes` | `17179869184` (16 GiB) | Memory cache budget, in bytes |
| `--disk-cache-dir` | The directory described below | Persistent file cache directory |
| `--disk-cache-capacity-mbytes` | `16384` MiB (16 GiB) | File cache budget, in MiB |

The memory cache evicts less recently used entries when it exceeds its budget.
Entries in use by active requests are retained, so the budget can be exceeded
while those entries are pinned. Individual uploads larger than one quarter of
the budget are not cached, but remain usable by the request that supplied them.

The file cache defaults to `$XDG_CACHE_HOME/kcoral/files` when `XDG_CACHE_HOME`
is an absolute path, otherwise `~/.cache/kcoral/files`. It evicts older entries
to stay within its budget. An empty directory option (`--disk-cache-dir ''`) or zero disk capacity
disables file caching without
falling back to the memory cache. Storage failures and oversized files do not
prevent execution when the request supplies the bytes.

Both caches store uploaded bytes, not execution state: each request creates
its own tensors, loads its libraries, and materializes its files. Changes made
during execution do not change the cached uploads.

See [upload caching](../client-guide/protocol.md#upload-caching) for
cache lookup and retry behavior, and
[file uploads](../client-guide/writing-a-program.md#upload-files-and-folders)
for working with request files.

### Logs

| Option | Default behavior | Effect |
| --- | --- | --- |
| `--log-dir` | `logs`, or `KCORAL_LOG_DIR` | Choose where to save logs and programs |
| `--no-log-console` | Console events enabled | Disable console events |
| `--no-log-programs` | Program recording enabled | Disable saved program JSON |

`--log-dir ''` disables
log files and saved programs; console events remain enabled unless you also
pass `--no-log-console`.
See [logs](logging.md) for locations, events and investigation commands.
