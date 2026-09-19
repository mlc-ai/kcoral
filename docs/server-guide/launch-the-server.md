# Launch the server

Install the [worker environment](../getting-started/installation.md) that matches
your programs first. A GPU worker executes programs on a graphics processing
unit; a CPU worker compiles CUDA C on a central processing unit without a GPU.
CUDA is NVIDIA's GPU programming platform.

## Start an instance

```bash
kcoral --host 127.0.0.1 --port 8000
```

The server binds to `127.0.0.1` by default. Use `--host` or `KCORAL_SERVER_HOST` to
select another address.

To compile on a machine without a GPU and execute on a separate GPU machine,
run two instances of this same command:

```bash
kcoral --device cpu --num-workers 16 --host 0.0.0.0 --port 8000
kcoral --device gpu --gpus 0 --workers-per-gpu 8 --host 0.0.0.0 --port 8001
```

See [configuration](#configuration) for every option and its default.

Check readiness with `GET /health`, which also reports the `target` an uploaded
library must be built for and the `versions` the worker runs. When compiling on
a CPU server, read the target from the GPU server. Submit programs with
`POST /execute` using `multipart/form-data`.

Several workers share each GPU, so one can compile while another measures on the
GPU it is not using; they take turns through a per-GPU lease and never run on it
at once. Raising `--workers-per-gpu` keeps the GPUs busier at the cost of dividing
their memory among more concurrent benchmarks.

Workers serve one request by default, then the pool replaces them before making
the slot idle again, so an out-of-bounds or race-sensitive kernel cannot make a
later request depend on its process history. `--max-requests-per-worker 0` reuses
workers instead: reset and poison detection still run, but undefined CUDA
behaviour is no longer contained, and replacement costs enough on short requests
that throughput numbers should record the setting.

## Choose a deployment

Start with one GPU instance for remote compilation and measurement. Use separate
CPU and GPU instances when compilation capacity should scale independently;
the [Remote Compilation tutorial](../tutorials/remote-compilation.md)
explains how to pass a compiled library between them.
The CPU compilation service does not provide general GPU execution.

`0.0.0.0` listens on every network interface. The default `127.0.0.1` listens
only on this machine. Disabling filesystem isolation lets workers execute
uploaded Python code with the server's permissions; a request working directory
alone does not isolate that code from the host.

See [logs](logging.md) to follow a request and diagnose worker replacement.

## Isolate worker files with bubblewrap

On Linux, install [bubblewrap](https://github.com/containers/bubblewrap) with
support for `--disable-userns` and allow unprivileged user namespaces on the
host. The server checks whether isolation can start before creating workers;
the default configuration requests it:

```bash
# A fresh isolated process for each program (the default request limit).
kcoral --device cpu --num-workers 2

# Reuse an isolated process, clearing its files after each program.
kcoral --device cpu --num-workers 2 --max-requests-per-worker 0

# The same isolation backend supports GPU workers.
kcoral --gpus 0 --workers-per-gpu 2
```

When enabled, each worker gets a separate filesystem view. Its only writable ordinary file
tree is `/work`; other workers' directories, the server's upload cache and log
directories, and the host home directory are not mounted. The operating system
rejects writes outside this tree, including writes from compiler subprocesses.
Python installations, system libraries, system device information and approved
dependencies remain readable. A private process view prevents inspecting other
workers through `/proc`. Network access is disabled.

The parent owns the backing directory. A program's file uploads and relative
paths resolve beneath `/work`. It is emptied between requests even when the
process is reused, so uploading the same path in successive programs creates
independent files. The interpreter and GPU context can remain alive. File and
folder returns keep their existing instruction-time snapshot semantics.

`/work/.kcoral` is reserved for runtime files and cannot be an upload
destination. Home, temporary files, shared-memory files, compiler caches,
uploaded libraries and captured output live below it. `/tmp`, `/var/tmp` and
`/dev/shm` refer into this private tree. Request-local caches are cleared;
the front-end upload cache is unaffected. Uploaded dynamic libraries are not
retained across sandboxed requests.

For dependencies installed outside the interpreter and system directories,
add only their necessary runtime paths:

```bash
kcoral --sandbox bubblewrap \
  --sandbox-readonly-path /opt/custom-compiler \
  --sandbox-readonly-path /opt/custom-python-packages
```

These paths are also added to the worker's Python module search path when they
are directories. Every file under an approved path becomes readable to every
worker. Do not approve private data, other workspaces, or broad host directories.
Run from a regular package installation; editable dependencies may need their
source and native-library directories approved explicitly.

This feature assumes **trusted programs**. It limits filesystem access; it
does not make arbitrary Python or native code in one interpreter mutually
untrusted, provide GPU memory isolation, or change upload-cache authorization.
Programs must finish their background work before returning. Before reuse,
KCoral removes workspace imports and checks for remaining Python/native threads,
child processes, open workspace files and workspace-backed memory mappings.
If cleanup cannot be confirmed, it retires the worker with
`finish_reason="sandbox_cleanup"` instead of exposing the next program's files.
Lazy dependency initialization that leaves new threads can also cause retirement.
Libraries that remain mapped after unloading also retire the process, including
libraries built with the linker's `-z nodelete` option.

GPU device files and the GPU worker's private `/proc` filesystem are approved
kernel interfaces, not ordinary writable files. NVIDIA drivers can require a
writable `/proc` mount to initialize CUDA. GPU isolation still relies on the
existing device selection and lease
mechanism. The sandbox exposes the selected NVIDIA device and required control
devices; custom driver/toolchain installations may require additional read-only
runtime paths.

Before creating the worker pool, the server runs a short Python process with
the same bubblewrap options and mounts as the workers. The check covers each
distinct selected GPU configuration without initializing CUDA. A missing binary,
denied namespace creation, mount failure, or a probe that does not finish within
10 seconds disables isolation for this server run. The server emits a
`RuntimeWarning` and a `sandbox_disabled` log event at `WARNING`, with the failure
reason, and continues starting workers without filesystem isolation.

The check runs only at server startup. Requests and worker replacements keep
the selected mode; restarting the server checks again. `server_started.config`
records the requested setting and `pool_ready.sandbox` records the effective
mode. Runtime initialization errors after a successful probe still fail startup.

Set `--sandbox none`, or `ServerConfig(sandbox="none")` in Python, to explicitly
disable isolation and skip the check. `--sandbox bubblewrap` and
`ServerConfig(sandbox="bubblewrap")` select the default startup check.


## Configuration

The command-line interface accepts the options below. Python applications pass
the corresponding fields to `ServerConfig`, the server configuration object.
Explicit command-line options take precedence over environment variables.
Defaults in this table assume none of those environment variables is set.

GPU means graphics processing unit; CPU means central processing unit. A worker
is a process that executes one request at a time. A lease gives a worker exclusive
access to its GPU while it executes or measures GPU work.

### Binding and worker selection

| Option | Default | Environment variable | Configuration field |
| --- | --- | --- | --- |
| `--host` | `127.0.0.1` | `KCORAL_SERVER_HOST` | Passed to the HTTP server, not `ServerConfig` |
| `--port` | `8000` | `KCORAL_SERVER_PORT` | Passed to the HTTP server, not `ServerConfig` |
| `--device` | `gpu` | `KCORAL_SERVER_DEVICE` | `device` |
| `--gpus` | `0` | `KCORAL_SERVER_GPUS` | `gpus`, a list, default `[0]` |
| `--num-workers` | `1` | — | `num_workers`, used in CPU mode |
| `--workers-per-gpu` | `8` | — | `workers_per_gpu`, used in GPU mode |
| `--max-requests-per-worker` | `1` | — | `max_requests_per_worker`; `0` reuses workers |
| `--worker-termination-grace-seconds` | `5` | — | `worker_termination_grace_seconds` |
| `--sandbox` | `bubblewrap` | — | `sandbox`; `none` explicitly disables filesystem isolation |
| `--sandbox-readonly-path` | No additional paths | — | `sandbox_readonly_paths`, a list of paths; repeatable |

`--gpus` takes comma-separated physical device numbers such as `0,1`. Workers
select their devices from this option, so setting `CUDA_VISIBLE_DEVICES` on the
front-end does not restrict the server. All selected GPUs must report the same
target architecture. CPU mode ignores `gpus` and uses `num_workers` instead.

The default replaces a worker after each request, giving the next request a
fresh process and GPU context. Reusing workers can reduce replacement overhead,
but reset and poison detection cannot contain every effect of invalid GPU code.
Record this setting when comparing throughput.

### Time and size limits

MiB means 1024 squared bytes; GiB means 1024 cubed bytes. All options ending in
`-bytes` take an integer number of bytes, not a value with a unit suffix.

| Option | Default | Configuration field | Meaning |
| --- | --- | --- | --- |
| `--worker-wait-timeout-seconds` | `1800` | `worker_wait_timeout_seconds` | Time to wait for a free worker before a 503 response |
| `--default-timeout-seconds` | `300` | `default_timeout_seconds` | Execution limit when a request omits its timeout |
| `--max-timeout-seconds` | `900` | `max_timeout_seconds` | Upper bound for a request's timeout |
| `--max-request-bytes` | `268435456` (256 MiB) | `max_request_bytes` | Maximum request body size |
| `--max-response-bytes` | `268435456` (256 MiB) | `max_response_bytes` | Maximum serialized response size |
| `--output-limit-bytes` | `1048576` (1 MiB) | `output_limit_bytes` | Default captured stdout/stderr limit per stream |
| `--max-output-limit-bytes` | `16777216` (16 MiB) | `max_output_limit_bytes` | Maximum requested captured output per stream |
| `--cache-capacity-bytes` | `17179869184` (16 GiB) | `cache_capacity_bytes` | Memory cache capacity for tensors, bytes and libraries |

Worker acquisition can wait up to 30 minutes by default. The execution budget
starts after worker assignment and excludes time waiting for another worker's
GPU lease. It defaults to 5 minutes and is capped at 15 minutes, so a long queue
wait does not give a running program a longer execution budget.

Request `timeout_seconds` and `output_limit_bytes` override their respective
defaults, up to these server maximums. See [protocol options](../client-guide/protocol.md#options)
for clamping and [errors](../client-guide/protocol.md#errors) for request failures.

### File upload cache

File uploads use a persistent disk cache; tensors, bytes and libraries use the
memory cache. The default directory is `$XDG_CACHE_HOME/kcoral/files` when
`XDG_CACHE_HOME` is an absolute path, otherwise `~/.cache/kcoral/files`.

| Option | Default | Configuration field |
| --- | --- | --- |
| `--disk-cache-dir` | The directory described above | `disk_cache_dir` |
| `--disk-cache-capacity-mbytes` | `16384` MiB (16 GiB) | `disk_cache_capacity_mbytes` |

An empty directory option (`None` in `ServerConfig`) or zero capacity disables
file caching without falling back to the memory cache. Cached content survives
server restarts. Caching is best-effort: storage failures and oversized objects
do not prevent execution when the request supplies the bytes.

File destinations are private to each request and removed when the request
ends. See [file uploads](../client-guide/writing-a-program.md#files-used-by-uploaded-scripts) for
path restrictions, snapshot behavior and directory uploads.

### Logs

| Option | Command-line default | Configuration field and Python default |
| --- | --- | --- |
| `--log-dir` | `logs`, or `KCORAL_LOG_DIR` | `log_dir=None` |
| `--no-log-console` | Console mirroring enabled | `log_console=True` |
| `--no-log-programs` | Program recording enabled | `log_programs=True` |

The two `--no-*` flags set their fields to `False`. `--log-dir ''` disables
logging; direct Python construction already defaults to `log_dir=None`.
See [logs](logging.md) for locations, events and investigation commands.

### Configure an application in Python

`create_app` builds the application; an HTTP server such as uvicorn runs it.
The server extra must be installed. This example defines an application and
does not start workers until the application enters its serving lifecycle.

```python
from pathlib import Path

from kcoral import ServerConfig, create_app

app = create_app(
    ServerConfig(
        gpus=[0],
        workers_per_gpu=8,
        log_dir=Path("logs"),
        disk_cache_capacity_mbytes=16384,
    )
)
```

See the {ref}`Python reference <server-integration>` for
the complete configuration signature and application factory.
