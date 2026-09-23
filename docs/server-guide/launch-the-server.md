# Launch the server

Install the [worker environment](../getting-started/installation.md) that matches
your programs first. A GPU worker executes programs on a graphics processing
unit; a CPU worker compiles CUDA C on a central processing unit without a GPU.
CUDA is NVIDIA's GPU programming platform.

## Start an instance

```bash
kcoral server --host 127.0.0.1 --port 8000
```

The server binds to `127.0.0.1` by default. Use `--host` or `KCORAL_SERVER_HOST` to
select another address.

To compile on a machine without a GPU and execute on a separate GPU machine,
run two instances of this same command:

```bash
kcoral server --device cpu --num-workers 16 --host 0.0.0.0 --port 8000
kcoral server --device gpu --gpus 0 --workers-per-gpu 8 --host 0.0.0.0 --port 8001
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

By default, the server checks whether bubblewrap can start before creating
workers. If the check fails or times out, it warns and disables isolation for
that server run. Restart to check again. Set `--sandbox none` or
`ServerConfig(sandbox="none")` to disable isolation and skip the check.
See [installation requirements](../getting-started/installation.md#install-the-server).

When enabled, each worker can write ordinary files only under its private
`/work`. Other workers' files and the server's cache and logs are hidden;
runtime dependencies are read-only, and network access is disabled.
`/work/.kcoral` is reserved for runtime files and cannot receive uploads.

`--max-requests-per-worker 0` reuses processes while clearing files and caches
between programs. Programs must finish background work before returning;
workers with remaining resources or failed cleanup are replaced.

For dependencies outside the standard runtime directories, add read-only paths:

```bash
kcoral server --sandbox-readonly-path /opt/custom-compiler
```

Repeat the option for multiple paths. Directories also enter the Python module
search path. All workers can read these paths, so exclude private data and
other workspaces.

This feature assumes **trusted programs**. It does not isolate hostile code
sharing an interpreter or provide GPU memory isolation.

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
