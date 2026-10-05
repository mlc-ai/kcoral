# Logging

When a request fails, runs slowly, or causes a worker to restart, the server logs
help you follow what happened. KCoral records request progress and worker events
in one log for each server run. You can use a request ID to trace one submission,
or a worker ID to investigate repeated problems on the same worker.

## Find and configure the logs

By default, `kcoral server` writes to
`logs/runs/<timestamp>/events.jsonl`. Each server start creates a new run
directory, and each line in the file is a JSON object with `timestamp`, `level`,
`event`, and fields specific to that event. The `server_started` event identifies
the run directory and records the server configuration.

The server also prints a shorter text version to standard error. This console
view omits bulky fields such as tracebacks and runtime versions and truncates
long field values; use the JSON log for the full record. KCoral does not rotate
or cap these log files.

Use `--log-dir` to choose the directory, `--no-log-console` to turn off console
events, and `--no-log-programs` to stop saving program JSON alongside the log.
`--log-dir ''` disables file logging while leaving console events enabled.
See [logging configuration](launch-the-server.md#logs) for defaults and
environment settings.

These JSON logs come from the Python server, including when it runs behind a
Router. The Router and its node supervisors write their own text logs to the
console. Use the request ID to correlate Router and server events.

## Follow a request

A typical request produces four events:

| Event | What it tells you |
| --- | --- |
| `request_received` | The request arrived; includes `request_id`, client address, and declared content length |
| `request_accepted` | The program passed validation; includes the timeout, request size, instruction and upload counts, and saved program filename when available |
| `request_routed` | A worker was assigned; includes `worker_id`, `gpu_id`, `generation`, `pid`, and `queue_ms` |
| `request_finished` | The request ended; includes the HTTP status, `finish_reason`, and available execution or error details |

A rejected request or cache miss finishes before worker assignment, so it does
not produce every event in this sequence. Cache retries are separate HTTP
requests and have separate request IDs.

On completed or failed executions, `request_finished` includes program `status`,
`response_bytes`, and timing fields: `queue_ms`, `elapsed_ms`, `lease_wait_ms`,
and `lease_held_ms`. Error details, when available, include `error_kind`,
`error_message`, `instruction_index`, `instruction_op`, and `instruction_id`.
Crashes can also include `exitcode`; crashes and timeouts can include captured
worker output in `output_tail`. Fields depend on how far the request progressed.

The `finish_reason` explains the outcome:

| `finish_reason` | Meaning | Level |
| --- | --- | --- |
| `completed` | The program ran to completion | `INFO` |
| `program_failed` | An instruction failed; error fields describe the failure | `INFO` |
| `timeout` | The request exceeded its execution budget | `WARNING` |
| `crashed` | The worker crashed while handling the request | `ERROR` |
| `no_worker` | No worker became available, or the pool was shutting down; nothing ran | `WARNING` |
| `rejected` | The request was rejected before execution | `WARNING` |
| `cache_miss` | Required uploaded content was missing from the cache | `INFO` |
| `server_error` | The server could not handle the request or produce its response | `ERROR` |

An ordinary program failure is logged at `INFO`, while a worker crash is
`ERROR` even if client code caused it. A function declared `cpu_only=True` that
is caught accessing the GPU also produces a `gpu_access_violation` event at
`WARNING`.

To follow one request, search the log for the `request_id` returned to the
client. To investigate a worker, search for its `worker_id`, such as `gpu0/w3`.

## Diagnose startup and worker replacement

During startup, `pool_ready` reports the worker count, target, runtime versions,
and active sandbox mode (`bubblewrap` or `none`). A `sandbox_disabled` warning
explains why the isolation startup check failed. If the server cannot start,
look for `server_start_failed`; worker initialization failures also produce
`worker_failed` with a `phase` and error description.

A `worker_id`, such as `gpu0/w3` or `cpu/w0`, identifies a position in the pool.
It stays the same when the process is replaced; `generation` and `pid`
distinguish the processes that occupy it. `worker_retired` records why a process
is being replaced:

| `reason` | Meaning | Level |
| --- | --- | --- |
| `request_limit` | The worker reached `--max-requests-per-worker` | `INFO` |
| `poisoned_context` | Runtime cleanup failed | `WARNING` |
| `sandbox_cleanup` | Resources survived the request or workspace cleanup failed | `WARNING` |
| `timeout` | The worker exceeded the execution timeout | `WARNING` |
| `crashed` | The worker crashed or could not execute a program | `ERROR` |

A replacement produces `worker_ready` once initialized. If replacement fails,
look for `worker_failed` or `worker_replace_failed`, both at `ERROR`. During
shutdown, `shutdown_started`, `shutdown_waiting`, and `shutdown_complete` show
the pool's progress; `server_stopped` records the end of the server lifecycle.

## Inspect a saved program

By default, accepted programs are saved as
`programs/<request-id>.json` inside the run directory. The `program` field in
`request_accepted` gives the filename when the save succeeds. Open it to inspect
the instructions and inline source associated with a request.

These files contain the submitted program JSON, including inline module source,
but not binary upload contents: those are referenced by hash. Their size depends
on the program, and they do not by themselves contain everything needed to
replay it. Use `--no-log-programs` to disable this recording.
