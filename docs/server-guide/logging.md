<a id="logs"></a>

# Logging

Each run appends to `<log-dir>/runs/<timestamp>/events.jsonl`, one JSON object
per line, mirrored to stderr and never capped or rotated. Only the front-end
writes it, so one file holds a request's whole history: `request_received`,
`request_accepted` with the shape of the workload, `request_routed` with the
worker that took it, and `request_finished`. One that never comes back stops
after the record naming its worker.

`sandbox_disabled` records why the startup check disabled isolation, at `WARNING`.
`pool_ready.sandbox` reports the active mode: `bubblewrap` or `none`.

`request_finished` carries a `finish_reason`: why the request ended when it did.

| `finish_reason` | what happened | worker |
|---|---|---|
| `completed` | the program ran to the end | keeps serving |
| `program_failed` | the program raised; `error_kind` and `instruction_index` say where | keeps serving |
| `request_limit` | `--max-requests-per-worker` reached, after answering | replaced |
| `poisoned_context` | cleanup after the program failed, after answering | replaced |
| `sandbox_cleanup` | sandbox resources survived the request or workspace cleanup failed | killed, replaced |
| `timeout` | no answer inside `timeout_seconds` | killed, replaced |
| `crashed` | the process exited mid-request; `exitcode` says how | killed, replaced |
| `no_worker` | saturated, nothing ran | untouched |
| `rejected`, `cache_miss`, `server_error` | never reached a worker | untouched |

A failing program is the client's kernel and stays `INFO`; only what the server
itself did wrong reaches `ERROR`. A `cpu_only` function caught touching the GPU
is a `gpu_access_violation` record at `WARNING`.

```bash
jq -c 'select(.level == "ERROR")' logs/runs/*/events.jsonl
jq -c 'select(.request_id == "<id>")' logs/runs/*/events.jsonl   # one request end to end
jq -c 'select(.worker_id == "gpu0/w3")' logs/runs/*/events.jsonl # one worker's history
```

`worker_id` names a seat on a GPU and outlives the processes that fill it, which
`generation` and `pid` identify; the `worker_*` records cover what happens
between requests. A killed worker's output reaches `request_finished` as
`output_tail`.

Each program's JSON is kept beside the log as `programs/<request-id>.json`, so a
failed request names the kernel that failed. Blob uploads travel by hash, so this
costs a few KB per request; `--no-log-programs` turns it off.
