# Async API

Submit a program to a Router and receive a durable job handle after its inputs
have been saved. The Router waits for a node, executes the program, and stores
the complete result. Clients can exit after submission and later reconnect using
the job ID to check status or download results.

## Enable jobs

Start the Router with a directory on a persistent volume to enable the job API.

```bash
kcoral router --port 9000 --job-dir /var/lib/kcoral/jobs
```

Connect worker nodes to this Router using the usual deployment commands. The
Router owns job storage and dispatches programs through its execution slots.
See the [Router deployment guide](../server-guide/router.md).

## Submit and reconnect

```python
from kcoral import Client, Program

program = Program()
value = program.upload(kind="bytes", value=b"hello")
program.return_(key="value", value=value)

with Client("http://127.0.0.1:9000") as client:
    job = client.submit(program, idempotency_key="experiment-001")
    job_id = job.id

# Save job_id to reconnect from another process.
with Client("http://127.0.0.1:9000") as client:
    job = client.get_job(job_id)
    print(job.status().state)
    result = job.result(timeout=60)
    assert result["value"] == b"hello"
```

`submit()` uploads every referenced blob and waits for the Router to accept the
job. It returns a `JobHandle` independently of worker availability. Keep its
canonical lowercase UUID in `job.id` and the Router's address to restore the
handle later. `get_job()` creates a handle locally. The associated `Client`
must remain open while using that handle.

`status()` fetches a lightweight `JobStatus` snapshot. `result()` polls until it
can download a `ProgramResult`, including returned tensors and files. Completed
results can be downloaded repeatedly until retention cleanup removes the job.
The Python methods perform blocking HTTP calls. The program runs independently
on the server after submission returns.

`submit()` accepts the same execution options as `execute()`, including
`gpu_count`, `timeout_seconds`, and `output_limit_bytes`. Full signatures and
return types are available in the [Python API reference](../python-api/index.rst).

## Timeouts and errors

`submit(..., timeout_seconds=300)` sets the node's execution budget. Worker and
GPU waiting follow the existing execution protocol. `job.result(timeout=60)`
limits client waiting and raises `TimeoutError` when that budget expires. The
job continues and can be queried again with the same ID. Omitting `timeout`
waits indefinitely. `poll_interval` defaults to one second and must be positive.

A `FAILED` job returns a `ProgramResult` with its instruction error, captured
output, and any partial results. An `ERROR` job raises `KCoralError` when its
result is retrieved. Unknown or expired job IDs return HTTP `404`.

## States and snapshots

| State | Meaning |
| --- | --- |
| `QUEUED` | Inputs are saved and the job is awaiting a Router slot |
| `DISPATCHED` | Assigned to a node, including any node-side worker or GPU waiting |
| `COMPLETED` | The complete successful program response is available |
| `FAILED` | The complete failed program response is available |
| `ERROR` | The stored result contains an execution, scheduling, transport or storage error |

Snapshots contain `job_id`, `state`, `submitted_at`, `dispatched_at`,
`finished_at`, `node_id`, `request_id`, and an optional `error`. Timestamps are
Unix milliseconds. Events that have not occurred use `null` in HTTP responses
and `None` in Python. The execution attempt's `request_id` matches its
`X-Request-ID` for log correlation.

## Retry submissions

The SDK generates an idempotency key before sending a submission and retries a
transport failure once with the same key. If both attempts fail,
`TransportError.idempotency_key` provides the key for another submission with
the same input. Supply an explicit `idempotency_key` to recover across client
process restarts, as in the example above.

Keys are scoped to the Router's job store. Repeating a key with identical program
JSON bytes, referenced blob hashes and `X-KCoral-Client` identity returns the
existing job. Multipart boundaries and part ordering do not affect this check.
Different input with the same key returns HTTP `409`.

Use the same key for retries of one submission and a new key for an intentional
new execution. Retention cleanup removes the key along with its job, so reusing
that key after expiration creates a new job.

## Storage and recovery

The Router uses SQLite for metadata and separate files for inputs and responses.
One Router holds an exclusive lock on the storage directory. Inputs are retained
until the result is saved. Terminal jobs, their results and idempotency keys are
removed after the configured retention interval by cleanup running every second.

On restart, the Router resumes `QUEUED` jobs. Jobs that were `DISPATCHED` become
`ERROR` with `error.kind="execution_unknown"`, since their programs may have run.
Connection loss after dispatch also produces an unknown-outcome error. Choose a
new submission explicitly if the program is safe to run again. Graceful shutdown
saves queued work and waits for dispatched jobs to finish.

| Router option | Default | Meaning |
| --- | --- | --- |
| `--job-dir` | Disabled | Enable the job API and store durable job data in this directory |
| `--job-retention-seconds` | `86400` | Retention interval for terminal jobs, results and idempotency keys |
| `--max-jobs` | `1024` | Maximum stored jobs, including active and terminal jobs |
| `--max-job-result-bytes` | `268435456` | Maximum complete response size per job, including multipart framing |

The existing `--max-request-bytes` bounds each upload, and at most four job
uploads are processed concurrently. Node assignment uses the Router's scheduling
queue and its wait timeout. A scheduling failure after admission becomes a stored
`ERROR` result. Configure capacity and retention together to fit the storage
volume.

## HTTP endpoints

| Endpoint | Response |
| --- | --- |
| `POST /jobs` | `202`, a job snapshot and `Location: /jobs/{job_id}` after durable admission |
| `GET /jobs/{job_id}` | `200` and the current snapshot |
| `GET /jobs/{job_id}/result` | `202` and `Retry-After: 1` while pending, then the stored execution response |

Submission uses the same multipart format as `/execute` and includes every
referenced blob. The Router validates multipart input, JSON syntax, blob hashes
and completeness before admission. Node-side validation of program instructions
and execution options happens after dispatch, with errors saved as job results.
See the [execution protocol](protocol.md#post-execute) for program and blob encodings.

`Idempotency-Key` is optional in HTTP and accepts one header containing 1–128
visible ASCII characters. A repeated submission returns `202` with the existing
job's current snapshot, which may already show a terminal state.

| HTTP status | Condition |
| --- | --- |
| `400` | Invalid submission |
| `409` | An idempotency key was reused with different input |
| `413` | Upload exceeds the configured size limit |
| `503` | Job store or upload admission limit is full, with `Retry-After: 1` |
| `404` | Queried job ID is unknown or expired |

Successful and failed program results preserve `/execute` encodings, including
multipart tensors and returned files. Node HTTP execution errors retain their
status and error payload. Router restart or connection loss after dispatch
produces `502` with `error.kind="execution_unknown"`. A result exceeding the
Router's configured size limit produces `500` with
`error.kind="response_too_large"`.
