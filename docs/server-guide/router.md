# Router

A Router gives clients one server address and selects an available compute
node for each request. A node supervisor starts, checks and restarts its local
Python server. The supervisor and Python server initiate their connections to
the Router, so compute nodes do not need to accept inbound network connections.

Install the client and Python server from the same revision as the Rust
binaries so their internal protocols match.

## Components and connections

| Component | Where it runs | Responsibility |
| --- | --- | --- |
| `kcoral router` | A host reachable by clients and nodes | Accept HTTP requests, manage a bounded wait queue and select nodes |
| `kcoral server --router ...` | Each compute node | Supervise one Python server process tree and report its health |
| Python server | Each compute node | Execute programs with the existing worker pool |

HTTP is the public request-and-response protocol. Internally, gRPC is the
streaming remote-call protocol connecting each node to the Router. It uses
Protocol Buffers, a binary message format, over HTTP/2.

The node manager opens `ConnectSupervisor`, a stream of health, instance and
capacity reports. The Python server opens one `ConnectSlot` stream per local
worker. A slot is one connection able to carry one execution at a time. Both
connections originate on the node; the Router sends work over these existing
streams. The node manager never handles execution bodies.

Request and response bytes travel in frames of at most 256 KiB, where KiB means
1024 bytes. Small bounded queues and HTTP/2 flow control limit buffering in the
tunnel. The Python `/execute` implementation still assembles the complete
request before parsing it.

## Build and launch

Use Rust 1.87 or newer and Cargo, Rust's build tool. The build supplies its own
Protocol Buffers compiler. Build from the repository root:

```bash
cargo build --release --locked
export PATH="$PWD/target/release:$PATH"
```

Install the [Python package](../getting-started/installation.md) on the Router host
and the worker environment on each compute node. Start the Router:

```bash
export KCORAL_NODE_TOKEN='<shared-node-token>'
kcoral router --host 127.0.0.1 --port 9000
```

On the same machine, start a node with the same token and a stable `node-id`:

```bash
export KCORAL_NODE_TOKEN='<shared-node-token>'
kcoral server \
  --router http://127.0.0.1:9000 \
  --node-id gpu-a
```

The examples run on one machine. For remote nodes, bind the Router with
`--host 0.0.0.0` and replace `127.0.0.1` with its reachable address. Give each node
a distinct stable `node-id`. The example uses
plain HTTP for a controlled network; an HTTPS endpoint requires TLS (transport
encryption) terminated by a compatible proxy. The optional token authenticates
node streams; it does not provide public client authorization or encryption.

With `--router`, `kcoral server` starts a supervisor (the node manager) that
manages the Python server using the current Python environment. Its health-check address follows
`--host` and `--port`; the Router does not connect to that address.
Without `--router`, the Python server runs directly.

Pass server options directly:

```bash
kcoral server \
  --router http://127.0.0.1:9000 \
  --node-id gpu-a \
  --host 0.0.0.0 --gpus 0 --log-dir /var/log/kcoral
```

The manager requires Linux 5.3 or newer, access to `/proc`, and permission to
signal its child processes. Run the Router
and node managers under your service manager so those processes are also
restarted if they fail.

Clients continue using `Client`, `Program`, `POST /execute` and `GET /health`:

```python
from kcoral import Client

with Client("http://127.0.0.1:9000") as client:
    print(client.target())
    # client.execute(program) uses the same program format as a direct server.
```

## Capacity and health

A node needs a healthy supervisor report and an idle slot for the same Python
server instance before it can receive work. All eligible nodes must match the
Router's target and runtime-version baseline. A Python restart invalidates slots
from the previous instance. Disconnected nodes without active requests expire
after the retention interval; once all records expire, the next healthy node
establishes a new compatibility baseline.

When choosing between nodes, the Router compares reported busy workers with
its own active requests and divides the larger number by worker capacity. It
samples two candidates and chooses the lower load, breaking ties toward the
least recently selected node. Python's worker pool remains the final capacity
boundary.

| Router option | Default | Meaning |
| --- | --- | --- |
| `--host`, `--port` | `127.0.0.1`, `9000` | Router listen address |
| `--status-check-interval-seconds` | `1` | Interval for checking status freshness |
| `--max-status-age-seconds` | `5` | Maximum age of a node health report |
| `--unhealthy-threshold` | `3` | Failed observations before removing a ready node |
| `--recovery-threshold` | `2` | Successful observations before a failed node returns |
| `--queue-wait-timeout-seconds` | `1800` | Maximum wait for capacity |
| `--max-queued-requests` | `1024` | Bound on requests waiting for capacity |
| `--node-retention-seconds` | `600` | Retention of disconnected, unused node records |
| `--max-request-bytes` | `268435456` | Maximum accepted request body size |

The Router uses the [health schema](../client-guide/protocol.md#get-health),
with router-local request counts.

The supervisor polls loopback-only `/internal/worker-status` and forwards worker
occupancy and environment information to the router. Node eligibility follows
these states:

| Node status | Meaning |
| --- | --- |
| `starting` | A node is registered but has not established healthy service |
| `ready` | Health and compatibility allow admission, subject to capacity |
| `recovering` | Successful observations are accumulating after a failure |
| `incompatible` | Target or runtime versions differ from the Router baseline |
| `unhealthy` | Health observations failed or became stale |

Start with one Router. With multiple replicas, direct each node's supervisor
stream and all of its slots to the same replica. Otherwise a replica may see
health without the corresponding execution connections.

## Cache negotiation and failures

`X-KCoral-Node` identifies the selected node. The client sends it back
as a preference during cache-miss retries. If the node or Python instance
changes, the client may need to resend bytes. Memory caches are instance-local;
persistent file caches follow their configured storage lifetime.

The Router creates a fresh `X-Request-ID` for every HTTP attempt, including
rejections, and propagates it through Router and Python logs. Cache negotiation
may therefore produce several request identifiers for one `Client.execute()`.
Router and node-manager logs are plain text without terminal color codes; the
Router's `request_finished` record can be correlated with Python's JSON events
using `request_id`.

| Failure | Client-visible behavior |
| --- | --- |
| Queue full or no capacity before timeout | HTTP 503 before node selection; retrying is safe |
| Tunnel fails after work may have reached a node | HTTP 502 with an unknown execution outcome; the Router does not replay the program |
| Client disconnects | Router capacity is released and the slot is discarded; this does not prove GPU computation has stopped |
| Router connection fails | Node connections reconnect; local server health checks and restart decisions continue independently |

The node manager checks local health every 2 seconds by default, with a 1-second
probe timeout, 3-failure threshold and 30-second startup grace. Restart delays
grow from 1 to 30 seconds with jitter, and reset after 60 seconds of stable running.

To stop a node, send SIGTERM, the normal termination signal, to the `kcoral server` process.
It withdraws healthy status and signals the Python child. Idle slots close and
active requests finish before the child exits. Normal shutdown waits for that
exit without imposing an additional timeout.

When restarting an unhealthy server, `--termination-grace-seconds` (default 5)
limits the wait after SIGTERM before forced cleanup. The node manager removes
descendant processes before starting the replacement Python server.

External shutdown deadlines can still interrupt work. Configure the service
manager's stop timeout, such as systemd's `TimeoutStopSec`, to allow the desired
request completion. The service manager must also clean up the entire process
tree if the node manager itself dies.

Implementation reference: [Router source and package guide](https://github.com/mlc-ai/kcoral/tree/main/rust/kcoral).
