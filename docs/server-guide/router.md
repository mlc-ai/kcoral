# Router

A Router gives clients one server address and selects an available compute
node for each request. A node supervisor starts, checks and restarts its local
Python server. The supervisor and Python server initiate their connections to
the Router, so compute nodes do not need to accept inbound network connections.

Install the client, Python server and native services from the same revision
so their internal protocols match. See [installation](../getting-started/installation.md#install-native-services).

## Components and connections

| Component | Where it runs | Responsibility |
| --- | --- | --- |
| `kcoral router` | A host reachable by clients and nodes | Accept HTTP requests, manage a bounded wait queue and select nodes |
| `kcoral server --router URL --node-id NAME` | Each compute node | Supervise the local execution service, report health and execute programs |

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

## Install and launch

Install the native services on the Router host and on each compute node using
[the installation guide](../getting-started/installation.md#install-native-services).
Compute nodes additionally need the Python worker environment. A Router host
needs only the client package and native binaries, without GPU libraries.

Start the Router at an address nodes can reach:

```bash
export KCORAL_NODE_TOKEN='<shared-node-token>'
kcoral router --host 0.0.0.0 --port 9000
```

On each compute node, use the same token and a distinct stable node ID:

```bash
export KCORAL_NODE_TOKEN='<shared-node-token>'
kcoral server --router http://router.example.com:9000 --node-id gpu-a --gpus 0
```

This one command starts both the process supervisor and the local Python
execution service. It checks health, restarts failed services and connects
outward to the Router. Omitting `--router` and `--node-id` starts a supervised
standalone server. Both modes accept the same server options:

```bash
kcoral server --router http://router.example.com:9000 --node-id cpu-a \
  --device cpu --num-workers 8 --port 8001 --log-dir /var/log/kcoral
```

`--router` and `--node-id` must be configured together. They default from
`KCORAL_ROUTER_ENDPOINT` and `KCORAL_NODE_ID`; command-line values take precedence.
`--node-token` defaults from `KCORAL_NODE_TOKEN`. Prefer the environment variable
so the token is not visible in the shell's command line. The launcher passes it
to child processes through the environment.

The Python service binds to `127.0.0.1:8000` by default. Its local health-check
address is derived automatically from `--host` and `--port`, including wildcard
and IPv6 bind addresses. The Router uses the node's outbound connections, so it
does not need access to the node's listening port. Multiple services on one host
need distinct ports and, when routed, distinct node IDs.

Replace the Router hostname and token for your deployment. The example uses
plain HTTP for a controlled network; an HTTPS endpoint requires TLS (transport
encryption) terminated by a compatible proxy. The optional token authenticates
node streams; it does not provide public client authorization or encryption.

The native supervisor requires Linux 5.3 or newer, access to `/proc`, and
permission to signal its child processes. Run `kcoral router` and each
`kcoral server` under your service manager so they also restart after a host
reboot or a supervisor crash. Service options are listed by
`kcoral router --help` and `kcoral server --help`.

Clients continue using `Client`, `Program`, `POST /execute` and `GET /health`:

```python
from kcoral import Client

with Client("http://router.example.com:9000") as client:
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
