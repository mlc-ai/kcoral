# KCoral Router and Node

This Cargo package builds two independent Rust processes:

- `kcoral-router`: the public KCoral HTTP API, outbound-node gRPC endpoint,
  admission queue, and scheduler.
- `kcoral-node`: a per-node process supervisor and outbound gRPC
  control client for the local Python server.

The public client protocol remains HTTP. Compute nodes may reject every inbound
connection: the supervisor opens `ConnectSupervisor`, and the Python server opens
one `ConnectSlot` call per local worker. The router sends each execution through
an already established data call. The supervisor never receives a request body.

Execution bodies are split into bounded 256 KiB protobuf `bytes` messages. Both
Rust and Python keep only a small number of messages queued and rely on HTTP/2
flow control for backpressure. A full request is assembled only by the existing
Python `/execute` implementation.

## Build and test

```bash
cargo build --release --locked
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

The build downloads a platform-specific Protocol Buffers compiler through a
build dependency. It does not use or require a system `protoc` executable. The
generated Python client modules are checked into `python/kcoral` so installing
the server does not require a compiler.

## Process ownership

Run the router and each node supervisor as separate systemd services or
container processes. The supervisor launches the Python server child, probes
its loopback-only `/internal/worker-status` endpoint, and restarts it independently of the router.
It passes `KCORAL_ROUTER_ENDPOINT`, `KCORAL_NODE_ID`, and the optional
`KCORAL_NODE_TOKEN` to the child so the child's data slots use the same identity
as its control stream. The host and port from `--server-url` become the child's
`KCORAL_SERVER_HOST` and `KCORAL_SERVER_PORT` defaults. Without a command after
`--`, the manager runs `kcoral`; explicit child flags override the defaults.
See the [Router deployment guide](../../docs/server-guide/router.md) for examples.

The router and supervisor themselves rely on systemd, Kubernetes, or an
equivalent service manager for process restart. A supervisor restart also
recreates its Python child. Router restarts do not require inbound access to a
node; its control and data clients reconnect with bounded exponential backoff.

The first deployment should run one router. A multi-router load balancer must
place each node's independently opened control and data calls on the same
router replica, normally by consistently hashing the authenticated node ID.

## Failure boundary

Queue rejection happens before a node is selected and is safe for a client to
retry. Once request data enters a selected outbound slot, the Python server may
have run the program. A tunnel failure is therefore returned as an unknown
outcome and is never replayed automatically on another node. A cancelled or
protocol-invalid slot is discarded and must reconnect before carrying another
request.

## Protocol and implementation

`router/pool.rs` owns node status, admission, capacity, and slot lifetime;
`router/gateway.rs` authenticates and registers outbound streams;
`router/http.rs` bridges HTTP bodies and records request completion. Internal
state stays private to the pool.

`ConnectSupervisor` returns one empty `SupervisorAck`. `ConnectSlot` first
accepts `SlotHello` and returns `SlotAck`; only then does Python reset reconnect
backoff. Subsequent frames have a shared request UUID. Both body directions may
progress concurrently, and the slot is reusable after both `EndOfBody` frames.
The HTTP response uses stream framing so the router consumes the tunnel's final
frame before returning capacity. Repeated application headers remain intact;
hop-by-hop headers and the backend Content-Length are removed.

Cancellation releases local capacity without waiting for a full transport queue.
The reason (`client_disconnected`, `request_rejected`, `invalid_response`, or
`tunnel_disconnected`) is recorded by the router and sent in a Cancel frame or
the terminating gRPC status. Expected cancellation of an accepted request
reconnects immediately; authentication, registration, and network failures use
exponential delay with jitter. Local shutdown never reconnects. Reconnecting a
slot does not imply its old computation has stopped, so scheduling also consults
reported busy workers.

The node binary runs as a Linux child subreaper: orphaned descendants are
reparented to it even if they created separate sessions. Repeated descendant
scans and stable process handles clean up a generation before the next server
starts. This requires Linux 5.3+, `/proc`, and permission to signal owned
processes using `pidfd_open` and `pidfd_send_signal`. Run one server tree per
dedicated node-manager process. Service-level cleanup remains the responsibility
of the external process manager if `kcoral-node` itself dies. Normal stop sends
SIGTERM and waits for the Python server to finish active requests and exit.
`--termination-grace-seconds` (default 5) only limits cleanup of an unhealthy
server during restart. An external service manager can still impose a shutdown
deadline; configure its stop timeout to allow the desired request completion.

Regenerate Python bindings from the repository root with a fixed compiler:

```bash
uv run --no-project --with grpcio-tools==1.66.2 --with protobuf==5.27.2 \
  python scripts/generate_gateway.py
```

Add `--check` to verify checked-in bindings without modifying them. Rust bindings
are generated by Cargo from the same file. These renamed RPCs require upgrading
router, node manager, and Python server together; they are not compatible with
the earlier unmerged protocol.

Run the real Rust/Python integration and process-tree tests without a GPU:

```bash
cargo build --locked
uv sync --frozen --no-editable --group test
KCORAL_REQUIRE_GATEWAY_TESTS=1 CUDA_VISIBLE_DEVICES='' \
  uv run --no-sync pytest -q -o pythonpath= tests/test_gateway.py tests/test_node_processes.py
```

The test suite uses simulated execution workers with the real application,
client, protobuf bindings, and Router. It verifies binary cache negotiation,
instance replacement during retries, large early responses, cancellation,
request IDs, graceful tunnel shutdown, and descendant cleanup. Setting
`KCORAL_REQUIRE_GATEWAY_TESTS=1` makes missing binaries fail instead of silently
skipping. GPU resource-release behavior is outside these CPU functional tests.
