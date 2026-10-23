# Worker Tensor Transfer

A KCoral server runs each request in a worker process and moves the request's
tensors across the process boundary. For small inputs this is cheap. For large
tensors the copy dominates the request, so the server passes big blobs through
shared memory instead of serializing them over the worker control socket. This
page explains what crosses the boundary, when the shared-memory path is used,
and what it saves.

## What crosses the boundary

The server and each worker communicate over a control socket. By default a
request's inputs and results are pickled and written over that socket, including
on cache hits. Pickling and copying a few kilobytes is negligible, but a
64 MiB tensor is copied several times on the way in and again on the way out,
and that copying, not the GPU work, sets the request's wall time.

## The shared-memory path

Blobs of at least 1 MiB — inputs and results alike — are written into a sealed
[`memfd`](https://man7.org/linux/man-pages/man2/memfd_create.2.html) and the
file descriptor is handed to the worker over the same socket. The receiver
checks the seals and the exact size, then maps the buffer read-only; no bytes
are copied over the socket. Smaller blobs, non-Linux hosts, and Python builds
without `memfd_create` keep the pickling path, so the change is transparent to
clients.

## Measured effect

Vector add with 64 MiB tensors, medians of 30 trials in balanced order,
sandbox off, measured with the benchmark harness on two consumer GPUs:

| GPU | worker | serialized | shared memory |
| --- | --- | ---: | ---: |
| RTX 5090 | reused | 997 ms | 643 ms (−36%) |
| RTX 5090 | fresh | 1335 ms | 940 ms (−30%) |
| RTX 2070 | reused | 982 ms | 657 ms (−33%) |
| RTX 2070 | fresh | 1310 ms | 940 ms (−28%) |

A 4 MiB reused-worker request drops 17–28%. Requests below the 1 MiB threshold
do not use the shared-memory path and are unchanged.

## Reproduce

The numbers above come from a vector-add workload driven through the server over
loopback. Follow [Benchmark a Kernel with KCoral](../tutorials/benchmark-kernel.md)
to run a kernel end to end, and vary the tensor size and
`--max-requests-per-worker` (reused vs. fresh workers) to see the transfer cost
and the shared-memory path at each size.
