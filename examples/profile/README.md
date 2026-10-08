# KCoral overhead: vector_add

How much time does running a job through a KCoral server add compared with running
it directly? This example runs `c = a + b` (float32, PyTorch) both ways on the same GPU.

| Mode | What is timed |
|---|---|
| local | in-process, warm CUDA context: numpy → GPU, `a + b`, GPU → numpy |
| local, new process | the same job in a fresh `python` (includes `import torch` and CUDA init) |
| KCoral warm | build `Program` + `client.execute`: upload tensors, run, return `c` (input blobs already cached) |
| KCoral cold | the same, with new tensor contents: `CACHE_MISS`, then the blobs are uploaded |
| kernel | GPU time of `a + b` from `kcoral.builtins.benchmark` (CUPTI, 100 warmup + 1000 timed runs), locally and in KCoral |

The modes are interleaved in each trial: 50 end-to-end trials and 30 kernel trials per
configuration. Before each KCoral request the script waits until all 8 workers are ready,
so the results do not include waiting for worker replacement. Sizes are 4 KiB, 4 MiB and
64 MiB per input tensor. Two server configurations are tested:
`--max-requests-per-worker 1` (**fresh**, the default: a new worker process per request)
and `0` (**reused**).

## Reproduce

```bash
examples/profile/run.sh [GPU_ID] [PORT]   # defaults: 0 8765; takes ~35 min
```

`run.sh` installs the locked environment (`uv sync --group server`), starts the server
for each configuration, runs `bench.py` for each size and writes `out/`
(raw samples in `*.json`, tables in `summary.md`, environment in `env.txt`).

## Results (2026-10-08)

Environment: RTX 2070 (sm_75, PCIe 3.0 x16), driver 595.58.03, Threadripper 3970X,
Ubuntu 24.04 (kernel 6.8), Python 3.12, torch 2.14.0+cu130, KCoral `dfcf30c`. The client
and server ran on the same host over loopback. The sandbox was **off**: bubblewrap could
not start because AppArmor restricts unprivileged user namespaces. We used GPU 1 because
another job was running on GPU 0.

End-to-end time per job, ms: median [min, max]

| workers | tensor | local | local, new process | KCoral warm | KCoral cold | first request |
|---|---|---|---|---|---|---|
| fresh | 4 KiB | 0.34 [0.10, 0.39] | 1779 [1718, 1814] | 337.9 [315.6, 343.4] | 344.1 [324.9, 350.2] | 351 |
| fresh | 4 MiB | 1.53 [1.41, 5.17] | 1773 [1742, 1833] | 381.1 [340.6, 392.3] | 404.8 [370.2, 439.3] | 436 |
| fresh | 64 MiB | 48.47 [47.39, 54.31] | 1942 [1919, 1957] | 1239.0 [1175.0, 1369.5] | 1654.6 [1591.4, 1755.2] | 1716 |
| reused | 4 KiB | 0.36 [0.10, 0.42] | 1717 [1692, 1741] | 16.3 [10.8, 73.4] | 20.3 [15.1, 21.6] | 39 |
| reused | 4 MiB | 1.62 [1.39, 4.44] | 1733 [1710, 1753] | 55.5 [41.5, 98.5] | 76.3 [53.1, 94.3] | 111 |
| reused | 64 MiB | 46.31 [46.04, 47.33] | 1909 [1889, 1938] | 934.7 [907.1, 1063.9] | 1289.8 [1196.5, 1402.4] | 1430 |

Breakdown of a warm KCoral request, ms (medians). "client build" includes hashing the
input blobs (SHA-256). "server elapsed" and "lease held" are the `ProgramResult`
fields. "rest" is wall time minus build minus elapsed: HTTP and decoding the response.

| workers | tensor | client build | server elapsed | lease held | rest |
|---|---|---|---|---|---|
| fresh | 4 KiB | 0.1 | 332.0 | 330.9 | 5.8 |
| fresh | 4 MiB | 9.0 | 360.5 | 342.8 | 13.3 |
| fresh | 64 MiB | 136.4 | 882.5 | 571.7 | 215.8 |
| reused | 4 KiB | 0.1 | 9.6 | 8.6 | 6.5 |
| reused | 4 MiB | 8.8 | 25.8 | 15.7 | 16.4 |
| reused | 64 MiB | 135.7 | 620.3 | 126.9 | 175.1 |

Kernel latency (CUPTI median of each run), µs: median [min, max]

| workers | tensor | local | KCoral |
|---|---|---|---|
| fresh | 4 KiB | 1.98 [1.73, 1.98] | 1.95 [1.70, 1.98] |
| fresh | 4 MiB | 33.50 [33.47, 33.50] | 33.47 [33.47, 33.50] |
| fresh | 64 MiB | 510.66 [510.21, 510.72] | 510.56 [510.34, 510.69] |
| reused | 4 KiB | 1.98 [1.98, 1.98] | 1.98 [1.98, 2.02] |
| reused | 4 MiB | 33.50 [33.50, 33.54] | 33.60 [33.57, 33.60] |
| reused | 64 MiB | 510.56 [510.00, 510.69] | 510.69 [510.40, 510.76] |

## Findings

- **Kernel timing is unaffected.** Medians inside and outside KCoral agree within 0.15 µs
  (≤0.3% at 4 MiB and 64 MiB).
- **The fixed per-request cost depends on whether workers are reused.** It is ~16 ms with
  reused workers and ~335 ms with the default fresh workers. With fresh workers, the
  response waits while the retired worker exits under the GPU lease (Python/CUDA teardown;
  `_wait_for_exit` in `runtime/worker.py`). Replacing a worker took 3.3 s (median;
  `worker_retired` → `worker_ready`). That is outside these timings, but more than 8
  back-to-back requests within that window would queue.
- **Tensor size adds overhead.** At 64 MiB per tensor with reused workers, a warm request
  adds ~0.9 s over local (46 ms). The time goes to the client hashing 128 MiB of inputs
  (136 ms), the server moving data outside the lease (elapsed − lease held ≈ 490 ms),
  and returning the 64 MiB result (~175 ms).
- **Cold start = blob upload.** With new tensor contents, the extra `CACHE_MISS` round
  trip and upload add 4–6 ms (4 KiB), 21–24 ms (4 MiB) and 355–415 ms (64 MiB) to the
  median. The first request after server start (one sample per run) was 7–140 ms slower
  than the cold median. There is no large startup cost, because workers start with a ready
  CUDA context.
- A KCoral request is still faster than starting a new local Python process
  (~1.7–1.9 s), which pays for `import torch` and CUDA init.

Not covered here: the bubblewrap sandbox, a network between client and server,
concurrent requests, and non-PyTorch kernels.
