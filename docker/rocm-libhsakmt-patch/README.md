# ROCm `libhsakmt` patch: sandboxed workers with only their own render node

## The bug

`hsakmt_fmm_init_process_apertures()` in ROCR-Runtime's
`libhsakmt/src/fmm.c` (ROCm 7.0.0 through 7.0.2, `develop` and `amd-staging`)
has two loops over the GPUs the kernel reports for the process:

1. The aperture and VM loop skips any GPU that is not in libhsakmt's usable list
   (`gpu_mem_find_by_gpu_id(...) < 0`), for example because its render node is
   not accessible.
2. The memory-policy loop that follows calls `fmm_set_memory_policy()` for
   **every** reported GPU, without that check, and aborts initialization on any
   failure.

In a bubblewrap sandbox exposing `/dev/kfd` and one render node, the kernel
still reports all GPUs the device cgroup allows. Loop 1 acquires a VM only for
the sandbox's GPU; loop 2 then sets a policy on a GPU without a VM, the kernel
returns `ESRCH`, libhsakmt prints `Failed to set mem policy for GPU [0x…]`, and
HIP reports `hipErrorNoDevice`. This matches the strace and debug logs from the
8× MI355X runs (`AMDKFD_IOC_SET_MEMORY_POLICY` → `ESRCH` was the only failing
KFD ioctl).

## The patch

`0001-libhsakmt-skip-memory-policy-for-unusable-gpus.patch` adds the same
check to loop 2. Failures on usable GPUs still abort initialization. The rest
of the function already iterates only usable GPUs (`gpu_mem[]`), as does
ROCr's agent discovery (the trace reads topology details only for the usable
GPU).

Validated on 8× MI355X with ROCm 7.0.0 (`ROCR_TAG=rocm-7.0.0`): in an all-GPU
container every sandboxed worker saw exactly its own GPU, PCI address and
render node; NCCL all-reduce ran at 2, 4 and 8 GPUs through sandboxed workers;
the full KCoral suite passed (568 passed, 38 skipped); and restoring the
original runtime made the sandbox probe fail again. It applies cleanly to
ROCm 7.0.0, 7.0.1 and 7.0.2.

## Build

`libhsakmt` is linked statically into `libhsa-runtime64.so`, so the patch means
rebuilding ROCR-Runtime and replacing that library. The Dockerfile clones the
ROCR-Runtime tag matching the image's ROCm, applies the patch, builds, and
replaces every `libhsa-runtime64.so*` the server can load (`/opt/rocm/lib` and
the copy bundled in the PyTorch wheel), keeping `.orig` backups.

From the KCoral repository root, on the node:

```bash
docker build -f docker/Dockerfile.rocm.dev -t kcoral:rocm-dev .
docker build -f docker/rocm-libhsakmt-patch/Dockerfile -t kcoral:rocm-dev-patched \
  --build-arg ROCR_TAG=rocm-7.0.0 .
```

Set `ROCR_TAG` to the image's ROCm release (`cat /opt/rocm/.info/version`). The
build needs `rocm-llvm-dev` (for `ClangConfig.cmake`); the Dockerfile installs
it from the image's ROCm apt repository.

## Validate

Start a container with all GPUs and the bubblewrap Docker options, then run
`validate.sh` inside it from the repository root:

```bash
docker run -d --name kcoral-patched \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_ADMIN --cap-add NET_ADMIN \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --security-opt systempaths=unconfined \
  -v "$PWD:/workspace/kcoral" -w /workspace/kcoral \
  kcoral:rocm-dev-patched sleep infinity
docker exec -i kcoral-patched bash docker/rocm-libhsakmt-patch/validate.sh
```

Logs go to `validation-results/libhsakmt-patch/`. Pass criteria:

| Step | Pass |
| --- | --- |
| 1 | The library torch loads is the patched one |
| 2 | Unsandboxed, all 8 GPUs initialize |
| 3 | `rocm.sandbox_problem()` returns OK for every GPU, so each sandboxed GPU sees exactly itself |
| 4 | A server started **without** `--sandbox none` logs `pool_ready sandbox=bubblewrap`, and the Triton example allocates, runs and prints `timing=trace` |
| 5 | Sandbox, isolation and multi-GPU tests pass with `KCORAL_REQUIRE_SANDBOX_TESTS=1` |
| 6 | Full suite: no new failures |

Isolation holds by construction: a sandboxed worker has no device files for
other GPUs, and step 3 checks it sees exactly one GPU with the assigned PCI
address.

## KCoral side

KCoral does not need this patch to stay safe. A ROCm server probes HIP inside
a real sandbox for each configured GPU at startup (`rocm.sandbox_problem`) and
enables bubblewrap only if every probe sees exactly its card; otherwise it
refuses and says to use this patch or `--sandbox none`. With an unpatched ROCm
it refuses; with this patch it allows bubblewrap for any number of GPUs in one
container, as on NVIDIA.

## Upstreaming

If validation passes, the patch is small enough to submit to ROCm (now
developed in `ROCm/rocm-systems`, `projects/rocr-runtime`) with the strace
evidence, under a title such as "libhsakmt: skip memory policy for GPUs
without an acquired VM".
