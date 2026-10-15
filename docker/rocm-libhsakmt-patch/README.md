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

## How KCoral uses it

`docker/Dockerfile.rocm.dev` always builds with it: it clones the ROCR-Runtime
tag matching the image's ROCm release (or `--build-arg ROCR_TAG=rocm-X.Y.Z`),
applies the patch, rebuilds `libhsa-runtime64` (which links libhsakmt in), and
`install_patched_rocr.sh` replaces every copy the server can load, in
`/opt/rocm/lib` and the one bundled with the PyTorch wheel, keeping `.orig`
backups. The build needs `rocm-llvm-dev`, installed from the image's ROCm apt
repository. The image's server runs bubblewrap workers by default.

Outside that image (for example `pip install 'kcoral[rocm]'` on a host's own
ROCm), the runtime is unpatched. A sandboxed worker then fails to start with
"HIP found no GPU inside the bubblewrap sandbox"; run the same build steps
against your ROCm, or start the server with `--sandbox none`.

## Upstreaming

If validation passes, the patch is small enough to submit to ROCm (now
developed in `ROCm/rocm-systems`, `projects/rocr-runtime`) with the strace
evidence, under a title such as "libhsakmt: skip memory policy for GPUs
without an acquired VM".
