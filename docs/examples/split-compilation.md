# Separate compilation and execution


Use a CPU (central processing unit) server for compilation and a GPU (graphics
processing unit) server for execution. The client installs only KCoral. Install
the [compiler environment](../getting-started/installation.md#running-cpu-compilation-workers)
on the CPU server and the [GPU environment](../getting-started/installation.md#running-gpu-programs)
on the execution server.

Start each server in a separate terminal or on its own machine:

```bash
kcoral --device cpu --num-workers 8 --port 8000
kcoral --device gpu --gpus 0 --port 8001
```

Then run the client from the repository checkout:

```bash
KCORAL_CPU_URL=http://localhost:8000 KCORAL_GPU_URL=http://localhost:8001   python examples/cpu_compile_gpu_execute.py
```

Use reachable server addresses for a deployment across machines. The client
reads the GPU target, sends a compilation request to the CPU server, then uploads
the returned library in a second request to the GPU server. Both use the same
`POST /execute` protocol. The script prints the library size, target and timing;
the CPU request reports zero time holding a GPU lease.


## Source

{download}`Download cpu_compile_gpu_execute.py <../../examples/cpu_compile_gpu_execute.py>`.

```{literalinclude} ../../examples/cpu_compile_gpu_execute.py
:language: python
:linenos:
```
