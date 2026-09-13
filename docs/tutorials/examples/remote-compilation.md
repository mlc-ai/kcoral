# Remote compilation


Compile, check and measure a kernel in each supported language: TIRx (TVM's
kernel language), CuTeDSL (NVIDIA's Python language for CuTe kernels), CUDA C
(NVIDIA's GPU extension to C++), and Triton (a GPU kernel language and compiler).
The client needs KCoral and NumPy; the server needs the full
[GPU worker environment](../../getting-started/installation.md#running-gpu-programs),
including the CUDA toolkit and a host C++ compiler.

Start the server on the GPU machine, then run the client from your checkout:

```bash
kcoral --gpus 0 --port 8000
KCORAL_URL=http://localhost:8000 python examples/remote_compile_client.py
```

Set `KCORAL_URL` to a reachable address when the server is on another machine.
The script prints total, GPU and kernel times for each language and checks the
returned output against a NumPy reference. A failed language prints its error.
Timing values vary by machine and should not be compared to fixed example numbers.


## Source

{download}`Download remote_compile_client.py <../../../examples/remote_compile_client.py>`.

```{literalinclude} ../../../examples/remote_compile_client.py
:language: python
:linenos:
```
