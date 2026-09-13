# Upload a compiled library


Compile on the client, then upload a shared library, a binary file containing
callable compiled code. This path supports custom builds beyond the server's
built-in compiler options.

The client needs the [compilation toolchain](../../getting-started/installation.md#running-cpu-compilation-workers),
including TVM FFI (TVM's foreign-function interface), `nvcc`, `ninja` and a host
C++ compiler. The GPU server needs PyTorch, TVM FFI and CUPTI, NVIDIA's CUDA
Profiling Tools Interface, for execution, validation and timing.

```bash
kcoral --gpus 0 --port 8000
KCORAL_URL=http://localhost:8000 python examples/library_upload_client.py
```

Start the server and client in separate terminals. Set `KCORAL_URL` when using a
remote server. The client reads the server's target architecture before compiling,
prints the library size and target, then reports correctness and kernel timing.
See the [library protocol](../../client-guide/protocol.md#library) for export and linking rules.


## Source

{download}`Download library_upload_client.py <../../../examples/library_upload_client.py>`.

```{literalinclude} ../../../examples/library_upload_client.py
:language: python
:linenos:
```
