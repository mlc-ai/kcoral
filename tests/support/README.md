# Test programs

`programs.py` constructs upload/get_function/run sequences used by engine and
protocol tests. `fake.py` supplies small process-lifecycle workloads; `core.py`
provides comparison reports. Tests initialize tensors with uploaded PyTorch code.

CUDA C, CuTeDSL and Triton compiler helpers remain here for integration tests.
Tests use the importable `kcoral.builtins.compile_tirx` and `kcoral.builtins.benchmark`
functions for TIRx compilation and GPU activity measurement.

GPU compilation helpers retain the lease because compilation can call CUDA or
load modules. `compile_cuda_binary` uses an explicit target and returns bytes,
so its calls are declared CPU-only. See `test_engine.py` for an explicit
host-build/GPU-load sequence.
