from benchmark_server.gpu_runtime import _CUDAErrorAPI


class _Function:
    def __init__(self, fn):
        self._fn = fn
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self._fn(*args)


class _Library:
    def __init__(self, errors):
        self.cudaGetLastError = _Function(lambda: errors.pop(0))
        self.cudaGetErrorName = _Function(lambda code: b"cudaErrorInvalidValue")
        self.cudaGetErrorString = _Function(lambda code: b"invalid argument")


def test_cuda_error_api_consumes_and_formats_last_error():
    api = _CUDAErrorAPI(_Library([1, 0]))

    assert api.take_last_error() == "CUDA error cudaErrorInvalidValue (1): invalid argument"
    assert api.take_last_error() is None
