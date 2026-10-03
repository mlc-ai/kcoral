# Remote functions

The function decorator is the simplest way to use KCoral. Add
`@client.function()` to a Python function, then call `.remote()` to run it on
the server and receive its return value. KCoral builds and submits the program
for you.

## Start the remote server

On a Linux machine with an NVIDIA GPU and a compatible driver, install and
start the server:

```bash
python -m pip install 'kcoral[server]'
kcoral server --device gpu --gpus 0 --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete.` and leave this terminal running.
`--host 0.0.0.0` lets clients connect from other machines. Run the server on a
trusted network accessible only to trusted clients. See the
[system requirements](../getting-started/installation.md#server-system-requirements)
and [server guide](../server-guide/launch-the-server.md) for setup details.

## Call a remote function

On the client machine, [install the client](../getting-started/installation.md#install-the-client)
and save this as `remote_sum.py`. Replace `server` in the URL with the GPU
machine's hostname or IP address:

```python
from kcoral import Client

with Client("http://server:8000") as client:

    @client.function(timeout=30)
    def gpu_sum(n):
        import torch

        return torch.arange(n, device="cuda").sum().item()

    print(gpu_sum.remote(4))  # 6
```

Run `python remote_sum.py` on the client. The server creates the values 0
through 3 on its GPU and returns their sum, `6`.
`timeout=30` sets the server execution limit in seconds.

- Define the function in a Python file so KCoral can read its source.
- Import dependencies inside the function and install them on the server.
- Pass inputs as arguments; surrounding variables and external globals are
  not captured. Keep the client open while making remote calls.
- `.remote()` runs on the server; an ordinary call such as `gpu_sum(4)` runs
  locally. Each remote call is an independent request.

Arguments can be JSON values, bytes, NumPy arrays, or DLPack-compatible tensors.
Pass bytes and tensors as whole arguments. Returned tensors become local NumPy
arrays. `.remote()` raises an exception if execution fails; use `.execute()`
to receive the full `ProgramResult`, including captured output and error details.

See the [Python API](../python-api/index.rst) for details, or
{download}`download a tensor example <../../examples/remote_function.py>`.
For more control over individual instructions, see
[Write a client program](writing-a-program.md).
