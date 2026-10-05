# Your First Program

Start a KCoral server, then submit a program that adds one to a four-element
tensor on its GPU and returns the result.

## Required hardware

You need one Linux machine with an NVIDIA GPU and a compatible driver. Follow
[Install the server](installation.md#install-the-server) and check the
[system requirements](installation.md#server-system-requirements); the server
installation also includes the client. The example uses PyTorch and does not
compile a custom kernel.

The steps below run the server and client on that same machine, in two
terminals. A CPU compilation server cannot run this program: the uploaded
tensor requires GPU support, and the function also checks that it is on a GPU
before doing arithmetic. Activate the same Python environment in both terminals.

## Launch the server

```{warning}
KCoral allows clients to execute arbitrary code on its workers. Only allow
trusted clients to access your KCoral server or Router. Deploy on a trusted,
isolated network and never expose these endpoints to the public internet.
Run workers in a sandbox with restricted permissions and access to host resources.
```

In the first terminal, start one worker on GPU 0:

```bash
kcoral server --device gpu --gpus 0 --workers-per-gpu 1 --host 127.0.0.1 --port 8000
```

GPU 0 is the first GPU listed by `nvidia-smi`. Wait for server startup to finish
and leave this terminal running. The client will connect to
`http://127.0.0.1:8000`.

## Submit the program

In a second terminal, save the following complete program as `first_program.py`:

```{literalinclude} ../../examples/basics/first_program.py
:language: python
```

You can also {download}`download first_program.py <../../examples/basics/first_program.py>`.
Run it from the directory where you saved it:

```bash
KCORAL_URL=http://127.0.0.1:8000 python first_program.py
```

Expected output:

```text
COMPLETED
[1. 2. 3. 4.]
```

The program checks that the request completed and that the returned array has
the expected values. When you finish, press `Ctrl+C` in the server terminal to
stop it.

### Use a remote server

To submit from another machine, start the server with `--host 0.0.0.0` so it
listens beyond the local machine. Install the
[client](installation.md#install-the-client) on the submitting machine, save the same
program there, and set `KCORAL_URL` to the server's reachable address on port
8000. The client machine does not need a GPU.
See [Launch the server](../server-guide/launch-the-server.md) for more configuration.

## How it works

The `Program` calls describe the work; `client.execute()` submits it. The server
then executes the instructions in order:

1. Load the uploaded source defining `add_one`.
2. Select that function from the module.
3. Transfer the uploaded NumPy input to the GPU.
4. Call `add_one` with that tensor.
5. Return the selected output. The client decodes it as a CPU NumPy array and
   checks the values.

The example also checks `result.completed` before reading the output. Instruction
failures are returned as data in `result.error`; connection failures and request
errors raise the exceptions documented in the {ref}`Python API <python-errors>`.

Continue to [Write a client program](../client-guide/writing-a-program.md) for
building programs and reading results, or [Benchmark a Kernel with KCoral](../tutorials/benchmark-kernel.md)
to compile a custom kernel, check correctness and measure its execution.
