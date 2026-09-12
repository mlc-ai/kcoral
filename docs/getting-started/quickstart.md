# Your First Program

Upload a four-element tensor, add one on the GPU (graphics processing unit),
and return the result. The uploaded function checks that its input is on the
GPU before doing arithmetic. The client only needs KCoral and NumPy, a Python
array library; the computation happens on the server.

## Before you start

Install the [client](installation.md#the-client) and use a running server with
the [GPU worker environment](installation.md#running-gpu-programs). A CPU
(central processing unit) compilation server cannot run this example. See
[Launch the server](../server/deployment.md) if you need to start one.

## Submit the program

Run from your repository checkout, replacing the address when using a remote server:

```bash
KCORAL_URL=http://localhost:8000 python examples/first_program.py
```

```{literalinclude} ../../examples/first_program.py
:language: python
```

{download}`Download the complete program <../../examples/first_program.py>`.

Expected output:

```text
COMPLETED
[1. 2. 3. 4.]
```

## How it works

1. `Program.upload(kind="module")` sends the source defining `add_one`.
2. `Program.get_function()` selects that function from the uploaded module.
3. The tensor upload transfers the NumPy input to the server's GPU.
4. `Program.run()` performs the addition on that GPU and binds the result to `y`.
5. `Program.return_()` selects the output for the response. The client decodes it
   as a CPU NumPy array and checks the values.

The example also checks `result.completed` before reading the output. Instruction
failures are returned as data in `result.error`; connection failures and request
errors raise the exceptions documented in the {ref}`Python API <python-errors>`.

Continue to [Writing a Program](../client_guide.md) for the builder methods and
request lifecycle, or [Benchmark a Kernel with KCoral](../tutorials/benchmark-kernel.md)
to compile a custom kernel, check correctness and measure its execution.
