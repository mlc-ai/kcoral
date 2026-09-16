# Whole-model inference on a GPU set

Start a server configured with at least the requested number of GPUs:

```bash
kcoral --gpus 0,1,2,3 --workers-per-gpu 1
python examples/multi_gpu_inference/client.py --gpus 4
python examples/multi_gpu_inference/client.py --gpus 4 --single-process
```

Both commands upload the same `infer.py`. It contains a small, randomly
initialized residual MLP (multilayer perceptron), sharded by intermediate
features. All layers and multiple inference steps execute in one request.
Every distributed rank checks its output against an unsharded reference. This
is a correctness example, not a pretrained language model or a latency claim.

The script can also run independently:

```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/multi_gpu_inference/infer.py
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  examples/multi_gpu_inference/infer.py
```

For an existing inference project, upload its code/configuration folder, and
replace the launcher's argument list with its normal command. Keep model weights
in a server-local or shared directory and pass that path. Preinstall its Python,
CUDA and communication dependencies on the server. Folder uploads include
hidden files and reject symlinks; do not include your virtual environment or
large checkpoints.

`Client.execute(..., gpu_count=N)` requests one fresh interpreter with all N
devices visible. KCoral does **not** launch the program once per GPU or
initialize a communication group. The script may control every device itself,
use `torchrun`, or use its own launcher. Wait for children before returning and
propagate nonzero exit codes. The complete GPU set remains reserved through
interpreter and descendant cleanup, including file returns and CPU-only steps.

The server creates a Linux supervisor that adopts orphaned descendants, even
when a launcher starts new process sessions. A background process left after
the program finishes is terminated and makes the request fail. Timeouts also
clean up the whole process tree before the set can be reused. The default
execution limit is 300 seconds, capped by a server default of 900 seconds;
configure the server for longer model initialization when needed.

Connect directly to the desired GPU server. The current router does not select
nodes by requested GPU count. This feature allocates devices on one machine;
the script must check any peer-access or topology requirements. Requests do
not preserve model state across calls. Several `run` instructions may still
share ordinary interpreter-local objects within one request; remote child
process state does not automatically become a KCoral register.
