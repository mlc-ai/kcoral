"""Uploaded by mcp_client.py and run on the worker: report its Python, torch, and GPUs."""

import json
import platform

import torch

print(
    json.dumps(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    )
)
