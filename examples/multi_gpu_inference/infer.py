"""Run a sharded residual MLP in one process or with ordinary torchrun.

No KCoral imports: this is the same script locally and on the server. Each
layer shards its intermediate features across GPUs and sums their contributions
before the next layer. Several inference steps reuse the loaded weights.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def weights(hidden, layers):
    generator = torch.Generator().manual_seed(7)
    return [
        (
            torch.randn(hidden, 4 * hidden, generator=generator) / hidden**0.5,
            torch.randn(4 * hidden, hidden, generator=generator) / (4 * hidden) ** 0.5,
        )
        for _ in range(layers)
    ]


def shards(parameters, rank, world, device):
    return [
        (
            torch.tensor_split(up, world, dim=1)[rank].contiguous().to(device),
            torch.tensor_split(down, world, dim=0)[rank].contiguous().to(device),
        )
        for up, down in parameters
    ]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--output", default="result.json")
    args = parser.parse_args()
    if min(args.steps, args.hidden, args.layers) < 1:
        parser.error("steps, hidden and layers must be positive")
    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ["WORLD_SIZE"]) if distributed else torch.cuda.device_count()
    if not 1 <= world <= 8:
        raise ValueError("this example needs 1-8 visible GPUs")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    try:
        parameters = weights(args.hidden, args.layers)
        device = torch.device("cuda", local_rank)
        reference_weights = [(up.to(device), down.to(device)) for up, down in parameters]
        local_weights = (
            [shards(parameters, rank, world, device)]
            if distributed
            else [shards(parameters, i, world, torch.device("cuda", i)) for i in range(world)]
        )
        generator = torch.Generator().manual_seed(42)
        x = torch.randn(8, args.hidden, generator=generator).to(device)
        expected = x.clone()
        for _ in range(args.steps):
            for layer in range(args.layers):
                up, down = reference_weights[layer]
                expected = expected + torch.nn.functional.gelu(expected @ up) @ down
                if distributed:
                    up, down = local_weights[0][layer]
                    contribution = torch.nn.functional.gelu(x @ up) @ down
                    dist.all_reduce(contribution)
                else:
                    parts = []
                    for i, partition in enumerate(local_weights):
                        up, down = partition[layer]
                        partial = torch.nn.functional.gelu(x.to(i) @ up) @ down
                        parts.append(partial.to(device))
                    contribution = torch.stack(parts).sum(dim=0)
                x = x + contribution
            torch.testing.assert_close(x, expected, rtol=2e-4, atol=2e-4)
        report = {
            "ok": True,
            "gpu_count": world,
            "processes": world if distributed else 1,
            "steps": args.steps,
            "layers": args.layers,
            "max_abs_error": float((x - expected).abs().max()),
        }
        if distributed:
            reports = [None] * world
            dist.all_gather_object(reports, report)
        else:
            reports = [report]
        if rank == 0:
            Path(args.output).write_text(json.dumps({"ranks": reports}, indent=2))
            print(json.dumps({"ranks": reports}))
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
