"""A simple candidate with an output buffer allocated outside timing."""

import torch


def prepare(x, y):
    output = torch.empty_like(x)

    def run():
        torch.add(x, y, out=output)
        return output

    return run
