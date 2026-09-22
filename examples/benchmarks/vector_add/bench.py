"""Inputs and correctness reference for the standalone vector-add benchmark."""

import torch


def make_inputs(case):
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(case["n"], device="cuda", generator=generator)
    y = torch.randn(case["n"], device="cuda", generator=generator)
    return x, y


def reference(x, y):
    return x + y
