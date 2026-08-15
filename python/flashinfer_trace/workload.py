"""Self-contained GPU input generation uploaded to the benchmark server."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any

_SAFETENSORS_DTYPES = {
    "BOOL": ("bool", 1),
    "U8": ("uint8", 1),
    "I8": ("int8", 1),
    "U16": ("uint16", 2),
    "I16": ("int16", 2),
    "U32": ("uint32", 4),
    "I32": ("int32", 4),
    "U64": ("uint64", 8),
    "I64": ("int64", 8),
    "F8_E4M3": ("float8_e4m3fn", 1),
    "F8_E5M2": ("float8_e5m2", 1),
    "F16": ("float16", 2),
    "BF16": ("bfloat16", 2),
    "F32": ("float32", 4),
    "F64": ("float64", 8),
}


def make_input_generator(
    normalized: dict[str, Any],
    resource_manifest: dict[str, int],
    *resource_bytes: bytes,
):
    """Return a callable that creates one trial's inputs on the current GPU."""

    if not isinstance(normalized, dict):
        raise TypeError("normalized evaluation must be an object")
    if not isinstance(resource_manifest, dict) or not all(
        isinstance(path, str) and isinstance(index, int) and not isinstance(index, bool)
        for path, index in resource_manifest.items()
    ):
        raise TypeError("resource manifest must map paths to integer handles")
    if sorted(resource_manifest.values()) != list(range(len(resource_bytes))):
        raise ValueError("resource manifest must use every bytes handle exactly once")
    if not all(isinstance(value, bytes) for value in resource_bytes):
        raise TypeError("safetensors resources must be bytes handles")

    definition = normalized["definition"]
    workload = normalized["workload"]
    if any(
        specification["dtype"] == "float4_e2m1" for specification in definition["inputs"].values()
    ):
        raise ValueError("float4_e2m1 input generation is not supported")
    referenced_paths = {
        input_spec["path"]
        for input_spec in workload["inputs"].values()
        if input_spec["type"] == "safetensors"
    }
    if set(resource_manifest) != referenced_paths:
        raise ValueError("resource manifest does not match workload safetensors paths")

    parsed_resources = {
        path: _parse_safetensors(resource_bytes[index]) for path, index in resource_manifest.items()
    }

    def generate(trial_index: int) -> list[Any]:
        if isinstance(trial_index, bool) or not isinstance(trial_index, int) or trial_index < 0:
            raise ValueError("trial index must be a non-negative integer")

        generated: list[Any] = []
        for input_name, tensor_spec in definition["inputs"].items():
            input_spec = workload["inputs"][input_name]
            shape = _resolve_shape(tensor_spec["shape"], definition["axes"], workload["axes"])
            dtype = tensor_spec["dtype"]
            if input_spec["type"] == "random":
                if shape is None:
                    raise ValueError(f"random input {input_name!r} requires a tensor shape")
                generated.append(
                    _random_tensor(
                        shape,
                        dtype,
                        _stable_seed(workload["uuid"], trial_index, input_name),
                    )
                )
            elif input_spec["type"] == "scalar":
                if shape is not None:
                    raise ValueError(f"scalar input {input_name!r} requires shape=null")
                generated.append(input_spec["value"])
            elif input_spec["type"] == "safetensors":
                generated.append(
                    _load_safetensors_tensor(
                        parsed_resources[input_spec["path"]],
                        input_spec["tensor_key"],
                        shape,
                        dtype,
                    )
                )
            else:
                raise ValueError(f"unsupported input type: {input_spec['type']!r}")
        return generated

    return generate


def _stable_seed(workload_uuid: str, trial_index: int, input_name: str) -> int:
    digest = hashlib.sha256(f"{workload_uuid}\0{trial_index}\0{input_name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _resolve_shape(
    symbolic_shape: list[str] | None,
    axes: dict[str, dict[str, Any]],
    workload_axes: dict[str, int],
) -> list[int] | None:
    if symbolic_shape is None:
        return None
    return [
        axes[name]["value"] if axes[name]["type"] == "const" else workload_axes[name]
        for name in symbolic_shape
    ]


def _torch_dtype(dtype: str):
    import torch

    try:
        return {
            "bool": torch.bool,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "float8_e5m2": torch.float8_e5m2,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"unsupported input dtype: {dtype!r}") from exc


def _random_tensor(shape: list[int] | None, dtype: str, seed: int):
    import torch

    if shape is None:
        raise ValueError("random tensor shape must not be null")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    torch_dtype = _torch_dtype(dtype)
    if dtype == "bool":
        return torch.randint(0, 2, shape, dtype=torch.int8, device="cuda", generator=generator).to(
            torch.bool
        )
    if dtype.startswith("int"):
        low, high = (-128, 128) if dtype == "int8" else (-1024, 1024)
        return torch.randint(
            low, high, shape, dtype=torch_dtype, device="cuda", generator=generator
        )
    values = torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator)
    if dtype.startswith("float8"):
        values.clamp_(-2.0, 2.0)
    return values.to(torch_dtype)


def _parse_safetensors(data: bytes) -> dict[str, tuple[str, list[int], bytes]]:
    if len(data) < 8:
        raise ValueError("safetensors file is shorter than its length prefix")
    header_length = struct.unpack("<Q", data[:8])[0]
    if header_length == 0 or header_length > len(data) - 8:
        raise ValueError("safetensors header length is invalid")
    header_bytes = data[8 : 8 + header_length]
    try:
        header = json.loads(
            header_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"safetensors header is invalid: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")

    metadata = header.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
        )
    ):
        raise ValueError("safetensors metadata must map strings to strings")

    payload = data[8 + header_length :]
    parsed: dict[str, tuple[str, list[int], bytes]] = {}
    ranges: list[tuple[int, int, str]] = []
    for tensor_name, tensor in header.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("safetensors tensor names must be non-empty strings")
        if not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"safetensors tensor {tensor_name!r} has invalid metadata")
        safetensors_dtype = tensor["dtype"]
        if safetensors_dtype in {"F4", "F4_E2M1"}:
            raise ValueError("float4_e2m1 safetensors data is not supported")
        try:
            dtype, item_size = _SAFETENSORS_DTYPES[safetensors_dtype]
        except (KeyError, TypeError):
            raise ValueError(
                f"safetensors tensor {tensor_name!r} has unsupported dtype {safetensors_dtype!r}"
            ) from None
        shape = tensor["shape"]
        if not isinstance(shape, list) or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
            for dimension in shape
        ):
            raise ValueError(f"safetensors tensor {tensor_name!r} has invalid shape")
        offsets = tensor["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets)
        ):
            raise ValueError(f"safetensors tensor {tensor_name!r} has invalid offsets")
        start, end = offsets
        if not 0 <= start <= end <= len(payload):
            raise ValueError(f"safetensors tensor {tensor_name!r} offsets are out of bounds")
        expected_length = math.prod(shape) * item_size
        if end - start != expected_length:
            raise ValueError(
                f"safetensors tensor {tensor_name!r} expects {expected_length} bytes, "
                f"got {end - start}"
            )
        ranges.append((start, end, tensor_name))
        parsed[tensor_name] = (dtype, shape, payload[start:end])

    position = 0
    for start, end, tensor_name in sorted(ranges):
        if start != position:
            raise ValueError(f"safetensors tensor {tensor_name!r} has a gap or overlapping offset")
        position = end
    if position != len(payload):
        raise ValueError("safetensors payload has unreferenced bytes")
    return parsed


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate safetensors header key: {key!r}")
        result[key] = value
    return result


def _load_safetensors_tensor(
    resource: dict[str, tuple[str, list[int], bytes]],
    tensor_key: str,
    expected_shape: list[int] | None,
    expected_dtype: str,
):
    import torch

    if expected_shape is None:
        raise ValueError("safetensors inputs require a declared tensor shape")
    try:
        dtype, shape, data = resource[tensor_key]
    except KeyError:
        raise ValueError(f"safetensors tensor key is missing: {tensor_key!r}") from None
    if dtype != expected_dtype:
        raise ValueError(
            f"safetensors tensor {tensor_key!r} has dtype {dtype!r}, expected {expected_dtype!r}"
        )
    if shape != expected_shape:
        raise ValueError(
            f"safetensors tensor {tensor_key!r} has shape {shape}, expected {expected_shape}"
        )
    if not data:
        return torch.empty(shape, dtype=_torch_dtype(dtype), device="cuda")
    return torch.frombuffer(bytearray(data), dtype=_torch_dtype(dtype)).reshape(shape).to("cuda")


__all__ = ["make_input_generator"]
