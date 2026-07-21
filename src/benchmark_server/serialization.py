from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._dlpack import cpu_tensor_bytes
from .models import BinaryPart, ServerConfig


class InvalidReturnValue(ValueError):
    pass


@dataclass
class _State:
    output_dir: Path
    config: ServerConfig
    nodes: int = 0
    binary_bytes: int = 0

    def __post_init__(self) -> None:
        self.parts: list[BinaryPart] = []
        self.by_content: dict[tuple[str, int], BinaryPart] = {}


def serialize_return(
    value: Any, output_dir: Path, config: ServerConfig
) -> tuple[dict[str, Any], list[BinaryPart]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = _State(output_dir, config)
    tree = _encode(value, "$", 0, set(), state)
    try:
        metadata = json.dumps(
            tree, allow_nan=False, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise InvalidReturnValue(f"return value cannot be encoded: {exc}") from exc
    if len(metadata) > config.max_json_metadata_bytes:
        raise InvalidReturnValue("return value metadata exceeds the configured limit")
    if len(metadata) + state.binary_bytes > config.max_response_bytes:
        raise InvalidReturnValue(
            "return value exceeds the configured response-size limit"
        )
    return tree, state.parts


def _encode(
    value: Any, path: str, depth: int, active: set[int], state: _State
) -> dict[str, Any]:
    if depth > state.config.max_nesting_depth:
        raise InvalidReturnValue(f"maximum nesting depth exceeded at {path}")
    state.nodes += 1
    if state.nodes > state.config.max_description_nodes:
        raise InvalidReturnValue(f"maximum description-node count exceeded at {path}")

    if _is_json_subtree(value, path, depth, active, state.config.max_nesting_depth):
        return {"type": "json", "value": value}

    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        part = _store_binary(data, state)
        return {
            "type": "bytes",
            "part": part.name,
            "size": part.size,
            "sha256": part.sha256,
        }

    if _is_tensor(value):
        try:
            import tvm_ffi

            tensor = (
                value
                if isinstance(value, tvm_ffi.Tensor)
                else tvm_ffi.from_dlpack(value, require_contiguous=True)
            )
            if not tensor.is_contiguous():
                raise InvalidReturnValue(f"tensor must be C-contiguous at {path}")
            shape = [int(dimension) for dimension in tensor.shape]
            if any(dimension < 0 for dimension in shape):
                raise InvalidReturnValue(f"tensor has an invalid shape at {path}")
            dtype = str(tensor.dtype)
            data = _tensor_bytes(tensor)
            expected = math.prod(shape) * int(tensor.dtype.itemsize)
            if len(data) != expected:
                raise InvalidReturnValue(
                    f"tensor byte size does not match shape and dtype at {path}"
                )
        except InvalidReturnValue:
            raise
        except Exception as exc:
            raise InvalidReturnValue(
                f"tensor cannot be serialized at {path}: {exc}"
            ) from exc
        part = _store_binary(data, state)
        return {
            "type": "tensor",
            "dtype": dtype,
            "shape": shape,
            "part": part.name,
            "size": part.size,
            "sha256": part.sha256,
        }

    if isinstance(value, (list, tuple, dict)):
        identity = id(value)
        if identity in active:
            raise InvalidReturnValue(f"circular reference at {path}")
        active.add(identity)
        try:
            if isinstance(value, list):
                return {
                    "type": "list",
                    "items": [
                        _encode(item, f"{path}[{index}]", depth + 1, active, state)
                        for index, item in enumerate(value)
                    ],
                }
            if isinstance(value, tuple):
                return {
                    "type": "tuple",
                    "items": [
                        _encode(item, f"{path}[{index}]", depth + 1, active, state)
                        for index, item in enumerate(value)
                    ],
                }
            items: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise InvalidReturnValue(
                        f"dictionary key is not a string at {path}"
                    )
                child_path = (
                    f"{path}.{key}" if key.isidentifier() else f"{path}[{key!r}]"
                )
                items[key] = _encode(item, child_path, depth + 1, active, state)
            return {"type": "dict", "items": items}
        finally:
            active.remove(identity)

    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidReturnValue(f"non-finite float at {path}")
    raise InvalidReturnValue(f"unsupported value at {path}")


def _is_json_subtree(
    value: Any, path: str, depth: int, active: set[int], maximum_depth: int
) -> bool:
    if depth > maximum_depth:
        raise InvalidReturnValue(f"maximum nesting depth exceeded at {path}")
    if (
        value is None
        or isinstance(value, (bool, str))
        or (isinstance(value, int) and not isinstance(value, bool))
    ):
        return True
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidReturnValue(f"non-finite float at {path}")
        return True
    if isinstance(value, tuple):
        return False
    if isinstance(value, list):
        identity = id(value)
        if identity in active:
            raise InvalidReturnValue(f"circular reference at {path}")
        active.add(identity)
        try:
            return all(
                _is_json_subtree(
                    item, f"{path}[{index}]", depth + 1, active, maximum_depth
                )
                for index, item in enumerate(value)
            )
        finally:
            active.remove(identity)
    if isinstance(value, dict):
        identity = id(value)
        if identity in active:
            raise InvalidReturnValue(f"circular reference at {path}")
        active.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise InvalidReturnValue(
                        f"dictionary key is not a string at {path}"
                    )
                child_path = (
                    f"{path}.{key}" if key.isidentifier() else f"{path}[{key!r}]"
                )
                if not _is_json_subtree(
                    item, child_path, depth + 1, active, maximum_depth
                ):
                    return False
            return True
        finally:
            active.remove(identity)
    return False


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "__dlpack__") and hasattr(value, "__dlpack_device__")


def _tensor_bytes(tensor: Any) -> bytes:
    try:
        import torch

        torch_tensor = torch.from_dlpack(tensor)
        if not torch_tensor.is_contiguous():
            raise InvalidReturnValue("tensor must be C-contiguous")
        if torch_tensor.device.type != "cpu":
            torch_tensor = torch_tensor.cpu()
        return torch_tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    except ImportError:
        pass

    if str(tensor.device).split(":", 1)[0] != "cpu":
        raise InvalidReturnValue(
            "serializing a GPU tensor requires PyTorch in the server environment"
        )
    size = math.prod(int(dimension) for dimension in tensor.shape) * int(
        tensor.dtype.itemsize
    )
    return cpu_tensor_bytes(tensor, size)


def _store_binary(data: bytes, state: _State) -> BinaryPart:
    size = len(data)
    if size > state.config.max_binary_value_bytes:
        raise InvalidReturnValue(
            "an individual binary return value exceeds the configured limit"
        )
    digest = hashlib.sha256(data).hexdigest()
    key = (digest, size)
    existing = state.by_content.get(key)
    if existing is not None:
        return existing
    name = f"return:{len(state.parts)}"
    path = state.output_dir / f"part-{len(state.parts)}.bin"
    path.write_bytes(data)
    part = BinaryPart(name, path, size, digest)
    state.parts.append(part)
    state.by_content[key] = part
    state.binary_bytes += size
    if state.binary_bytes > state.config.max_response_bytes:
        raise InvalidReturnValue(
            "binary return values exceed the configured response-size limit"
        )
    return part
