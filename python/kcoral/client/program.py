"""Build self-contained programs and snapshot their uploads."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kcoral.artifacts import _folder_files
from kcoral.protocol import (
    compute_blob_hash,
    expected_tensor_nbytes,
    normalize_file_path,
    validate_and_add_file_path,
    validate_and_add_file_paths,
)
from kcoral.schemas import DTYPE_ITEM_SIZES


@dataclass(frozen=True)
class Register:
    """Reference to a value produced by an earlier instruction in one request.

    :param id: The producing instruction's unique identifier.

    Pass the register to later instructions in the same :class:`Program`.
    Registers do not refer to persistent server-side objects.
    """

    id: str
    """The unique identifier of the instruction that produced this value."""


@dataclass
class Program:
    """Build an ordered, self-contained sequence of remote instructions.

    Construction does not contact a server or compile a kernel. Uploads
    snapshot local input, and :meth:`Client.execute` sends the instructions
    and required binary data. Each instruction identifier and return key must
    be unique within this program. Only outputs selected by :meth:`return_`,
    :meth:`return_file` or :meth:`return_folder` appear in the response.

    Each value-producing instruction automatically receives an ID such as
    ``upload_0``, ``get_function_1``, or ``run_2``. Use the optional ``id`` field
    to customize it.
    """

    _instructions: list[dict[str, Any]] = field(default_factory=list, init=False)
    _blobs: dict[str, bytes] = field(default_factory=dict, init=False)
    _ids: set[str] = field(default_factory=set, init=False)
    _next_id: int = field(default=0, init=False)
    _return_keys: set[str] = field(default_factory=set, init=False)
    _file_paths: set[str] = field(default_factory=set, init=False)

    @property
    def instructions(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the ordered wire instruction list.

        The contained dictionaries are shared with this program. Treat them
        as read-only; use the builder methods to add instructions.
        """
        return list(self._instructions)

    def upload_file(self, *, blob: Any, path: str, id: str | None = None) -> Register:
        """Snapshot bytes-like data as a file in the request workspace.

        The destination must be a relative POSIX path without ``..`` components
        and cannot conflict with another file upload.

        :param blob: Bytes-like content, copied when this method is called.
        :param path: Destination relative to the request's working directory.
        :param id: Optional custom instruction identifier; generated when omitted.
        :returns: A register containing the normalized relative path string.
        :raises TypeError: If the content is not bytes-like.
        :raises ValueError: If the destination is invalid or conflicts with a file.

        Add this instruction before any code that reads the destination.
        Files are removed when execution ends; cached content may persist.
        """
        normalized_path = normalize_file_path(path)
        try:
            raw = blob if isinstance(blob, bytes) else bytes(memoryview(blob))
        except TypeError as exc:
            raise TypeError("file upload requires a bytes-like 'blob'") from exc
        paths = self._file_paths.copy()
        validate_and_add_file_path(normalized_path, paths)
        blob_hash = compute_blob_hash(raw)
        id = self._add_id(id, op="upload")
        self._file_paths = paths
        self._blobs.setdefault(blob_hash, raw)
        self._instructions.append(
            {"op": "upload", "id": id, "kind": "file", "blob": blob_hash, "path": normalized_path}
        )
        return Register(id)

    def upload_folder(self, folder: str | os.PathLike[str], *, path: str) -> None:
        """Snapshot a directory as ordinary file uploads at this program position.

        Includes hidden files; empty directories and original file metadata are
        not uploaded. Symbolic links, special files, and repeated directories
        are rejected. Failed calls leave the program unchanged.

        :param folder: Local directory whose contents should be uploaded.
        :param path: Relative destination directory in the request workspace.
        :raises ValueError: If traversal or destination validation fails.
        :raises OSError: If the local directory cannot be read.

        For example, uploading ``assets`` with ``path="inputs"`` maps
        ``assets/a.bin`` to ``inputs/a.bin``. An empty folder adds no instructions.
        """
        destination = normalize_file_path(path)
        instructions = []
        blobs = {}
        for remote, data in _folder_files(folder, destination):
            digest = compute_blob_hash(data)
            blobs.setdefault(digest, data)
            instructions.append({"op": "upload", "kind": "file", "blob": digest, "path": remote})
        # One batch validation avoids a quadratic scan for folders with many
        # files and commits no state until traversal and validation both succeed.
        validate_and_add_file_paths([item["path"] for item in instructions], self._file_paths)
        for instruction in sorted(instructions, key=lambda item: item["path"]):
            instruction["id"] = self._add_id(None, op="upload")
            self._instructions.append(instruction)
        for digest, data in blobs.items():
            self._blobs.setdefault(digest, data)

    def upload(
        self,
        *,
        id: str | None = None,
        kind: str,
        source: str | None = None,
        value: Any = None,
        dtype: str | None = None,
        shape: list[int] | None = None,
    ) -> Register:
        """Upload source or binary content and return its request-local register.

        :param kind: One of ``module``, ``tensor``, ``bytes`` or ``library``.
        :param source: Python source text for a module upload.
        :param value: Tensor input or bytes-like data for a binary upload.
            Tensors accept NumPy arrays, PyTorch tensors, objects implementing
            the DLPack tensor exchange protocol, or raw bytes.
        :param dtype: Element type for a tensor supplied as raw bytes.
        :param shape: Dimensions for a tensor supplied as raw bytes.
        :param id: Optional custom instruction identifier. Must be nonempty and unique.
        :returns: A register usable by later instructions.
        :raises TypeError: If the input does not match the upload kind.
        :raises ValueError: If the kind, identifier or tensor metadata is invalid.

        A library is a precompiled TVM FFI module packaged as a shared library;
        a module executes Python source.
        Use :meth:`upload_file` or :meth:`upload_folder` for filesystem uploads.
        ``kind="file"`` is a wire-protocol option, not accepted by this method.
        """
        if kind == "module":
            if not isinstance(source, str):
                raise TypeError("module upload requires string 'source'")
            if value is not None or dtype is not None or shape is not None:
                raise TypeError("module upload does not accept tensor fields")
            instruction = {"op": "upload", "kind": "module", "source": source}
        elif kind == "tensor":
            if source is not None:
                raise TypeError("tensor upload does not accept 'source'")
            tensor_dtype, tensor_shape, raw = _tensor_fields(value, dtype=dtype, shape=shape)
            blob_hash = compute_blob_hash(raw)
            self._blobs.setdefault(blob_hash, raw)
            instruction = {
                "op": "upload",
                "kind": "tensor",
                "blob": blob_hash,
                "dtype": tensor_dtype,
                "shape": tensor_shape,
            }
        elif kind == "bytes":
            if source is not None:
                raise TypeError("bytes upload does not accept 'source'")
            if dtype is not None or shape is not None:
                raise TypeError("bytes upload does not accept tensor fields")
            try:
                raw = value if isinstance(value, bytes) else bytes(memoryview(value))
            except TypeError as exc:
                raise TypeError("bytes upload requires a bytes-like 'value'") from exc
            blob_hash = compute_blob_hash(raw)
            self._blobs.setdefault(blob_hash, raw)
            instruction = {
                "op": "upload",
                "kind": "bytes",
                "blob": blob_hash,
            }
        elif kind == "library":
            if source is not None:
                raise TypeError("library upload does not accept 'source'")
            if dtype is not None or shape is not None:
                raise TypeError("library upload does not accept tensor fields")
            raw = value if isinstance(value, bytes) else bytes(memoryview(value))
            blob_hash = compute_blob_hash(raw)
            self._blobs.setdefault(blob_hash, raw)
            instruction = {
                "op": "upload",
                "kind": "library",
                "blob": blob_hash,
            }
        else:
            raise ValueError("upload kind must be 'module', 'tensor', 'bytes', or 'library'")
        id = self._add_id(id, op="upload")
        instruction["id"] = id
        self._instructions.append(instruction)
        return Register(id)

    def get_function(
        self,
        *,
        id: str | None = None,
        module: Register | dict[str, str],
        name: str,
        cpu_only: bool = False,
    ) -> Register:
        """Select a named function from an earlier module or library upload.

        :param module: An earlier upload register or ``{"$ref": "id"}`` reference.
        :param name: Nonempty function name exported by the module or library.
        :param cpu_only: Declare that the function does not access the GPU.
        :param id: Optional custom instruction identifier. Must be nonempty and unique.
        :returns: A function register for a later :meth:`run` instruction.
        :raises TypeError: If the reference or flag has an invalid type.
        :raises ValueError: If an identifier, reference or function name is invalid.

        Running a ``cpu_only`` function as its own instruction releases the
        exclusive GPU lease while it executes. This declaration is checked
        on a best-effort basis; it does not make GPU operations safe to call.
        """
        reference = _reference(module) if isinstance(module, Register) else module
        if not (
            isinstance(reference, dict)
            and set(reference) == {"$ref"}
            and isinstance(reference["$ref"], str)
        ):
            raise TypeError("module must be a Register or {'$ref': id}")
        if reference["$ref"] not in self._ids:
            raise ValueError(f"get_function {id!r} references unknown handle {reference['$ref']!r}")
        if not isinstance(name, str) or not name:
            raise ValueError("function name must be a non-empty string")
        if not isinstance(cpu_only, bool):
            raise TypeError("cpu_only must be a bool")
        id = self._add_id(id, op="get_function")
        instruction = {"op": "get_function", "id": id, "module": reference, "name": name}
        if cpu_only:
            instruction["cpu_only"] = True
        self._instructions.append(instruction)
        return Register(id)

    def run(
        self, *, id: str | None = None, fn: Register, args: list[Any] | None = None
    ) -> Register:
        """Append a function call and return a register for its result.

        :param fn: The :class:`Register` returned by :meth:`get_function`,
            or by an earlier :meth:`run` that returned a callable.
        :param args: Positional arguments; omitted or ``None`` means no arguments.
            Top-level registers are encoded automatically. Nested lists and
            dictionaries remain JSON literals, including reference-shaped objects.
        :param id: Optional custom instruction identifier. Must be nonempty and unique.
        :returns: A register, without automatically returning the value to the client.
        :raises TypeError: If ``fn`` is not a :class:`Register`.
        :raises ValueError: If an identifier is invalid or a reference is unknown.
        """
        if not isinstance(fn, Register):
            raise TypeError("fn must be a Register")
        if fn.id not in self._ids:
            raise ValueError(f"run {id!r} references unknown handle {fn.id!r}")
        wire_fn = _reference(fn)
        wire_args = [
            _reference(argument) if isinstance(argument, Register) else argument
            for argument in (args or [])
        ]
        id = self._add_id(id, op="run")
        self._instructions.append({"op": "run", "id": id, "fn": wire_fn, "args": wire_args})
        return Register(id)

    def return_file(self, *, key: str, path: str | Register) -> None:
        """Select a regular file at this instruction, relative to the request workspace."""
        self._return_path(key=key, path=path, kind="file")

    def return_folder(self, *, key: str, path: str | Register) -> None:
        """Select a complete folder, including hidden files and empty directories."""
        self._return_path(key=key, path=path, kind="folder")

    def _return_path(self, *, key: str, path: str | Register, kind: str) -> None:
        self._check_return_key(key)
        if isinstance(path, Register):
            if path.id not in self._ids:
                raise ValueError(f"return {key!r} references unknown handle {path.id!r}")
            wire_path = _reference(path)
        else:
            wire_path = normalize_file_path(path)
        self._instructions.append({"op": "return", "key": key, "kind": kind, "path": wire_path})
        self._return_keys.add(key)

    def _check_return_key(self, key: str) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("return key must be a non-empty string")
        if key in self._return_keys:
            raise ValueError(f"duplicate return key: {key!r}")

    def return_(self, *, key: str, value: Register | dict[str, str]) -> None:
        """Select an earlier value for the response's results mapping.

        :param key: Unique, nonempty response key.
        :param value: An earlier register or ``{"$ref": "id"}`` reference.
        :raises TypeError: If the value is not a valid reference.
        :raises ValueError: If the key or referenced identifier is invalid.

        A return instruction that runs before a later failure preserves its
        entry in the partial result. Only serializable values can be returned;
        compiled modules and function handles cannot be sent back.
        """
        self._check_return_key(key)
        reference = _reference(value) if isinstance(value, Register) else value
        if not (
            isinstance(reference, dict)
            and set(reference) == {"$ref"}
            and isinstance(reference["$ref"], str)
        ):
            raise TypeError("return value must be a Register or {'$ref': id}")
        if reference["$ref"] not in self._ids:
            raise ValueError(f"return {key!r} references unknown handle {reference['$ref']!r}")
        self._return_keys.add(key)
        self._instructions.append({"op": "return", "key": key, "value": reference})

    def _add_id(self, instruction_id: str | None, *, op: str) -> str:
        if instruction_id is None:
            while (instruction_id := f"{op}_{self._next_id}") in self._ids:
                self._next_id += 1
            self._next_id += 1
        if not isinstance(instruction_id, str) or not instruction_id:
            raise ValueError("instruction id must be a non-empty string")
        if instruction_id in self._ids:
            raise ValueError(f"duplicate instruction id: {instruction_id!r}")
        self._ids.add(instruction_id)
        return instruction_id


def _reference(register: Register) -> dict[str, str]:
    return {"$ref": register.id}


def _tensor_fields(
    value: Any, *, dtype: str | None, shape: list[int] | None
) -> tuple[str, list[int], bytes]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        if dtype is None or shape is None:
            raise TypeError("raw tensor bytes require 'dtype' and 'shape'")
        tensor_dtype, tensor_shape, raw = dtype, list(shape), bytes(value)
    else:
        tensor_dtype, tensor_shape, raw = _array_fields(value)
        if dtype is not None and dtype != tensor_dtype:
            raise ValueError(
                f"declared dtype {dtype!r} does not match value dtype {tensor_dtype!r}"
            )
        if shape is not None and list(shape) != tensor_shape:
            raise ValueError(
                f"declared shape {shape!r} does not match value shape {tensor_shape!r}"
            )
    if tensor_dtype not in DTYPE_ITEM_SIZES:
        raise ValueError(f"unsupported tensor dtype: {tensor_dtype!r}")
    try:
        expected_size = expected_tensor_nbytes(tensor_dtype, tensor_shape)
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    if len(raw) != expected_size:
        raise ValueError(f"tensor metadata expects {expected_size} bytes, got {len(raw)}")
    return tensor_dtype, tensor_shape, raw


def _array_fields(value: Any) -> tuple[str, list[int], bytes]:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            return (
                str(tensor.dtype).removeprefix("torch."),
                [int(dimension) for dimension in tensor.shape],
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
            )
    except ImportError:
        pass

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.byteorder == ">" or (
            array.dtype.byteorder == "=" and sys.byteorder == "big"
        ):
            array = array.astype(array.dtype.newbyteorder("<"))
        return array.dtype.name, [int(dimension) for dimension in array.shape], array.tobytes()
    if hasattr(value, "__dlpack__"):
        try:
            array = np.from_dlpack(value)
        except Exception:
            array = None
        if array is not None:
            return _array_fields(array)

    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        return (
            str(value.dtype),
            [int(dimension) for dimension in value.shape],
            value.tobytes(),
        )
    raise TypeError(f"cannot upload {type(value).__name__!r} as a tensor")
