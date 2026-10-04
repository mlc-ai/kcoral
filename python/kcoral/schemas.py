"""Validated instruction and outcome types for the execution protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DTYPE_ITEM_SIZES: dict[str, int] = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}


@dataclass
class Upload:
    id: str
    kind: Literal["module", "tensor", "bytes", "library"]
    source: str | None = None
    blob: str | None = None
    dtype: str | None = None
    shape: list[int] | None = None
    op: Literal["upload"] = "upload"


@dataclass
class FileUpload:
    """A regular file whose handle binds its relative request-workspace path."""

    id: str
    blob: str
    path: str
    kind: Literal["file"] = "file"
    op: Literal["upload"] = "upload"


@dataclass(frozen=True)
class Ref:
    """A validated reference to an earlier handle; wire form is ``{"$ref": id}``."""

    id: str


@dataclass
class Run:
    id: str
    fn: Ref
    args: list[Any] = field(default_factory=list)  # ``Ref`` or a JSON literal
    op: Literal["run"] = "run"


@dataclass
class GetFunction:
    id: str
    module: Ref
    name: str
    cpu_only: bool = False
    op: Literal["get_function"] = "get_function"


@dataclass
class Return:
    key: str
    value: Ref
    op: Literal["return"] = "return"


@dataclass
class FileReturn:
    key: str
    kind: Literal["file", "folder"]
    path: str | Ref
    op: Literal["return"] = "return"


Instruction = Upload | FileUpload | GetFunction | Run | Return | FileReturn


@dataclass
class Program:
    instructions: list[Instruction]
    options: dict[str, Any] = field(default_factory=dict)
    # Filled by the HTTP front-end after multipart validation and cache lookup.
    blob_bytes: dict[str, bytes] = field(default_factory=dict)

    # Trusted limits supplied by the front-end; never accepted in wire options.
    max_return_bytes: int = 256 * 1024**2

    def blob_uploads(self) -> list[Upload | FileUpload]:
        """Uploads whose payload comes from the content-addressed blob cache."""
        return [
            instruction
            for instruction in self.instructions
            if isinstance(instruction, FileUpload)
            or (isinstance(instruction, Upload) and instruction.blob is not None)
        ]


@dataclass
class ProgramOutcome:
    """What running one program produced, before the front-end adds request metadata."""

    status: str
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    binary_parts: dict[str, bytes] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
