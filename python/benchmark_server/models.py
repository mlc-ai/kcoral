from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class Entry:
    file: str = "main.py"
    function: str = "main"


@dataclass(frozen=True)
class RegisterReference:
    index: int


@dataclass(frozen=True)
class UploadModuleInstruction:
    blob: str


@dataclass(frozen=True)
class UploadTensorInstruction:
    destination: int
    blob: str
    shape: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True)
class CallInstruction:
    destination: int | None
    function: str
    arguments: tuple[Any, ...]


@dataclass(frozen=True)
class ReturnInstruction:
    register: int
    key: str


Instruction = (
    UploadModuleInstruction
    | UploadTensorInstruction
    | CallInstruction
    | ReturnInstruction
)


@dataclass(frozen=True)
class PreparedFiles:
    manifest: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


@dataclass(frozen=True)
class ExecutionWarning:
    code: str
    blobs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionResult:
    request_id: str
    value: Any
    elapsed_ms: float
    queue_ms: float
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    warnings: tuple[ExecutionWarning, ...] = ()


@dataclass(frozen=True)
class WorkerHealth:
    gpu_id: int
    status: str
    uptime_seconds: float


@dataclass(frozen=True)
class Health:
    status: str
    gpu_count: int
    queue_length: int
    workers: tuple[WorkerHealth, ...]


FileContent = str | bytes | bytearray | memoryview | Path


class BenchmarkServerError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        request_id: str | None = None,
        stdout: str = "",
        stderr: str = "",
        traceback: str | None = None,
        missing_blobs: tuple[str, ...] = (),
        instruction_index: int | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id
        self.stdout = stdout
        self.stderr = stderr
        self.traceback = traceback
        self.missing_blobs = missing_blobs
        self.instruction_index = instruction_index


class TransportError(Exception):
    pass


class ProtocolError(Exception):
    pass


@dataclass(frozen=True)
class ServerConfig:
    devices: tuple[str, ...] = ("0",)
    cache_dir: Path = Path("cache")
    cache_capacity_bytes: int = 10 * 1024**3
    work_dir: Path = Path("work")
    log_dir: Path = Path("logs")
    default_timeout_seconds: float = 60
    max_timeout_seconds: float = 3600
    default_stdout_limit_bytes: int = 1024**2
    default_stderr_limit_bytes: int = 1024**2
    max_stdout_limit_bytes: int = 16 * 1024**2
    max_stderr_limit_bytes: int = 16 * 1024**2
    worker_termination_grace_seconds: float = 5
    max_nesting_depth: int = 64
    max_description_nodes: int = 100_000
    max_json_metadata_bytes: int = 16 * 1024**2
    max_binary_value_bytes: int = 1024**3
    max_response_bytes: int = 2 * 1024**3

    def validated(self) -> ServerConfig:
        if not self.devices or any(not str(device).strip() for device in self.devices):
            raise ValueError("at least one non-empty GPU device identifier is required")
        positive = {
            "cache_capacity_bytes": self.cache_capacity_bytes,
            "default_timeout_seconds": self.default_timeout_seconds,
            "max_timeout_seconds": self.max_timeout_seconds,
            "worker_termination_grace_seconds": self.worker_termination_grace_seconds,
            "max_nesting_depth": self.max_nesting_depth,
            "max_description_nodes": self.max_description_nodes,
            "max_json_metadata_bytes": self.max_json_metadata_bytes,
            "max_binary_value_bytes": self.max_binary_value_bytes,
            "max_response_bytes": self.max_response_bytes,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        non_negative = {
            "default_stdout_limit_bytes": self.default_stdout_limit_bytes,
            "default_stderr_limit_bytes": self.default_stderr_limit_bytes,
            "max_stdout_limit_bytes": self.max_stdout_limit_bytes,
            "max_stderr_limit_bytes": self.max_stderr_limit_bytes,
        }
        for name, value in non_negative.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.default_timeout_seconds > self.max_timeout_seconds:
            raise ValueError("default timeout cannot exceed maximum timeout")
        if self.default_stdout_limit_bytes > self.max_stdout_limit_bytes:
            raise ValueError("default stdout limit cannot exceed maximum")
        if self.default_stderr_limit_bytes > self.max_stderr_limit_bytes:
            raise ValueError("default stderr limit cannot exceed maximum")
        return self


@dataclass(frozen=True)
class ValidatedJob:
    language: str
    entry: Entry
    files: Mapping[str, str]
    timeout_seconds: float
    stdout_limit_bytes: int
    stderr_limit_bytes: int


@dataclass(frozen=True)
class ValidatedProgram:
    instructions: tuple[Instruction, ...]
    blob_digests: frozenset[str]
    timeout_seconds: float
    stdout_limit_bytes: int
    stderr_limit_bytes: int


ValidatedRequest = ValidatedJob | ValidatedProgram


@dataclass
class BinaryPart:
    name: str
    path: Path
    size: int
    sha256: str


@dataclass
class WorkerOutcome:
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)
    binaries: list[BinaryPart] = field(default_factory=list)
