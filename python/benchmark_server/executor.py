from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import tvm_ffi

from .models import (
    CallInstruction,
    RegisterReference,
    ReturnInstruction,
    UploadModuleInstruction,
    UploadTensorInstruction,
    ValidatedProgram,
)
from .validation import TENSOR_DTYPE_SIZES


class InvalidProgramError(ValueError):
    def __init__(self, message: str, instruction_index: int) -> None:
        super().__init__(message)
        self.instruction_index = instruction_index


class InstructionExecutionError(RuntimeError):
    def __init__(self, message: str, instruction_index: int) -> None:
        super().__init__(message)
        self.instruction_index = instruction_index


@dataclass
class ExecutionContext:
    blob_paths: Mapping[str, Path]
    work_dir: Path
    request_id: str
    registers: dict[int, Any] = field(default_factory=dict)
    loaded_modules: list[ModuleType] = field(default_factory=list)
    returned_values: dict[str, Any] = field(default_factory=dict)

    def get_blob_path(self, digest: str, instruction_index: int) -> Path:
        path = self.blob_paths.get(digest)
        if path is None or not path.is_file():
            raise InvalidProgramError(
                f"blob {digest} is unavailable", instruction_index
            )
        return path

    def get_register(self, register: int, instruction_index: int) -> Any:
        if register not in self.registers:
            raise InvalidProgramError(
                f"register r{register} is undefined", instruction_index
            )
        return self.registers[register]


def execute_program(
    program: ValidatedProgram,
    blob_paths: Mapping[str, Path],
    work_dir: Path,
    request_id: str,
) -> dict[str, Any]:
    context = ExecutionContext(blob_paths, work_dir, request_id)
    for instruction_index, instruction in enumerate(program.instructions):
        if isinstance(instruction, UploadModuleInstruction):
            _upload_module(context, instruction, instruction_index)
        elif isinstance(instruction, UploadTensorInstruction):
            _upload_tensor(context, instruction, instruction_index)
        elif isinstance(instruction, CallInstruction):
            _call(context, instruction, instruction_index)
        elif isinstance(instruction, ReturnInstruction):
            context.returned_values[instruction.key] = context.get_register(
                instruction.register, instruction_index
            )
    return context.returned_values


def _upload_module(
    context: ExecutionContext,
    instruction: UploadModuleInstruction,
    instruction_index: int,
) -> None:
    source = context.get_blob_path(instruction.blob, instruction_index)
    module_dir = context.work_dir / ".benchmark-server" / "modules"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / f"{instruction_index}-{instruction.blob}.py"
    module_path.write_bytes(source.read_bytes())
    module_name = (
        f"_benchmark_server_instruction_{context.request_id.replace('-', '_')}"
        f"_{instruction_index}"
    )
    specification = importlib.util.spec_from_file_location(module_name, module_path)
    if specification is None or specification.loader is None:
        raise InvalidProgramError(
            f"cannot import module blob {instruction.blob}", instruction_index
        )
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException as error:
        sys.modules.pop(module_name, None)
        raise InstructionExecutionError(str(error), instruction_index) from error
    context.loaded_modules.append(module)


def _upload_tensor(
    context: ExecutionContext,
    instruction: UploadTensorInstruction,
    instruction_index: int,
) -> None:
    blob_path = context.get_blob_path(instruction.blob, instruction_index)
    expected_size = math.prod(instruction.shape) * TENSOR_DTYPE_SIZES[instruction.dtype]
    actual_size = blob_path.stat().st_size
    if actual_size != expected_size:
        raise InvalidProgramError(
            "upload_tensor blob size mismatch: "
            f"expected {expected_size} bytes, got {actual_size}",
            instruction_index,
        )
    try:
        import torch
    except ImportError as error:
        raise InstructionExecutionError(
            "upload_tensor requires PyTorch", instruction_index
        ) from error

    try:
        host_tensor = torch.from_dlpack(
            tvm_ffi.frombuffer(blob_path.read_bytes(), instruction.dtype)
        ).reshape(instruction.shape)
        tensor = (
            host_tensor
            if instruction.device == "cpu"
            else host_tensor.to(instruction.device)
        )
    except BaseException as error:
        raise InstructionExecutionError(str(error), instruction_index) from error
    context.registers[instruction.destination] = tensor


def _call(
    context: ExecutionContext,
    instruction: CallInstruction,
    instruction_index: int,
) -> None:
    function = tvm_ffi.get_global_func(instruction.function, allow_missing=True)
    if function is None:
        raise InvalidProgramError(
            f"function {instruction.function!r} is not registered",
            instruction_index,
        )
    arguments = [
        _resolve_operand(context, argument, instruction_index)
        for argument in instruction.arguments
    ]
    try:
        result = function(*arguments)
    except BaseException as error:
        raise InstructionExecutionError(str(error), instruction_index) from error
    if instruction.destination is not None:
        context.registers[instruction.destination] = result


def _resolve_operand(
    context: ExecutionContext, operand: Any, instruction_index: int
) -> Any:
    if isinstance(operand, RegisterReference):
        return context.get_register(operand.index, instruction_index)
    if isinstance(operand, list):
        return [
            _resolve_operand(context, item, instruction_index) for item in operand
        ]
    return operand
