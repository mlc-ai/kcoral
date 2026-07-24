"""Wire types for the instruction protocol: Program, Instruction, Result.

A request is a `Program` — an ordered list of instructions (`upload` / `run`).
Parsing validates structure up front (unique ids, known ops/kinds, `$ref`
shape); GPU work happens later, in the worker's engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import ValidationError

# Per-instruction result status.
Status = Literal["OK", "FAILED", "SKIPPED"]


@dataclass
class Upload:
    """An upload instruction. ``inline`` is the wire payload — sent only when the
    object may not already be cached under ``key``. The resolved bytes are not
    stored here but in ``Program.upload_bytes`` (keyed by instruction id), so this
    stays the pure wire form."""

    id: str
    kind: str
    key: str
    inline: dict | None = None
    op: str = "upload"


@dataclass
class Run:
    id: str
    fn: Any  # a function name (str) or {"$ref": "<id>"}
    args: list  # each element: {"$ref": "<id>"} (handle) or a literal
    op: str = "run"


Instruction = Any  # Upload | Run


@dataclass
class Program:
    instructions: list
    options: dict = field(default_factory=dict)
    # Resolved canonical bytes for each upload (id -> bytes). Empty on a freshly
    # parsed (wire) program; the front-end fills it from the cache/inline before
    # the program is dispatched to a worker.
    upload_bytes: dict = field(default_factory=dict)

    def uploads(self) -> list:
        return [i for i in self.instructions if i.op == "upload"]


@dataclass
class Result:
    id: str
    op: str
    status: Status
    value: Any = None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    error: dict | None = None


# --- reference / structural helpers ----------------------------------------


def is_ref(x: Any) -> bool:
    """True if ``x`` is a handle reference ``{"$ref": "<id>"}``."""
    return isinstance(x, dict) and set(x.keys()) == {"$ref"} and isinstance(x["$ref"], str)


def is_json_structural(value: Any) -> bool:
    """True if ``value`` is entirely JSON-structural (no opaque objects)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(is_json_structural(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and is_json_structural(v) for k, v in value.items())
    return False


def to_structural(value: Any, instr_id: str) -> Any:
    """Return the JSON-structural form of a result value.

    Structural values pass through; an opaque object (a tensor, a module) is not
    transmitted back — it stays server-side and is referenced by a handle.
    """
    if is_json_structural(value):
        return value
    return {"handle": instr_id}


# --- parsing ---------------------------------------------------------------

_KINDS = {"function", "tensor", "object", "package"}


def parse_program(body: Any) -> Program:
    if not isinstance(body, dict):
        raise ValidationError("request body must be a JSON object")
    raw = body.get("instructions")
    if not isinstance(raw, list) or not raw:
        raise ValidationError("'instructions' must be a non-empty list")
    options = body.get("options") or {}
    if not isinstance(options, dict):
        raise ValidationError("'options' must be an object")

    seen: set[str] = set()
    instructions: list = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValidationError(f"instruction {i} must be an object")
        iid = item.get("id")
        if not isinstance(iid, str) or not iid:
            raise ValidationError(f"instruction {i} needs a string 'id'")
        if iid in seen:
            raise ValidationError(f"duplicate instruction id: {iid!r}")
        seen.add(iid)
        op = item.get("op")
        if op == "upload":
            instructions.append(_parse_upload(item, seen_before=seen))
        elif op == "run":
            instructions.append(_parse_run(item))
        else:
            raise ValidationError(f"instruction {iid!r}: unknown op {op!r}")
    _check_refs(instructions)
    return Program(instructions=instructions, options=options)


def _parse_upload(item: dict, seen_before: set[str]) -> Upload:
    kind = item.get("kind")
    if kind not in _KINDS:
        raise ValidationError(f"upload {item['id']!r}: unknown kind {kind!r}")
    key = item.get("key")
    if not isinstance(key, str) or not key:
        raise ValidationError(f"upload {item['id']!r}: needs a string 'key'")
    inline = item.get("inline")
    if inline is not None and not isinstance(inline, dict):
        raise ValidationError(f"upload {item['id']!r}: 'inline' must be an object")
    return Upload(id=item["id"], kind=kind, key=key, inline=inline)


def _parse_run(item: dict) -> Run:
    fn = item.get("fn")
    if not (isinstance(fn, str) or is_ref(fn)):
        raise ValidationError(f"run {item['id']!r}: 'fn' must be a name or {{'$ref': id}}")
    args = item.get("args", [])
    if not isinstance(args, list):
        raise ValidationError(f"run {item['id']!r}: 'args' must be a list")
    return Run(id=item["id"], fn=fn, args=args)


def _check_refs(instructions: list) -> None:
    """Every ``$ref`` must point to an earlier instruction (straight-line DAG)."""
    produced: set[str] = set()
    for ins in instructions:
        refs: list = []
        if ins.op == "run":
            if is_ref(ins.fn):
                refs.append(ins.fn["$ref"])
            refs += [a["$ref"] for a in ins.args if is_ref(a)]
        for r in refs:
            if r not in produced:
                raise ValidationError(f"{ins.id!r} references unknown/forward handle {r!r}")
        produced.add(ins.id)
