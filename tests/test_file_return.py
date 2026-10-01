"""Core file-return behavior and protection against unsafe reads or local data loss."""

import os

import pytest

from kcoral import Program, Register, ReturnedFile, ReturnedFolder
from kcoral.client import _decode_value
from kcoral.engine import execute
from kcoral.keys import compute_blob_hash
from kcoral.schemas import parse_program
from kcoral.testing import UNSHARED_GPU, FakeRuntime


def run(program, workspace, *, max_bytes=1024):
    parsed = parse_program({"instructions": program.instructions})
    parsed.blob_bytes = program._blobs
    parsed.max_return_bytes = max_bytes
    return execute(parsed, FakeRuntime(), UNSHARED_GPU, workspace_dir=str(workspace))


def values(outcome):
    return _decode_value({"type": "object", "value": outcome.results}, outcome.binary_parts, set())


def test_selector_validation_leaves_program_unchanged():
    program = Program()
    program.return_file(key="file", path="out/a")
    before = program.instructions
    for path in ("../escape", Register("missing")):
        with pytest.raises(ValueError):
            program.return_folder(key="bad", path=path)
        assert program.instructions == before
    with pytest.raises(ValueError, match="duplicate"):
        program.return_folder(key="file", path="out")


def test_returns_snapshot_contents_and_survive_later_failure(tmp_path):
    program = Program()
    module = program.upload(
        id="module",
        kind="module",
        source="""
from pathlib import Path
def write(data):
    Path("out").mkdir(exist_ok=True)
    Path("out/a").write_text(data)
    return "out/a"
""",
    )
    writer = program.get_function(id="writer", module=module, name="write", cpu_only=True)
    path = program.run(id="path", fn=writer, args=["first"])
    program.return_file(key="first", path=path)
    program.return_folder(key="tree", path="out")
    program.run(id="change", fn=writer, args=["second"])
    program.return_file(key="second", path="out/a")
    program.return_file(key="missing", path="missing")
    outcome = run(program, tmp_path)
    assert outcome.error["kind"] == "serialization"
    assert values(outcome) == {
        "first": ReturnedFile(b"first"),
        "tree": ReturnedFolder({"a": ReturnedFile(b"first")}),
        "second": ReturnedFile(b"second"),
    }


def test_file_registers_bind_normalized_relative_paths(tmp_path):
    program = Program()
    first = program.upload_file(blob=b"original\x00\xff", path="./inputs//first")
    second = program.upload_file(blob=b"original\x00\xff", path="inputs/second")
    module = program.upload(
        kind="module",
        source="""
import os
from pathlib import Path

def edit(first, second):
    assert first == "inputs/first" and second == "inputs/second"
    assert Path(first).read_bytes() == Path(second).read_bytes()
    Path(first).write_bytes(b"edited")
    os.chdir("inputs")
    return str(Path(first).parent)
""",
    )
    edit = program.get_function(module=module, name="edit", cpu_only=True)
    directory = program.run(fn=edit, args=[first, second])
    # File returns remain relative to the original workspace after chdir.
    program.return_(key="path", value=first)
    program.return_file(key="first", path=first)
    program.return_file(key="second", path=second)
    program.return_folder(key="folder", path=directory)
    outcome = run(program, tmp_path)
    assert outcome.status == "COMPLETED", outcome.error
    assert values(outcome) == {
        "path": "inputs/first",
        "first": ReturnedFile(b"edited"),
        "second": ReturnedFile(b"original\x00\xff"),
        "folder": ReturnedFolder(
            {
                "first": ReturnedFile(b"edited"),
                "second": ReturnedFile(b"original\x00\xff"),
            }
        ),
    }
    assert program._blobs == {compute_blob_hash(b"original\x00\xff"): b"original\x00\xff"}


@pytest.mark.parametrize("kind", ["file", "folder"])
def test_return_rejects_absolute_path_registers_even_inside_workspace(tmp_path, kind):
    target = tmp_path / "target"
    target.mkdir()
    (target / "data").write_bytes(b"data")
    path = str(target / "data" if kind == "file" else target)
    program = Program()
    module = program.upload(kind="module", source=f"def path(): return {path!r}")
    fn = program.get_function(module=module, name="path")
    result = program.run(fn=fn)
    getattr(program, f"return_{kind}")(key="bad", path=result)
    outcome = run(program, tmp_path)
    assert outcome.status == "FAILED"
    assert outcome.error["kind"] == "serialization"
    assert "relative" in outcome.error["message"]
    assert outcome.results == {} and outcome.binary_parts == {}


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_folder_return_rejects_unsafe_entries(tmp_path, kind):
    folder = tmp_path / "out"
    folder.mkdir()
    (folder / "a").write_bytes(b"ok")
    if kind == "symlink":
        (folder / "bad").symlink_to(folder / "a")
    else:
        os.mkfifo(folder / "bad")
    program = Program()
    program.return_file(key="kept", path="out/a")
    program.return_folder(key="bad", path="out")
    outcome = run(program, tmp_path)
    assert outcome.error["kind"] == "serialization"
    assert values(outcome) == {"kept": ReturnedFile(b"ok")}
    assert outcome.binary_parts == {"return:0": b"ok"}


def test_byte_limit_counts_all_returns_and_rolls_back_failed_collection(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out/a").write_bytes(b"1234")
    (tmp_path / "out/b").write_bytes(b"5")
    program = Program()
    data = program.upload(id="bytes", kind="bytes", value=b"12")
    program.return_(key="bytes", value=data)
    program.return_file(key="kept", path="out/a")
    program.return_folder(key="tree", path="out")
    outcome = run(program, tmp_path, max_bytes=10)
    assert outcome.error["kind"] == "serialization"
    assert values(outcome) == {"bytes": b"12", "kept": ReturnedFile(b"1234")}
    assert outcome.binary_parts == {"return:0": b"12", "return:1": b"1234"}
    assert run(program, tmp_path, max_bytes=11).status == "COMPLETED"


def test_file_save_requires_explicit_overwrite(tmp_path):
    destination = tmp_path / "file"
    ReturnedFile(b"original").save(destination)
    with pytest.raises(FileExistsError):
        ReturnedFile(b"new").save(destination)
    assert destination.read_bytes() == b"original"
    ReturnedFile(b"new").save(destination, overwrite=True)
    assert destination.read_bytes() == b"new"


def test_folder_save_preserves_structure_and_refuses_existing_destination(tmp_path):
    folder = ReturnedFolder(
        {".hidden": ReturnedFile(b""), "nested/a": ReturnedFile(b"a")},
        ("empty", "nested"),
    )
    destination = tmp_path / "tree"
    folder.save(destination)
    assert (destination / "nested/a").read_bytes() == b"a"
    assert (destination / ".hidden").read_bytes() == b""
    assert (destination / "empty").is_dir()
    with pytest.raises(FileExistsError):
        folder.save(destination)
    empty = tmp_path / "existing"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        folder.save(empty)
    assert list(empty.iterdir()) == []


@pytest.mark.parametrize("folder", [False, True])
def test_save_rejects_symlink_parents(tmp_path, folder):
    (tmp_path / "actual").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "actual", target_is_directory=True)
    result = ReturnedFolder({}) if folder else ReturnedFile(b"data")
    with pytest.raises(OSError):
        result.save(tmp_path / "link/output")
    assert list((tmp_path / "actual").iterdir()) == []


@pytest.mark.parametrize("folder", [False, True])
def test_failed_save_cleans_output_and_preserves_existing_files(tmp_path, monkeypatch, folder):
    destination = tmp_path / "out"
    fdopen = os.fdopen

    def failing_write(fd, mode):
        stream = fdopen(fd, mode)
        write = stream.write

        def partial(data):
            write(data[:1])
            raise OSError("disk full")

        stream.write = partial
        return stream

    monkeypatch.setattr("kcoral.artifacts.os.fdopen", failing_write)
    if folder:
        result = ReturnedFolder({"nested/a": ReturnedFile(b"new")}, ("empty", "nested"))
        kwargs = {}
    else:
        result = ReturnedFile(b"new")
        destination.write_bytes(b"original")
        kwargs = {"overwrite": True}
    with pytest.raises(OSError, match="disk full"):
        result.save(destination, **kwargs)
    assert sorted(path.name for path in tmp_path.iterdir()) == ([] if folder else ["out"])
    if not folder:
        assert destination.read_bytes() == b"original"


@pytest.mark.parametrize("path,directories", [("../escape", []), ("a", ["a"])])
def test_decoder_rejects_unsafe_folder_manifests(path, directories):
    file = {"type": "file", "size": 0, "part": "return:0", "sha256": compute_blob_hash(b"")}
    with pytest.raises(ValueError):
        _decode_value(
            {"type": "folder", "files": {path: file}, "directories": directories},
            {"return:0": b""},
            set(),
        )
