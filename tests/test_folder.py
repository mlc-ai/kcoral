import os
import stat
import sys

import pytest
from support.programs import harness_function

from kcoral import Program
from kcoral.keys import compute_blob_hash
from kcoral.schemas import parse_program


def test_folder_expands_to_fixed_file_uploads_and_deduplicates(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "empty").mkdir()
    (tmp_path / "a").write_bytes(b"shared")
    (tmp_path / "sub" / "b").write_bytes(b"shared")
    (tmp_path / ".hidden").write_bytes(b"")
    program = Program()
    program.run(id="before", fn=harness_function(program, "structural", "before"))
    assert program.upload_folder(tmp_path, path="./assets//") is None
    program.run(id="after", fn=harness_function(program, "structural", "after"))
    original = program.instructions
    items = [
        item for item in program.instructions if item["op"] in {"run"} or item.get("kind") == "file"
    ]
    assert items[0]["id"] == "before" and items[-1]["id"] == "after"
    assert [item["path"] for item in items[1:-1]] == ["assets/.hidden", "assets/a", "assets/sub/b"]
    assert all(set(item) == {"op", "id", "kind", "blob", "path"} for item in items[1:-1])
    assert all(item["op"] == "upload" and item["kind"] == "file" for item in items[1:-1])
    assert [item["id"] for item in items[1:-1]] == ["upload_0", "upload_1", "upload_2"]
    assert program._blobs == {compute_blob_hash(b"shared"): b"shared", compute_blob_hash(b""): b""}
    parse_program({"instructions": program.instructions})
    (tmp_path / "a").write_bytes(b"changed")
    assert program.instructions == original
    assert program._blobs[compute_blob_hash(b"shared")] == b"shared"


def test_folder_generated_ids_skip_existing_handles_and_are_referenceable(tmp_path):
    (tmp_path / "a").write_bytes(b"a")
    program = Program()
    program.upload(kind="bytes", id="upload_0", value=b"reserved")
    program.upload_folder(tmp_path, path="inputs")
    assert program.instructions[1]["id"] == "upload_1"
    program.return_(key="path", value={"$ref": "upload_1"})
    parse_program({"instructions": program.instructions})


def test_empty_folder_is_a_noop(tmp_path):
    program = Program()
    program.upload_folder(tmp_path, path="empty")
    assert program.instructions == [] and program._blobs == {}


@pytest.mark.parametrize("path", ["/absolute", "a/../b"])
def test_folder_rejects_invalid_destinations(tmp_path, path):
    with pytest.raises(ValueError):
        Program().upload_folder(tmp_path, path=path)


@pytest.mark.parametrize("kind", ["loop", "file_link", "root_link", "fifo"])
def test_folder_rejects_links_and_special_files_without_partial_changes(tmp_path, kind):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a").write_bytes(b"ordinary")
    if kind == "loop":
        (source / "loop").symlink_to(source, target_is_directory=True)
    elif kind == "file_link":
        (source / "link").symlink_to(source / "a")
    elif kind == "root_link":
        link = tmp_path / "link"
        link.symlink_to(source, target_is_directory=True)
        source = link
    else:
        os.mkfifo(source / "fifo")
    program = Program()
    program.upload_file(blob=b"kept", path="kept")
    before = (program.instructions, program._blobs.copy(), program._file_paths.copy())
    with pytest.raises(ValueError, match=r"symbolic links|special files"):
        program.upload_folder(source, path="data")
    assert (program.instructions, program._blobs, program._file_paths) == before


def test_repeated_directory_identity_is_rejected(tmp_path, monkeypatch):
    (tmp_path / "child").mkdir()
    root_info = tmp_path.stat()
    fstat = os.fstat

    def duplicate_directory(fd):
        info = fstat(fd)
        # Simulate a bind mount revisiting an ancestor without requiring mount privileges.
        return root_info if stat.S_ISDIR(info.st_mode) else info

    monkeypatch.setattr("kcoral.client.os.fstat", duplicate_directory)
    with pytest.raises(ValueError, match="repeated directory"):
        Program().upload_folder(tmp_path, path="data")


def test_deep_folder_does_not_use_the_python_call_stack(tmp_path):
    directory = tmp_path
    for _ in range(100):
        directory = directory / "d"
        directory.mkdir()
    (directory / "file").write_bytes(b"deep")
    program = Program()
    original_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(80)
        program.upload_folder(tmp_path, path="data")
    finally:
        sys.setrecursionlimit(original_limit)
    assert len(program.instructions) == 1
    assert program._blobs == {compute_blob_hash(b"deep"): b"deep"}


@pytest.mark.parametrize("existing", ["assets", "assets/a", "assets/a/child"])
def test_folder_conflicts_roll_back_the_whole_call(tmp_path, existing):
    (tmp_path / "a").write_bytes(b"new")
    (tmp_path / "b").write_bytes(b"new")
    program = Program()
    program.upload_file(blob=b"original", path=existing)
    before = (program.instructions, program._blobs.copy(), program._file_paths.copy())
    with pytest.raises(ValueError, match=r"duplicate|conflicting"):
        program.upload_folder(tmp_path, path="assets")
    assert (program.instructions, program._blobs, program._file_paths) == before


def test_file_upload_after_folder_still_checks_conflicts(tmp_path):
    (tmp_path / "a").write_bytes(b"x")
    program = Program()
    program.upload_folder(tmp_path, path="assets")
    with pytest.raises(ValueError, match="conflicting"):
        program.upload_file(blob=b"y", path="assets")
