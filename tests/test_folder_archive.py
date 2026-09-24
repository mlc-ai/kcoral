"""Archive validation, extraction boundaries, and whole-archive caching."""

import gzip
import io
import os
import tarfile

import pytest

from kcoral import Program
from kcoral.app import _parse_execute_request
from kcoral.cache import ByteCache, DiskFileCache
from kcoral.engine import execute
from kcoral.errors import ValidationError
from kcoral.folder_archive import archive_files, pack_files
from kcoral.keys import compute_blob_hash
from kcoral.multipart import MultipartPart, encode_multipart
from kcoral.schemas import FolderUpload, parse_program
from kcoral.testing import FakeRuntime


class NoGPULease:
    held = False

    def acquire(self):
        raise AssertionError("folder extraction must not acquire a GPU")


def archive_with(*names, kind=tarfile.REGTYPE):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name in names:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = "../outside"
            member.mode = 0o777
            member.size = 1 if kind == tarfile.REGTYPE else 0
            archive.addfile(member, io.BytesIO(b"x"))
    return output.getvalue()


def run_archive(tmp_path, data, *, path="inputs"):
    program = parse_program(
        {
            "instructions": [
                {
                    "op": "upload",
                    "kind": "folder",
                    "blob": compute_blob_hash(data),
                    "path": path,
                }
            ]
        }
    )
    assert isinstance(program.instructions[0], FolderUpload)
    program.blob_bytes = {compute_blob_hash(data): data}
    return execute(program, FakeRuntime(), NoGPULease(), workspace_dir=str(tmp_path))


def test_extracts_nested_and_hidden_files_without_archived_permissions(tmp_path):
    data = archive_with("nested/data", ".hidden", "empty")
    result = run_archive(tmp_path, data)
    assert result.status == "COMPLETED", result.error
    for name in ["nested/data", ".hidden", "empty"]:
        target = tmp_path / "inputs" / name
        assert target.read_bytes() == b"x"
        assert target.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("name", ["../outside", "/outside", "a/../../outside", "a\\b"])
def test_rejects_unsafe_archive_paths_before_writing_any_entry(tmp_path, name):
    result = run_archive(tmp_path, archive_with("good", name))
    assert result.status == "FAILED"
    assert result.error["kind"] == "runtime"
    assert result.error["instruction_index"] == 0
    assert result.error["instruction_id"] is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "kind",
    [
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.FIFOTYPE,
        tarfile.CHRTYPE,
        tarfile.DIRTYPE,
        tarfile.GNUTYPE_SPARSE,
    ],
)
def test_rejects_links_special_files_and_sparse_archives(kind):
    with pytest.raises(ValueError):
        archive_files(archive_with("bad", kind=kind))


@pytest.mark.parametrize("names", [("a", "./a"), ("a", "a/b"), ("a/b", "a")])
def test_rejects_duplicate_and_conflicting_members(names):
    with pytest.raises(ValueError, match=r"duplicate|conflicting"):
        archive_files(archive_with(*names))


def test_rejects_compressed_and_truncated_archives():
    data = archive_with("a")
    for invalid in [b"not a tar", gzip.compress(data), data[:512]]:
        with pytest.raises(ValueError):
            archive_files(invalid)


@pytest.mark.parametrize("where", ["inputs", "inputs/nested", "inputs/nested/file"])
def test_extraction_does_not_follow_existing_symlinks(tmp_path, where):
    workspace = tmp_path / "work"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = workspace / where
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    outcome = run_archive(workspace, pack_files({"nested/file": b"stay inside"}))
    assert outcome.status == "FAILED"
    assert list(outside.iterdir()) == []
    assert link.is_symlink()


def test_extraction_refuses_to_overwrite_existing_files(tmp_path):
    (tmp_path / "inputs").mkdir()
    target = tmp_path / "inputs/a"
    target.write_bytes(b"keep")
    outcome = run_archive(tmp_path, pack_files({"a": b"replace"}))
    assert outcome.status == "FAILED"
    assert target.read_bytes() == b"keep"


@pytest.mark.parametrize(
    "other_kind,other_path", [("file", "inputs/a"), ("file", "inputs"), ("folder", "inputs")]
)
def test_frontend_rejects_conflicts_between_archives_and_other_uploads(
    tmp_path, other_kind, other_path
):
    import json

    archive = pack_files({"a": b"x"})
    digest = compute_blob_hash(archive)
    instructions = [
        {"op": "upload", "kind": "folder", "path": "inputs", "blob": digest},
        {"op": "upload", "kind": other_kind, "path": other_path, "blob": digest},
    ]
    body, content_type = encode_multipart(
        [
            MultipartPart(
                "program", "application/json", json.dumps({"instructions": instructions}).encode()
            ),
            MultipartPart("blob:" + digest, "application/octet-stream", archive),
        ]
    )
    with pytest.raises(ValidationError, match=r"duplicate|conflicting"):
        _parse_execute_request(
            content_type, body, ByteCache(1024), DiskFileCache(tmp_path, 1024**2)
        )


def test_archive_is_standard_tar_and_deterministic_across_enumeration_orders():
    files = {"z": b"last", "nested/\u6587\u4ef6": b"unicode", "long/" + "x" * 200: b"long"}
    first = pack_files(files)
    assert first == pack_files(dict(reversed(list(files.items()))))
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:") as archive:
        assert {member.name: archive.extractfile(member).read() for member in archive} == files
    assert {
        name: first[offset : offset + size] for name, offset, size in archive_files(first)
    } == files


def test_source_rename_changes_archive_identity(tmp_path):
    (tmp_path / "before").write_bytes(b"same bytes")
    before = Program()
    before.upload_folder(tmp_path, path="data")
    os.rename(tmp_path / "before", tmp_path / "after")
    after = Program()
    after.upload_folder(tmp_path, path="data")
    assert before._blobs.keys().isdisjoint(after._blobs)
