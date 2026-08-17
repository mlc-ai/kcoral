import hashlib
import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_accrl_triton_kernels.py"
_SPEC = importlib.util.spec_from_file_location("extract_accrl_triton_kernels", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
extractor = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = extractor
_SPEC.loader.exec_module(extractor)


TRITON_SOURCE = """\
import triton
import triton.language as tl

@triton.jit
def kernel(a, b, c):
    pass

def run(a, b, c):
    kernel[(1,)](a, b, c)
"""


def _fake_extract(text, **_kwargs):
    marker = "```python\n"
    if marker not in text:
        return ""
    return text.split(marker, 1)[1].split("\n```", 1)[0]


def test_extract_turns_skips_non_code_but_preserves_turn_index():
    trajectory = {
        "messages": [
            {"role": "assistant", "content": "no code"},
            {"role": "user", "content": "try again"},
            {"role": "assistant", "content": f"```python\n{TRITON_SOURCE}\n```"},
        ]
    }

    assert extractor.extract_turns(trajectory, _fake_extract) == [(1, TRITON_SOURCE)]


def test_discovers_nested_triton_run_from_plan(tmp_path):
    run_dir = tmp_path / "smoke-group" / "mha-fwd"
    (run_dir / "trajectories").mkdir(parents=True)
    (run_dir / "plan.json").write_text(
        json.dumps(
            {
                "plan": [
                    {
                        "exp_index": 0,
                        "prompt_tag": "triton-blackwell-010",
                        "definition": "mha_with_lse_d128",
                    }
                ]
            }
        )
    )

    assert extractor.discover_triton_runs(tmp_path) == [run_dir]


def test_extract_corpus_includes_trajectory_success_and_workspace(tmp_path, monkeypatch):
    eval_root = tmp_path / "eval_runs"
    run_dir = eval_root / "nested-smoke" / "gemm"
    trajectory_dir = run_dir / "trajectories"
    trajectory_dir.mkdir(parents=True)
    plan = {
        "plan": [
            {
                "exp_index": 0,
                "prompt_tag": "triton-hopper",
                "definition": "gemm_n7168_k5120",
                "test_path": "gemm_triton.py",
            }
        ]
    }
    (run_dir / "plan.json").write_text(json.dumps(plan))
    trajectory = {
        "info": {
            "exit_status": "LimitsExceeded",
            "config": {"environment": {"env": {"TRITON_GPU_ARCH": "hopper"}}},
        },
        "messages": [
            {"role": "assistant", "content": "no code"},
            {"role": "assistant", "content": f"```python\n{TRITON_SOURCE}\n```"},
        ],
    }
    trajectory_path = trajectory_dir / "exp_000.json"
    trajectory_path.write_text(json.dumps(trajectory))

    success_dir = run_dir / "success" / "exp_000"
    success_dir.mkdir(parents=True)
    (success_dir / "kernel_v0.py").write_text(TRITON_SOURCE)
    (success_dir / "record.json").write_text(
        json.dumps(
            [
                {
                    "version": 0,
                    "turn": 1,
                    "traces": [{"evaluation": {"status": "PASSED"}}],
                }
            ]
        )
    )
    workspace_dir = run_dir / "exp_000"
    workspace_dir.mkdir()
    (workspace_dir / "kernel.py").write_text(TRITON_SOURCE)

    monkeypatch.setattr(extractor, "load_accrl_extractor", lambda _root: _fake_extract)
    output = tmp_path / "corpus"
    summary = extractor.extract_corpus(
        accrl_root=tmp_path / "AccRL", eval_root=eval_root, output=output
    )

    assert summary["runs"] == 1
    assert summary["kernels"] == 3
    assert summary["by_source"] == {
        "success": 1,
        "trajectory": 1,
        "workspace": 1,
    }
    records = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert {record["source_kind"] for record in records} == {
        "trajectory",
        "success",
        "workspace",
    }
    assert all(record["source_arch"] == "h100" for record in records)
    assert all(record["language"] == "triton" for record in records)
    assert records[0]["turn"] == 1
    assert any(record["evaluation_status"] == "PASSED" for record in records)
    assert all((output / record["path"]).is_file() for record in records)


def test_checked_in_manifest_covers_all_triton_sources():
    corpus = Path(__file__).parents[1] / "triton_kernels"
    records = [
        json.loads(line)
        for line in (corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) >= 3296
    assert {record["source_kind"] for record in records} == {
        "trajectory",
        "success",
        "workspace",
    }
    assert {record["source_arch"] for record in records} == {"h100", "b200"}
    assert len({record["run"] for record in records}) >= 30
    paths = {record["path"] for record in records}
    assert paths == {path.relative_to(corpus).as_posix() for path in corpus.rglob("*.py")}
    for record in records:
        source = (corpus / record["path"]).read_bytes()
        assert hashlib.sha256(source).hexdigest() == record["sha256"]
