import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_accrl_cuda_kernels.py"
_SPEC = importlib.util.spec_from_file_location("extract_accrl_cuda_kernels", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
extractor = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = extractor
_SPEC.loader.exec_module(extractor)


def test_source_architecture_prefers_gencode():
    trajectory = {
        "info": {
            "config": {"environment": {"env": {"NVCC_GENCODE": "arch=compute_100a,code=sm_100a"}}}
        }
    }

    assert extractor.source_architecture(trajectory, {"prompt_tag": "hopper-no-hint"}) == "b200"


def test_source_architecture_uses_b200_and_h100_plan_tags():
    assert extractor.source_architecture({}, {"prompt_tag": "b200-bf16-010"}) == "b200"
    assert extractor.source_architecture({}, {"prompt_tag": "hopper-no-hint"}) == "h100"
    assert extractor.source_architecture({}, {"prompt_tag": "h100-fp8"}) == "h100"
    assert extractor.source_architecture({}, {"prompt_tag": "triton-blackwell"}) is None


def test_definition_comes_from_evaluation_trace():
    trajectory = {
        "messages": [
            {
                "role": "user",
                "extra": {"traces": [{"definition": "gemm_n7168_k5120"}]},
            }
        ]
    }

    assert extractor.trajectory_definition(trajectory, {}, None) == "gemm_n7168_k5120"


def test_definition_falls_back_to_plan_then_summary():
    assert extractor.trajectory_definition({}, {"definition": "from_plan"}, "summary") == (
        "from_plan"
    )
    assert extractor.trajectory_definition({}, {}, "from_summary") == "from_summary"


def test_copy_turn_correctness_csvs_records_copied_and_missing_runs(tmp_path):
    eval_root = tmp_path / "eval_runs"
    output = tmp_path / "corpus"
    source = eval_root / "run-a" / "figures" / "turn_correctness_arch.csv"
    source.parent.mkdir(parents=True)
    source.write_text("turn,correct,arch\n0,1,sm_100a\n", encoding="utf-8")
    output.mkdir()

    result = extractor.copy_turn_correctness_csvs(
        eval_root=eval_root, output=output, runs={"run-b", "run-a"}
    )

    metadata = output / "turn_correctness_arch"
    assert result == {"copied": 1, "missing": 1}
    assert (metadata / "run-a.csv").read_bytes() == source.read_bytes()
    assert (metadata / "missing.txt").read_text(encoding="utf-8") == "run-b\n"
    manifest = (metadata / "manifest.jsonl").read_text(encoding="utf-8")
    assert '"run":"run-a"' in manifest
    assert '"path":"turn_correctness_arch/run-a.csv"' in manifest


def test_extract_corpus_filters_and_labels_source_architecture(tmp_path, monkeypatch):
    eval_root = tmp_path / "eval_runs"
    for run, prompt_tag in (("run-b200", "b200-bf16"), ("run-h100", "hopper-no-hint")):
        run_dir = eval_root / run
        (run_dir / "trajectories").mkdir(parents=True)
        (run_dir / "plan.json").write_text(
            json.dumps(
                {
                    "plan": [
                        {
                            "exp_index": 0,
                            "prompt_tag": prompt_tag,
                            "definition": "gemm_n7168_k5120",
                        }
                    ]
                }
            )
        )
        (run_dir / "trajectories" / "exp_000.json").write_text("{}")

    monkeypatch.setattr(
        extractor,
        "load_accrl_extractor",
        lambda _root: lambda _trajectory: [(0, "void run() {}", "")],
    )
    output = tmp_path / "corpus"

    summary = extractor.extract_corpus(
        accrl_root=tmp_path / "AccRL",
        eval_root=eval_root,
        output=output,
        source_arches={"h100"},
    )

    assert summary["kernels"] == 1
    assert summary["by_source_arch"] == {"h100": 1}
    record = json.loads((output / "manifest.jsonl").read_text())
    assert record["language"] == "cuda"
    assert record["source_arch"] == "h100"
    assert record["run"] == "run-h100"


def test_checked_in_cuda_manifest_is_explicitly_b200():
    corpus = Path(__file__).parents[1] / "cuda_kernels"
    records = [
        json.loads(line)
        for line in (corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) >= 3362
    assert {record["language"] for record in records} == {"cuda"}
    assert {record["source_arch"] for record in records} == {"b200"}
    paths = {record["path"] for record in records}
    assert paths == {path.relative_to(corpus).as_posix() for path in corpus.rglob("kernel_t*.cu")}
