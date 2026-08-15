import importlib.util
import sys
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_accrl_blackwell_kernels.py"
_SPEC = importlib.util.spec_from_file_location("extract_accrl_blackwell_kernels", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
extractor = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = extractor
_SPEC.loader.exec_module(extractor)


def test_blackwell_detection_prefers_gencode():
    trajectory = {
        "info": {
            "config": {
                "environment": {
                    "env": {"NVCC_GENCODE": "arch=compute_100a,code=sm_100a"}
                }
            }
        }
    }

    assert extractor.is_blackwell_trajectory(trajectory, {})


def test_blackwell_detection_uses_b200_plan_tag_fallback():
    assert extractor.is_blackwell_trajectory({}, {"prompt_tag": "b200-bf16-010"})
    assert not extractor.is_blackwell_trajectory({}, {"prompt_tag": "hopper-no-hint"})
    assert not extractor.is_blackwell_trajectory({}, {"prompt_tag": "triton-blackwell"})


def test_definition_comes_from_evaluation_trace():
    trajectory = {
        "messages": [
            {
                "role": "user",
                "extra": {"traces": [{"definition": "gemm_n7168_k5120"}]},
            }
        ]
    }

    assert (
        extractor.trajectory_definition(trajectory, {}, None) == "gemm_n7168_k5120"
    )


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
