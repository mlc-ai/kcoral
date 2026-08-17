#!/usr/bin/env python3
"""Extract every CUDA turn from Blackwell trajectories using AccRL's extractor."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACCRL_ROOT = Path("/home/yixind/AccRL")
DEFAULT_EVAL_ROOT = Path("/home/yixind/AccRL-exps/eval_runs")
DEFAULT_OUTPUT = REPO_ROOT / "b200_cuda_kernels"
CORRECTNESS_DIR = "turn_correctness_arch"


def load_accrl_extractor(accrl_root: Path) -> Callable[[dict], list[tuple[int, str, str]]]:
    script = (
        accrl_root
        / "fib_runtime"
        / "mini_swe_agent_docker"
        / "plots"
        / "analyze_kernel_per_turn.py"
    )
    if not script.is_file():
        raise ValueError(f"AccRL extraction script not found: {script}")
    spec = importlib.util.spec_from_file_location("accrl_analyze_kernel_per_turn", script)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load AccRL extraction script: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.extract_turns_from_trajectory


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc


def plan_metadata(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "plan.json"
    if not path.is_file():
        return {}
    data = load_json(path)
    result: dict[str, dict[str, Any]] = {}
    for item in data.get("plan", []):
        index = item.get("exp_index")
        if isinstance(index, int):
            result[f"exp_{index:03d}"] = item
    return result


def summary_definitions(run_dir: Path) -> dict[str, str]:
    path = run_dir / "summary.json"
    if not path.is_file():
        return {}
    data = load_json(path)
    return {
        item["exp_name"]: item["definition"]
        for item in data.get("results", [])
        if isinstance(item.get("exp_name"), str) and isinstance(item.get("definition"), str)
    }


def is_blackwell_trajectory(trajectory: dict, plan: dict[str, Any]) -> bool:
    environment = trajectory.get("info", {}).get("config", {}).get("environment", {})
    variables = environment.get("env", {}) if isinstance(environment, dict) else {}
    gencode = str(variables.get("NVCC_GENCODE", "")).lower()
    if "compute_100a" in gencode or "sm_100a" in gencode:
        return True
    prompt_tag = str(plan.get("prompt_tag", "")).lower()
    return prompt_tag == "b200" or prompt_tag.startswith("b200-")


def trajectory_definition(
    trajectory: dict,
    plan: dict[str, Any],
    summary_definition: str | None,
) -> str | None:
    definitions = {
        trace.get("definition")
        for message in trajectory.get("messages", [])
        for trace in (message.get("extra", {}).get("traces", []) or [])
        if isinstance(trace, dict) and isinstance(trace.get("definition"), str)
    }
    if len(definitions) == 1:
        return definitions.pop()
    planned = plan.get("definition")
    if isinstance(planned, str):
        return planned
    return summary_definition


def copy_turn_correctness_csvs(*, eval_root: Path, output: Path, runs: set[str]) -> dict[str, int]:
    """Copy the run-level correctness/architecture tables available in AccRL."""
    destination = output / CORRECTNESS_DIR
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.jsonl"
    missing_path = destination / "missing.txt"
    copied = 0
    missing: list[str] = []
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for run in sorted(runs):
            source = eval_root / run / "figures" / "turn_correctness_arch.csv"
            if not source.is_file():
                missing.append(run)
                continue
            target = destination / f"{run}.csv"
            shutil.copy2(source, target)
            record = {
                "run": run,
                "path": target.relative_to(output).as_posix(),
                "source": source.relative_to(eval_root).as_posix(),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
            manifest.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            copied += 1
    missing_path.write_text("".join(f"{run}\n" for run in missing), encoding="utf-8")
    return {"copied": copied, "missing": len(missing)}


def extract_corpus(
    *,
    accrl_root: Path,
    eval_root: Path,
    output: Path,
) -> dict[str, Any]:
    extract_turns = load_accrl_extractor(accrl_root)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.jsonl"
    manifest = manifest_path.open("w", encoding="utf-8")
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    trajectories = 0
    represented_runs: set[str] = set()
    try:
        for run_dir in sorted(path for path in eval_root.iterdir() if path.is_dir()):
            trajectory_dir = run_dir / "trajectories"
            if not trajectory_dir.is_dir():
                continue
            plans = plan_metadata(run_dir)
            summaries = summary_definitions(run_dir)
            for trajectory_path in sorted(trajectory_dir.glob("exp_*.json")):
                exp_name = trajectory_path.stem
                try:
                    trajectory = load_json(trajectory_path)
                except ValueError:
                    skipped["unreadable_trajectory"] += 1
                    continue
                plan = plans.get(exp_name, {})
                if not is_blackwell_trajectory(trajectory, plan):
                    continue
                turns = extract_turns(trajectory)
                if not turns:
                    skipped["no_cuda_turns"] += 1
                    continue
                definition = trajectory_definition(trajectory, plan, summaries.get(exp_name))
                if not definition:
                    skipped["unknown_definition"] += len(turns)
                    continue
                trajectories += 1
                represented_runs.add(run_dir.name)
                destination = output / definition / run_dir.name / exp_name
                destination.mkdir(parents=True, exist_ok=True)
                for turn, kernel_source, _observation in turns:
                    kernel_path = destination / f"kernel_t{turn}.cu"
                    kernel_path.write_text(kernel_source, encoding="utf-8")
                    digest = hashlib.sha256(kernel_source.encode("utf-8")).hexdigest()
                    record = {
                        "workload": definition,
                        "run": run_dir.name,
                        "experiment": exp_name,
                        "turn": turn,
                        "path": kernel_path.relative_to(output).as_posix(),
                        "sha256": digest,
                        "trajectory": trajectory_path.relative_to(eval_root).as_posix(),
                        "prompt_tag": plan.get("prompt_tag"),
                        "exit_status": trajectory.get("info", {}).get("exit_status"),
                    }
                    manifest.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    counts[definition] += 1
    finally:
        manifest.close()
    correctness_csvs = copy_turn_correctness_csvs(
        eval_root=eval_root, output=output, runs=represented_runs
    )
    return {
        "trajectories": trajectories,
        "kernels": sum(counts.values()),
        "by_workload": dict(sorted(counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "turn_correctness_arch_csvs": correctness_csvs,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--accrl-root", type=Path, default=DEFAULT_ACCRL_ROOT)
    result.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--force", action="store_true", help="replace an existing output directory")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output.resolve()
    try:
        if not args.eval_root.is_dir():
            raise ValueError(f"evaluation root not found: {args.eval_root}")
        if output.exists():
            if not args.force:
                raise ValueError(f"output already exists (pass --force to replace it): {output}")
            protected = {Path(output.anchor), Path.home().resolve(), REPO_ROOT.resolve()}
            if output in protected or len(output.parts) < 3:
                raise ValueError(f"refusing to replace unsafe output path: {output}")
            shutil.rmtree(output)
        summary = extract_corpus(
            accrl_root=args.accrl_root.resolve(),
            eval_root=args.eval_root.resolve(),
            output=output,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Output: {output}", file=sys.stderr)
        return 0
    except (OSError, ValueError) as exc:
        print(f"extract_accrl_blackwell_kernels: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
