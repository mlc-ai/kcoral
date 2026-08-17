#!/usr/bin/env python3
"""Extract every Triton candidate available in AccRL evaluation runs."""

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
DEFAULT_OUTPUT = REPO_ROOT / "triton_kernels"


def load_accrl_extractor(accrl_root: Path) -> Callable[..., str]:
    script = accrl_root / "accrl" / "utils" / "code_utils.py"
    if not script.is_file():
        raise ValueError(f"AccRL code extractor not found: {script}")
    spec = importlib.util.spec_from_file_location("accrl_code_utils", script)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load AccRL code extractor: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.extract_code_block


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc


def plan_items(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "plan.json"
    if not path.is_file():
        return []
    data = load_json(path)
    items = data.get("plan", []) if isinstance(data, dict) else data
    return [item for item in items if isinstance(item, dict)]


def plan_metadata(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        f"exp_{item['exp_index']:03d}": item
        for item in plan_items(run_dir)
        if isinstance(item.get("exp_index"), int)
    }


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


def is_triton_run(run_dir: Path) -> bool:
    for item in plan_items(run_dir):
        prompt_tag = str(item.get("prompt_tag", "")).lower()
        test_path = str(item.get("test_path", "")).lower()
        if prompt_tag.startswith("triton-") or test_path.endswith("_triton.py"):
            return True

    trajectory_dir = run_dir / "trajectories"
    for path in sorted(trajectory_dir.glob("exp_*.json")):
        try:
            trajectory = load_json(path)
        except ValueError:
            continue
        environment = trajectory.get("info", {}).get("config", {}).get("environment", {})
        variables = environment.get("env", {}) if isinstance(environment, dict) else {}
        if variables.get("TRITON_GPU_ARCH"):
            return True
    return False


def discover_triton_runs(eval_root: Path) -> list[Path]:
    """Find Triton runs, including runs nested below a smoke-test group."""
    runs: list[Path] = []
    for trajectory_dir in sorted(eval_root.rglob("trajectories")):
        run_dir = trajectory_dir.parent
        if (run_dir / "plan.json").is_file() and is_triton_run(run_dir):
            runs.append(run_dir)
    return runs


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


def source_architecture(trajectory: dict, plan: dict[str, Any] | None = None) -> str | None:
    environment = trajectory.get("info", {}).get("config", {}).get("environment", {})
    variables = environment.get("env", {}) if isinstance(environment, dict) else {}
    value = str(variables.get("TRITON_GPU_ARCH", "")).strip().lower()
    aliases = {
        "b200": "b200",
        "blackwell": "b200",
        "sm_100a": "b200",
        "h100": "h100",
        "hopper": "h100",
        "sm_90a": "h100",
    }
    if value in aliases:
        return aliases[value]
    prompt_tag = str((plan or {}).get("prompt_tag", "")).lower()
    if prompt_tag.startswith("triton-blackwell"):
        return "b200"
    if prompt_tag.startswith("triton-hopper"):
        return "h100"
    return None


def require_source_architecture(trajectory: dict, plan: dict[str, Any]) -> str:
    source_arch = source_architecture(trajectory, plan)
    if source_arch is None:
        raise ValueError(
            f"cannot determine Triton source architecture for prompt {plan.get('prompt_tag')!r}"
        )
    return source_arch


def extract_turns(
    trajectory: dict, extract_code_block: Callable[..., str]
) -> list[tuple[int, str]]:
    """Extract Python blocks while preserving assistant-turn numbering."""
    turns: list[tuple[int, str]] = []
    turn = 0
    for message in trajectory.get("messages", []):
        if message.get("role") != "assistant":
            continue
        source = extract_code_block(
            message.get("content", "") or "",
            languages=["python"],
            keep_separators=False,
        )
        if source:
            turns.append((turn, source))
        turn += 1
    return turns


def success_records(exp_dir: Path) -> dict[int, dict[str, Any]]:
    path = exp_dir / "record.json"
    if not path.is_file():
        return {}
    try:
        data = load_json(path)
    except ValueError:
        return {}
    return {
        item["version"]: item
        for item in data
        if isinstance(item, dict) and isinstance(item.get("version"), int)
    }


def record_source(
    *,
    manifest: Any,
    counts: Counter[str],
    output: Path,
    eval_root: Path,
    run_relative: Path,
    workload: str,
    experiment: str,
    source_kind: str,
    source_path: Path,
    destination_name: str,
    plan: dict[str, Any],
    trajectory_path: Path | None,
    trajectory: dict,
    turn: int | None = None,
    version: int | None = None,
    evaluation_status: str | None = None,
) -> None:
    source = source_path.read_text(encoding="utf-8")
    destination = output / workload / run_relative / experiment / source_kind / destination_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
    record = {
        "language": "triton",
        "workload": workload,
        "run": run_relative.as_posix(),
        "experiment": experiment,
        "source_kind": source_kind,
        "turn": turn,
        "version": version,
        "path": destination.relative_to(output).as_posix(),
        "source": source_path.relative_to(eval_root).as_posix(),
        "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "trajectory": (
            trajectory_path.relative_to(eval_root).as_posix()
            if trajectory_path is not None
            else None
        ),
        "prompt_tag": plan.get("prompt_tag"),
        "source_arch": require_source_architecture(trajectory, plan),
        "exit_status": trajectory.get("info", {}).get("exit_status"),
        "evaluation_status": evaluation_status,
        "valid_triton_shape": "@triton.jit" in source and "def run(" in source,
    }
    manifest.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    counts[f"source:{source_kind}"] += 1
    counts[f"workload:{workload}"] += 1


def extract_corpus(
    *,
    accrl_root: Path,
    eval_root: Path,
    output: Path,
) -> dict[str, Any]:
    extract_code_block = load_accrl_extractor(accrl_root)
    run_dirs = discover_triton_runs(eval_root)
    output.mkdir(parents=True, exist_ok=False)
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    manifest_path = output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for run_dir in run_dirs:
            run_relative = run_dir.relative_to(eval_root)
            plans = plan_metadata(run_dir)
            summaries = summary_definitions(run_dir)
            trajectories: dict[str, tuple[Path, dict]] = {}

            for trajectory_path in sorted((run_dir / "trajectories").glob("exp_*.json")):
                experiment = trajectory_path.stem
                try:
                    trajectory = load_json(trajectory_path)
                except ValueError:
                    skipped["unreadable_trajectory"] += 1
                    continue
                trajectories[experiment] = (trajectory_path, trajectory)
                plan = plans.get(experiment, {})
                workload = trajectory_definition(trajectory, plan, summaries.get(experiment))
                if not workload:
                    skipped["unknown_definition"] += 1
                    continue
                turns = extract_turns(trajectory, extract_code_block)
                if not turns:
                    skipped["no_python_turns"] += 1
                    continue
                counts["trajectories"] += 1
                for turn, source in turns:
                    # record_source accepts paths for all source kinds; trajectory sources only
                    # exist inside JSON, so write directly without fabricating a source file.
                    destination = (
                        output
                        / workload
                        / run_relative
                        / experiment
                        / "trajectory"
                        / f"kernel_t{turn}.py"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(source, encoding="utf-8")
                    record = {
                        "language": "triton",
                        "workload": workload,
                        "run": run_relative.as_posix(),
                        "experiment": experiment,
                        "source_kind": "trajectory",
                        "turn": turn,
                        "version": None,
                        "path": destination.relative_to(output).as_posix(),
                        "source": None,
                        "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                        "trajectory": trajectory_path.relative_to(eval_root).as_posix(),
                        "prompt_tag": plan.get("prompt_tag"),
                        "source_arch": require_source_architecture(trajectory, plan),
                        "exit_status": trajectory.get("info", {}).get("exit_status"),
                        "evaluation_status": None,
                        "valid_triton_shape": "@triton.jit" in source and "def run(" in source,
                    }
                    manifest.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    counts["source:trajectory"] += 1
                    counts[f"workload:{workload}"] += 1

            experiments = set(trajectories)
            experiments.update(path.name for path in (run_dir / "success").glob("exp_*"))
            experiments.update(path.name for path in run_dir.glob("exp_*") if path.is_dir())
            for experiment in sorted(experiments):
                plan = plans.get(experiment, {})
                trajectory_entry = trajectories.get(experiment)
                trajectory_path, trajectory = (
                    trajectory_entry if trajectory_entry is not None else (None, {})
                )
                workload = trajectory_definition(trajectory, plan, summaries.get(experiment))
                if not workload:
                    skipped["unknown_definition_file"] += 1
                    continue

                success_dir = run_dir / "success" / experiment
                records = success_records(success_dir)
                for kernel_path in sorted(success_dir.glob("kernel_v*.py")):
                    try:
                        version = int(kernel_path.stem.removeprefix("kernel_v"))
                    except ValueError:
                        version = None
                    metadata = records.get(version, {}) if version is not None else {}
                    traces = metadata.get("traces", [])
                    evaluation_status = None
                    if traces and isinstance(traces[0], dict):
                        evaluation_status = traces[0].get("evaluation", {}).get("status")
                    record_source(
                        manifest=manifest,
                        counts=counts,
                        output=output,
                        eval_root=eval_root,
                        run_relative=run_relative,
                        workload=workload,
                        experiment=experiment,
                        source_kind="success",
                        source_path=kernel_path,
                        destination_name=kernel_path.name,
                        plan=plan,
                        trajectory_path=trajectory_path,
                        trajectory=trajectory,
                        turn=metadata.get("turn"),
                        version=version,
                        evaluation_status=evaluation_status,
                    )

                workspace_kernel = run_dir / experiment / "kernel.py"
                if workspace_kernel.is_file():
                    record_source(
                        manifest=manifest,
                        counts=counts,
                        output=output,
                        eval_root=eval_root,
                        run_relative=run_relative,
                        workload=workload,
                        experiment=experiment,
                        source_kind="workspace",
                        source_path=workspace_kernel,
                        destination_name="kernel.py",
                        plan=plan,
                        trajectory_path=trajectory_path,
                        trajectory=trajectory,
                    )

    return {
        "runs": len(run_dirs),
        "trajectories": counts["trajectories"],
        "kernels": sum(value for key, value in counts.items() if key.startswith("source:")),
        "by_source": {
            key.removeprefix("source:"): value
            for key, value in sorted(counts.items())
            if key.startswith("source:")
        },
        "by_workload": {
            key.removeprefix("workload:"): value
            for key, value in sorted(counts.items())
            if key.startswith("workload:")
        },
        "skipped": dict(sorted(skipped.items())),
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
        print(f"extract_accrl_triton_kernels: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
