from __future__ import annotations

import argparse
import json
from pathlib import Path

from trs.data.registry import get_dataset_adapter
from trs.domain.enums import RunMode, TaskType
from trs.domain.schemas import ProblemRecord, RunManifest
from trs.experiments.compare import compare_summaries
from trs.experiments.reporting import write_comparison_report, write_run_outputs
from trs.experiments.runner import ExperimentRunner
from trs.models.base import ModelConfig
from trs.models.openai_compatible import build_model_client
from trs.prompts.loader import load_mapping
from trs.retrieval.selector import build_retriever
from trs.service.app import create_app
from trs.service.runtime import ReasoningService
from trs.skills.compiler import build_library_snapshot, load_skill_cards, save_skill_cards
from trs.skills.distiller import SkillDistiller
from trs.storage.jsonl_store import read_jsonl, write_jsonl
from trs.storage.manifest import read_manifest, write_manifest
from trs.traces.generator import TraceGenerator
from trs.traces.store import load_traces, save_traces
from trs.utils.ids import make_run_id
from trs.utils.logging import configure_logging


def load_problem_records(path: str | Path) -> list[ProblemRecord]:
    return [ProblemRecord.from_dict(row) for row in read_jsonl(path)]


def save_problem_records(path: str | Path, records: list[ProblemRecord]) -> None:
    write_jsonl(path, [record.to_dict() for record in records])


def command_datasets_prepare(args: argparse.Namespace) -> None:
    adapter = get_dataset_adapter(args.dataset)
    records = adapter.load(args.source)
    save_problem_records(args.output, records)
    print(json.dumps({"dataset": args.dataset, "count": len(records), "output": str(args.output)}, indent=2))


def _load_client(config_path: str | Path):
    config = ModelConfig.from_dict(load_mapping(config_path))
    return build_model_client(config)


def command_traces_generate(args: argparse.Namespace) -> None:
    problems = load_problem_records(args.dataset_file)
    client = _load_client(args.model_config)
    prompt_config = load_mapping(args.prompt_config)
    traces = TraceGenerator(client, prompt_config).generate_batch(problems)
    save_traces(args.output, traces)
    manifest = RunManifest(
        run_id=make_run_id("traces", str(args.output)),
        task_type=problems[0].task_type if problems else TaskType.MATH,
        dataset=problems[0].dataset if problems else "",
        mode=RunMode.DIRECT,
        model_name=client.name,
        source_path=str(args.dataset_file),
        output_path=str(args.output),
        metadata={"prompt_config": str(args.prompt_config), "model_config": str(args.model_config), "count": len(traces)},
    )
    write_manifest(Path(args.output).with_suffix(".manifest.json"), manifest.to_dict())
    print(json.dumps({"trace_count": len(traces), "output": str(args.output)}, indent=2))


def command_skills_distill(args: argparse.Namespace) -> None:
    problems = load_problem_records(args.dataset_file)
    traces = load_traces(args.traces_file)
    client = _load_client(args.model_config)
    prompt_config = load_mapping(args.prompt_config)
    cards = SkillDistiller(client, prompt_config).distill_batch(problems, traces)
    save_skill_cards(args.output, cards)
    manifest = RunManifest(
        run_id=make_run_id("skills", str(args.output)),
        task_type=problems[0].task_type if problems else TaskType.MATH,
        dataset=problems[0].dataset if problems else "",
        mode=RunMode.DIRECT,
        model_name=client.name,
        source_path=str(args.traces_file),
        output_path=str(args.output),
        metadata={"prompt_config": str(args.prompt_config), "model_config": str(args.model_config), "count": len(cards)},
    )
    write_manifest(Path(args.output).with_suffix(".manifest.json"), manifest.to_dict())
    print(json.dumps({"skill_count": len(cards), "output": str(args.output)}, indent=2))


def command_library_build(args: argparse.Namespace) -> None:
    cards = load_skill_cards(args.cards_file)
    profile = load_mapping(args.profile_config)
    snapshot = build_library_snapshot(cards, args.output_dir, profile)
    print(json.dumps(snapshot.to_dict(), indent=2))


def command_eval_run(args: argparse.Namespace) -> None:
    problems = load_problem_records(args.dataset_file)
    client = _load_client(args.model_config)
    prompt_config = load_mapping(args.prompt_config)
    mode = RunMode(args.mode)
    cards = []
    retriever = None
    profile = {}
    if mode == RunMode.TRS:
        if not args.library_dir:
            raise ValueError("TRS mode requires --library-dir")
        cards = load_skill_cards(Path(args.library_dir) / "cards.jsonl")
        manifest = read_manifest(Path(args.library_dir) / "library_manifest.json")
        profile = dict(manifest.get("metadata", {}).get("profile", {}))
        retriever = build_retriever(profile)
    runner = ExperimentRunner(client, prompt_config, mode, retriever=retriever, profile=profile, cards=cards)
    results, summary = runner.run(problems)
    write_run_outputs(args.output_dir, results, summary)
    manifest = RunManifest(
        run_id=make_run_id("eval", str(args.output_dir)),
        task_type=problems[0].task_type if problems else TaskType.MATH,
        dataset=problems[0].dataset if problems else "",
        mode=mode,
        model_name=client.name,
        source_path=str(args.dataset_file),
        output_path=str(args.output_dir),
        metadata={
            "prompt_config": str(args.prompt_config),
            "model_config": str(args.model_config),
            "library_dir": str(args.library_dir or ""),
            "count": len(results),
        },
    )
    write_manifest(Path(args.output_dir) / "manifest.json", manifest.to_dict())
    print(json.dumps(summary.to_dict(), indent=2))


def command_compare(args: argparse.Namespace) -> None:
    baseline_summary = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
    candidate_summary = json.loads(Path(args.candidate_summary).read_text(encoding="utf-8"))
    candidate_results_payload = json.loads(Path(args.candidate_results).read_text(encoding="utf-8"))
    from trs.domain.results import MetricsSummary
    from trs.domain.schemas import InferenceRequest, InferenceResult, RetrievedSkill, UsageStats

    candidate_results = []
    for row in candidate_results_payload:
        request_payload = row["request"]
        request = InferenceRequest(
            task_type=TaskType(request_payload["task_type"]),
            question=request_payload["question"],
            dataset=request_payload["dataset"],
            problem_id=request_payload["problem_id"],
            mode=RunMode(request_payload["mode"]),
            profile=request_payload.get("profile", ""),
            library_id=request_payload.get("library_id", ""),
            tests=list(request_payload.get("tests", [])),
            answer=request_payload.get("answer", ""),
            entry_point=request_payload.get("entry_point", ""),
            metadata={k: v for k, v in request_payload.items() if k not in {"task_type", "question", "dataset", "problem_id", "mode", "profile", "library_id", "tests", "answer", "entry_point"}},
        )
        candidate_results.append(
            InferenceResult(
                request=request,
                response_text=row["response_text"],
                final_answer=row["final_answer"],
                usage=UsageStats.from_dict(row["usage"]),
                used_skills=[RetrievedSkill.from_dict(skill) for skill in row.get("used_skills", [])],
                is_correct=bool(row["is_correct"]),
                model_name=row["model_name"],
                fallback_used=bool(row.get("fallback_used", False)),
                cost_estimate=float(row["cost_estimate"]),
                created_at=row.get("created_at", ""),
                metadata=dict(row.get("metadata", {})),
            )
        )
    baseline = MetricsSummary(**baseline_summary)
    candidate = MetricsSummary(**candidate_summary)
    comparison = compare_summaries(baseline, candidate, candidate_results)
    write_comparison_report(args.output_dir, comparison)
    print(json.dumps(comparison.to_dict(), indent=2))


def command_serve_api(args: argparse.Namespace) -> None:
    service = ReasoningService(args.model_config, args.prompt_config, library_dir=args.library_dir)
    app = create_app(service)
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is not installed. Install optional api dependencies first.") from exc
    uvicorn.run(app, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trs", description="Thinking with Reasoning Skills CLI")
    parser.add_argument("--log-config", default="configs/logging.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    datasets = subparsers.add_parser("datasets")
    datasets_sub = datasets.add_subparsers(dest="datasets_command", required=True)
    prepare = datasets_sub.add_parser("prepare")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(func=command_datasets_prepare)

    traces = subparsers.add_parser("traces")
    traces_sub = traces.add_subparsers(dest="traces_command", required=True)
    generate = traces_sub.add_parser("generate")
    generate.add_argument("--dataset-file", required=True)
    generate.add_argument("--task-type", required=False)
    generate.add_argument("--dataset-name", required=False)
    generate.add_argument("--model-config", required=True)
    generate.add_argument("--prompt-config", required=True)
    generate.add_argument("--output", required=True)
    generate.set_defaults(func=command_traces_generate)

    skills = subparsers.add_parser("skills")
    skills_sub = skills.add_subparsers(dest="skills_command", required=True)
    distill = skills_sub.add_parser("distill")
    distill.add_argument("--dataset-file", required=True)
    distill.add_argument("--traces-file", required=True)
    distill.add_argument("--model-config", required=True)
    distill.add_argument("--prompt-config", required=True)
    distill.add_argument("--output", required=True)
    distill.set_defaults(func=command_skills_distill)

    library = subparsers.add_parser("library")
    library_sub = library.add_subparsers(dest="library_command", required=True)
    build = library_sub.add_parser("build")
    build.add_argument("--cards-file", required=True)
    build.add_argument("--profile-config", required=True)
    build.add_argument("--output-dir", required=True)
    build.set_defaults(func=command_library_build)

    eval_parser = subparsers.add_parser("eval")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    run = eval_sub.add_parser("run")
    run.add_argument("--mode", choices=[RunMode.DIRECT.value, RunMode.TRS.value], required=True)
    run.add_argument("--dataset-file", required=True)
    run.add_argument("--task-type", required=False)
    run.add_argument("--dataset-name", required=False)
    run.add_argument("--model-config", required=True)
    run.add_argument("--prompt-config", required=True)
    run.add_argument("--library-dir")
    run.add_argument("--output-dir", required=True)
    run.set_defaults(func=command_eval_run)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline-summary", required=True)
    compare.add_argument("--candidate-summary", required=True)
    compare.add_argument("--candidate-results", required=True)
    compare.add_argument("--output-dir", required=True)
    compare.set_defaults(func=command_compare)

    serve = subparsers.add_parser("serve")
    serve_sub = serve.add_subparsers(dest="serve_command", required=True)
    api = serve_sub.add_parser("api")
    api.add_argument("--model-config", required=True)
    api.add_argument("--prompt-config", required=True)
    api.add_argument("--library-dir")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)
    api.set_defaults(func=command_serve_api)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_config)
    args.func(args)


if __name__ == "__main__":
    main()
