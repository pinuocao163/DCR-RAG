import argparse
import json
from pathlib import Path

from .config import load_config
from .data import DATASETS, assert_disjoint, convert, prepare_splits
from .features import initial_pair
from .gain import (
    choose_pair,
    fit_gain_ensemble,
    gain_diagnostics,
    label_gains,
    tune_gate,
)
from .io import ROOT, digest, load_rows, output_path, save_json, save_rows
from .metrics import evaluate, paired_bootstrap
from .prompts import qa_prompt
from .teacher import fit_teacher, necessity_scores
from .reading import read_selected


def read_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_for(args):
    rows = load_rows(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("Input is empty")
    assert_disjoint({"input": rows})
    return rows


def generator(config, role, device=None, for_scoring=False):
    from .models import CausalGenerator, VLLMGenerator, seed_everything

    seed_everything(config["seed"])
    path = config["models"][role]
    backend = config["generation"].get(
        f"{role}_backend", config["generation"].get("backend", "hf")
    )
    if not for_scoring and backend == "vllm":
        return VLLMGenerator(path, {"seed": config["seed"], **config["generation"]})
    return CausalGenerator(
        path,
        device or config["device"],
        config["generation"]["dtype"],
        trust_remote_code=config["generation"].get(f"{role}_trust_remote_code", False),
        device_map=config["generation"].get(f"{role}_device_map"),
    )


def stage_prepare(args, config):
    specification = config["datasets"][args.dataset]
    train_paths = args.train_source or specification["train_sources"]
    eval_paths = args.eval_source or specification["eval_sources"]
    prepare_splits(
        config,
        args.dataset,
        output_path(args.output),
        [Path(path) for path in train_paths],
        [Path(path) for path in eval_paths],
        args.exclude,
    )


def stage_bge(args, config):
    from .models import PassageScorer

    scorer = None
    rows = rows_for(args)
    output = []
    for index, original in enumerate(rows):
        row = convert(original, args.dataset)
        if row is None:
            raise ValueError(f"Invalid input question {original.get('id')}")
        if not args.use_cached_scores:
            if scorer is None:
                scorer = PassageScorer(
                    config["models"]["bge"],
                    args.device or config["device"],
                    semantic=True,
                    token_config=config["tokenization"],
                )
            scores = scorer.score(row, batch_size=config["scoring_batch_size"])
            for item, score in zip(row["ctxs"], scores):
                item["rerank_score"] = score
        elif any("rerank_score" not in item for item in row["ctxs"]):
            raise ValueError("Cached BGE reuse requires rerank_score on every passage")
        row["ctxs"].sort(key=lambda item: -float(item["rerank_score"]))
        cap = config["datasets"][args.dataset]["pool_cap"]
        row["ctxs"] = row["ctxs"][:cap]
        for rank, item in enumerate(row["ctxs"]):
            item["bge_rank"] = rank
        row["bge_ordered"] = True
        output.append(row)
        print(f"bge {index + 1}/{len(rows)}", flush=True)
    save_rows(args.output, output)


def stage_draft(args, config):
    model = generator(config, "reference", args.device)
    rows = rows_for(args)
    for index, row in enumerate(rows):
        if not row.get("bge_ordered"):
            raise ValueError("Draft input must be the fixed BGE-ordered pool")
        row["draft"] = model.generate(
            qa_prompt(row),
            config["generation"]["draft_input_cap"],
            config["generation"]["draft_output_cap"],
        )
        print(f"draft {index + 1}/{len(rows)}", flush=True)
    save_rows(args.output, rows)


def stage_teacher(args, config):
    model = generator(config, "reference", args.device, for_scoring=True)
    rows = rows_for(args)
    for index, row in enumerate(rows):
        raw = necessity_scores(row, model, config["generation"]["draft_input_cap"])
        for item, value in zip(row["ctxs"], raw):
            item["necessity_raw"] = value
        print(f"teacher {index + 1}/{len(rows)}", flush=True)
    calibration = fit_teacher(rows)
    save_rows(args.output, rows)
    save_json(str(args.output) + ".calibration.json", calibration)


def stage_train(args, config):
    from .training import train_scorer

    rows = rows_for(args)
    if args.device:
        config["device"] = args.device
    train_scorer(
        rows, config, args.output, args.initial_model, include_draft=not args.no_draft
    )


def stage_score(args, config):
    from .models import PassageScorer

    rows = rows_for(args)
    scorer = PassageScorer(
        args.checkpoint,
        args.device or config["device"],
        token_config=config["tokenization"],
    )
    for index, row in enumerate(rows):
        scores = scorer.score(
            row, config["scoring_batch_size"], include_draft=not args.no_draft
        )
        for item, score in zip(row["ctxs"], scores):
            item["student_score"] = score
        print(f"score {index + 1}/{len(rows)}", flush=True)
    save_rows(args.output, rows)


def stage_gain_labels(args, config):
    rows = rows_for(args)
    model = generator(config, "reader", args.device)
    output = []
    for index, row in enumerate(rows):
        output.append(label_gains(row, model, config))
        print(f"gain-labels {index + 1}/{len(rows)}", flush=True)
    save_rows(args.output, output)


def stage_fit_gain(args, config):
    rows = rows_for(args)
    save_json(args.output, fit_gain_ensemble(rows, config))


def stage_tune(args, config):
    bundle = read_config(args.bundle)
    gate = tune_gate(rows_for(args), bundle, config)
    gate["bundle_sha256"] = digest(args.bundle)
    save_json(args.output, gate)


def stage_select(args, config):
    rows = rows_for(args)
    bundle = read_config(args.bundle) if args.bundle else None
    gate = read_config(args.gate) if args.gate else None
    if args.mode == "conditional" and (bundle is None or gate is None):
        raise ValueError(
            "Conditional selection requires a fitted bundle and development gate"
        )
    if bundle and gate and gate.get("bundle_sha256") != digest(args.bundle):
        raise ValueError("Gate was tuned for a different gain bundle")
    output = []
    for row in rows:
        if args.mode == "bge":
            selected, audit = (0, 1), {"accepted": False, "route": "bge"}
        elif args.mode == "independent":
            selected, audit = initial_pair(row), {
                "accepted": False,
                "route": "independent",
            }
        else:
            if row["dataset"] != bundle["dataset"]:
                raise ValueError("Dataset does not match gain bundle")
            expected = config["datasets"][row["dataset"]]["auxiliary"]
            if bundle["auxiliary"] != expected:
                raise ValueError("Configuration differs from fitted feature schema")
            selected, audit = choose_pair(
                row, bundle, gate["best"]["kappa"], gate["best"]["tau"]
            )
        output.append(
            {
                **row,
                "selected_indices": sorted(selected),
                "selected_ids": [row["ctxs"][i]["id"] for i in sorted(selected)],
                "selection": audit,
            }
        )
    save_rows(args.output, output)


def stage_read(args, config):
    rows = rows_for(args)
    model = generator(config, "reader", args.device)
    for index, row in enumerate(rows):
        read_selected(row, model, config["generation"])
        if row.get("generation_error"):
            print(f"read failed for {row['id']}: {row['generation_error']}", flush=True)
        print(f"read {index + 1}/{len(rows)}", flush=True)
    save_rows(args.output, rows)
    if all(row.get("answers") for row in rows):
        save_json(str(args.output) + ".metrics.json", evaluate(rows))


def stage_evaluate(args, config):
    rows = rows_for(args)
    result = evaluate(rows)
    if args.compare:
        result["paired_comparison"] = paired_bootstrap(
            rows,
            load_rows(args.compare),
            config.get("bootstrap_seed", 2092),
            config["bootstrap_samples"],
        )
    save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def stage_diagnostics(args, config):
    bundle = read_config(args.bundle)
    gate = read_config(args.gate)["best"]
    save_json(args.output, gain_diagnostics(rows_for(args), bundle, gate))


def stage_check(args, config):
    from importlib.util import find_spec
    from importlib.metadata import version, PackageNotFoundError
    import sys

    result = {
        "dependencies": {
            name: find_spec(name) is not None
            for name in (
                "torch",
                "transformers",
                "numpy",
                "sklearn",
                "accelerate",
                "ijson",
                "pyarrow",
                "datasets",
                "vllm",
            )
        },
        "models": {
            name: {"path": path, "exists": Path(path).is_dir()}
            for name, path in config["models"].items()
        },
        "sources": {
            key: {
                field: [
                    {"path": path, "exists": Path(path).is_file()}
                    for path in specification[field]
                ]
                for field in ("train_sources", "eval_sources")
            }
            for key, specification in config["datasets"].items()
        },
    }
    result["python_version"] = sys.version
    result["package_versions"] = {}
    for name in (
        "torch",
        "transformers",
        "scikit-learn",
        "numpy",
        "accelerate",
        "datasets",
        "pyarrow",
        "vllm",
    ):
        try:
            result["package_versions"][name] = version(name)
        except PackageNotFoundError:
            result["package_versions"][name] = None
    if result["dependencies"]["torch"]:
        import torch

        result["torch_version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        result["gpu_count"] = torch.cuda.device_count()
    save_json(args.output, result)
    print(json.dumps(result, indent=2))


def stage_policy_answers(args, config):
    from .policies import attach_policy_answers

    rows = rows_for(args)
    model = generator(config, "reader", args.device)
    output = []
    for index, row in enumerate(rows):
        output.append(attach_policy_answers(row, model, config))
        print(f"policy-answers {index + 1}/{len(rows)}", flush=True)
    save_rows(args.output, output)


STAGES = {
    "prepare": stage_prepare,
    "bge": stage_bge,
    "draft": stage_draft,
    "teacher": stage_teacher,
    "train": stage_train,
    "score": stage_score,
    "gain-labels": stage_gain_labels,
    "policy-answers": stage_policy_answers,
    "fit-gain": stage_fit_gain,
    "tune": stage_tune,
    "select": stage_select,
    "read": stage_read,
    "evaluate": stage_evaluate,
    "diagnostics": stage_diagnostics,
    "check": stage_check,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", default=str(ROOT / "configs/paper.json"))
    parser.add_argument("--dataset", choices=DATASETS, default="hotpotqa")
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--bundle")
    parser.add_argument("--gate")
    parser.add_argument("--compare")
    parser.add_argument("--initial-model")
    parser.add_argument("--no-draft", action="store_true")
    parser.add_argument("--use-cached-scores", action="store_true")
    parser.add_argument(
        "--mode", choices=("conditional", "independent", "bge"), default="conditional"
    )
    parser.add_argument("--train-source", nargs="+")
    parser.add_argument("--eval-source", nargs="+")
    parser.add_argument("--exclude", nargs="*", default=[])
    args = parser.parse_args()
    config = load_config(args.config)
    output_path(args.output)
    STAGES[args.stage](args, config)
    if args.stage != "prepare":
        save_json(
            str(args.output) + ".provenance.json",
            {
                "stage": args.stage,
                "config": config,
                "input": args.input,
                "input_sha256": digest(args.input) if args.input else None,
                "arguments": vars(args),
            },
        )


if __name__ == "__main__":
    main()
