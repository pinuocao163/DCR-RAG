import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from birrag.config import load_config
from birrag.data import assert_disjoint
from birrag.io import digest, load_rows, output_path, save_json, save_rows
from birrag.features import draft_match_diagnostics
from birrag.gain import tune_shared_gate
from birrag.support import route_support_diagnostics


def run(command, config, output, **kwargs):
    arguments = [
        sys.executable,
        "-m",
        "birrag.cli",
        command,
        "--config",
        str(config),
        "--output",
        str(output),
    ]
    for name, value in kwargs.items():
        if value is not None:
            arguments.extend(["--" + name.replace("_", "-"), str(value)])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    environment["HF_HOME"] = str(ROOT / ".cache/huggingface")
    environment["TORCH_HOME"] = str(ROOT / ".cache/torch")
    environment["WANDB_DISABLED"] = "true"
    print("Running:", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, env=environment, check=True)


def artifact_checksums(checkpoint, bundle, gate):
    return {
        "scorer": {
            path.relative_to(checkpoint).as_posix(): digest(path)
            for path in sorted(checkpoint.rglob("*"))
            if path.is_file()
        },
        "gain_ensemble": digest(bundle),
        "gate": digest(gate),
    }


def evaluate_routes(config_path, target, split, bundle, gate):
    predictions_by_route = {}
    for mode in ("conditional", "independent", "bge"):
        selected = target / f"{split}.{mode}.selected.jsonl"
        predictions = target / f"{split}.{mode}.predictions.jsonl"
        run(
            "select",
            config_path,
            selected,
            input=target / f"{split}.scored.jsonl",
            mode=mode,
            bundle=bundle,
            gate=gate,
        )
        run("read", config_path, predictions, input=selected)
        predictions_by_route[mode] = load_rows(predictions)
    run(
        "evaluate",
        config_path,
        target
        / ("paired_effect.json" if split == "test" else f"{split}.paired_effect.json"),
        input=target / f"{split}.conditional.predictions.jsonl",
        compare=target / f"{split}.independent.predictions.jsonl",
    )
    save_json(
        target / f"{split}.draft_match_diagnostics.json",
        draft_match_diagnostics(predictions_by_route["conditional"]),
    )
    config = load_config(config_path)
    save_json(
        target / f"{split}.support_diagnostics.json",
        route_support_diagnostics(
            predictions_by_route,
            config.get("bootstrap_seed", 2092),
            config.get("bootstrap_samples", 10000),
        ),
    )
    return predictions_by_route


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/paper.json"))
    parser.add_argument(
        "--dataset",
        default="all",
        choices=("all", "hotpotqa", "2wiki", "musique", "naturalqa", "triviaqa"),
    )
    parser.add_argument("--output-root", default=str(ROOT / "runs/paper"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split-dir")
    parser.add_argument("--label-test-proposals", action="store_true")
    parser.add_argument("--shared-gate")
    args = parser.parse_args()
    root = output_path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    config_path = root / "resolved_config.json"
    save_json(config_path, config)
    datasets = list(config["datasets"]) if args.dataset == "all" else [args.dataset]
    all_predictions = {mode: [] for mode in ("conditional", "independent", "bge")}
    all_confirmation_predictions = {mode: [] for mode in all_predictions}
    for dataset in datasets:
        target = root / dataset
        target.mkdir(parents=True, exist_ok=True)
        split_dir = (
            Path(args.split_dir).resolve() if args.split_dir else target / "splits"
        )
        if args.split_dir and args.dataset == "all":
            split_dir = split_dir / dataset
        if not args.split_dir:
            run("prepare", config_path, split_dir, dataset=dataset)
        groups = {
            name: load_rows(split_dir / f"{name}.jsonl")
            for name in ("scorer_train", "gain_train", "dev", "test")
        }
        confirmation_count = config["datasets"][dataset].get(
            "confirmation_questions", 0
        )
        confirmation = (
            load_rows(split_dir / "confirmation.jsonl") if confirmation_count else []
        )
        if confirmation_count and len(confirmation) != confirmation_count:
            raise ValueError("Confirmation split does not match its configured size")
        assert_disjoint({**groups, "confirmation": confirmation})
        for split in groups:
            run(
                "bge",
                config_path,
                target / f"{split}.bge.jsonl",
                input=split_dir / f"{split}.jsonl",
                dataset=dataset,
            )
            run(
                "draft",
                config_path,
                target / f"{split}.draft.jsonl",
                input=target / f"{split}.bge.jsonl",
            )
        teacher = target / "scorer_train.teacher.jsonl"
        run("teacher", config_path, teacher, input=target / "scorer_train.draft.jsonl")
        checkpoint = target / "scorer"
        run("train", config_path, checkpoint, input=teacher)
        for split in ("gain_train", "dev", "test"):
            run(
                "score",
                config_path,
                target / f"{split}.scored.jsonl",
                input=target / f"{split}.draft.jsonl",
                checkpoint=checkpoint,
            )
        for split in ("gain_train", "dev"):
            run(
                "gain-labels",
                config_path,
                target / f"{split}.gains.jsonl",
                input=target / f"{split}.scored.jsonl",
            )
        bundle, gate = target / "gain_ensemble.json", target / "gate.json"
        run("fit-gain", config_path, bundle, input=target / "gain_train.gains.jsonl")
    if args.shared_gate:
        shared = json.loads(Path(args.shared_gate).read_text())
        if "best" not in shared:
            raise ValueError("The supplied shared gate must contain best parameters")
    elif len(datasets) == len(config["datasets"]):
        shared = tune_shared_gate(
            {
                dataset: load_rows(root / dataset / "dev.gains.jsonl")
                for dataset in datasets
            },
            {
                dataset: json.loads((root / dataset / "gain_ensemble.json").read_text())
                for dataset in datasets
            },
            config,
        )
    else:
        shared = {
            "best": dict(config["selector"]["paper_gate"]),
            "scope": "published_fixed",
            "selection_source": "paper",
            "test_used": False,
        }
    save_json(root / "shared_gate.json", shared)
    for dataset in datasets:
        target = root / dataset
        checkpoint, bundle, gate = (
            target / "scorer",
            target / "gain_ensemble.json",
            target / "gate.json",
        )
        save_json(
            gate,
            {
                **shared,
                "dataset": dataset,
                "shared_gate_sha256": digest(root / "shared_gate.json"),
                "bundle_sha256": digest(bundle),
            },
        )
        split_dir = (
            Path(args.split_dir).resolve() if args.split_dir else target / "splits"
        )
        if args.split_dir and args.dataset == "all":
            split_dir = split_dir / dataset
        confirmation_count = config["datasets"][dataset].get(
            "confirmation_questions", 0
        )
        confirmation = (
            load_rows(split_dir / "confirmation.jsonl") if confirmation_count else []
        )
        frozen = artifact_checksums(checkpoint, bundle, gate)
        save_json(target / "frozen_model_manifest.json", frozen)
        routes = evaluate_routes(config_path, target, "test", bundle, gate)
        for mode, predictions in routes.items():
            all_predictions[mode].extend(predictions)
        if args.label_test_proposals:
            run(
                "gain-labels",
                config_path,
                target / "test.gains.jsonl",
                input=target / "test.scored.jsonl",
            )
            run(
                "diagnostics",
                config_path,
                target / "test.gain_diagnostics.json",
                input=target / "test.gains.jsonl",
                bundle=bundle,
                gate=gate,
            )
        if confirmation:
            run(
                "bge",
                config_path,
                target / "confirmation.bge.jsonl",
                input=split_dir / "confirmation.jsonl",
                dataset=dataset,
            )
            run(
                "draft",
                config_path,
                target / "confirmation.draft.jsonl",
                input=target / "confirmation.bge.jsonl",
            )
            run(
                "score",
                config_path,
                target / "confirmation.scored.jsonl",
                input=target / "confirmation.draft.jsonl",
                checkpoint=checkpoint,
            )
            routes = evaluate_routes(config_path, target, "confirmation", bundle, gate)
            for mode, predictions in routes.items():
                all_confirmation_predictions[mode].extend(predictions)
            after = artifact_checksums(checkpoint, bundle, gate)
            if after != frozen:
                raise ValueError(
                    "Main trained artifacts changed during held-out evaluation"
                )
            save_json(
                target / "confirmation_protocol.json",
                {
                    "questions": len(confirmation),
                    "reuses_main_scorer": True,
                    "reuses_main_gain_ensemble": True,
                    "reuses_frozen_development_gate": True,
                    "retrained_on_confirmation": False,
                    "retuned_on_confirmation": False,
                    "artifact_checksums_unchanged": after == frozen,
                    "main_artifacts": frozen,
                },
            )
    from birrag.metrics import evaluate, paired_bootstrap

    for mode, rows in all_predictions.items():
        save_rows(root / f"test.{mode}.predictions.jsonl", rows)
    save_json(
        root / "support_diagnostics.json",
        route_support_diagnostics(
            all_predictions,
            config["bootstrap_seed"],
            config["bootstrap_samples"],
        ),
    )
    save_json(root / "metrics.json", evaluate(all_predictions["conditional"]))
    save_json(
        root / "route_metrics.json",
        {mode: evaluate(rows) for mode, rows in all_predictions.items()},
    )
    save_json(
        root / "paired_effect.json",
        paired_bootstrap(
            all_predictions["conditional"],
            all_predictions["independent"],
            config["bootstrap_seed"],
            config["bootstrap_samples"],
        ),
    )

    if all_confirmation_predictions["conditional"]:
        save_json(
            root / "confirmation_metrics.json",
            evaluate(all_confirmation_predictions["conditional"]),
        )
        save_json(
            root / "confirmation_route_metrics.json",
            {
                mode: evaluate(rows)
                for mode, rows in all_confirmation_predictions.items()
            },
        )
        save_json(
            root / "confirmation_paired_effect.json",
            paired_bootstrap(
                all_confirmation_predictions["conditional"],
                all_confirmation_predictions["independent"],
                config["bootstrap_seed"],
                config["bootstrap_samples"],
            ),
        )


if __name__ == "__main__":
    main()
