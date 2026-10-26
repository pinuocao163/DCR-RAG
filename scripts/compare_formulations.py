import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from birrag.config import load_config
from birrag.data import assert_disjoint
from birrag.gain import (
    FORMULATIONS,
    fit_gain_ensemble,
    gain_diagnostics,
    tune_shared_gate,
)
from birrag.io import digest, load_rows, output_path, save_json, save_rows
from birrag.experiments import aggregate_diagnostics, gain_predictions
from birrag.metrics import paired_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--formulations", nargs="+", choices=FORMULATIONS)
    parser.add_argument("--mlp-config")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    config_path = (
        Path(args.config) if args.config else run_root / "resolved_config.json"
    )
    config = load_config(config_path)
    if args.mlp_config:
        config.setdefault("controls", {})["mlp"] = json.loads(
            Path(args.mlp_config).read_text()
        )
    args.formulations = args.formulations or [
        name
        for name in FORMULATIONS
        if name != "mlp" or config.get("controls", {}).get("mlp")
    ]
    output_root = output_path(args.output_root)
    datasets = list(config["datasets"]) if args.dataset == "all" else [args.dataset]
    results = {formulation: {} for formulation in args.formulations}
    predictions = {formulation: [] for formulation in args.formulations}
    grouped = {}
    for dataset in datasets:
        paths = {
            split: run_root / dataset / f"{split}.gains.jsonl"
            for split in ("gain_train", "dev", "test")
        }
        grouped[dataset] = {split: load_rows(path) for split, path in paths.items()}
        assert_disjoint(grouped[dataset])
        save_json(
            output_root / dataset / "input_manifest.json",
            {
                split: {
                    "path": str(path),
                    "sha256": digest(path),
                    "questions": len(grouped[dataset][split]),
                }
                for split, path in paths.items()
            },
        )
    for formulation in args.formulations:
        current = copy.deepcopy(config)
        current["selector"]["formulation"] = formulation
        bundles = {
            dataset: fit_gain_ensemble(grouped[dataset]["gain_train"], current)
            for dataset in datasets
        }
        gate = tune_shared_gate(
            {dataset: grouped[dataset]["dev"] for dataset in datasets}, bundles, current
        )
        save_json(output_root / formulation / "shared_gate.json", gate)
        for dataset in datasets:
            destination = output_root / dataset / formulation
            save_json(destination / "gain_ensemble.json", bundles[dataset])
            dataset_gate = {
                **gate,
                "dataset": dataset,
                "bundle_sha256": digest(destination / "gain_ensemble.json"),
            }
            save_json(destination / "gate.json", dataset_gate)
            diagnostics = gain_diagnostics(
                grouped[dataset]["test"], bundles[dataset], gate["best"]
            )
            save_json(destination / "diagnostics.json", diagnostics)
            results[formulation][dataset] = diagnostics
            predicted = gain_predictions(
                grouped[dataset]["test"], bundles[dataset], gate["best"]
            )
            save_rows(destination / "predictions.jsonl", predicted)
            predictions[formulation].extend(predicted)
    macro = {}
    for formulation, per_dataset in results.items():
        values = list(per_dataset.values())
        accepted = sum(item["accepted"] for item in values)
        harmed = sum(item["harmful"] for item in values)
        errors = [
            item["question_mean_gain_mae"]
            for item in values
            if item["question_mean_gain_mae"] is not None
        ]
        macro[formulation] = {
            "macro_f1": 100 * float(np.mean([item["mean_f1"] for item in values])),
            "macro_mae": float(np.mean(errors)) if errors else None,
            "accepted": accepted,
            "harmful": harmed,
            "harm_per_accepted": harmed / accepted if accepted else 0.0,
            "acceptance_rate": accepted / sum(item["questions"] for item in values),
            "macro_independent_f1": 100
            * float(
                np.mean(
                    [
                        np.mean(
                            [row["baseline_f1"] for row in grouped[dataset]["test"]]
                        )
                        for dataset in per_dataset
                    ]
                )
            ),
            "datasets": per_dataset,
            "pooled_initial_f1_strata": {
                key: aggregate_diagnostics(
                    [item["initial_pair_quality_strata"][key] for item in values]
                )
                for key in values[0]["initial_pair_quality_strata"]
            },
        }
    contrasts = {}
    if "direct" in predictions:
        contrasts = {
            name: paired_bootstrap(
                predictions["direct"],
                values,
                config["bootstrap_seed"],
                config["bootstrap_samples"],
            )
            for name, values in predictions.items()
            if name != "direct"
        }
    save_json(
        output_root / "comparison.json",
        {
            "results": macro,
            "direct_minus_alternative_paired_intervals": contrasts,
            "same_questions_proposals_and_reader_outputs": True,
            "test_used_for_fitting_or_tuning": False,
            "zero_is_prediction_only_in_paper": True,
        },
    )
    print(json.dumps(macro, indent=2))


if __name__ == "__main__":
    main()
