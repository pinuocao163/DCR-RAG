import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from birrag.config import load_config
from birrag.data import assert_disjoint
from birrag.experiments import gain_predictions, validate_development
from birrag.gain import gain_diagnostics
from birrag.io import digest, load_rows, output_path, save_json, save_rows
from birrag.metrics import evaluate, paired_bootstrap
from birrag.policies import POLICIES, policy_gain_rows, tune_heuristic
from run_pipeline import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run_root, output_root = Path(args.run_root).resolve(), output_path(args.output_root)
    config_path = run_root / "resolved_config.json"
    config = load_config(config_path)
    datasets = list(config["datasets"]) if args.dataset == "all" else [args.dataset]
    development, testing, artifacts, manifests = {}, {}, {}, {}
    for dataset in datasets:
        source = run_root / dataset
        bundle_path, gate_path = source / "gain_ensemble.json", source / "gate.json"
        bundle, gate = json.loads(bundle_path.read_text()), json.loads(
            gate_path.read_text()
        )
        if bundle.get("formulation", "direct") != "direct":
            raise ValueError(
                "Fixed-policy transfer requires the frozen main Direct Gain ensemble"
            )
        frozen = {"bundle": digest(bundle_path), "gate": digest(gate_path)}
        if gate["bundle_sha256"] != frozen["bundle"]:
            raise ValueError("Gate was not selected for the supplied frozen ensemble")
        for split, container in (("dev", development), ("test", testing)):
            source_path = source / f"{split}.gains.jsonl"
            cache = output_root / dataset / f"{split}.pair_answers.jsonl"
            run("policy-answers", config_path, cache, input=source_path)
            container[dataset] = load_rows(cache)
        assert_disjoint({"development": development[dataset], "test": testing[dataset]})
        validate_development(development[dataset], bundle)
        artifacts[dataset] = (bundle, gate["best"])
        manifests[dataset] = (bundle_path, gate_path, frozen)
    grid = config.get("controls", {}).get("heuristic_grid", [0, 0.25, 0.5, 0.75, 1])
    replacements = config["selector"]["replacements_per_position"]
    results, predictions = {}, {}
    for policy in POLICIES:
        selected = tune_heuristic(development, policy, grid, replacements)
        save_json(output_root / policy / "development_policy.json", selected)
        predictions[policy] = {"initial": [], "after": []}
        per_dataset = {}
        for dataset in datasets:
            bundle, gate = artifacts[dataset]
            rows, baselines = policy_gain_rows(
                testing[dataset], policy, selected["best"]["parameters"], replacements
            )
            initial = gain_predictions(rows, bundle, gate, baselines, initial_only=True)
            after = gain_predictions(rows, bundle, gate, baselines)
            diagnostic = gain_diagnostics(rows, bundle, gate, baselines)
            predictions[policy]["initial"].extend(initial)
            predictions[policy]["after"].extend(after)
            per_dataset[dataset] = diagnostic
            destination = output_root / policy / dataset
            save_rows(destination / "initial.predictions.jsonl", initial)
            save_rows(destination / "after.predictions.jsonl", after)
            save_json(destination / "diagnostics.json", diagnostic)
        initial_metrics = evaluate(predictions[policy]["initial"])
        after_metrics = evaluate(predictions[policy]["after"])
        results[policy] = {
            "initial": initial_metrics,
            "after": after_metrics,
            "macro_f1_gain": after_metrics["macro"]["f1"]
            - initial_metrics["macro"]["f1"],
            "macro_gain_mae": (
                float(
                    np.mean(
                        [
                            value["question_mean_gain_mae"]
                            for value in per_dataset.values()
                            if value["question_mean_gain_mae"] is not None
                        ]
                    )
                )
                if any(
                    value["question_mean_gain_mae"] is not None
                    for value in per_dataset.values()
                )
                else None
            ),
            "within_policy_paired_interval": paired_bootstrap(
                predictions[policy]["after"],
                predictions[policy]["initial"],
                config["bootstrap_seed"],
                config["bootstrap_samples"],
            ),
            "datasets": per_dataset,
        }
    for policy in POLICIES:
        results[policy]["main_direct_minus_initial_paired_interval"] = paired_bootstrap(
            predictions["top2"]["after"],
            predictions[policy]["initial"],
            config["bootstrap_seed"],
            config["bootstrap_samples"],
        )
    for dataset, (bundle_path, gate_path, frozen) in manifests.items():
        if frozen != {"bundle": digest(bundle_path), "gate": digest(gate_path)}:
            raise ValueError(
                "Policy comparison altered frozen main predictors or acceptance settings"
            )
        save_json(output_root / dataset / "frozen_artifacts.json", frozen)
    save_json(
        output_root / "comparison.json",
        {
            "results": results,
            "gain_models_retrained": False,
            "gain_preprocessing_or_gate_retuned": False,
            "heuristics_use_gain_training_labels": False,
            "test_used_for_heuristic_tuning": False,
        },
    )


if __name__ == "__main__":
    main()
