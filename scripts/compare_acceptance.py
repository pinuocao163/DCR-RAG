import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from birrag.config import load_config
from birrag.data import assert_disjoint
from birrag.experiments import (
    aggregate_diagnostics,
    acceptance_rate,
    fit_acceptance_threshold,
    gain_predictions,
)
from birrag.gain import gain_diagnostics
from birrag.io import digest, load_rows, output_path, save_json, save_rows
from birrag.metrics import evaluate, paired_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run_root, output_root = Path(args.run_root).resolve(), output_path(args.output_root)
    config = load_config(run_root / "resolved_config.json")
    datasets = list(config["datasets"]) if args.dataset == "all" else [args.dataset]
    targets = config.get("controls", {}).get(
        "acceptance_targets", [0.2, 0.25, 0.317, 0.4, 0.5]
    )
    modes = ("full", "mean_only_matched", "kappa_zero", "tau_zero")
    diagnostics = {mode: {} for mode in modes}
    predictions = {mode: [] for mode in modes}
    curves = {
        str(target): {
            mode: {"predictions": [], "diagnostics": {}}
            for mode in ("full", "mean_only")
        }
        for target in targets
    }
    for dataset in datasets:
        source = run_root / dataset
        development, test = load_rows(source / "dev.gains.jsonl"), load_rows(
            source / "test.gains.jsonl"
        )
        assert_disjoint({"dev": development, "test": test})
        bundle_path, gate_path = source / "gain_ensemble.json", source / "gate.json"
        bundle, gate_file = json.loads(bundle_path.read_text()), json.loads(
            gate_path.read_text()
        )
        frozen = {"bundle": digest(bundle_path), "gate": digest(gate_path)}
        if gate_file["bundle_sha256"] != frozen["bundle"]:
            raise ValueError("Development gate differs from the frozen gain model")
        gate = gate_file["best"]
        target = acceptance_rate(development, bundle, gate)
        mean_only = fit_acceptance_threshold(development, bundle, 0.0, target)
        gates = {
            "full": gate,
            "mean_only_matched": mean_only,
            "kappa_zero": {"kappa": 0.0, "tau": gate["tau"]},
            "tau_zero": {"kappa": gate["kappa"], "tau": 0.0},
        }
        for mode, current in gates.items():
            destination = output_root / dataset / mode
            predicted = gain_predictions(test, bundle, current)
            diagnostic = gain_diagnostics(test, bundle, current)
            save_json(destination / "gate.json", current)
            save_json(destination / "diagnostics.json", diagnostic)
            save_rows(destination / "predictions.jsonl", predicted)
            diagnostics[mode][dataset] = diagnostic
            predictions[mode].extend(predicted)
        for target in targets:
            for mode, kappa in (("full", gate["kappa"]), ("mean_only", 0.0)):
                selected = fit_acceptance_threshold(development, bundle, kappa, target)
                destination = output_root / dataset / "targets" / str(target) / mode
                save_json(destination / "gate.json", selected)
                diagnostic = gain_diagnostics(test, bundle, selected)
                save_json(destination / "diagnostics.json", diagnostic)
                curves[str(target)][mode]["predictions"].extend(
                    gain_predictions(test, bundle, selected)
                )
                curves[str(target)][mode]["diagnostics"][dataset] = diagnostic
        if frozen != {"bundle": digest(bundle_path), "gate": digest(gate_path)}:
            raise ValueError(
                "Acceptance control changed trained models or main thresholds"
            )
        save_json(output_root / dataset / "frozen_artifacts.json", frozen)

    def summarize(values, per_dataset):
        accepted = sum(item["accepted"] for item in per_dataset.values())
        harmful = sum(item["harmful"] for item in per_dataset.values())
        return {
            "metrics": evaluate(values),
            "pooled_decisions": aggregate_diagnostics(list(per_dataset.values())),
            "accepted": accepted,
            "harmful": harmful,
            "harm_per_accepted": harmful / accepted if accepted else 0.0,
            "datasets": per_dataset,
            "pooled_initial_f1_strata": {
                key: aggregate_diagnostics(
                    [
                        item["initial_pair_quality_strata"][key]
                        for item in per_dataset.values()
                    ]
                )
                for key in next(iter(per_dataset.values()))[
                    "initial_pair_quality_strata"
                ]
            },
        }

    result = {mode: summarize(predictions[mode], diagnostics[mode]) for mode in modes}
    contrasts = {
        mode: paired_bootstrap(
            predictions["full"],
            predictions[mode],
            config["bootstrap_seed"],
            config["bootstrap_samples"],
        )
        for mode in modes
        if mode != "full"
    }
    curve_summary = {
        target: {
            mode: summarize(value["predictions"], value["diagnostics"])
            for mode, value in items.items()
        }
        for target, items in curves.items()
    }
    save_json(
        output_root / "comparison.json",
        {
            "results": result,
            "full_minus_control_paired_intervals": contrasts,
            "development_acceptance_targets": curve_summary,
            "test_used_for_threshold_selection": False,
            "gain_models_retrained": False,
        },
    )


if __name__ == "__main__":
    main()
