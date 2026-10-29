import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from birrag.config import load_config
from birrag.data import assert_disjoint
from birrag.experiments import gain_predictions
from birrag.gain import fit_gain_ensemble, gain_diagnostics, tune_shared_gate
from birrag.io import digest, load_rows, output_path, save_json, save_rows
from birrag.metrics import evaluate, paired_bootstrap
from run_pipeline import artifact_checksums


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--targets", nargs="+")
    args = parser.parse_args()
    source = Path(args.run_root).resolve()
    output = output_path(args.output_root)
    config = load_config(source / "resolved_config.json")
    datasets = list(config["datasets"])
    targets = args.targets or datasets
    if len(datasets) < 2 or not set(targets) <= set(datasets):
        raise ValueError("LODO requires multiple registered datasets and valid targets")
    aggregate, independent, fitted = [], [], []
    reports = {}
    for target in targets:
        root = source / target
        frozen = artifact_checksums(
            root / "scorer", root / "gain_ensemble.json", root / "gate.json"
        )
        training, development, hashes = [], {}, {}
        for dataset in datasets:
            if dataset == target:
                continue
            for split in ("gain_train", "dev"):
                path = source / dataset / f"{split}.gains.jsonl"
                rows = load_rows(path)
                if any(row["dataset"] != dataset for row in rows):
                    raise ValueError("Source dataset labels are inconsistent")
                hashes[str(path)] = digest(path)
                if split == "gain_train":
                    training.extend(rows)
                else:
                    development[dataset] = rows
        test_path = root / "test.gains.jsonl"
        test = load_rows(test_path)
        hashes[str(test_path)] = digest(test_path)
        if any(row["dataset"] != target for row in test):
            raise ValueError("Target test dataset is inconsistent")
        groups = {
            "gain_train": training,
            "development": [row for rows in development.values() for row in rows],
            "test": test,
        }
        assert_disjoint(
            {
                name: [{**row, "id": f"{row['dataset']}:{row['id']}"} for row in rows]
                for name, rows in groups.items()
            }
        )
        bundle = fit_gain_ensemble(training, config, target_dataset=target)
        gate = tune_shared_gate(
            development,
            {
                dataset: {**deepcopy(bundle), "dataset": dataset}
                for dataset in development
            },
            config,
        )
        destination = output / target
        save_json(destination / "gain_ensemble.json", bundle)
        gate["bundle_sha256"] = digest(destination / "gain_ensemble.json")
        save_json(destination / "gate.json", gate)
        predictions = gain_predictions(test, bundle, gate["best"])
        baseline = gain_predictions(test, bundle, gate["best"], initial_only=True)
        main_bundle = json.loads((root / "gain_ensemble.json").read_text())
        main_gate = json.loads((root / "gate.json").read_text())
        main = gain_predictions(test, main_bundle, main_gate["best"])
        save_rows(destination / "predictions.jsonl", predictions)
        report = {
            "metrics": evaluate(predictions),
            "lodo_minus_independent": paired_bootstrap(
                predictions,
                baseline,
                config["bootstrap_seed"],
                config["bootstrap_samples"],
            ),
            "lodo_minus_target_fitting": paired_bootstrap(
                predictions, main, config["bootstrap_seed"], config["bootstrap_samples"]
            ),
            "diagnostics": gain_diagnostics(test, bundle, gate["best"]),
            "input_checksums": hashes,
            "target_gain_training_used": False,
            "target_development_used": False,
            "target_scorer_retrained": False,
        }
        if (
            artifact_checksums(
                root / "scorer", root / "gain_ensemble.json", root / "gate.json"
            )
            != frozen
        ):
            raise ValueError("Target scorer or main selector artifacts changed")
        report["main_artifacts_unchanged"] = True
        save_json(destination / "evaluation.json", report)
        reports[target] = report
        aggregate.extend(predictions)
        independent.extend(baseline)
        fitted.extend(main)
    save_json(
        output / "comparison.json",
        {
            "datasets": reports,
            "metrics": evaluate(aggregate),
            "lodo_minus_independent": paired_bootstrap(
                aggregate,
                independent,
                config["bootstrap_seed"],
                config["bootstrap_samples"],
            ),
            "lodo_minus_target_fitting": paired_bootstrap(
                aggregate, fitted, config["bootstrap_seed"], config["bootstrap_samples"]
            ),
        },
    )


if __name__ == "__main__":
    main()
