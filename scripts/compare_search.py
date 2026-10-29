import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from birrag.cli import generator
from birrag.config import load_config
from birrag.features import proposals
from birrag.gain import choose_pair
from birrag.io import digest, load_rows, output_path, save_json, save_rows
from birrag.metrics import answer_metrics, evaluate, paired_bootstrap
from birrag.reading import read_selected
from birrag.support import support_diagnostics
from run_pipeline import artifact_checksums


def search_candidates(row, scope):
    baseline, small = proposals(row, 4)
    if scope == "8":
        return small
    if scope == "16":
        return proposals(row, 8)[1]
    if scope == "all":
        return [
            pair
            for pair in itertools.combinations(range(len(row["ctxs"])), 2)
            if pair != baseline
        ]
    raise ValueError("Unknown proposal search scope")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()
    source, output = Path(args.run_root).resolve(), output_path(args.output_root)
    config = load_config(source / "resolved_config.json")
    datasets = list(config["datasets"]) if args.dataset == "all" else [args.dataset]
    model = generator(config, "reader", None)
    predictions = {scope: [] for scope in ("8", "16", "all")}
    oracle_predictions, costs, checksums = [], [], {}
    for dataset in datasets:
        target = source / dataset
        bundle = json.loads((target / "gain_ensemble.json").read_text())
        gate = json.loads((target / "gate.json").read_text())["best"]
        frozen = artifact_checksums(
            target / "scorer", target / "gain_ensemble.json", target / "gate.json"
        )
        path = target / "test.scored.jsonl"
        checksums[str(path)] = digest(path)
        for row in load_rows(path):
            if row["dataset"] != dataset:
                raise ValueError("Search dataset differs from fitted artifacts")
            cache, audits = {}, {}

            def answer(pair):
                pair = tuple(pair)
                if pair not in cache:
                    selected = deepcopy(row)
                    selected["selected_indices"] = list(pair)
                    selected["selected_ids"] = [
                        row["ctxs"][index]["id"] for index in pair
                    ]
                    cache[pair] = read_selected(selected, model, config["generation"])
                return deepcopy(cache[pair])

            for scope in predictions:
                pairs = search_candidates(row, scope)
                selected, audit = choose_pair(
                    row, bundle, gate["kappa"], gate["tau"], candidates=pairs
                )
                audits[scope] = audit
                predicted = answer(selected)
                predicted["selection"] = audit
                predicted["search_scope"] = scope
                predictions[scope].append(predicted)
            best_small = max(
                (item["adjusted_gain"] for item in audits["8"]["proposals"]),
                default=None,
            )
            best_all = max(
                (item["adjusted_gain"] for item in audits["all"]["proposals"]),
                default=None,
            )
            costs.append(
                {
                    "dataset": dataset,
                    "id": row["id"],
                    "pairs": {
                        scope: len(audit["proposals"])
                        for scope, audit in audits.items()
                    },
                    "eight_contains_exhaustive_maximum": (
                        bool(np.isclose(best_small, best_all, rtol=1e-12, atol=1e-12))
                        if best_small is not None and best_all is not None
                        else True
                    ),
                }
            )
            if args.oracle:
                baseline = proposals(row)[0]
                pairs = [baseline, *search_candidates(row, "all")]
                values = [answer(pair) for pair in pairs]
                scores = [
                    (
                        answer_metrics(
                            value.get("prediction", ""), value["answers"], dataset
                        )["f1"]
                        if not value.get("generation_error")
                        else 0.0
                    )
                    for value in values
                ]
                oracle = values[int(np.argmax(scores))]
                oracle["offline_oracle"] = True
                oracle_predictions.append(oracle)
        if (
            artifact_checksums(
                target / "scorer", target / "gain_ensemble.json", target / "gate.json"
            )
            != frozen
        ):
            raise ValueError("Search comparison modified the main artifacts")
    results = {}
    for scope, rows in predictions.items():
        save_rows(output / f"search{scope}.predictions.jsonl", rows)
        results[scope] = {
            "metrics": evaluate(rows),
            "mean_evaluated_pairs": float(
                np.mean([row["pairs"][scope] for row in costs])
            ),
            "support": {
                view: support_diagnostics(rows, view)
                for view in ("selected", "reader_visible", "reader_fully_visible")
            },
        }
    report = {
        "results": results,
        "questions": len(costs),
        "eight_contains_exhaustive_maximum_rate": float(
            np.mean([row["eight_contains_exhaustive_maximum"] for row in costs])
        ),
        "paired_effects_against_eight": {
            scope: paired_bootstrap(
                rows,
                predictions["8"],
                config["bootstrap_seed"],
                config["bootstrap_samples"],
            )
            for scope, rows in predictions.items()
            if scope != "8"
        },
        "models_refitted": False,
        "acceptance_retuned": False,
        "input_checksums": checksums,
    }
    if args.oracle:
        save_rows(output / "oracle.predictions.jsonl", oracle_predictions)
        report["offline_oracle"] = evaluate(oracle_predictions)
    save_json(output / "comparison.json", report)


if __name__ == "__main__":
    main()
