import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from birrag.budgets import budget_indices
from birrag.cli import generator
from birrag.config import load_config
from birrag.io import digest, load_rows, output_path, save_json, save_rows
from birrag.metrics import evaluate, paired_bootstrap
from birrag.reading import read_selected
from birrag.support import compare_support
from run_pipeline import artifact_checksums


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--dataset",
        default="musique",
        choices=("hotpotqa", "2wiki", "musique", "naturalqa", "triviaqa"),
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 3, 4])
    args = parser.parse_args()
    if not args.budgets or len(set(args.budgets)) != len(args.budgets):
        raise ValueError("Passage budgets must be nonempty and distinct")
    run_root = Path(args.run_root).resolve()
    source = run_root / args.dataset
    destination = output_path(args.output_root)
    destination.mkdir(parents=True, exist_ok=True)
    config = load_config(run_root / "resolved_config.json")
    scored_path = source / "test.scored.jsonl"
    selected_path = source / "test.conditional.predictions.jsonl"
    scored, selected = load_rows(scored_path), load_rows(selected_path)
    first = {str(row["id"]): row for row in scored}
    second = {str(row["id"]): row for row in selected}
    if (
        not first
        or len(first) != len(scored)
        or len(second) != len(selected)
        or first.keys() != second.keys()
    ):
        raise ValueError(
            "Frozen scored pools and main predictions must have identical unique question IDs"
        )
    for key in first:
        a, b = first[key], second[key]
        if (
            any(
                a.get(field) != b.get(field)
                for field in ("dataset", "question", "answers")
            )
            or a["dataset"] != args.dataset
        ):
            raise ValueError("Frozen prediction metadata differs from the scored pool")
        if [
            (item["id"], item.get("title", ""), item.get("text", ""))
            for item in a["ctxs"]
        ] != [
            (item["id"], item.get("title", ""), item.get("text", ""))
            for item in b["ctxs"]
        ]:
            raise ValueError(
                "Frozen predictions must use identical passage IDs, text and order"
            )
        for budget in args.budgets:
            budget_indices(a, b["selected_indices"], budget)
    paths = source / "scorer", source / "gain_ensemble.json", source / "gate.json"
    before = artifact_checksums(*paths)
    model = generator(config, "reader", None)
    seed, samples = config["bootstrap_seed"], config["bootstrap_samples"]
    results, predictions = {}, {}
    for budget in args.budgets:
        routes = {}
        for policy in ("dcrrag", "bge"):
            rows = []
            for row in scored:
                current = deepcopy(row)
                current.pop("reader_input", None)
                current["selected_indices"] = budget_indices(
                    row, second[str(row["id"])]["selected_indices"], budget, policy
                )
                current["selected_ids"] = [
                    row["ctxs"][index]["id"] for index in current["selected_indices"]
                ]
                current["passage_budget"] = budget
                current["budget_policy"] = policy
                rows.append(read_selected(current, model, config["generation"]))
            routes[policy] = rows
            save_rows(
                destination / f"{args.dataset}.k{budget}.{policy}.predictions.jsonl",
                rows,
            )
        predictions[budget] = routes
        results[str(budget)] = {
            "metrics": {policy: evaluate(rows) for policy, rows in routes.items()},
            "dcrrag_minus_bge_f1": paired_bootstrap(
                routes["dcrrag"], routes["bge"], seed, samples
            ),
            "support": {
                view: compare_support(
                    routes["dcrrag"], routes["bge"], seed, samples, view
                )
                for view in ("selected", "reader_visible", "reader_fully_visible")
            },
        }
        print(f"Finished {args.dataset} passage budget {budget}", flush=True)
    consecutive = {}
    for budget in sorted(predictions):
        if budget - 1 in predictions:
            consecutive[f"{budget}_minus_{budget - 1}"] = paired_bootstrap(
                predictions[budget]["dcrrag"],
                predictions[budget - 1]["dcrrag"],
                seed,
                samples,
            )
    after = artifact_checksums(*paths)
    if after != before:
        raise ValueError("Frozen main artifacts changed during budget comparison")
    save_json(
        destination / "comparison.json",
        {
            "dataset": args.dataset,
            "budgets": args.budgets,
            "results": results,
            "consecutive_dcrrag_f1_effects": consecutive,
            "reader_input_cap": config["generation"]["reader_input_cap"],
            "reader_output_cap": config["generation"]["reader_output_cap"],
            "selection_rule": "k=1: highest student score; k=2: frozen main pair; k>2: append highest remaining student scores; always display in BGE order",
            "scorer_retrained": False,
            "gain_models_retrained": False,
            "acceptance_retuned": False,
            "artifact_checksums_unchanged": before == after,
            "main_artifacts": before,
            "inputs": {
                str(path): digest(path)
                for path in (
                    scored_path,
                    selected_path,
                    run_root / "resolved_config.json",
                )
            },
        },
    )


if __name__ == "__main__":
    main()
