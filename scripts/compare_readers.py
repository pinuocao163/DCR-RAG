import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from birrag.cli import generator
from birrag.config import load_config
from birrag.io import digest, load_rows, output_path, save_json, save_rows
from birrag.metrics import (
    answer_metrics,
    evaluate,
    paired_bootstrap,
    seed_averaged_paired_bootstrap,
    joint_paired_bootstrap,
)
from birrag.reading import read_selected
from run_pipeline import artifact_checksums


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-roots", nargs="+", required=True)
    parser.add_argument("--readers-config", required=True)
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    roots = [Path(value).resolve() for value in args.run_roots]
    configurations = [load_config(root / "resolved_config.json") for root in roots]
    seeds = [config["seed"] for config in configurations]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Reader transfer requires distinct paired training seeds")
    config = configurations[0]
    if any(
        any(
            current["generation"][field] != config["generation"][field]
            for field in ("reader_input_cap", "reader_output_cap")
        )
        for current in configurations
    ):
        raise ValueError(
            "Reader-transfer seeds must use identical input/output budgets"
        )
    datasets = list(config["datasets"]) if args.dataset == "all" else [args.dataset]
    readers = json.loads(Path(args.readers_config).read_text())
    if not readers or not all(
        isinstance(name, str)
        and name
        and name not in (".", "..")
        and "/" not in name
        and "\\" not in name
        for name in readers
    ):
        raise ValueError("Reader names must be nonempty directory names")
    destination = output_path(args.output_root)
    sources, frozen, manifests = [], {}, {}
    for root, current in zip(roots, configurations):
        routes = {mode: [] for mode in ("conditional", "independent")}
        for dataset in datasets:
            path = root / dataset
            key = f"{current['seed']}/{dataset}"
            frozen[key] = artifact_checksums(
                path / "scorer", path / "gain_ensemble.json", path / "gate.json"
            )
            for mode in routes:
                prediction_file = path / f"test.{mode}.predictions.jsonl"
                rows = load_rows(prediction_file)
                if any(row["dataset"] != dataset for row in rows):
                    raise ValueError("Reader-transfer prediction dataset mismatch")
                if any(
                    row.get("selected_ids")
                    != [row["ctxs"][index]["id"] for index in row["selected_indices"]]
                    for row in rows
                ):
                    raise ValueError(
                        "Frozen selected IDs differ from original passages"
                    )
                routes[mode].extend(rows)
                manifests[str(prediction_file)] = digest(prediction_file)
        paired_bootstrap(routes["conditional"], routes["independent"], samples=1)
        first = {(row["dataset"], str(row["id"])): row for row in routes["conditional"]}
        for row in routes["independent"]:
            other = first[row["dataset"], str(row["id"])]
            if [
                (item["id"], item.get("title", ""), item.get("text", ""))
                for item in row["ctxs"]
            ] != [
                (item["id"], item.get("title", ""), item.get("text", ""))
                for item in other["ctxs"]
            ]:
                raise ValueError(
                    "Reader transfer requires identical original candidate pools"
                )
        sources.append(routes)
    seed_averaged_paired_bootstrap(
        [routes["conditional"] for routes in sources],
        [routes["independent"] for routes in sources],
        samples=1,
    )
    summary = {}
    for name, specification in readers.items():
        current = deepcopy(config)
        model_path = Path(specification["model"]).expanduser()
        current["models"]["reader"] = str(
            model_path if model_path.is_absolute() else ROOT / model_path
        )
        current["generation"]["reader_backend"] = specification.get("backend", "hf")
        current["generation"]["reader_trust_remote_code"] = specification.get(
            "trust_remote_code", False
        )
        current["generation"]["reader_device_map"] = specification.get("device_map")
        model = generator(current, "reader", None)
        runs = {mode: [] for mode in ("conditional", "independent")}
        outcomes = []
        metrics = []
        for seed, routes in zip(seeds, sources):
            predicted = {}
            for mode, rows in routes.items():
                predicted[mode] = [
                    read_selected(deepcopy(row), model, current["generation"])
                    for row in rows
                ]
                save_rows(
                    destination / name / f"seed{seed}.{mode}.predictions.jsonl",
                    predicted[mode],
                )
                runs[mode].append(predicted[mode])
            counts = {"accepted": 0, "beneficial": 0, "unchanged": 0, "harmful": 0}
            independent = {
                (row["dataset"], str(row["id"])): row
                for row in predicted["independent"]
            }
            for row in predicted["conditional"]:
                baseline = independent[row["dataset"], str(row["id"])]
                if row["selected_indices"] == baseline["selected_indices"]:
                    continue
                counts["accepted"] += 1
                scores = [
                    (
                        answer_metrics(
                            value.get("prediction", ""),
                            value["answers"],
                            value["dataset"],
                        )["f1"]
                        if not value.get("generation_error")
                        else 0.0
                    )
                    for value in (row, baseline)
                ]
                key = (
                    "beneficial"
                    if scores[0] > scores[1]
                    else "harmful" if scores[0] < scores[1] else "unchanged"
                )
                counts[key] += 1
            outcomes.append(
                {
                    "seed": seed,
                    **counts,
                    "harm_per_accepted": (
                        counts["harmful"] / counts["accepted"]
                        if counts["accepted"]
                        else 0.0
                    ),
                    "acceptance_rate": counts["accepted"]
                    / len(predicted["conditional"]),
                }
            )
            metrics.append(
                {
                    "seed": seed,
                    **{mode: evaluate(rows) for mode, rows in predicted.items()},
                }
            )
        summary[name] = {
            "per_seed_metrics": metrics,
            "accepted_swap_outcomes": outcomes,
            "seed_averaged_paired_effect": seed_averaged_paired_bootstrap(
                runs["conditional"],
                runs["independent"],
                config["bootstrap_seed"],
                config["bootstrap_samples"],
            ),
            "joint_paired_effect": joint_paired_bootstrap(
                runs["conditional"],
                runs["independent"],
                config["bootstrap_seed"],
                config["bootstrap_samples"],
            ),
        }
        del model
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for root, current in zip(roots, configurations):
        for dataset in datasets:
            path = root / dataset
            if (
                artifact_checksums(
                    path / "scorer", path / "gain_ensemble.json", path / "gate.json"
                )
                != frozen[f"{current['seed']}/{dataset}"]
            ):
                raise ValueError(
                    "Main selector artifacts changed during reader transfer"
                )
    save_json(
        destination / "comparison.json",
        {
            "readers": summary,
            "seeds": seeds,
            "selector_refitted": False,
            "acceptance_retuned": False,
            "selected_ids_frozen": True,
            "artifact_checksums_unchanged": True,
            "reader_input_cap": config["generation"]["reader_input_cap"],
            "reader_output_cap": config["generation"]["reader_output_cap"],
            "input_checksums": manifests,
        },
    )


if __name__ == "__main__":
    main()
