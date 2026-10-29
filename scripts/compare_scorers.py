import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from birrag.config import load_config
from birrag.io import load_rows, output_path, save_json
from birrag.metrics import answer_metrics, bootstrap_effects, evaluate, paired_bootstrap
from run_pipeline import artifact_checksums, evaluate_routes, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()
    source, output = Path(args.run_root).resolve(), output_path(args.output_root)
    config_path = source / "resolved_config.json"
    config = load_config(config_path)
    datasets = list(config["datasets"]) if args.dataset == "all" else [args.dataset]
    routes = {
        name: []
        for name in (
            "native_independent",
            "native_conditional",
            "learned_independent",
            "learned_conditional",
        )
    }
    for dataset in datasets:
        main = source / dataset
        frozen = artifact_checksums(
            main / "scorer", main / "gain_ensemble.json", main / "gate.json"
        )
        target = output / dataset
        target.mkdir(parents=True, exist_ok=True)
        run(
            "score",
            config_path,
            target / "test.scored.jsonl",
            input=main / "test.draft.jsonl",
            checkpoint=config["models"]["bge"],
            native_head=True,
        )
        native = evaluate_routes(
            config_path, target, "test", main / "gain_ensemble.json", main / "gate.json"
        )
        for mode in ("independent", "conditional"):
            routes["native_" + mode].extend(native[mode])
            routes["learned_" + mode].extend(
                load_rows(main / f"test.{mode}.predictions.jsonl")
            )
        if (
            artifact_checksums(
                main / "scorer", main / "gain_ensemble.json", main / "gate.json"
            )
            != frozen
        ):
            raise ValueError("Scorer comparison changed the main gain models or gate")
    contrasts = {
        "scorer_independent": ("learned_independent", "native_independent"),
        "scorer_conditional": ("learned_conditional", "native_conditional"),
        "selector_native": ("native_conditional", "native_independent"),
        "selector_learned": ("learned_conditional", "learned_independent"),
    }
    effects = {
        name: paired_bootstrap(
            routes[first],
            routes[second],
            config["bootstrap_seed"],
            config["bootstrap_samples"],
        )
        for name, (first, second) in contrasts.items()
    }
    indexed = {
        mode: {(row["dataset"], str(row["id"])): row for row in rows}
        for mode, rows in routes.items()
    }
    interaction = {}
    for key in indexed["learned_conditional"]:
        values = {
            mode: (
                answer_metrics(
                    rows[key].get("prediction", ""), rows[key]["answers"], key[0]
                )["f1"]
                if not rows[key].get("generation_error")
                else 0.0
            )
            for mode, rows in indexed.items()
        }
        interaction.setdefault(key[0], []).append(
            values["learned_conditional"]
            - values["learned_independent"]
            - values["native_conditional"]
            + values["native_independent"]
        )
    effects["interaction"] = bootstrap_effects(
        interaction, config["bootstrap_seed"], config["bootstrap_samples"]
    )
    save_json(
        output / "comparison.json",
        {
            "metrics": {mode: evaluate(rows) for mode, rows in routes.items()},
            "effects": effects,
            "gain_predictors_refitted": False,
            "acceptance_retuned": False,
            "native_scorer": "unchanged BGE classification head with question and the same cached pool draft",
        },
    )


if __name__ == "__main__":
    main()
