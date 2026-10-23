import argparse
from pathlib import Path

from run_pipeline import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    target, artifacts = Path(args.output_root).resolve(), Path(args.artifacts).resolve()
    config = (
        Path(args.config).resolve()
        if args.config
        else artifacts.parent / "resolved_config.json"
    )
    run(
        "bge",
        config,
        target / "pool.jsonl",
        input=Path(args.input).resolve(),
        dataset=args.dataset,
    )
    run("draft", config, target / "draft.jsonl", input=target / "pool.jsonl")
    run(
        "score",
        config,
        target / "scored.jsonl",
        input=target / "draft.jsonl",
        checkpoint=artifacts / "scorer",
    )
    run(
        "select",
        config,
        target / "selected.jsonl",
        input=target / "scored.jsonl",
        bundle=artifacts / "gain_ensemble.json",
        gate=artifacts / "gate.json",
    )
    run("read", config, target / "predictions.jsonl", input=target / "selected.jsonl")


if __name__ == "__main__":
    main()
