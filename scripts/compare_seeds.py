import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from birrag.io import digest, load_rows, save_json
from birrag.metrics import seed_averaged_paired_bootstrap, joint_paired_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", nargs="+", required=True)
    parser.add_argument("--second", nargs="+", required=True)
    parser.add_argument("--seed-ids", nargs="+", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=2092)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--resampling", choices=("seed_average", "joint"), default="seed_average"
    )
    args = parser.parse_args()
    if len(args.first) != len(args.second):
        raise ValueError("Provide one prediction file per route for each paired seed")
    if args.seed_ids and (
        len(args.seed_ids) != len(args.first)
        or len(set(args.seed_ids)) != len(args.seed_ids)
    ):
        raise ValueError("Seed IDs must be distinct and correspond to paired files")
    first, second = [load_rows(path) for path in args.first], [
        load_rows(path) for path in args.second
    ]
    compare = (
        joint_paired_bootstrap
        if args.resampling == "joint"
        else seed_averaged_paired_bootstrap
    )
    result = compare(first, second, args.bootstrap_seed, args.bootstrap_samples)
    result["input_pairs"] = [
        {
            "seed_id": args.seed_ids[index] if args.seed_ids else index,
            "first": str(Path(a).resolve()),
            "first_sha256": digest(a),
            "second": str(Path(b).resolve()),
            "second_sha256": digest(b),
        }
        for index, (a, b) in enumerate(zip(args.first, args.second))
    ]
    save_json(args.output, result)


if __name__ == "__main__":
    main()
