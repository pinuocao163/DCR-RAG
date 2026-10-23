import re
import string
from collections import Counter

import numpy as np


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(
        character for character in text if character not in string.punctuation
    )
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_metrics(prediction, aliases, dataset="hotpotqa"):
    prediction = normalize_answer(prediction)
    best_em, best_f1 = 0.0, 0.0
    for alias in aliases:
        truth = normalize_answer(alias)
        em = float(prediction == truth)
        if (
            dataset == "hotpotqa"
            and prediction != truth
            and ({prediction, truth} & {"yes", "no", "noanswer"})
        ):
            f1 = 0.0
        else:
            predicted, gold = prediction.split(), truth.split()
            common = sum((Counter(predicted) & Counter(gold)).values())
            f1 = 2 * common / (len(predicted) + len(gold)) if common else 0.0
        best_em, best_f1 = max(best_em, em), max(best_f1, f1)
    return {"em": best_em, "f1": best_f1}


def evaluate(rows):
    groups = {}
    for row in rows:
        dataset = row["dataset"]
        values = answer_metrics(row.get("prediction", ""), row["answers"], dataset)
        if row.get("generation_error"):
            values = {"em": 0.0, "f1": 0.0}
        groups.setdefault(dataset, []).append(values)
    if not groups:
        raise ValueError("Cannot evaluate an empty file")
    summary = {
        dataset: {
            "count": len(values),
            **{
                key: 100 * float(np.mean([item[key] for item in values]))
                for key in ("em", "f1")
            },
        }
        for dataset, values in groups.items()
    }
    summary["macro"] = {
        key: float(np.mean([item[key] for item in summary.values()]))
        for key in ("em", "f1")
    }
    return summary


def paired_bootstrap(first, second, seed=2092, samples=10000):
    first_count, second_count = len(first), len(second)
    first = {(row["dataset"], str(row["id"])): row for row in first}
    second = {(row["dataset"], str(row["id"])): row for row in second}
    if len(first) != first_count or len(second) != second_count:
        raise ValueError("Paired runs must not contain duplicate question IDs")
    if not first or first.keys() != second.keys():
        raise ValueError(
            "Paired runs must have identical nonempty dataset/question IDs"
        )
    by_dataset = {}
    for key, row in first.items():
        if (
            row["question"] != second[key]["question"]
            or row["answers"] != second[key]["answers"]
        ):
            raise ValueError("Paired records have mismatched questions or aliases")
        a = (
            answer_metrics(row.get("prediction", ""), row["answers"], key[0])["f1"]
            if not row.get("generation_error")
            else 0
        )
        b = (
            answer_metrics(second[key].get("prediction", ""), row["answers"], key[0])[
                "f1"
            ]
            if not second[key].get("generation_error")
            else 0
        )
        by_dataset.setdefault(key[0], []).append(a - b)
    return bootstrap_effects(by_dataset, seed, samples)


def bootstrap_effects(by_dataset, seed=2092, samples=10000):
    if (
        not by_dataset
        or samples <= 0
        or any(not len(values) for values in by_dataset.values())
    ):
        raise ValueError(
            "Bootstrap needs nonempty datasets and a positive sample count"
        )
    rng = np.random.default_rng(seed)
    bootstraps = np.zeros(samples)
    per_dataset = {}
    for dataset, values in by_dataset.items():
        values = np.asarray(values)
        distribution = (
            np.asarray(
                [
                    values[rng.integers(0, len(values), size=len(values))].mean()
                    for _ in range(samples)
                ]
            )
            * 100
        )
        bootstraps += distribution / len(by_dataset)
        per_dataset[dataset] = {
            "difference": float(values.mean() * 100),
            "ci95": np.quantile(distribution, [0.025, 0.975]).tolist(),
        }
    return {
        "per_dataset": per_dataset,
        "macro_difference": float(
            np.mean([np.mean(values) for values in by_dataset.values()]) * 100
        ),
        "macro_ci95": np.quantile(bootstraps, [0.025, 0.975]).tolist(),
        "bootstrap_samples": samples,
    }


def seed_averaged_paired_bootstrap(first_runs, second_runs, seed=2092, samples=10000):
    if not first_runs or len(first_runs) != len(second_runs):
        raise ValueError("Paired seed runs must be nonempty and have the same count")
    reference, effects, seed_effects = None, [], []
    for first_rows, second_rows in zip(first_runs, second_runs):
        first = {(row["dataset"], str(row["id"])): row for row in first_rows}
        second = {(row["dataset"], str(row["id"])): row for row in second_rows}
        if len(first) != len(first_rows) or len(second) != len(second_rows):
            raise ValueError("Duplicate question IDs within a seed run")
        if not first or first.keys() != second.keys():
            raise ValueError("Paired seed runs must contain identical questions")
        if reference is None:
            reference = first
        if first.keys() != reference.keys():
            raise ValueError("Seed runs must share the same retained questions")
        current = {}
        for key in reference:
            a, b, expected = first[key], second[key], reference[key]
            if any(
                row["question"] != expected["question"]
                or row["answers"] != expected["answers"]
                for row in (a, b)
            ):
                raise ValueError("Questions or aliases differ across seeds/routes")
            scores = [
                (
                    answer_metrics(
                        row.get("prediction", ""), row["answers"], row["dataset"]
                    )["f1"]
                    if not row.get("generation_error")
                    else 0.0
                )
                for row in (a, b)
            ]
            current[key] = scores[0] - scores[1]
        effects.append(current)
        per_dataset = {}
        for (dataset, row_id), value in current.items():
            per_dataset.setdefault(dataset, []).append(value)
        seed_effects.append(
            100 * float(np.mean([np.mean(values) for values in per_dataset.values()]))
        )
    by_dataset = {}
    for key in reference:
        by_dataset.setdefault(key[0], []).append(
            float(np.mean([values[key] for values in effects]))
        )
    result = bootstrap_effects(by_dataset, seed, samples)
    result.update(
        {
            "seed_runs": len(effects),
            "unique_questions": len(reference),
            "resampling_unit": "question after averaging paired effects across seeds",
            "macro_weighting": "equal dataset weight within every bootstrap replicate",
            "per_seed_macro_differences": seed_effects,
            "macro_difference_sample_sd": (
                float(np.std(seed_effects, ddof=1)) if len(effects) > 1 else None
            ),
            "per_dataset_question_counts": {
                dataset: len(values) for dataset, values in by_dataset.items()
            },
        }
    )
    return result
