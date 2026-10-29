import math

import numpy as np

from .data import assert_disjoint, question_key
from .gain import choose_pair


def validate_development(rows, bundle):
    if not rows:
        raise ValueError("Development is empty")
    assert_disjoint({"development": rows})
    ids = set(map(str, bundle["training_question_ids"]))
    keys = {tuple(key) for key in bundle.get("training_question_keys", [])}
    questions = {question_key(text) for text in bundle["training_question_texts"]}
    if any(
        ((row["dataset"], str(row["id"])) in keys if keys else str(row["id"]) in ids)
        or question_key(row["question"]) in questions
        for row in rows
    ):
        raise ValueError("Development overlaps gain training")


def fit_acceptance_threshold(rows, bundle, kappa, target):
    if not 0 <= target <= 1 or kappa < 0:
        raise ValueError("Invalid acceptance target or disagreement coefficient")
    validate_development(rows, bundle)
    scores = []
    for row in rows:
        _, audit = choose_pair(row, bundle, kappa, 0.0)
        scores.append(
            max(
                (item["adjusted_gain"] for item in audit["proposals"]),
                default=-math.inf,
            )
        )
    values = sorted(set(value for value in scores if math.isfinite(value)))
    thresholds = (
        [float(np.nextafter(values[0], -math.inf)), *values] if values else [0.0]
    )
    candidates = [
        {"tau": tau, "accepted": sum(value > tau for value in scores)}
        for tau in thresholds
    ]
    best = min(
        candidates,
        key=lambda item: (
            abs(item["accepted"] - target * len(rows)),
            item["accepted"],
            -item["tau"],
        ),
    )
    return {
        "kappa": float(kappa),
        "tau": best["tau"],
        "target_acceptance_rate": float(target),
        "development_acceptance_rate": best["accepted"] / len(rows),
        "development_question_ids": [row["id"] for row in rows],
        "test_used_for_selection": False,
        "uses_gold_for_threshold_matching": False,
    }


def acceptance_rate(rows, bundle, gate):
    return sum(
        choose_pair(row, bundle, gate["kappa"], gate["tau"])[1]["accepted"]
        for row in rows
    ) / len(rows)


def gain_predictions(rows, bundle, gate, baselines=None, initial_only=False):
    output = []
    for row in rows:
        baseline = tuple(row["baseline_indices"])
        if initial_only:
            selected, audit = baseline, {"accepted": False}
        else:
            selected, audit = choose_pair(
                row,
                bundle,
                gate["kappa"],
                gate["tau"],
                None if baselines is None else baselines[str(row["id"])],
            )
        selected_label = (
            None
            if tuple(selected) == baseline
            else next(
                label
                for label in row["gain_labels"]
                if tuple(label["indices"]) == tuple(selected)
            )
        )
        response = (
            row["baseline_answer"]
            if selected_label is None
            else selected_label["answer"]
        )
        error = (
            row.get("baseline_generation_error")
            if selected_label is None
            else selected_label.get("generation_error")
        )
        output.append(
            {
                "id": row["id"],
                "question": row["question"],
                "dataset": row["dataset"],
                "answers": row["answers"],
                "prediction": response,
                "generation_error": error,
                "selected_indices": list(selected),
                "selection": audit,
            }
        )
    return output


def aggregate_diagnostics(items):
    fields = (
        "questions",
        "questions_with_proposals",
        "accepted",
        "harmful",
        "beneficial",
        "accepted_unchanged",
        "changed",
        "proposal_count",
        "proposal_absolute_error_sum",
    )
    result = {field: sum(item.get(field, 0) for item in items) for field in fields}
    result["question_mean_gain_mae"] = (
        sum(
            item["question_mean_gain_mae"] * item["questions_with_proposals"]
            for item in items
            if item["question_mean_gain_mae"] is not None
        )
        / result["questions_with_proposals"]
        if result["questions_with_proposals"]
        else None
    )
    result["proposal_weighted_gain_mae"] = (
        result["proposal_absolute_error_sum"] / result["proposal_count"]
        if result["proposal_count"]
        else None
    )
    result["mean_f1"] = (
        sum(
            item["mean_f1"] * item["questions"]
            for item in items
            if item["mean_f1"] is not None
        )
        / result["questions"]
        if result["questions"]
        else None
    )
    for field, count_field in (
        ("mean_positive_gain", "beneficial"),
        ("mean_negative_gain", "harmful"),
    ):
        result[field] = (
            sum(
                item[field] * item[count_field]
                for item in items
                if item.get(field) is not None
            )
            / result[count_field]
            if result[count_field]
            else None
        )
    result["acceptance_rate"] = result["accepted"] / max(1, result["questions"])
    result["harm_per_accepted"] = result["harmful"] / max(1, result["accepted"])
    result["harm_per_question"] = result["harmful"] / max(1, result["questions"])
    return result
