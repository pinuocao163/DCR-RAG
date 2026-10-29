import itertools

import numpy as np

from .data import assert_disjoint
from .features import FEATURE_NAMES, feature_vector, initial_pair, proposals
from .metrics import answer_metrics
from .reading import generate_pair

POLICIES = ("top2", "mmr", "coverage", "combined")


def heuristic_candidates(row, replacements=4):
    baseline, alternatives = proposals(row, replacements)
    return baseline, [baseline, *sorted(alternatives)]


def heuristic_terms(row, pair):
    vector = dict(zip(FEATURE_NAMES, feature_vector(row, pair)))
    return {
        "A": vector["student_mean"],
        "J": vector["passage_jaccard"],
        "C": vector["question_union_coverage"],
        "R": (
            vector["residual_first_given_second"]
            + vector["residual_second_given_first"]
        )
        / 2,
    }


def heuristic_score(terms, policy, parameters):
    if policy == "mmr":
        return terms["A"] - parameters["lambda"] * terms["J"]
    if policy == "coverage":
        return terms["A"] + parameters["beta"] * terms["C"]
    if policy == "combined":
        return (
            terms["A"]
            + parameters["beta"] * terms["C"]
            + parameters["gamma"] * terms["R"]
            - parameters["delta"] * terms["J"]
        )
    if policy == "top2":
        return terms["A"]
    raise ValueError(f"Unknown initial-pair policy: {policy}")


def choose_heuristic(row, policy, parameters, replacements=4):
    baseline, candidates = heuristic_candidates(row, replacements)
    best = baseline
    best_score = heuristic_score(heuristic_terms(row, baseline), policy, parameters)
    for pair in candidates[1:]:
        score = heuristic_score(heuristic_terms(row, pair), policy, parameters)
        if score > best_score:
            best, best_score = pair, score
    return best


def parameter_grid(policy, grid):
    if policy == "top2":
        return [{}]
    names = {
        "mmr": ("lambda",),
        "coverage": ("beta",),
        "combined": ("beta", "gamma", "delta"),
    }[policy]
    return [
        dict(zip(names, values))
        for values in itertools.product(grid, repeat=len(names))
    ]


def answer_map(row):
    return {tuple(item["indices"]): item for item in row["pair_answers"]}


def policy_answer_pairs(row, replacements=4):
    _, candidates = heuristic_candidates(row, replacements)
    pairs = set(candidates)
    for baseline in candidates:
        pairs.update(proposals(row, replacements, baseline)[1])
    return sorted(pairs)


def attach_policy_answers(row, reader, config):
    cached = {}
    if "baseline_answer" in row:
        cached[tuple(row["baseline_indices"])] = (
            row["baseline_answer"],
            row.get("baseline_generation_error"),
        )
    for label in row.get("gain_labels", []):
        cached[tuple(label["indices"])] = (
            label["answer"],
            label.get("generation_error"),
        )
    answers = []
    for pair in policy_answer_pairs(
        row, config["selector"]["replacements_per_position"]
    ):
        response, error = (
            cached[pair]
            if pair in cached
            else generate_pair(row, reader, pair, config["generation"])
        )
        metrics = (
            answer_metrics(response, row["answers"], row["dataset"])
            if error is None
            else {"em": 0.0, "f1": 0.0}
        )
        answers.append(
            {
                "indices": list(pair),
                "answer": response,
                "generation_error": error,
                **metrics,
            }
        )
    return {**row, "pair_answers": answers}


def tune_heuristic(development, policy, grid, replacements=4):
    if not development or any(not rows for rows in development.values()):
        raise ValueError("Each policy-development dataset must be nonempty")
    for dataset, rows in development.items():
        assert_disjoint({dataset: rows})
    prepared = {}
    for dataset, rows in development.items():
        prepared[dataset] = []
        for row in rows:
            baseline, candidates = heuristic_candidates(row, replacements)
            answers = answer_map(row)
            prepared[dataset].append(
                (
                    baseline,
                    candidates,
                    [heuristic_terms(row, pair) for pair in candidates],
                    answers,
                )
            )
    table = []
    for parameters in parameter_grid(policy, grid):
        dataset_f1, changes = [], 0
        for questions in prepared.values():
            values = []
            for baseline, candidates, terms, answers in questions:
                scores = [heuristic_score(item, policy, parameters) for item in terms]
                selected = candidates[int(np.argmax(scores))]
                changes += int(selected != baseline)
                values.append(answers[selected]["f1"])
            dataset_f1.append(float(np.mean(values)))
        table.append(
            {
                "parameters": parameters,
                "macro_f1": float(np.mean(dataset_f1)),
                "changed": changes,
            }
        )
    best = max(
        table,
        key=lambda item: (
            item["macro_f1"],
            -item["changed"],
            tuple(-x for x in item["parameters"].values()),
        ),
    )
    return {
        "policy": policy,
        "best": best,
        "grid": table,
        "development_question_ids": {
            key: [row["id"] for row in rows] for key, rows in development.items()
        },
        "uses_gain_training_labels": False,
        "test_used_for_selection": False,
    }


def policy_gain_rows(rows, policy, parameters, replacements=4):
    output, baselines = [], {}
    for row in rows:
        baseline = choose_heuristic(row, policy, parameters, replacements)
        answers = answer_map(row)
        _, alternatives = proposals(row, replacements, baseline)
        labels = [
            {
                "indices": list(pair),
                "features": feature_vector(row, pair).tolist(),
                "answer": answers[pair]["answer"],
                "generation_error": answers[pair].get("generation_error"),
                "f1": answers[pair]["f1"],
                "gain": answers[pair]["f1"] - answers[baseline]["f1"],
            }
            for pair in alternatives
        ]
        output.append(
            {
                **row,
                "baseline_indices": list(baseline),
                "baseline_answer": answers[baseline]["answer"],
                "baseline_generation_error": answers[baseline].get("generation_error"),
                "baseline_f1": answers[baseline]["f1"],
                "gain_labels": labels,
            }
        )
        baselines[str(row["id"])] = baseline
    return output, baselines
