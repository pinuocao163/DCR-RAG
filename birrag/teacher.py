import math

import numpy as np

from .data import minmax
from .prompts import qa_prompt


def teacher_top2(row, utilities):
    finite = [i for i, value in enumerate(utilities) if math.isfinite(float(value))]
    if len(finite) < 2:
        return [0, 1]
    return sorted(
        sorted(
            finite, key=lambda i: (-float(utilities[i]), i, str(row["ctxs"][i]["id"]))
        )[:2]
    )


def necessity_scores(row, generator, input_cap=4072):
    if not row["draft"].strip():
        return [0.0] * len(row["ctxs"])
    indices = list(range(len(row["ctxs"])))
    full = generator.mean_logprob(qa_prompt(row), row["draft"], input_cap)
    return [
        max(
            0.0,
            full
            - generator.mean_logprob(
                qa_prompt(row, [j for j in indices if j != i]), row["draft"], input_cap
            ),
        )
        for i in indices
    ]


def fit_teacher(rows):
    positives = [
        float(item["necessity_raw"])
        for row in rows
        for item in row["ctxs"]
        if item["necessity_raw"] > 0
    ]
    scale = max(1e-8, float(np.quantile(positives, 0.9))) if positives else 1.0
    for row in rows:
        semantic = minmax([float(item["rerank_score"]) for item in row["ctxs"]])
        utilities = []
        for item, score in zip(row["ctxs"], semantic):
            item["necessity"] = min(max(0.0, float(item["necessity_raw"])) / scale, 1.0)
            utilities.append(score + 0.75 * item["necessity"])
        row["teacher_utilities"] = utilities
        row["teacher_top2"] = teacher_top2(row, utilities)
    return {
        "positive_quantile": 0.9,
        "necessity_scale": scale,
        "semantic_weight": 1.0,
        "necessity_weight": 0.75,
        "question_ids": [row["id"] for row in rows],
        "gold_used": False,
    }
