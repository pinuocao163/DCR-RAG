import math

import numpy as np

from .data import minmax
from .prompts import qa_prompt, SYSTEM
from .visibility import fixed_prefix_deletions


def teacher_top2(row, utilities):
    finite = [
        i
        for i, value in enumerate(utilities)
        if value is not None and math.isfinite(float(value))
    ]
    if len(finite) < 2:
        return [0, 1]
    return sorted(
        sorted(
            finite, key=lambda i: (-float(utilities[i]), i, str(row["ctxs"][i]["id"]))
        )[:2]
    )


def necessity_scores(row, generator, input_cap=4072, deletion_context="backfill"):
    if not row["draft"].strip():
        return [0.0] * len(row["ctxs"])
    if deletion_context == "fixed_prefix":
        full_ids, deletions = fixed_prefix_deletions(
            generator.tokenizer, row, input_cap, getattr(generator, "system", SYSTEM)
        )
        full = generator.mean_logprob_ids(full_ids, row["draft"])
        return [
            max(0.0, full - generator.mean_logprob_ids(ids, row["draft"]))
            for ids in deletions
        ]
    if deletion_context != "backfill":
        raise ValueError("Unknown deletion-context rule")
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


def fit_teacher(rows, variant="hybrid"):
    if variant not in ("hybrid", "bge_only", "deletion_only"):
        raise ValueError("Unknown teacher supervision variant")
    semantic_weight = 0.0 if variant == "deletion_only" else 1.0
    necessity_weight = 0.0 if variant == "bge_only" else 0.75
    positives = [
        float(item["necessity_raw"])
        for row in rows
        for item in row["ctxs"]
        if item["necessity_raw"] is not None
        and math.isfinite(float(item["necessity_raw"]))
        and item["necessity_raw"] > 0
    ]
    scale = max(1e-8, float(np.quantile(positives, 0.9))) if positives else 1.0
    for row in rows:
        scores = [
            (
                float(item["rerank_score"])
                if item["rerank_score"] is not None
                else float("nan")
            )
            for item in row["ctxs"]
        ]
        finite = [score for score in scores if math.isfinite(score)]
        normal = iter(minmax(finite) if finite else [])
        semantic = [
            next(normal) if math.isfinite(score) else float("nan") for score in scores
        ]
        utilities = []
        for item, score in zip(row["ctxs"], semantic):
            raw = (
                float(item["necessity_raw"])
                if item["necessity_raw"] is not None
                else float("nan")
            )
            item["necessity_raw"] = raw if math.isfinite(raw) else None
            if item["rerank_score"] is None or not math.isfinite(
                float(item["rerank_score"])
            ):
                item["rerank_score"] = None
            item["necessity"] = (
                min(max(0.0, raw) / scale, 1.0) if math.isfinite(raw) else 0.0
            )
            utility = (
                semantic_weight * score if semantic_weight else 0.0
            ) + necessity_weight * item["necessity"]
            if necessity_weight and not math.isfinite(raw):
                utility = float("nan")
            utilities.append(utility if math.isfinite(utility) else None)
        row["teacher_utilities"] = utilities
        row["teacher_top2"] = teacher_top2(row, utilities)
    return {
        "positive_quantile": 0.9,
        "necessity_scale": scale,
        "variant": variant,
        "semantic_weight": semantic_weight,
        "necessity_weight": necessity_weight,
        "question_ids": [row["id"] for row in rows],
        "gold_used": False,
    }
