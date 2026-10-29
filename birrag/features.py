import math

import numpy as np

from .data import contains_phrase, minmax, passage, tokens

FEATURE_NAMES = (
    "question_length_log",
    "draft_length_log",
    "pool_size_log",
    "student_max",
    "student_min",
    "student_mean",
    "student_abs_diff",
    "bge_max",
    "bge_min",
    "bge_mean",
    "bge_abs_diff",
    "draft_match_count",
    "question_union_coverage",
    "question_mean_coverage",
    "residual_first_given_second",
    "residual_second_given_first",
    "passage_jaccard",
    "passage_mean_novelty",
    "title_link_either",
    "passage_length_max_log",
    "passage_length_min_log",
    "passage_length_mean_log",
    "bge_rank_min_log",
    "bge_rank_max_log",
    "bge_rank_mean_log",
    "student_rank_sum",
    "bge_rank_sum",
    "starts_who",
    "starts_what",
    "starts_when",
    "starts_where",
    "starts_which",
    "starts_how",
    "starts_why",
)


def order(scores):
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def initial_pair(row):
    return tuple(
        sorted(order([context["student_score"] for context in row["ctxs"]])[:2])
    )


def proposals(row, replacements_per_position=4, baseline=None):
    baseline = initial_pair(row) if baseline is None else tuple(sorted(baseline))
    if len(set(baseline)) != 2 or any(i < 0 or i >= len(row["ctxs"]) for i in baseline):
        raise ValueError(
            "An initial pair must contain two valid distinct candidate indices"
        )
    alternatives = [
        i
        for i in order([context["student_score"] for context in row["ctxs"]])
        if i not in baseline
    ]
    result, seen = [], set()
    for keeper in baseline:
        for alternative in alternatives[:replacements_per_position]:
            pair = tuple(sorted((keeper, alternative)))
            if pair not in seen:
                result.append(pair)
                seen.add(pair)
    return baseline, result


def ratio(numerator, denominator):
    return numerator / max(1, denominator)


def stable_ranks(values):
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order(values)):
        ranks[index] = rank / max(1, len(values) - 1)
    return ranks


def feature_names(auxiliary="none"):
    if auxiliary != "none":
        raise ValueError("The configured pair features require auxiliary=none")
    return list(FEATURE_NAMES)


def feature_vector(row, pair, auxiliary="none"):
    first, second = sorted(pair)
    contexts = row["ctxs"]
    student = minmax([float(item["student_score"]) for item in contexts])
    semantic = minmax([float(item["rerank_score"]) for item in contexts])
    documents = [tokens(passage(item)) for item in contexts]
    sets = [set(item) for item in documents]
    question = set(tokens(row["question"]))
    a, b = sets[first], sets[second]
    first_residual, second_residual = question - b, question - a
    draft = row["draft"] if row.get("draft_features_enabled", True) else ""
    student_values = [student[first], student[second]]
    semantic_values = [semantic[first], semantic[second]]
    values = [
        math.log1p(len(tokens(row["question"]))),
        math.log1p(len(tokens(draft))),
        math.log1p(len(contexts)),
    ]
    for scores in (student_values, semantic_values):
        values.extend(
            [max(scores), min(scores), sum(scores) / 2, abs(scores[0] - scores[1])]
        )
    values.extend(
        [
            sum(
                float(contains_phrase(passage(contexts[i]), draft))
                for i in (first, second)
            ),
            ratio(len(question & (a | b)), len(question)),
            (
                ratio(len(question & a), len(question))
                + ratio(len(question & b), len(question))
            )
            / 2,
            ratio(len(first_residual & a), len(first_residual)),
            ratio(len(second_residual & b), len(second_residual)),
            ratio(len(a & b), len(a | b)),
            (ratio(len(a - b), len(a)) + ratio(len(b - a), len(b))) / 2,
            float(
                contains_phrase(
                    passage(contexts[first]), contexts[second].get("title", "")
                )
                or contains_phrase(
                    passage(contexts[second]), contexts[first].get("title", "")
                )
            ),
        ]
    )
    lengths = [len(documents[first]), len(documents[second])]
    ranks = [first + 1, second + 1]
    values.extend(
        [
            math.log1p(max(lengths)),
            math.log1p(min(lengths)),
            math.log1p(sum(lengths) / 2),
        ]
    )
    values.extend(
        [math.log1p(min(ranks)), math.log1p(max(ranks)), math.log1p(sum(ranks) / 2)]
    )
    values.extend(
        [
            sum(stable_ranks(student)[i] for i in (first, second)),
            sum(stable_ranks(semantic)[i] for i in (first, second)),
        ]
    )
    question_start = next(iter(tokens(row["question"])), "")
    values.extend(
        float(question_start == word)
        for word in ("who", "what", "when", "where", "which", "how", "why")
    )
    result = np.asarray(values, dtype=np.float64)
    if len(result) != len(feature_names(auxiliary)) or not np.isfinite(result).all():
        raise ValueError("Invalid pair features")
    return result


def draft_match_diagnostics(rows):
    groups = {}
    for row in rows:
        group = groups.setdefault(
            row["dataset"], {"questions": 0, "candidate_passages": 0, "nonzero": 0}
        )
        group["questions"] += 1
        for context in row["ctxs"]:
            group["candidate_passages"] += 1
            group["nonzero"] += int(contains_phrase(passage(context), row["draft"]))
    for group in groups.values():
        group["nonzero_rate"] = group["nonzero"] / max(1, group["candidate_passages"])
        group["nonzero_percent"] = 100 * group["nonzero_rate"]
    return {
        "denominator": "all evaluated candidate passages, once per question/passage",
        "per_dataset": groups,
    }
