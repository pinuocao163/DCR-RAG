import numpy as np

from .metrics import bootstrap_effects


def question_support(row, view="selected"):
    contexts = row.get("ctxs", [])
    supporting = {
        str(item["id"]) for item in contexts if item.get("is_supporting", False)
    }
    if not supporting:
        return None
    selected = row["selected_indices"]
    if len(set(selected)) != len(selected) or any(
        index < 0 or index >= len(contexts) for index in selected
    ):
        raise ValueError("Invalid selected passage indices for support evaluation")
    selected_ids = {str(contexts[index]["id"]) for index in selected}
    if view not in ("selected", "reader_visible", "reader_fully_visible"):
        raise ValueError("Unknown support measurement view")
    if view != "selected":
        trace = row.get("reader_input", {})
        if not trace.get("available"):
            return None
        passages = trace["passages"]
        expected = [(index, str(contexts[index]["id"])) for index in sorted(selected)]
        if [(item["index"], item["id"]) for item in passages] != expected:
            raise ValueError(
                "Reader trace does not match the selected original passages"
            )
        field = "text_visible" if view == "reader_visible" else "text_fully_visible"
        selected_ids = {item["id"] for item in passages if item[field]}
    steps = row.get("question_decomposition", [])
    count = len(steps) if steps else row.get("hop_count")
    source_ids = {
        str(item.get("source_index", str(item["id"]).split(":")[-1])): str(item["id"])
        for item in contexts
    }
    positions, position_ids = [], []
    for step in steps:
        source = step.get("paragraph_support_idx") if isinstance(step, dict) else None
        passage_id = source_ids.get(str(source)) if source is not None else None
        position_ids.append(passage_id)
        positions.append(
            None if passage_id is None else float(passage_id in selected_ids)
        )
    return {
        "id": str(row["id"]),
        "dataset": row["dataset"],
        "question": row["question"],
        "support_ids": sorted(supporting),
        "support_recall": len(supporting & selected_ids) / len(supporting),
        "complete_coverage": float(supporting <= selected_ids),
        "support_hit": float(bool(supporting & selected_ids)),
        "hop_count": int(count) if count is not None else None,
        "position_hits": positions,
        "position_support_ids": position_ids,
    }


def support_summary(values):
    return {
        "questions": len(values),
        "support_recall": (
            float(np.mean([row["support_recall"] for row in values]))
            if values
            else None
        ),
        "complete_count": sum(int(row["complete_coverage"]) for row in values),
        "complete_coverage": (
            float(np.mean([row["complete_coverage"] for row in values]))
            if values
            else None
        ),
        "support_hit_count": sum(int(row["support_hit"]) for row in values),
        "support_hit_rate": (
            float(np.mean([row["support_hit"] for row in values])) if values else None
        ),
    }


def support_diagnostics(rows, view="selected"):
    groups = {}
    for row in rows:
        group = groups.setdefault(
            row["dataset"],
            {"total_questions": 0, "annotated_questions": 0, "records": []},
        )
        group["total_questions"] += 1
        group["annotated_questions"] += int(
            any(item.get("is_supporting", False) for item in row.get("ctxs", []))
        )
        value = question_support(row, view)
        if value is not None:
            group["records"].append(value)
    output = {}
    for dataset, group in groups.items():
        values = group["records"]
        report = {
            **support_summary(values),
            "total_questions": group["total_questions"],
            "questions_without_nonempty_support_annotation": group["total_questions"]
            - group["annotated_questions"],
            "annotated_questions_without_reader_trace": (
                group["annotated_questions"] - len(values) if view != "selected" else 0
            ),
        }
        if dataset == "musique":
            report["hop_subsets"] = {}
            for hops in sorted(
                {row["hop_count"] for row in values if row["hop_count"] is not None}
            ):
                subset = [row for row in values if row["hop_count"] == hops]
                positions = []
                for index in range(hops):
                    hits = [
                        row["position_hits"][index]
                        for row in subset
                        if len(row["position_hits"]) > index
                        and row["position_hits"][index] is not None
                    ]
                    positions.append(
                        {
                            "annotated_questions": len(hits),
                            "recall": float(np.mean(hits)) if hits else None,
                        }
                    )
                report["hop_subsets"][str(hops)] = {
                    **support_summary(subset),
                    "decomposition_position_recall": positions,
                }
            report["questions_without_hop_metadata"] = sum(
                row["hop_count"] is None for row in values
            )
        output[dataset] = report
    return {
        "measurement": {
            "selected": "selected original passages before reader token truncation",
            "reader_visible": "selected passages with at least one text token retained in the reader input",
            "reader_fully_visible": "selected passages with all text tokens retained in the reader input",
        }[view],
        "rate_scale": "fraction",
        "denominator": "questions with nonempty annotated support and, for reader views, an available input trace; missing annotations and traces reported separately",
        "per_dataset": output,
    }


def compare_support(first, second, seed=2092, samples=10000, view="selected"):
    def records(rows):
        result = {}
        seen = set()
        for row in rows:
            key = row["dataset"], str(row["id"])
            if key in seen:
                raise ValueError("Duplicate question ID in support comparison")
            seen.add(key)
            record = question_support(row, view)
            if record is not None:
                result[key] = record
        return seen, result

    first_ids, a = records(first)
    second_ids, b = records(second)
    if first_ids != second_ids or (view == "selected" and a.keys() != b.keys()):
        raise ValueError(
            "Support comparison requires identical questions and annotations"
        )
    paired = a.keys() & b.keys()
    a = {key: value for key, value in a.items() if key in paired}
    b = {key: value for key, value in b.items() if key in paired}
    if any(
        a[key]["support_ids"] != b[key]["support_ids"]
        or a[key]["question"] != b[key]["question"]
        or a[key]["hop_count"] != b[key]["hop_count"]
        or a[key]["position_support_ids"] != b[key]["position_support_ids"]
        for key in a
    ):
        raise ValueError("Support annotations differ between paired routes")
    effects = {}
    for metric in ("support_recall", "complete_coverage"):
        per_dataset = {}
        for key in a:
            per_dataset.setdefault(key[0], []).append(a[key][metric] - b[key][metric])
        effects[metric] = (
            bootstrap_effects(per_dataset, seed, samples) if per_dataset else None
        )
    subsets = {}
    for hops in sorted(
        {
            a[key]["hop_count"]
            for key in a
            if key[0] == "musique" and a[key]["hop_count"] is not None
        }
    ):
        keys = [key for key in a if key[0] == "musique" and a[key]["hop_count"] == hops]
        subsets[str(hops)] = {
            metric: bootstrap_effects(
                {"musique": [a[key][metric] - b[key][metric] for key in keys]},
                seed,
                samples,
            )
            for metric in ("support_recall", "complete_coverage")
        }
        positions = []
        for index in range(hops):
            paired = [
                key
                for key in keys
                if len(a[key]["position_hits"]) > index
                and len(b[key]["position_hits"]) > index
                and a[key]["position_hits"][index] is not None
                and b[key]["position_hits"][index] is not None
            ]
            positions.append(
                {
                    "paired_annotated_questions": len(paired),
                    "effect": (
                        bootstrap_effects(
                            {
                                "musique": [
                                    a[key]["position_hits"][index]
                                    - b[key]["position_hits"][index]
                                    for key in paired
                                ]
                            },
                            seed,
                            samples,
                        )
                        if paired
                        else None
                    ),
                }
            )
        subsets[str(hops)]["decomposition_position_effects"] = positions
    return {
        "first": support_diagnostics(first, view),
        "second": support_diagnostics(second, view),
        "paired_effects": effects,
        "paired_questions": len(a),
        "questions_without_paired_measurement": len(first_ids) - len(a),
        "paired_effect_scale": "percentage_points",
        "musique_hop_subset_effects": subsets,
    }


def route_support_diagnostics(routes, seed=2092, samples=10000):
    report = compare_support(
        routes["conditional"], routes["independent"], seed, samples
    )
    report["conditional_minus_bge"] = compare_support(
        routes["conditional"], routes["bge"], seed, samples
    )
    for view in ("reader_visible", "reader_fully_visible"):
        report[view] = compare_support(
            routes["conditional"], routes["independent"], seed, samples, view
        )
        report[view]["conditional_minus_bge"] = compare_support(
            routes["conditional"], routes["bge"], seed, samples, view
        )
    return report
