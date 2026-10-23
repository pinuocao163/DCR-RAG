import hashlib

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .data import assert_disjoint, question_key
from .features import feature_names, feature_vector, proposals
from .metrics import answer_metrics
from .prompts import qa_prompt

FORMULATIONS = (
    "direct",
    "baseline_aware",
    "baseline_aware_two_block",
    "difference",
    "shared_absolute",
    "single_ridge",
    "zero",
)


def fold_index(seed, dataset, row_id, folds=5):
    return (
        int(hashlib.sha256(f"{seed}{dataset}{row_id}".encode("utf-8")).hexdigest(), 16)
        % folds
    )


def baseline_features(row, auxiliary):
    return feature_vector(
        row, row.get("baseline_indices", proposals(row)[0]), auxiliary
    )


def label_gains(row, reader, config):
    baseline, alternatives = proposals(
        row, config["selector"]["replacements_per_position"]
    )
    input_cap = config["generation"]["reader_input_cap"]
    output_cap = config["generation"]["reader_output_cap"]
    base_answer = reader.generate(qa_prompt(row, baseline), input_cap, output_cap)
    base_metrics = answer_metrics(base_answer, row["answers"], row["dataset"])
    labels = []
    auxiliary = config["datasets"][row["dataset"]]["auxiliary"]
    for pair in alternatives:
        answer = reader.generate(qa_prompt(row, pair), input_cap, output_cap)
        metrics = answer_metrics(answer, row["answers"], row["dataset"])
        labels.append(
            {
                "indices": list(pair),
                "features": feature_vector(row, pair, auxiliary).tolist(),
                "answer": answer,
                "f1": metrics["f1"],
                "gain": metrics["f1"] - base_metrics["f1"],
            }
        )
    return {
        **row,
        "baseline_indices": list(baseline),
        "baseline_answer": base_answer,
        "baseline_f1": base_metrics["f1"],
        "gain_labels": labels,
        "feature_names": feature_names(auxiliary),
    }


def regression_features(vectors, baseline_vectors, formulation):
    x = np.asarray(vectors, dtype=np.float64)
    if formulation == "difference":
        return x - baseline_vectors
    if formulation == "baseline_aware":
        return np.concatenate([x, baseline_vectors, x - baseline_vectors], axis=-1)
    if formulation == "baseline_aware_two_block":
        return np.concatenate([x, baseline_vectors], axis=-1)
    return x


def fitting_examples(rows, auxiliary, formulation):
    features, targets = [], []
    for row in rows:
        base = (
            baseline_features(row, auxiliary)
            if formulation
            in (
                "difference",
                "baseline_aware",
                "baseline_aware_two_block",
                "shared_absolute",
            )
            else None
        )
        if formulation == "shared_absolute":
            features.append(base)
            targets.append(row["baseline_f1"])
        for label in row["gain_labels"]:
            features.append(regression_features(label["features"], base, formulation))
            targets.append(
                label["f1"] if formulation == "shared_absolute" else label["gain"]
            )
    return np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def fit_gain_ensemble(rows, config):
    if not rows:
        raise ValueError("Gain training is empty")
    dataset = rows[0]["dataset"]
    if any(row["dataset"] != dataset for row in rows):
        raise ValueError("Gain predictors are fitted separately for each dataset")
    assert_disjoint({"gain_train": rows})
    folds = config["selector"]["folds"]
    formulation = config["selector"].get("formulation", "direct")
    if formulation not in FORMULATIONS:
        raise ValueError(f"Unknown gain formulation: {formulation}")
    auxiliary = config["datasets"][dataset]["auxiliary"]
    names = feature_names(auxiliary)
    if any(row.get("feature_names") != names for row in rows):
        raise ValueError("Inconsistent feature schema")
    assignment = {
        str(row["id"]): fold_index(config["seed"], dataset, row["id"], folds)
        for row in rows
    }
    members = []
    holdouts = (
        []
        if formulation == "zero"
        else [None] if formulation == "single_ridge" else range(folds)
    )
    for held_out in holdouts:
        selected = [
            row
            for row in rows
            if held_out is None or assignment[str(row["id"])] != held_out
        ]
        x, y = fitting_examples(selected, auxiliary, formulation)
        if x.size == 0:
            raise ValueError(f"No fitting proposals for fold {held_out}")
        blocks = (
            3
            if formulation == "baseline_aware"
            else 2 if formulation == "baseline_aware_two_block" else 1
        )
        dimension = len(names) * blocks
        if (
            x.ndim != 2
            or x.shape[1] != dimension
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or np.any(np.abs(y) > 1)
        ):
            raise ValueError("Invalid gain training data")
        normalizer = StandardScaler().fit(x)
        predictor = Ridge(
            alpha=config["selector"]["ridge_alpha"], fit_intercept=True
        ).fit(normalizer.transform(x), y)
        members.append(
            {
                "held_out_fold": held_out,
                "mean": normalizer.mean_.tolist(),
                "scale": normalizer.scale_.tolist(),
                "coefficients": predictor.coef_.tolist(),
                "intercept": float(predictor.intercept_),
                "fitting_question_ids": [row["id"] for row in selected],
            }
        )
    return {
        "version": 2,
        "dataset": dataset,
        "seed": config["seed"],
        "fold_assignment": assignment,
        "formulation": formulation,
        "feature_names": names,
        "auxiliary": auxiliary,
        "members": members,
        "alpha": config["selector"]["ridge_alpha"],
        "replacements_per_position": config["selector"]["replacements_per_position"],
        "training_question_ids": [row["id"] for row in rows],
        "training_question_texts": [row["question"] for row in rows],
        "ddof": 0,
    }


def predict_gain(bundle, vectors, baseline_vectors=None):
    x = np.asarray(vectors, dtype=np.float64)
    if x.size == 0:
        return np.array([]), np.array([])
    if (
        x.ndim != 2
        or x.shape[1] != len(bundle["feature_names"])
        or not np.isfinite(x).all()
    ):
        raise ValueError("Gain features differ from fitted schema")
    formulation = bundle.get("formulation", "direct")
    if formulation == "zero":
        return np.zeros(len(x)), np.zeros(len(x))
    base = None
    if formulation in (
        "difference",
        "baseline_aware",
        "baseline_aware_two_block",
        "shared_absolute",
    ):
        if baseline_vectors is None:
            raise ValueError("This gain formulation requires initial-pair features")
        base = np.broadcast_to(np.asarray(baseline_vectors, dtype=np.float64), x.shape)
        if not np.isfinite(base).all():
            raise ValueError("Invalid initial-pair features")
    encoded = regression_features(x, base, formulation)
    predictions = []
    for member in bundle["members"]:
        scale = np.asarray(member["scale"])
        if np.any(scale <= 0):
            raise ValueError("Invalid fitted feature scale")
        prediction = ((encoded - member["mean"]) / scale) @ np.asarray(
            member["coefficients"]
        ) + member["intercept"]
        if formulation == "shared_absolute":
            prediction -= ((base - member["mean"]) / scale) @ np.asarray(
                member["coefficients"]
            ) + member["intercept"]
        predictions.append(prediction)
    if not predictions:
        raise ValueError("Gain bundle has no fitted models")
    predictions = np.asarray(predictions)
    return predictions.mean(axis=0), predictions.std(axis=0, ddof=0)


def choose_pair(row, bundle, kappa, tau, baseline=None):
    if kappa < 0:
        raise ValueError("kappa must be nonnegative")
    baseline, alternatives = proposals(
        row, bundle["replacements_per_position"], baseline
    )
    if not alternatives:
        return baseline, {
            "accepted": False,
            "baseline": list(baseline),
            "proposals": [],
            "reason": "empty_proposals",
        }
    base = (
        feature_vector(row, baseline, bundle["auxiliary"])
        if bundle.get("formulation")
        in (
            "difference",
            "baseline_aware",
            "baseline_aware_two_block",
            "shared_absolute",
        )
        else None
    )
    means, deviations = predict_gain(
        bundle,
        [feature_vector(row, pair, bundle["auxiliary"]) for pair in alternatives],
        base,
    )
    risk_adjusted = means - kappa * deviations
    best = int(np.argmax(risk_adjusted))
    accepted = bool(risk_adjusted[best] > tau)
    audit = {
        "accepted": accepted,
        "baseline": list(baseline),
        "kappa": kappa,
        "tau": tau,
        "proposals": [
            {
                "indices": list(pair),
                "mean": float(mean),
                "std": float(std),
                "adjusted_gain": float(risk),
            }
            for pair, mean, std, risk in zip(
                alternatives, means, deviations, risk_adjusted
            )
        ],
        "best_proposal": best,
    }
    return alternatives[best] if accepted else baseline, audit


def tune_gate(rows, bundle, config):
    training_ids = set(map(str, bundle["training_question_ids"]))
    training_questions = {
        question_key(text) for text in bundle["training_question_texts"]
    }
    if any(
        str(row["id"]) in training_ids
        or question_key(row["question"]) in training_questions
        for row in rows
    ):
        raise ValueError("Development overlaps gain training")
    if not rows:
        raise ValueError("Development is empty")
    table = []
    kappas = (
        [0.0]
        if bundle.get("formulation") in ("single_ridge", "zero")
        else config["selector"]["kappa_grid"]
    )
    for kappa in kappas:
        for tau in config["selector"]["tau_grid"]:
            selected_f1, accepted, harmed = [], 0, 0
            for row in rows:
                pair, audit = choose_pair(row, bundle, kappa, tau)
                labels = {
                    tuple(label["indices"]): label for label in row["gain_labels"]
                }
                f1 = (
                    labels[tuple(pair)]["f1"]
                    if audit["accepted"]
                    else row["baseline_f1"]
                )
                selected_f1.append(f1)
                accepted += int(audit["accepted"])
                harmed += int(f1 < row["baseline_f1"])
            table.append(
                {
                    "kappa": float(kappa),
                    "tau": float(tau),
                    "mean_f1": float(np.mean(selected_f1)),
                    "accepted": accepted,
                    "harmful": harmed,
                }
            )
    best = max(
        table,
        key=lambda item: (
            item["mean_f1"],
            -item["accepted"],
            item["kappa"],
            item["tau"],
        ),
    )
    return {
        "dataset": bundle["dataset"],
        "seed": config["seed"],
        "best": best,
        "grid": table,
        "development_question_ids": [row["id"] for row in rows],
        "development_question_texts": [row["question"] for row in rows],
        "test_used": False,
    }


def diagnostic_summary(records):
    errors = [row["mae"] for row in records if row["mae"] is not None]
    accepted = sum(row["accepted"] for row in records)
    harmful = sum(row["harmful"] for row in records)
    count = len(records)
    positive = [row["gain"] for row in records if row.get("gain", 0) > 0]
    negative = [row["gain"] for row in records if row.get("gain", 0) < 0]
    proposal_count = sum(row.get("proposal_count", 0) for row in records)
    proposal_error_sum = sum(row.get("proposal_error_sum", 0) for row in records)
    return {
        "questions": count,
        "questions_with_proposals": len(errors),
        "question_mean_gain_mae": float(np.mean(errors)) if errors else None,
        "accepted": accepted,
        "harmful": harmful,
        "beneficial": len(positive),
        "accepted_unchanged": accepted - len(positive) - len(negative),
        "mean_positive_gain": float(np.mean(positive)) if positive else None,
        "mean_negative_gain": float(np.mean(negative)) if negative else None,
        "proposal_count": proposal_count,
        "proposal_absolute_error_sum": proposal_error_sum,
        "proposal_weighted_gain_mae": (
            proposal_error_sum / proposal_count if proposal_count else None
        ),
        "changed": sum(row["changed"] for row in records),
        "acceptance_rate": accepted / count if count else 0.0,
        "harm_per_accepted": harmful / accepted if accepted else 0.0,
        "harm_per_question": harmful / count if count else 0.0,
        "mean_f1": float(np.mean([row["f1"] for row in records])) if count else None,
    }


def gain_diagnostics(rows, bundle, gate, baselines=None):
    records = []
    for row in rows:
        error = None
        absolute_errors = []
        if row["gain_labels"]:
            base = (
                baseline_features(row, bundle["auxiliary"])
                if bundle.get("formulation")
                in (
                    "difference",
                    "baseline_aware",
                    "baseline_aware_two_block",
                    "shared_absolute",
                )
                else None
            )
            means, _ = predict_gain(
                bundle, [label["features"] for label in row["gain_labels"]], base
            )
            absolute_errors = np.abs(
                means - [label["gain"] for label in row["gain_labels"]]
            ).tolist()
            error = float(np.mean(absolute_errors))
        pair, audit = choose_pair(
            row,
            bundle,
            gate["kappa"],
            gate["tau"],
            None if baselines is None else baselines[str(row["id"])],
        )
        f1 = row["baseline_f1"]
        if audit["accepted"]:
            chosen = next(
                label
                for label in row["gain_labels"]
                if tuple(label["indices"]) == tuple(pair)
            )
            f1 = chosen["f1"]
        records.append(
            {
                "mae": error,
                "accepted": int(audit["accepted"]),
                "harmful": int(f1 < row["baseline_f1"]),
                "changed": int(tuple(pair) != tuple(row["baseline_indices"])),
                "f1": f1,
                "baseline_f1": row["baseline_f1"],
                "gain": f1 - row["baseline_f1"],
                "proposal_error_sum": float(sum(absolute_errors)),
                "proposal_count": len(absolute_errors),
            }
        )
    result = diagnostic_summary(records)
    result["initial_pair_quality_strata"] = {}
    for lower, upper in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        group = [
            record
            for record in records
            if lower <= record["baseline_f1"]
            and (
                record["baseline_f1"] < upper
                or upper == 1.0
                and record["baseline_f1"] == 1.0
            )
        ]
        result["initial_pair_quality_strata"][f"{lower:.2f}-{upper:.2f}"] = (
            diagnostic_summary(group)
        )
    return result
