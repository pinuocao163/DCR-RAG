import hashlib

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .data import assert_disjoint, question_key
from .features import feature_names, feature_vector, proposals
from .metrics import answer_metrics
from .reading import generate_pair

FORMULATIONS = (
    "direct",
    "baseline_aware",
    "baseline_aware_two_block",
    "difference",
    "shared_absolute",
    "single_ridge",
    "zero",
    "mlp",
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
    base_answer, base_error = generate_pair(row, reader, baseline, config["generation"])
    base_metrics = (
        answer_metrics(base_answer, row["answers"], row["dataset"])
        if base_error is None
        else {"em": 0.0, "f1": 0.0}
    )
    labels = []
    auxiliary = config["datasets"][row["dataset"]]["auxiliary"]
    for pair in alternatives:
        answer, error = generate_pair(row, reader, pair, config["generation"])
        metrics = (
            answer_metrics(answer, row["answers"], row["dataset"])
            if error is None
            else {"em": 0.0, "f1": 0.0}
        )
        labels.append(
            {
                "indices": list(pair),
                "features": feature_vector(row, pair, auxiliary).tolist(),
                "answer": answer,
                "generation_error": error,
                "f1": metrics["f1"],
                "gain": metrics["f1"] - base_metrics["f1"],
            }
        )
    return {
        **row,
        "baseline_indices": list(baseline),
        "baseline_answer": base_answer,
        "baseline_generation_error": base_error,
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


def fit_gain_ensemble(rows, config, target_dataset=None):
    if not rows:
        raise ValueError("Gain training is empty")
    dataset = target_dataset or rows[0]["dataset"]
    sources = sorted({row["dataset"] for row in rows})
    if target_dataset is None and len(sources) != 1:
        raise ValueError("Gain predictors are fitted separately for each dataset")
    if target_dataset is not None and target_dataset in sources:
        raise ValueError("LODO gain fitting cannot include the target dataset")
    for source in sources:
        assert_disjoint(
            {"gain_train": [row for row in rows if row["dataset"] == source]}
        )
    folds = config["selector"]["folds"]
    formulation = config["selector"].get("formulation", "direct")
    if formulation not in FORMULATIONS:
        raise ValueError(f"Unknown gain formulation: {formulation}")
    auxiliary = config["datasets"][dataset]["auxiliary"]
    names = feature_names(auxiliary)
    if any(row.get("feature_names") != names for row in rows):
        raise ValueError("Inconsistent feature schema")

    def key(row):
        return (
            f"{row['dataset']}:{row['id']}"
            if target_dataset is not None
            else str(row["id"])
        )

    assignment = {
        key(row): fold_index(
            config["selector"].get("fold_seed", config["seed"]),
            row["dataset"],
            row["id"],
            folds,
        )
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
            row for row in rows if held_out is None or assignment[key(row)] != held_out
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
        parameters = config.get("controls", {}).get("mlp")
        if formulation == "mlp":
            required = {
                "hidden_layer_sizes",
                "activation",
                "solver",
                "alpha",
                "learning_rate_init",
                "max_iter",
            }
            if parameters is None or not required <= parameters.keys():
                raise ValueError(
                    "MLP control requires explicit hidden_layer_sizes, activation, solver, alpha, learning_rate_init and max_iter"
                )
            settings = dict(parameters)
            settings.setdefault("random_state", config["seed"] + int(held_out))
            settings.setdefault("early_stopping", False)
            predictor = MLPRegressor(**settings).fit(normalizer.transform(x), y)
            weights = {
                "kind": "mlp",
                "activation": predictor.activation,
                "layers": [value.tolist() for value in predictor.coefs_],
                "biases": [value.tolist() for value in predictor.intercepts_],
                "parameters": settings,
                "iterations": int(predictor.n_iter_),
            }
        else:
            predictor = Ridge(
                alpha=config["selector"]["ridge_alpha"], fit_intercept=True
            ).fit(normalizer.transform(x), y)
            weights = {
                "kind": "ridge",
                "coefficients": predictor.coef_.tolist(),
                "intercept": float(predictor.intercept_),
            }
        members.append(
            {
                "held_out_fold": held_out,
                "mean": normalizer.mean_.tolist(),
                "scale": normalizer.scale_.tolist(),
                **weights,
                "fitting_question_ids": [row["id"] for row in selected],
            }
        )
    return {
        "version": 2,
        "dataset": dataset,
        "seed": config["seed"],
        "fold_assignment": assignment,
        "fold_seed": config["selector"].get("fold_seed", config["seed"]),
        "formulation": formulation,
        "feature_names": names,
        "auxiliary": auxiliary,
        "members": members,
        "alpha": (
            config["controls"]["mlp"]["alpha"]
            if formulation == "mlp"
            else config["selector"]["ridge_alpha"]
        ),
        "replacements_per_position": config["selector"]["replacements_per_position"],
        "source_datasets": sources,
        "training_question_keys": [[row["dataset"], str(row["id"])] for row in rows],
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
        normalized = (encoded - member["mean"]) / scale
        if member.get("kind") == "mlp":
            hidden = normalized
            for index, (weights, biases) in enumerate(
                zip(member["layers"], member["biases"])
            ):
                hidden = hidden @ np.asarray(weights) + np.asarray(biases)
                if index < len(member["layers"]) - 1:
                    activation = member["activation"]
                    if activation == "relu":
                        hidden = np.maximum(hidden, 0)
                    elif activation == "tanh":
                        hidden = np.tanh(hidden)
                    elif activation == "logistic":
                        hidden = 1 / (1 + np.exp(-np.clip(hidden, -700, 700)))
                    elif activation != "identity":
                        raise ValueError("Unsupported fitted MLP activation")
            prediction = hidden.reshape(-1)
        else:
            prediction = (
                normalized @ np.asarray(member["coefficients"]) + member["intercept"]
            )
        if formulation == "shared_absolute":
            prediction -= ((base - member["mean"]) / scale) @ np.asarray(
                member["coefficients"]
            ) + member["intercept"]
        predictions.append(prediction)
    if not predictions:
        raise ValueError("Gain bundle has no fitted models")
    predictions = np.asarray(predictions)
    return predictions.mean(axis=0), predictions.std(axis=0, ddof=0)


def choose_pair(row, bundle, kappa, tau, baseline=None, candidates=None):
    if kappa < 0:
        raise ValueError("kappa must be nonnegative")
    baseline, alternatives = proposals(
        row, bundle["replacements_per_position"], baseline
    )
    if candidates is not None:
        alternatives = [tuple(sorted(pair)) for pair in candidates]
        if len(set(alternatives)) != len(alternatives) or any(
            len(pair) != 2
            or len(set(pair)) != 2
            or pair == baseline
            or any(index < 0 or index >= len(row["ctxs"]) for index in pair)
            for pair in alternatives
        ):
            raise ValueError("Invalid or duplicate replacement candidates")
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
    training_keys = {tuple(key) for key in bundle.get("training_question_keys", [])}
    training_questions = {
        question_key(text) for text in bundle["training_question_texts"]
    }
    if any(
        (
            (row["dataset"], str(row["id"])) in training_keys
            if training_keys
            else str(row["id"]) in training_ids
        )
        or question_key(row["question"]) in training_questions
        for row in rows
    ):
        raise ValueError("Development overlaps gain training")
    if not rows:
        raise ValueError("Development is empty")
    prepared = []
    for row in rows:
        _, audit = choose_pair(row, bundle, 0.0, 0.0)
        labels = {tuple(label["indices"]): label for label in row["gain_labels"]}
        prepared.append(
            (
                row["baseline_f1"],
                [
                    (item["mean"], item["std"], labels[tuple(item["indices"])]["f1"])
                    for item in audit["proposals"]
                ],
            )
        )
    table = []
    kappas = (
        [0.0]
        if bundle.get("formulation") in ("single_ridge", "zero")
        else config["selector"]["kappa_grid"]
    )
    for kappa in kappas:
        for tau in config["selector"]["tau_grid"]:
            scores, accepted, harmed = [], 0, 0
            for baseline, candidates in prepared:
                risk = [mean - kappa * std for mean, std, f1 in candidates]
                best_index = int(np.argmax(risk)) if risk else None
                changed = best_index is not None and risk[best_index] > tau
                f1 = candidates[best_index][2] if changed else baseline
                scores.append(f1)
                accepted += int(changed)
                harmed += int(f1 < baseline)
            table.append(
                {
                    "kappa": float(kappa),
                    "tau": float(tau),
                    "mean_f1": float(np.mean(scores)),
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


def tune_shared_gate(rows_by_dataset, bundles, config):
    if not rows_by_dataset or rows_by_dataset.keys() != bundles.keys():
        raise ValueError(
            "Shared acceptance search needs development rows and a bundle for each dataset"
        )
    training_texts = {
        question_key(text)
        for bundle in bundles.values()
        for text in bundle["training_question_texts"]
    }
    if any(
        question_key(row["question"]) in training_texts
        for rows in rows_by_dataset.values()
        for row in rows
    ):
        raise ValueError("Shared development overlaps gain training")
    searches = {}
    for dataset, rows in rows_by_dataset.items():
        if (
            any(row["dataset"] != dataset for row in rows)
            or bundles[dataset]["dataset"] != dataset
        ):
            raise ValueError("Development dataset differs from gain bundle")
        assert_disjoint({"development": rows})
        searches[dataset] = tune_gate(rows, bundles[dataset], config)
    combined = {}
    for dataset, result in searches.items():
        for candidate in result["grid"]:
            key = candidate["kappa"], candidate["tau"]
            item = combined.setdefault(
                key,
                {
                    "kappa": key[0],
                    "tau": key[1],
                    "mean_f1": 0.0,
                    "accepted": 0,
                    "harmful": 0,
                    "per_dataset_mean_f1": {},
                },
            )
            item["mean_f1"] += candidate["mean_f1"] / len(searches)
            item["accepted"] += candidate["accepted"]
            item["harmful"] += candidate["harmful"]
            item["per_dataset_mean_f1"][dataset] = candidate["mean_f1"]
    table = list(combined.values())
    if any(len(item["per_dataset_mean_f1"]) != len(searches) for item in table):
        raise ValueError("Datasets must use a common acceptance grid")
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
        "best": best,
        "grid": table,
        "scope": "shared_development_macro",
        "datasets": list(searches),
        "seed": config["seed"],
        "development_question_ids": {
            dataset: result["development_question_ids"]
            for dataset, result in searches.items()
        },
        "development_question_texts": {
            dataset: result["development_question_texts"]
            for dataset, result in searches.items()
        },
        "development_questions": sum(len(rows) for rows in rows_by_dataset.values()),
        "test_used": False,
    }
