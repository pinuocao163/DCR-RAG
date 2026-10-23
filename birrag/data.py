import hashlib
import json
import math
import re
import unicodedata

from .io import digest, iter_rows, save_json, save_rows

DATASETS = ("hotpotqa", "2wiki", "musique", "naturalqa", "triviaqa")


def tokens(text):
    return re.findall(
        r"\w+",
        unicodedata.normalize("NFD", str(text or "")).casefold(),
        flags=re.UNICODE,
    )


def question_key(question):
    return " ".join(tokens(question))


def passage(context):
    return (
        str(context.get("title", "")).strip()
        + "\n"
        + str(context.get("text", "")).strip()
    ).strip()


def contains_phrase(text, phrase):
    document, target = tokens(text), tokens(phrase)
    return bool(target) and any(
        document[i : i + len(target)] == target
        for i in range(len(document) - len(target) + 1)
    )


def minmax(values):
    if not values or not all(math.isfinite(float(x)) for x in values):
        raise ValueError("Scores must be finite and nonempty")
    low, high = min(values), max(values)
    return (
        [(float(value) - low) / (high - low) for value in values]
        if high > low
        else [0.0] * len(values)
    )


def parse(value):
    return json.loads(value) if isinstance(value, str) else value


def convert(record, dataset):
    row = dict(record)
    if row.get("answerable") is False:
        return None
    question = str(row.get("question", row.get("question_text", ""))).strip()
    raw_id = row.get("id", row.get("_id", row.get("question_id")))
    row_id = (
        str(raw_id)
        if raw_id is not None
        else hashlib.sha256(question.encode()).hexdigest()
    )
    answers = row.get("answers", row.get("golden_answers"))
    if answers is None:
        answers = ([row["answer"]] if "answer" in row else []) + list(
            parse(row.get("answer_aliases", [])) or []
        )
        for item in parse(row.get("answers_objects", [])):
            answers.extend(item.get("spans", []))
    if isinstance(answers, str):
        answers = [answers]
    answers = list(dict.fromkeys(str(answer) for answer in answers))
    if "candidate_ctxs" in row or "ctxs" in row:
        contexts = row.get("candidate_ctxs", row.get("ctxs"))
    elif "retrieval_docs" in row:
        contexts = []
        for item in row["retrieval_docs"][:10]:
            title, _, text = str(item.get("contents", "")).partition("\n")
            contexts.append(
                {
                    "id": item.get("id"),
                    "title": title.strip().strip('"'),
                    "text": text.strip(),
                }
            )
    elif "context" in row:
        context = parse(row["context"])
        support = parse(row.get("supporting_facts", []))
        support_titles = (
            set(support["title"])
            if isinstance(support, dict)
            else {item[0] for item in support}
        )
        pairs = (
            zip(context["title"], context["sentences"])
            if isinstance(context, dict)
            else context
        )
        contexts = [
            {
                "title": title,
                "text": " ".join(sentences),
                "is_supporting": title in support_titles,
            }
            for title, sentences in pairs
        ]
    else:
        contexts = []
        for i, item in enumerate(parse(row.get("paragraphs", row.get("contexts", [])))):
            contexts.append(
                {
                    "id": item.get("idx", i),
                    "source_index": item.get("idx", i),
                    "title": item.get("title", ""),
                    "text": item.get("paragraph_text", item.get("text", "")),
                    "is_supporting": bool(item.get("is_supporting", False)),
                }
            )
    converted, seen_ids = [], set()
    for index, item in enumerate(contexts):
        item = dict(item)
        item["id"] = str(
            item.get("id") if item.get("id") is not None else f"{row_id}:{index}"
        )
        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate passage ID for {row_id}: {item['id']}")
        seen_ids.add(item["id"])
        item.setdefault("title", "")
        item.setdefault("text", "")
        item.setdefault("retrieval_rank", index + 1)
        if not passage(item):
            continue
        converted.append(item)
    if len(converted) < 2 or not question:
        return None
    result = {
        "id": row_id,
        "question": question,
        "answers": answers,
        "dataset": dataset,
        "ctxs": converted,
    }
    for key in (
        "type",
        "level",
        "question_decomposition",
        "hop_count",
        "bir_necessity",
        "draft",
        "bge_ordered",
    ):
        if key in row:
            result[key] = row[key]
    if "draft" not in result and "bir_necessity" in result:
        result["draft"] = result["bir_necessity"].get("reference_answer", "")
    return result


def hash_order(seed, dataset, row):
    return hashlib.sha256(f"{seed}:{dataset}:{row['id']}".encode()).hexdigest()


def assert_disjoint(groups):
    ids, questions = {}, {}
    for name, rows in groups.items():
        local_ids, local_questions = set(), set()
        for row in rows:
            row_id, question = str(row["id"]), question_key(row["question"])
            if (
                row_id in ids
                or question in questions
                or row_id in local_ids
                or question in local_questions
            ):
                raise ValueError(f"Split leakage or duplicate in {name}: {row_id}")
            local_ids.add(row_id)
            local_questions.add(question)
        ids.update({value: name for value in local_ids})
        questions.update({value: name for value in local_questions})


def prepare_splits(
    config, dataset, destination, train_paths, eval_paths, exclude_paths=()
):
    specification = config["datasets"][dataset]
    blocked_ids, blocked_questions = set(), set()
    for path in exclude_paths:
        for row in iter_rows(path):
            blocked_ids.add(str(row["id"]))
            blocked_questions.add(question_key(row["question"]))
    groups = {}
    sources = {}

    def select(paths, sizes):
        candidates = []
        seen_ids, seen_questions = set(), set()
        for path in paths:
            sources[str(path)] = digest(path)
            for raw in iter_rows(path):
                row = convert(raw, dataset)
                if row is None or not row["answers"]:
                    continue
                row_id, question = row["id"], question_key(row["question"])
                if (
                    row_id in blocked_ids
                    or question in blocked_questions
                    or row_id in seen_ids
                    or question in seen_questions
                ):
                    continue
                seen_ids.add(row_id)
                seen_questions.add(question)
                candidates.append(row)
        candidates.sort(key=lambda row: hash_order(config["split_seed"], dataset, row))
        for name, count in sizes.items():
            eligible = [
                row
                for row in candidates
                if name != "scorer_train" or len(row["ctxs"]) >= 3
            ]
            quotas = specification.get("test_hop_counts") if name == "test" else None
            if quotas is None:
                chosen = eligible[:count]
            else:
                if sum(quotas.values()) != count or any(
                    int(hop) < 1 or not isinstance(size, int) or size < 0
                    for hop, size in quotas.items()
                ):
                    raise ValueError(
                        "test_hop_counts must contain nonnegative counts summing to test_questions"
                    )
                chosen = []
                for hop, size in quotas.items():
                    subset = [
                        row
                        for row in eligible
                        if len(row.get("question_decomposition", [])) == int(hop)
                        or (
                            not row.get("question_decomposition")
                            and row.get("hop_count") == int(hop)
                        )
                    ]
                    if len(subset) < size:
                        raise ValueError(
                            f"{dataset}/test needs {size} {hop}-hop rows, only {len(subset)} eligible"
                        )
                    chosen.extend(subset[:size])
                chosen.sort(
                    key=lambda row: hash_order(config["split_seed"], dataset, row)
                )
            if len(chosen) != count:
                raise ValueError(
                    f"{dataset}/{name} needs {count} rows, only {len(chosen)} eligible"
                )
            groups[name] = chosen
            chosen_ids = {row["id"] for row in chosen}
            candidates = [row for row in candidates if row["id"] not in chosen_ids]
            for row in chosen:
                blocked_ids.add(row["id"])
                blocked_questions.add(question_key(row["question"]))

    select(
        train_paths,
        {
            "scorer_train": config["scorer_train_questions"],
            "gain_train": config["gain_train_questions"],
        },
    )
    select(
        eval_paths,
        {
            "dev": config["dev_questions"],
            "test": specification["test_questions"],
            "confirmation": specification.get("confirmation_questions", 0),
        },
    )
    assert_disjoint(groups)
    for name, rows in groups.items():
        save_rows(destination / f"{name}.jsonl", rows)
    save_json(
        destination / "split_manifest.json",
        {
            "dataset": dataset,
            "source_sha256": sources,
            "split_seed": config["split_seed"],
            "test_hop_counts_requested": specification.get("test_hop_counts"),
            "test_hop_counts_observed": (
                {
                    str(hop): sum(
                        (
                            len(row.get("question_decomposition", []))
                            or row.get("hop_count")
                        )
                        == hop
                        for row in groups["test"]
                    )
                    for hop in (2, 3, 4)
                }
                if dataset == "musique"
                else None
            ),
            "exclude_paths": list(map(str, exclude_paths)),
            "splits": {
                name: {
                    "count": len(rows),
                    "ids": [row["id"] for row in rows],
                    "sha256": digest(destination / f"{name}.jsonl"),
                }
                for name, rows in groups.items()
            },
        },
    )
    return groups
