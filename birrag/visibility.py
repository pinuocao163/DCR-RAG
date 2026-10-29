import hashlib
import json

from .prompts import SYSTEM, qa_prompt_with_spans


def rendered_token_offsets(tokenizer, prompt, system=SYSTEM):
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if rendered.count(prompt) != 1:
        raise ValueError("The chat template must preserve one contiguous user prompt")
    origin = rendered.index(prompt)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    chat_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if list(chat_ids) != ids:
        raise ValueError("Chat-template tokens differ from rendered reader tokens")
    if list(encoded["input_ids"]) != ids:
        raise ValueError("Offset tokenization differs from actual reader tokenization")
    offsets = encoded["offset_mapping"]
    if len(offsets) != len(ids):
        raise ValueError("Tokenizer offsets do not match reader tokens")
    return ids, offsets, origin


def fixed_prefix_deletions(tokenizer, row, input_cap, system=SYSTEM):
    prompt, spans = qa_prompt_with_spans(row)
    ids, offsets, origin = rendered_token_offsets(tokenizer, prompt, system)
    ids, offsets = ids[:input_cap], offsets[:input_cap]
    deletions = []
    for span in spans:
        start, end = [origin + value for value in span["block"]]
        deletions.append(
            [
                token
                for token, (left, right) in zip(ids, offsets)
                if not (right > left and right > start and left < end)
            ]
        )
    return ids, deletions


def reader_input_trace(tokenizer, row, indices, input_cap, system=SYSTEM):
    indices = sorted(indices)
    if (
        input_cap < 1
        or len(set(indices)) != len(indices)
        or any(index < 0 or index >= len(row["ctxs"]) for index in indices)
    ):
        raise ValueError("Invalid reader input cap or passage indices")
    prompt, spans = qa_prompt_with_spans(row, indices)
    ids, offsets, origin = rendered_token_offsets(tokenizer, prompt, system)
    passages = []
    for span in spans:
        item = {"index": span["index"], "id": str(row["ctxs"][span["index"]]["id"])}
        for field in ("title", "text"):
            start, end = [origin + position for position in span[field]]
            positions = [
                index
                for index, (left, right) in enumerate(offsets)
                if right > left and right > start and left < end
            ]
            retained = sum(index < input_cap for index in positions)
            item[field + "_tokens"] = len(positions)
            item["retained_" + field + "_tokens"] = retained
            item[field + "_visible"] = retained > 0
            item[field + "_fully_visible"] = bool(positions) and retained == len(
                positions
            )
        passages.append(item)
    retained_ids = ids[:input_cap]
    return {
        "available": True,
        "input_cap": input_cap,
        "truncation_side": "right",
        "prompt_tokens_before_truncation": len(ids),
        "input_tokens": len(retained_ids),
        "truncated": len(ids) > input_cap,
        "input_ids_sha256": hashlib.sha256(
            json.dumps(retained_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "visibility_rule": "at least one original passage text token retained; titles and prompt labels do not count",
        "fully_visible_rule": "all original passage text tokens retained",
        "passages": passages,
    }


def attach_reader_trace(row, model, input_cap):
    row.pop("reader_input", None)
    try:
        row["reader_input"] = reader_input_trace(
            model.tokenizer,
            row,
            row["selected_indices"],
            input_cap,
            getattr(model, "system", SYSTEM),
        )
    except Exception as error:
        row["reader_input"] = {
            "available": False,
            "reason": f"{type(error).__name__}: {error}",
        }
    return row
