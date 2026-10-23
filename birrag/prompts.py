SYSTEM = "You answer questions using the supplied evidence. Be concise and do not add an explanation unless it is required by the question."
INSTRUCTION = "Answer the question below concisely in a few words."


def qa_prompt_with_spans(row, indices=None):
    indices = list(range(len(row["ctxs"]))) if indices is None else sorted(indices)
    blocks, spans, offset = [], [], 0
    for index in indices:
        context = row["ctxs"][index]
        title, text = str(context.get("title", "")), str(context.get("text", ""))
        prefix = f"Passage #{index + 1} title: "
        middle = f"\nPassage #{index + 1} text: "
        block = prefix + title + middle + text
        title_start = offset + len(prefix)
        text_start = title_start + len(title) + len(middle)
        spans.append(
            {
                "index": index,
                "title": (title_start, title_start + len(title)),
                "text": (text_start, text_start + len(text)),
            }
        )
        blocks.append(block)
        offset += len(block) + 2
    evidence = "\n\n".join(blocks)
    suffix = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{row['question']}\n"
    return (f"{evidence}\n\n{suffix}" if evidence else suffix), spans


def qa_prompt(row, indices=None):
    return qa_prompt_with_spans(row, indices)[0]


def student_query(row, include_draft=True):
    if not include_draft:
        return row["question"]
    if "draft" not in row:
        raise ValueError("A cached non-oracle draft is required")
    return f"Question: {row['question']}\nDraft answer: {row['draft']}"
