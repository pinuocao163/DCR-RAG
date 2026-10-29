from .prompts import qa_prompt
from .visibility import attach_reader_trace


def read_selected(row, model, generation):
    row.pop("generation_error", None)
    attach_reader_trace(row, model, generation["reader_input_cap"])
    response, error = generate_pair(row, model, row["selected_indices"], generation)
    row["prediction"] = response
    if error:
        row["generation_error"] = error
    return row


def generate_pair(row, model, indices, generation):
    try:
        return (
            model.generate(
                qa_prompt(row, indices),
                generation["reader_input_cap"],
                generation["reader_output_cap"],
            ),
            None,
        )
    except Exception as error:
        return "", f"{type(error).__name__}: {error}"
