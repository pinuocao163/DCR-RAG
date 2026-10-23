from .prompts import qa_prompt
from .visibility import attach_reader_trace


def read_selected(row, model, generation):
    row.pop("generation_error", None)
    attach_reader_trace(row, model, generation["reader_input_cap"])
    try:
        row["prediction"] = model.generate(
            qa_prompt(row, row["selected_indices"]),
            generation["reader_input_cap"],
            generation["reader_output_cap"],
        )
    except Exception as error:
        row["prediction"] = ""
        row["generation_error"] = f"{type(error).__name__}: {error}"
    return row
