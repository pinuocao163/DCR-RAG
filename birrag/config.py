import json
from pathlib import Path

from .io import ROOT


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for role, value in config["models"].items():
        model_path = Path(value).expanduser()
        config["models"][role] = str(
            model_path if model_path.is_absolute() else ROOT / model_path
        )
    for specification in config["datasets"].values():
        for name in ("train_sources", "eval_sources"):
            specification[name] = [
                str(
                    Path(value).expanduser()
                    if Path(value).expanduser().is_absolute()
                    else ROOT / Path(value).expanduser()
                )
                for value in specification[name]
            ]
    return config
