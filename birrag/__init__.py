import os
from pathlib import Path

__version__ = "1.0.0"
_root = Path(__file__).resolve().parents[1]
for _name, _directory in (
    ("HF_HOME", "huggingface"),
    ("TORCH_HOME", "torch"),
    ("TRITON_CACHE_DIR", "triton"),
):
    os.environ.setdefault(_name, str(_root / ".cache" / _directory))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
