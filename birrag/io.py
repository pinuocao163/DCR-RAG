import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT / "main.pdf":
        raise ValueError(
            f"Output must be inside {ROOT} and cannot replace main.pdf: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, value):
    path = output_path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_rows(path, rows):
    path = output_path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def iter_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches():
            yield from batch.to_pylist()
    elif path.suffix == ".arrow":
        from datasets import Dataset

        yield from Dataset.from_file(str(path))
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        try:
            import ijson
        except ImportError:
            with path.open(encoding="utf-8") as stream:
                data = json.load(stream)
            if not isinstance(data, list):
                raise ValueError("JSON input must contain a list")
            yield from data
        else:
            with path.open("rb") as stream:
                yield from ijson.items(stream, "item", use_float=True)


def load_rows(path):
    return list(iter_rows(path))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def identity(row):
    return str(row["id"])
