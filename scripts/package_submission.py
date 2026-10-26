import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "main.pdf",
    "paper.sha256",
    "pyproject.toml",
    "requirements.txt",
    "requirements-vllm.txt",
    "configs/paper.json",
    "configs/paper_hf.json",
    "configs/readers.json",
    "docs/DATA.md",
    "docs/METHOD.md",
)
SCRIPTS = (
    "run_pipeline.py",
    "infer.py",
    "compare_formulations.py",
    "compare_policies.py",
    "compare_acceptance.py",
    "compare_seeds.py",
    "compare_budgets.py",
    "compare_readers.py",
    "package_submission.py",
)


def submission_files():
    paths = [ROOT / name for name in FILES]
    paths.extend(sorted((ROOT / "birrag").glob("*.py")))
    paths.extend(ROOT / "scripts" / name for name in SCRIPTS)
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Invalid submission file: {path.relative_to(ROOT)}")
        yield path, path.relative_to(ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist/dcrrag_submission.zip")
    args = parser.parse_args()
    output = Path(args.output)
    output = output if output.is_absolute() else ROOT / output
    output = output.resolve()
    if not output.is_relative_to(ROOT) or output.suffix != ".zip":
        raise ValueError(
            "The archive output must be a ZIP file inside the project directory"
        )
    files = [
        (path, relative) for path, relative in submission_files() if path != output
    ]
    expected = (ROOT / "paper.sha256").read_text().split()[0]
    if hashlib.sha256((ROOT / "main.pdf").read_bytes()).hexdigest() != expected:
        raise ValueError("Manuscript checksum verification failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in files:
            info = zipfile.ZipInfo(
                "dcrrag_submission/" + relative.as_posix(),
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive checksum verification failed")
    print(f"Created {output.name}: {len(files)} files, {output.stat().st_size} bytes")
    print("SHA-256:", hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
