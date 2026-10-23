# DCR-RAG

Code for **DCR-RAG: Draft-Conditioned Ranking with Conservative Passage Replacement**. The manuscript is provided in [main.pdf](main.pdf).

DCR-RAG generates a candidate-pool answer draft, trains a draft-conditioned passage scorer, and selects two original passages. Five Ridge gain predictors evaluate bounded single-passage replacements. The final reader answers from the question and selected passages.

## Installation

Use Python 3.10 on Linux with CUDA. Run commands from the project root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[vllm]"
```

For Transformers-only generation, install with `python -m pip install -e .` and use `configs/paper_hf.json`. Dependencies are pinned in [requirements.txt](requirements.txt) and [requirements-vllm.txt](requirements-vllm.txt). The Python module is `birrag`.

## Models and data

Place complete local model snapshots, including tokenizers, in the configured directories:

| Role | Model | Directory |
|---|---|---|
| BGE prior and scorer initialization | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | `models/bge-reranker-base` |
| Draft and deletion teacher | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | `models/Qwen2.5-3B-Instruct` |
| Final reader and gain labels | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | `models/Qwen2.5-7B-Instruct` |

Prepare HotpotQA, 2WikiMultiHopQA, MuSiQue-Ans, and the E5 candidate exports for Natural Questions and TriviaQA as described in [docs/DATA.md](docs/DATA.md). Relative paths in configurations resolve from the project root. Dataset keys are `hotpotqa`, `2wiki`, `musique`, `naturalqa`, and `triviaqa`.

## Training and evaluation

Run all five datasets:

```bash
python scripts/run_pipeline.py --config configs/paper.json --dataset all --output-root runs/paper_seed2094 --label-test-proposals
```

Use `--dataset hotpotqa` to run one dataset. The pipeline prepares disjoint splits, fixes BGE candidate order, generates drafts, constructs teacher preferences, trains the scorer, labels gains, fits Ridge models, selects acceptance parameters on development data, and evaluates the test routes. HotpotQA and 2Wiki confirmation sets reuse the trained artifacts and development-selected parameters.

| Setting | Value |
|---|---|
| Scorer / gain / development questions per dataset | 100 / 100 / 100, disjoint |
| Scorer optimizer | AdamW, learning rate `1e-6`, weight decay `0.01` |
| Training | 3 epochs, batch 1 question, accumulation 4 |
| Schedule | Linear decay, warm-up ratio `0.1` |
| Precision | FP16 training, bfloat16 generation |
| Query / passage / joint caps | 200 / 464 / 512 tokens |
| Draft input / output caps | 4072 / 24 tokens |
| Reader input / output caps | 2048 / 32 tokens |
| Gain predictors | 5 question folds, Ridge `alpha=10` |
| Replacement search | 4 alternatives per position, at most 8 proposals |
| Main / split / bootstrap seeds | 2094 / 2091 / 2092 |
| Bootstrap replicates | 10,000 |

Run stability seeds with the same split seed:

```bash
python scripts/run_pipeline.py --config configs/paper.json --dataset all --seed 2095 --output-root runs/paper_seed2095 --label-test-proposals
python scripts/run_pipeline.py --config configs/paper.json --dataset all --seed 2096 --output-root runs/paper_seed2096 --label-test-proposals
```

Each run saves `resolved_config.json`, aggregate route predictions, EM/F1 summaries, and paired bootstrap intervals. Dataset subdirectories contain split manifests, scorer checkpoints, `gain_ensemble.json`, `gate.json`, predictions, support statistics, and proposal diagnostics. Main-test and confirmation results are reported separately.

Evaluate saved predictions:

```bash
python -m birrag.cli evaluate --input runs/paper_seed2094/hotpotqa/test.conditional.predictions.jsonl --compare runs/paper_seed2094/hotpotqa/test.independent.predictions.jsonl --output runs/evaluation.json
```

## Experimental comparisons

Use the main run generated with `--label-test-proposals`:

```bash
python scripts/compare_formulations.py --run-root runs/paper_seed2094 --output-root runs/formulations
python scripts/compare_policies.py --run-root runs/paper_seed2094 --output-root runs/policies
python scripts/compare_acceptance.py --run-root runs/paper_seed2094 --output-root runs/acceptance
python scripts/compare_budgets.py --run-root runs/paper_seed2094 --dataset musique --output-root runs/budgets
```

These scripts evaluate gain formulations, fixed-scorer heuristic policies, development-matched acceptance, and 1–4 passage budgets. Results include metrics, paired intervals, and artifact checksums. Configuration and statistical definitions are documented in [docs/METHOD.md](docs/METHOD.md).

For seed-averaged paired comparisons, supply one prediction file per arm and seed:

```bash
python scripts/compare_seeds.py --first runs/paper_seed2094/test.conditional.predictions.jsonl runs/paper_seed2095/test.conditional.predictions.jsonl runs/paper_seed2096/test.conditional.predictions.jsonl --second runs/paper_seed2094/test.independent.predictions.jsonl runs/paper_seed2095/test.independent.predictions.jsonl runs/paper_seed2096/test.independent.predictions.jsonl --seed-ids 2094 2095 2096 --output runs/seed_comparison.json
```

## Inference

Provide JSONL records containing `id`, `question`, and `ctxs`:

```bash
python scripts/infer.py --dataset hotpotqa --input data/new_questions.jsonl --artifacts runs/paper_seed2094/hotpotqa --output-root runs/inference
```

The script uses the saved experiment configuration and writes selected passage IDs, reader-input statistics, and final answers to `predictions.jsonl`.

## Submission package

```bash
python scripts/package_submission.py
```

The upload archive is `dist/dcrrag_submission.zip`. It contains the manuscript, source code, configurations, dependencies, and documentation. The packager verifies the manuscript checksum and archive integrity.

External source, model, and dataset acknowledgments are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
