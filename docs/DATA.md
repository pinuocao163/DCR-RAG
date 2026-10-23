# Data Preparation

## Dataset sources

DCR-RAG operates on fixed candidate pools. Obtain the following datasets and candidate exports from their providers:

| Dataset | Training source | Evaluation source | Candidate budget |
|---|---|---|---:|
| [HotpotQA](https://hotpotqa.github.io/) | Distractor validation | Distractor validation | Up to 10 |
| [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) | Training split | Development split | 10 |
| [MuSiQue-Ans](https://github.com/StonyBrookNLP/musique) | Answerable training split | Answerable development split | Up to 20 |
| [Natural Questions](https://ai.google.com/research/NaturalQuestions/) | E5 `train_1w.json` candidates | E5 `dev.json` candidates | 10 |
| [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) | E5 `train_1w.json` candidates | E5 `dev.json` candidates | 10 |

For NQ and TriviaQA, use the E5 Top-10 pre-retrieved files provided with [RAG-Critic](https://github.com/RUC-NLPIR/RAG-Critic). Candidate exports contain `golden_answers` and ordered `retrieval_docs`; each document has an `id` and `contents` containing its title and text.

Place the files in this layout:

```text
data/
├── hotpotqa/
│   └── hotpot_dev_distractor_v1.json
├── 2wiki/
│   ├── train.json
│   └── dev.json
├── musique/
│   ├── musique_ans_v1.0_train.jsonl
│   └── musique_ans_v1.0_dev.jsonl
├── naturalqa/
│   ├── train_1w.json
│   └── dev.json
└── triviaqa/
    ├── train_1w.json
    └── dev.json
```

`configs/paper.json` records these relative paths. JSON arrays, JSONL, Parquet, and Arrow input files are supported. To use a Parquet export of 2Wiki, set its `train_sources` and `eval_sources` to the corresponding `.parquet` files. The same conversion interface accepts native HotpotQA/2Wiki contexts, MuSiQue paragraphs, and canonical candidate-pool records.

## Canonical record format

Each JSONL line contains one question:

```json
{
  "id": "example-001",
  "dataset": "hotpotqa",
  "question": "Which university did the author attend?",
  "answers": ["Example University"],
  "ctxs": [
    {"id": "passage-001", "title": "The work", "text": "The work was written by the author."},
    {"id": "passage-002", "title": "The author", "text": "The author attended Example University."}
  ]
}
```

IDs are normalized to strings. Passage IDs must be unique within a question. Training questions for passage preferences require at least three candidates; two-candidate pools remain valid for inference. Answer aliases are required for gain labeling and evaluation and are optional for new-question inference.

The BGE stage assigns `rerank_score`, fixes the candidate order, and applies the configured pool cap. Later stages add `draft` and `student_score`. Support annotations and answer-bearing flags are preserved as metadata and are excluded from inference features.

## Split construction

Each dataset uses disjoint sets of 100 scorer-training questions, 100 gain-training questions, and 100 development questions. Main-test sizes are 1,000 for HotpotQA, 2Wiki, NQ, and TriviaQA, and 500 for MuSiQue. HotpotQA and 2Wiki additionally reserve 1,000 confirmation questions each. The pipeline evaluates those questions with the main trained scorer, gain ensemble, and development-selected gate, without retraining or retuning. Confirmation predictions and paired metrics are reported separately from the main test.

Questions are ordered by `SHA256(split_seed:dataset:id)` with split seed 2091. Training sets are selected first from training sources; development, main test, and confirmation are selected from evaluation sources after exclusions. Both question IDs and Unicode-normalized question text are checked for overlap and duplication. The split manifest stores selected IDs and source and output checksums.

The MuSiQue test split contains 24 two-hop, 324 three-hop, and 152 four-hop questions, specified by `test_hop_counts`. Preparation selects eligible hash-ordered questions within each hop stratum and restores hash order. Split manifests record requested and observed counts. Hop metadata is used for partitioning and offline diagnostics. When changing test size, update the corresponding hop counts.

Prepare a dataset independently:

```bash
python -m birrag.cli prepare --config configs/paper.json --dataset hotpotqa --output runs/splits/hotpotqa
```

To exclude registered questions or use different source exports:

```bash
python -m birrag.cli prepare --config configs/paper.json --dataset hotpotqa --train-source data/hotpotqa/training.jsonl --eval-source data/hotpotqa/evaluation.jsonl --exclude data/registered_questions.jsonl --output runs/custom_splits/hotpotqa
```

Exclusion files contain question IDs and question text in JSONL format. All requested split sizes must remain available after exclusions.

To train on prepared split files:

```bash
python scripts/run_pipeline.py --config configs/paper.json --dataset hotpotqa --split-dir runs/custom_splits/hotpotqa --output-root runs/custom_experiment
```

A per-dataset split directory contains `scorer_train.jsonl`, `gain_train.jsonl`, `dev.jsonl`, and `test.jsonl`, with `confirmation.jsonl` and `split_manifest.json` produced by the preparation stage. For `--dataset all`, `--split-dir` points to a parent containing one subdirectory per dataset.

## Reusing fixed BGE candidate pools

When importing an established pool, provide `rerank_score` on every passage and run:

```bash
python -m birrag.cli bge --config configs/paper.json --dataset hotpotqa --input data/fixed_pools.jsonl --use-cached-scores --output runs/fixed_pools.bge.jsonl
```

Subsequent draft, scorer, gain, and reader stages consume this fixed pool. Pair indices refer to the fixed candidate order, and final evidence is displayed in that order.
