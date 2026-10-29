# DCR-RAG

Official implementation of **DCR-RAG: Draft-Conditioned Ranking with Conservative Passage Replacement for Retrieval-Augmented Generation**.

We introduce DCR-RAG to select a compact context for retrieval-augmented question answering. Our method uses a candidate-pool answer draft to condition passage ranking, then accepts a single-passage replacement only when its predicted reader benefit exceeds a conservative acceptance criterion. The final reader receives the question and two selected original passages.

The accompanying manuscript is [main.pdf](main.pdf). Implementation details and feature definitions are provided in [docs/METHOD.md](docs/METHOD.md), and dataset formats are specified in [docs/DATA.md](docs/DATA.md).

## Method overview

1. **Pool-conditioned drafting.** A frozen BGE reranker establishes candidate order. Qwen2.5-3B-Instruct generates one short draft from the ordered pool.
2. **Answer-free passage supervision.** The teacher combines normalized BGE relevance with the decrease in the fixed draft's mean answer-token log probability after passage deletion. Its Top-2 membership defines selected–unselected preferences.
3. **Draft-conditioned ranking.** A BGE-initialized XLM-R cross-encoder jointly encodes the question, draft, and each passage. A ReLU scoring head is trained using an equally weighted pairwise logistic loss.
4. **Conservative replacement.** At most eight candidate pairs each retain one passage from the initial Top-2. Five question-fold Ridge predictors estimate reader-F1 gain. The best proposal is accepted only if `mean_gain - kappa * population_std > tau`; otherwise the initial pair is retained.
5. **Final reading.** Qwen2.5-7B-Instruct answers from the selected passages in BGE order. The draft is excluded from the reader input.

Gold answers supervise offline gain labels and development selection. Inference-time selection uses the question, candidate text, draft, and fitted models.

## Project structure

```text
.
├── README.md
├── main.pdf
├── paper.sha256
├── pyproject.toml
├── requirements.txt
├── requirements-vllm.txt
├── THIRD_PARTY_NOTICES.md
├── birrag/
│   ├── __init__.py
│   ├── modeling.py
│   ├── models.py
│   ├── training.py
│   ├── data.py
│   ├── prompts.py
│   ├── teacher.py
│   ├── features.py
│   ├── gain.py
│   ├── policies.py
│   ├── experiments.py
│   ├── reading.py
│   ├── metrics.py
│   ├── support.py
│   ├── visibility.py
│   ├── budgets.py
│   ├── config.py
│   ├── io.py
│   └── cli.py
├── configs/
│   ├── paper.json
│   ├── paper_hf.json
│   └── readers.json
├── scripts/
│   ├── run_pipeline.py
│   ├── infer.py
│   ├── compare_formulations.py
│   ├── compare_policies.py
│   ├── compare_acceptance.py
│   ├── compare_scorers.py
│   ├── compare_budgets.py
│   ├── compare_search.py
│   ├── compare_lodo.py
│   ├── compare_readers.py
│   ├── compare_seeds.py
│   └── package_submission.py
└── docs/
    ├── DATA.md
    └── METHOD.md
```

`birrag` is the implementation package. Model definitions and training are separated from data conversion, teacher construction, gain prediction, reading, and evaluation. The `scripts` directory provides experiment entry points. Runtime datasets, model snapshots, caches, and run outputs are stored separately from the submitted source files.

## Environment and installation

Use Python 3.10 on Linux with CUDA for the full experiments. The paper configuration uses PyTorch 2.5.1, Transformers 4.50.0, scikit-learn 1.6.1, and vLLM 0.7.3. All commands below are executed from the extracted project root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[vllm]"
```

For a Transformers-only installation:

```bash
python -m pip install -e .
```

Use `configs/paper_hf.json` with this installation. Core dependencies are pinned in [requirements.txt](requirements.txt), with the optional vLLM environment specified in [requirements-vllm.txt](requirements-vllm.txt).

| Configuration | Draft generation | Final reading |
|---|---|---|
| `configs/paper.json` | vLLM | Transformers |
| `configs/paper_hf.json` | Transformers | Transformers |
| `configs/readers.json` | Reader-transfer model paths and loading options | Used by `compare_readers.py` |

The main configuration selects `cuda:0`. For CPU execution with a reduced experiment, set `device` to `cpu`, disable `training.fp16`, and use the Transformers backend. Reader-transfer settings support `device_map` for model placement and `trust_remote_code` for locally supplied model implementations.

## Data and pretrained models

Obtain complete local model snapshots, including tokenizer and configuration files:

| Component | Model | Default path |
|---|---|---|
| Relevance prior and scorer initialization | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | `models/bge-reranker-base` |
| Draft and deletion teacher | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | `models/Qwen2.5-3B-Instruct` |
| Reader and offline gain labeling | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | `models/Qwen2.5-7B-Instruct` |

Place the dataset source files under the paths listed in [docs/DATA.md](docs/DATA.md). The supported sources are HotpotQA distractor data, 2WikiMultiHopQA, MuSiQue-Ans, and the E5 Top-10 retrieval exports for Natural Questions and TriviaQA.

| Dataset key | Maximum pool size | Main-test questions | Confirmation questions |
|---|---:|---:|---:|
| `hotpotqa` | 10 | 1,000 | 1,000 |
| `2wiki` | 10 | 1,000 | 1,000 |
| `musique` | 20 | 500 | 0 |
| `naturalqa` | 10 | 1,000 | 0 |
| `triviaqa` | 10 | 1,000 | 0 |

Each dataset additionally uses disjoint sets of 100 scorer-training, 100 gain-training, and 100 development questions. The MuSiQue test configuration specifies 24 two-hop, 324 three-hop, and 152 four-hop questions. Preparation records selected IDs and source checksums and checks overlap by question ID and normalized text.

Relative asset paths resolve from the project root. Update `models`, `train_sources`, and `eval_sources` in the configuration to use another asset location. Keep output paths inside the project root, conventionally under `runs/`, and use a separate output directory for each experiment.

Inspect installed dependencies, CUDA availability, and configured asset locations:

```bash
python -m birrag.cli check --config configs/paper.json --output runs/environment.json
```

## Main training and evaluation

After preparing the model snapshots and datasets, run the complete five-dataset protocol:

```bash
python scripts/run_pipeline.py --config configs/paper.json --dataset all --output-root runs/paper_seed2094 --label-test-proposals
```

For Transformers-only drafting, substitute `configs/paper_hf.json`. The pipeline prepares splits, scores candidate pools, generates drafts, constructs teacher preferences, trains scorers, labels reader gains, and fits dataset-specific gain ensembles. Once all datasets finish training, one shared development-macro search selects acceptance parameters. Test and confirmation evaluation follow.

| Setting | Main configuration |
|---|---|
| Scorer optimizer | AdamW; learning rate `1e-6`; weight decay `0.01` |
| Training schedule | 3 epochs; linear decay; warm-up ratio `0.1` |
| Batch and accumulation | 1 question group per device; 4 accumulation steps |
| Training precision | FP16 with gradient checkpointing |
| Query / passage / joint caps | 200 / 464 / 512 tokens; `only_second` joint truncation |
| Draft input / output caps | 4072 / 24 tokens; greedy decoding |
| Reader input / output caps | 2048 / 32 tokens; greedy decoding |
| Generation precision | bfloat16 on CUDA |
| Draft vLLM context / memory fraction | 8192 tokens / `0.45` |
| Teacher necessity coefficient | `0.75`; positive-response 90th-percentile scale |
| Gain ensemble | 5 question folds; member-specific standardization; Ridge `alpha=10` |
| Local proposals | Up to 4 alternatives per position; at most 8 distinct pairs |
| Training / split / bootstrap seeds | 2094 / 2091 / 2092 |
| Gain-fold salt | 2094, fixed across training seeds |
| Bootstrap replicates | 10,000 |

The shared search covers `kappa={0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3}` and `tau={0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1}` on gains in `[-1, 1]`. It maximizes equally weighted development F1 across datasets. The paper's selected values, `kappa=1` and `tau=0.04`, are recorded in `selector.paper_gate` for standalone single-dataset runs.

To run only HotpotQA:

```bash
python scripts/run_pipeline.py --config configs/paper.json --dataset hotpotqa --output-root runs/hotpotqa_seed2094 --label-test-proposals
```

A standalone main-method run uses the recorded paper gate. Supply `--shared-gate runs/paper_seed2094/shared_gate.json` to use a gate selected by an existing joint run. For registered split files, use `--split-dir` as described in the data guide; `--reuse-splits-from` reads the exact partitions of an existing experiment.

`--label-test-proposals` generates offline test proposal answers for gain diagnostics and the cached-output comparisons below. These labels are computed after training and acceptance selection.

## Testing, metrics, and outputs

The main pipeline evaluates BGE Top-2, independent Top-2 from our learned scorer, and conservative replacement with the same final reader. EM and token F1 use complete normalized answers and maximize over gold aliases. Generation failures remain in the denominators. Dataset macro averages assign equal weight to each dataset.

The experiment root contains:

| Output | Contents |
|---|---|
| `resolved_config.json` | Effective model, data, training, and experiment settings |
| `shared_gate.json` | Shared acceptance grid and selected parameters |
| `metrics.json`, `route_metrics.json` | Main-test EM/F1 summaries |
| `paired_effect.json` | Conditional-minus-independent paired F1 interval |
| `test.<route>.predictions.jsonl` | Aggregated predictions for `conditional`, `independent`, and `bge` |
| `support_diagnostics.json` | Selected and reader-visible support statistics |

Each dataset directory contains `splits/`, `scorer/`, `gain_ensemble.json`, `gate.json`, per-route predictions, and diagnostic outputs. HotpotQA and 2Wiki confirmation questions reuse the main scorer, ensemble, and gate. Confirmation outputs are reported separately as `confirmation_metrics.json`, `confirmation_route_metrics.json`, and `confirmation_paired_effect.json`.

Evaluate a saved pair of routes independently:

```bash
python -m birrag.cli evaluate --config runs/paper_seed2094/resolved_config.json --input runs/paper_seed2094/hotpotqa/test.conditional.predictions.jsonl --compare runs/paper_seed2094/hotpotqa/test.independent.predictions.jsonl --output runs/hotpotqa_evaluation.json
```

## Experimental comparisons

The commands in this section use the complete five-dataset run at `runs/paper_seed2094`, including `--label-test-proposals` outputs.

```bash
python scripts/compare_formulations.py --run-root runs/paper_seed2094 --output-root runs/formulations
python scripts/compare_policies.py --run-root runs/paper_seed2094 --output-root runs/policies
python scripts/compare_acceptance.py --run-root runs/paper_seed2094 --output-root runs/acceptance
python scripts/compare_scorers.py --run-root runs/paper_seed2094 --output-root runs/scorers
python scripts/compare_budgets.py --run-root runs/paper_seed2094 --dataset musique --output-root runs/budgets
python scripts/compare_search.py --run-root runs/paper_seed2094 --output-root runs/search
python scripts/compare_lodo.py --run-root runs/paper_seed2094 --output-root runs/lodo
```

| Script | Comparison |
|---|---|
| `compare_formulations.py` | Direct, difference, baseline-aware, shared-absolute, single-Ridge, and zero-gain formulations; shared development selection per formulation |
| `compare_policies.py` | Joint-pair MMR, coverage, and combined heuristics; frozen gain-model transfer |
| `compare_acceptance.py` | Mean-only, `kappa=0`, `tau=0`, and development-matched acceptance curves |
| `compare_scorers.py` | Learned/native BGE scoring crossed with independent/conditional selection; paired interaction |
| `compare_budgets.py` | One to four passages; fixed reader caps; quality and support coverage |
| `compare_search.py` | Eight-pair, sixteen-pair, and exhaustive search; proposal counts and quality |
| `compare_lodo.py` | Source-only gain fitting and development selection while retaining each target scorer |

Add `--oracle` to the search command for the separately labeled offline observed-F1 oracle. Single-dataset comparisons support `--dataset` where shown by the script's `--help`; LODO requires source experiments from multiple datasets. MLP gain prediction is opt-in via `--formulations direct mlp --mlp-config` and a JSON file defining `hidden_layer_sizes`, `activation`, `solver`, `alpha`, `learning_rate_init`, and `max_iter`.

For scorer and supervision ablations, retain the same questions, candidate sources, and training configuration:

```bash
python scripts/run_pipeline.py --config runs/paper_seed2094/resolved_config.json --dataset all --reuse-splits-from runs/paper_seed2094 --ablation no_draft --output-root runs/no_draft_seed2094 --label-test-proposals
```

Supported variants are `no_draft`, `question_only`, `bge_only`, `deletion_only`, and `no_backfill`. Each variant retrains the affected components and repeats development selection. No-draft scoring and gain features exclude draft content while the answer-free teacher retains its reference draft. No-backfill deletion removes tokens only from the original truncated prompt. Detailed definitions are given in the method guide.

For fixed-selector reader transfer, obtain the snapshots specified in `configs/readers.json`:

```bash
python scripts/compare_readers.py --run-roots runs/paper_seed2094 --readers-config configs/readers.json --output-root runs/reader_transfer
```

Selected passage IDs, trained gain models, acceptance parameters, prompts, and token caps remain fixed. The script reports each reader's F1 and beneficial, unchanged, and harmful outcomes on the same accepted swaps.

## Reproducing the multi-seed experiments

Run the additional training seeds while retaining the split seed and gain-fold salt:

```bash
python scripts/run_pipeline.py --config configs/paper.json --dataset all --seed 2095 --output-root runs/paper_seed2095 --label-test-proposals
python scripts/run_pipeline.py --config configs/paper.json --dataset all --seed 2096 --output-root runs/paper_seed2096 --label-test-proposals
```

Compare the paired conditional and independent predictions:

```bash
python scripts/compare_seeds.py --first runs/paper_seed2094/test.conditional.predictions.jsonl runs/paper_seed2095/test.conditional.predictions.jsonl runs/paper_seed2096/test.conditional.predictions.jsonl --second runs/paper_seed2094/test.independent.predictions.jsonl runs/paper_seed2095/test.independent.predictions.jsonl runs/paper_seed2096/test.independent.predictions.jsonl --seed-ids 2094 2095 2096 --output runs/seed_comparison.json
```

The default first averages each question's paired effects across seeds, then resamples questions within datasets. Set `--resampling joint` to resample questions and training seeds with pairing. Reports include input checksums and question counts. Reader transfer accepts multiple seed directories through `--run-roots` and produces both interval types.

All metrics and diagnostic tables are calculated from generated predictions and cached reader outputs. The source archive contains the implementation and configurations; datasets, pretrained snapshots, trained checkpoints, and measured run outputs are obtained or generated through the procedures above.

## Inference on new questions

Create a canonical JSONL input using the format in the data guide. Answer aliases are optional. The following synthetic record illustrates the schema; inference uses the HotpotQA artifacts from the main run:

```bash
mkdir -p data
cat > data/query_example.jsonl <<'JSON'
{"id":"query-example","question":"Which university did the author of The Example Novel attend?","ctxs":[{"id":"work","title":"The Example Novel","text":"The Example Novel was written by Author A."},{"id":"author","title":"Author A","text":"Author A attended Example University."}]}
JSON
python scripts/infer.py --dataset hotpotqa --input data/query_example.jsonl --artifacts runs/paper_seed2094/hotpotqa --output-root runs/inference
```

Inference loads the saved experiment configuration by default. Its `predictions.jsonl` records selected original passages, replacement decisions, reader-input statistics, and the final answer.

## Submission archive and acknowledgments

```bash
python scripts/package_submission.py
```

The upload artifact is `dist/dcrrag_submission.zip`. The packager includes only the manuscript, implementation, configurations, dependencies, and documentation. It verifies the PDF against `paper.sha256` and checks ZIP integrity. Runtime data, weights, caches, logs, and experiment outputs are excluded.

Source, dependency, model, and dataset acknowledgments are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). External datasets and model snapshots retain their providers' terms and accompanying licenses.
