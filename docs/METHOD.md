# Method and Experimental Configuration

## Passage scoring and teacher supervision

`birrag/modeling.py` defines the draft-conditioned XLM-R passage scorer. It retains the pretrained parameter layout and uses a classification head with a dense projection, ReLU, dropout, and a scalar output. Question and draft form one query field; passage title and text form the second field.

The tokenizer pretruncates query and passage to 200 and 464 tokens. Joint encoding has a 512-token cap with `only_second` truncation. The scorer is initialized from BGE-reranker-base, and all parameters are updated with equal-weight selected–unselected logistic preferences. AdamW uses learning rate `1e-6`, three epochs, one question group per device, accumulation four, warm-up `0.1`, weight decay `0.01`, FP16, and gradient checkpointing. The checkpoint is the final epoch.

For a fixed draft, deletion necessity is the positive part of the full-pool minus deleted-pool mean answer-token log probability. Prompt, padding, and an additionally appended EOS are excluded from the likelihood average. Deletion rebuilds the prompt in the original candidate order under the same input cap. The draft remains fixed.

The teacher utility is the pool-normalized BGE score plus `0.75 * min(necessity / scale, 1)`. The scale is the 90th percentile of positive necessity values from scorer-training questions, floored at `1e-8`; with no positive values it is one. Empty drafts have zero necessity. Utility ties follow BGE order and passage ID. The teacher's two selected passages define positive membership for pairwise training.

## Local pair proposals

`birrag/features.py` forms the initial pair from the two highest student logits. For each retained initial passage, it combines that passage with at most four highest-scoring candidates outside the initial pair. Duplicate pairs are removed. At most eight proposals are evaluated; every proposal shares one passage with the initial pair.

Pair indices refer to the fixed BGE-ordered pool. Final reading preserves that order. Ties use deterministic candidate order. With no replacement proposal, the initial pair is returned.

## Pair features

The main method uses the 34 features in `FEATURE_NAMES`, with the same ordering across datasets:

| Feature group | Fields |
|---|---|
| Question and draft context | Log question length, log draft length, log pool size |
| Student scores | Pair maximum, minimum, mean, absolute difference |
| BGE scores | Pair maximum, minimum, mean, absolute difference |
| Draft alignment | Number of pair passages containing the complete normalized draft |
| Question coverage | Union coverage, mean per-passage coverage, two directional residual coverages |
| Passage relationship | Token-set Jaccard, mean directional novelty, title-link indicator |
| Passage length | Log maximum, minimum, mean |
| Fixed BGE rank | Log minimum, maximum, mean |
| Normalized ranks | Student-rank sum, BGE-rank sum |
| Question type | Indicators for who, what, when, where, which, how, why |

Length and fixed-rank transforms use `log1p`. Student and BGE scores are normalized independently within each pool; constant pools return zero. Stable ranks are normalized by `max(1, pool_size - 1)`. Lexical tokens use Unicode NFD, case folding, and Unicode `\w+`, retaining articles and stopwords. A draft match requires the entire normalized draft-token sequence to occur contiguously in passage title and text.

Question coverage divides intersection size by `max(1, question_token_count)`. Directional residual coverage considers question tokens absent from the other passage. Jaccard uses the union-size denominator, and directional novelty measures tokens unique to each passage. The title-link indicator checks whether either passage contains the other passage's normalized title.

Feature extraction uses question, draft, candidate text, and model scores. Gold answers, support flags, and observed reader F1 enter offline labeling or evaluation rather than inference features. Direct Gain encodes the proposal pair; it does not concatenate a separate initial-pair representation. Dataset configurations select `auxiliary=none`.

## Gain fitting and acceptance

`birrag/gain.py` labels each proposal with its complete-answer reader F1 minus the initial pair's reader F1. The frozen reader, decoding, and token budgets are shared across all pairs. Scorer-training and gain-training questions are disjoint.

Question folds use the integer value of `SHA256(fold_seed || dataset || id)` modulo five. The fields are concatenated as UTF-8 strings without a separator. Every proposal for one question belongs to its question's fold. `fold_seed=2094` is frozen across training seeds. Each Ridge member fits its own `StandardScaler` on four folds, then fits an intercept and coefficients with `alpha=10`. Constant features use scale one.

Inference computes the mean and population standard deviation (`ddof=0`) of the five gain predictions. A proposal's conservative score is `mean - kappa * std`. The best proposal is accepted only when this score is strictly greater than `tau`; otherwise the initial pair remains selected.

The main acceptance grid is `kappa={0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3}` and `tau={0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1}` on normalized F1 gains. Five development sets of 100 questions select one shared pair by equal-weight dataset macro F1. All dataset-specific gain models are fitted before the search; test questions never enter it. Grid ties prefer fewer accepts, then larger kappa and tau. The paper's selected pair is kappa=1 and tau=0.04. A single-dataset pipeline uses this published fixed pair unless `--shared-gate` is supplied. Each gate records its scope and gain-bundle checksum. All learned artifacts and acceptance parameters remain fixed during testing and confirmation.

## Gain-formulation controls

The comparison script uses shared question sets, proposals, and cached reader outputs:

- **Direct Gain:** proposal features predict F1 gain.
- **Baseline-aware:** concatenated proposal, initial-pair, and proposal-minus-initial features predict gain.
- **Baseline-aware two-block:** concatenated proposal and initial-pair features predict gain, omitting the linearly dependent difference block.
- **Difference:** proposal-minus-initial features predict gain.
- **Shared absolute:** a common Ridge member predicts absolute F1 for both pairs; their predictions are subtracted. Each initial pair contributes once per training question.
- **Single Ridge:** one predictor fits all gain-training questions, with no disagreement penalty.
- **Zero Gain:** predicted gains and disagreement are zero.
- **MLP gain:** five fold-specific, standardized networks regress proposal gains. Network structure and optimizer settings are supplied explicitly with `--mlp-config`; the fitted weights and activation are serialized with the same feature schema.

Feature transformations precede member-specific standardization. Initial-pair observed F1 is used only for offline diagnostic strata. Harm means a strict decrease in reader F1 relative to the initial pair.

## Generation and evaluation

The draft uses Qwen2.5-3B-Instruct with 4072 input tokens and 24 new tokens. The final reader uses Qwen2.5-7B-Instruct with 2048 input tokens and 32 new tokens. Generation is greedy with bfloat16 on CUDA; input caps apply to the rendered chat prompt with right truncation. The paper configuration uses vLLM for the draft with context 8192 and GPU memory fraction `0.45`. The 7B reader and teacher-forced likelihood computation use Transformers. The `reader_backend` setting selects the reader independently of the draft backend. The executable prompt text is in `birrag/prompts.py`.

The main seed is 2094, with stability seeds 2094–2096 and fixed split seed 2091. Complete answers are normalized and compared against all gold aliases, taking the maximum EM and token F1. HotpotQA yes/no/noanswer mismatch rules are applied. Generation failures and empty-normalized aliases remain in metric denominators.

Dataset metrics are macro-averaged with equal weight. Paired F1 intervals use 10,000 question-bootstrap samples within each dataset, preserve route pairing, and use seed 2092. Gain MAE first averages over proposals within each question, then over questions within each dataset and equally over datasets.

## Held-out confirmation and common controls

The main-test and confirmation routes share the main trained scorer, gain ensemble, feature schema, decoding, development grid, and chosen threshold. Confirmation changes only the held-out questions. `run_pipeline.py` includes the confirmation split in ID/text isolation checks and evaluates it after fitting and development selection. It hashes the trained scorer files, gain ensemble, and gate before held-out evaluation and verifies the same hashes afterwards. No confirmation question is passed to the training or gate-selection stages.

Main-test and confirmation compare conditional and independent selection using the same trained scorer. Their populations and macro results are reported separately.

All gain-formulation controls reuse fitting questions, development questions, proposals, reader outputs, and features. Ridge formulations use alpha=10; the MLP uses its supplied configuration. Two-block concatenation retains 68 standardized inputs for the 34-dimensional base schema; three-block retains 102. Each model fits its own training-fold standardizer after feature transformation, and all use the same development search.

## Draft-match sparsity

`draft_match_diagnostics` reports the lexical flag's nonzero count divided by all candidate passages evaluated for the question population, counted once per question/passage. Empty drafts never match; passage scoring uses joint question–draft encoding.

## Fixed-scorer heuristics and frozen transfer

Appendix B.3 (equations 18–19) defines `A=(p_u+p_v)/2` for pool-normalized student scores, passage Jaccard `J`, union question coverage `C`, and mean bidirectional residual coverage `R`. The implemented pair scores are `A-lambda*J`, `A+beta*C`, and `A+beta*C+gamma*R-delta*J`. The search domain is the original Top-2 pair plus its at-most-eight replacements. MMR compares whole pairs; it does not fix the first passage. Equal scores retain the original pair, then BGE order.

Policy weights use the common development datasets and maximize equal-weight dataset macro F1 over `{0,.25,.5,.75,1}`: 5, 5, and 125 configurations. They use development absolute answer F1 but no reader-gain training targets. Weight-grid ties prefer fewer departures from original Top-2, then smaller weights.

Policy-specific replacement proposals retain one passage from the chosen heuristic pair. The main Direct Gain ensemble, fitted preprocessing, kappa, and threshold are applied without refitting. New observed gain targets are computed only for offline error reporting relative to that policy's initial-pair answer. Cached reader answers are reused, and missing bounded-local-search pairs are generated with the same reader/settings. The code saves initial and after predictions for paired analysis and verifies unchanged ensemble/gate checksums.

## Matched mean-only acceptance

Mean-only retains every ensemble member and its predicted gain, setting kappa to zero. Its matched threshold is selected from development best-proposal means to approximate the development acceptance of the frozen full rule. The acceptance curves similarly select thresholds for targets 20%, 25%, 31.7%, 40%, and 50%, with main kappa for full and zero for mean-only. No development gold answer or observed reader F1 is consulted by the threshold-matching function.

Thresholds are chosen at observed score cut points plus a point below the minimum, enforcing the same strict `score > tau` rule. Equal-score groups are never split. If a target is unattainable, the closest development count is selected, then fewer accepts and a larger threshold break ties. Negative thresholds are permitted for target curves; neither main model weights nor the quality-selected main gate is altered.

Diagnostics separately count beneficial, unchanged, and harmful accepts; preserve failures and zero-change questions in the denominator; and report positive and negative gain means. Formal gain MAE uses per-question proposal averages, followed by equal dataset macro weighting. Initial-F1-bin diagnostics additionally report proposal-weighted MAE and raw error sums/counts; those pooled bin values must not be confused with the formal dataset-macro metric. Gain-formulation and policy contrasts use question-paired bootstrap at seed 2092 with 10,000 draws.

## Paired statistical comparisons

`scripts/compare_seeds.py` compares paired prediction files across training seeds. All arms and seeds share dataset/question IDs, question text, and answer aliases. Paired F1 differences are first averaged across seeds for each question. Each bootstrap replicate resamples questions within datasets and equally averages dataset effects. The report contains per-dataset and macro confidence intervals, per-seed macro effects, sample standard deviation, and input checksums. Failed generations contribute zero F1.

## Support and reader-input accounting

Support recall averages the selected fraction of annotated supporting passages per question. Complete coverage requires every annotated support passage. MuSiQue diagnostics include hop subsets and decomposition-position recall. Supporting paragraph indices are retained through candidate reordering.

Reader predictions record rendered-chat token counts, the input cap, a token-ID checksum, and retained title/text token counts per passage. Token offsets and chat-template token IDs are checked against the generation input.

Support reports provide three views:

- `selected`: selected original passage IDs before reader truncation.
- `reader_visible`: passages with at least one original text token retained.
- `reader_fully_visible`: passages with all original text tokens retained.

Title-only and passage-header tokens are excluded from text visibility. Each view retains the original annotated-support denominator. Reports give annotation and trace availability counts. Paired reader-view intervals use jointly traceable, annotated questions. Answer F1 and input-visibility statistics are computed separately.

## Passage-budget comparison

`compare_budgets.py` uses the highest student score at k=1, the saved main conditional pair at k=2, and appends highest-scored remaining passages at k=3/4. Score ties follow BGE order, as does final display order. The reader input/output caps remain 2048/32 tokens. BGE selects the first k candidates in the same pool.

The script verifies question metadata, passage IDs/text/order, and trained-artifact checksums. It writes per-budget DCR-RAG/BGE predictions, support metrics, paired F1 intervals, and consecutive-budget F1 effects. Scorer, gain ensemble, preprocessing, and acceptance settings remain fixed.

## Joint uncertainty and reader transfer

`joint_paired_bootstrap` samples training seeds and question IDs with route pairing. Every replicate shares its seed draws across datasets and averages dataset effects equally. Question draws within a dataset are shared across selected seeds and routes. This differs from seed-averaged intervals, which average per-question effects before resampling only questions.

`compare_readers.py` freezes the independent and conditional passage IDs produced with the main learned scorer and selector. Each new reader receives the same original passage text, BGE order, prompt, and input/output caps. The scorer, gain models, and acceptance settings are neither fitted nor tuned for the new reader. Checksums are verified before and after transfer. The accepted-swap population is determined by the frozen selection, and beneficial/unchanged/harmful outcomes are recomputed from the new reader's answers. Single-seed and multi-seed outputs include paired F1 effects, counts, and harm/accepted rates.

## Scorer and teacher ablations

The `method` configuration records the draft source, scorer draft input, teacher utility, and deletion context. Main defaults are `pool`, `true`, `hybrid`, and `backfill`. `run_pipeline.py --ablation` changes the investigated factor, retrains affected models, and reselects acceptance on the development splits. `--reuse-splits-from` reads the source run's exact split files, verifies configured split sizes, and writes all new artifacts under a separate output root.

For `no_draft`, the scorer sees the question alone; draft length and draft-match gain features are zero. The answer-free teacher still uses its cached reference draft for preference construction. The question-only-draft arm uses the same drafting model and decoding with an empty evidence list. BGE-only and deletion-only teachers retain their respective terms of the hybrid utility and use the same Top-2 preference loss and training settings.

For `no_backfill`, token offsets identify complete passage blocks in the original rendered prompt. The prompt is first truncated at 4072 tokens. Deletion removes only tokens belonging to that passage from this retained sequence; it neither retokenizes the remaining text nor admits later tokens. The fixed draft is evaluated with `mean_logprob_ids`. The main backfill arm rebuilds the prompt and reapplies the cap after each deletion.

`compare_scorers.py` uses the native, unchanged BGE classification head on the same question–draft field. Main gain predictors, fitted preprocessing, and gate remain frozen. Both native and learned scorers serve independent and conditional routes. The interaction is `(learned full - learned independent) - (native full - native independent)`, evaluated on paired questions.

## Leave-one-dataset-out gain transfer

`compare_lodo.py` pools gain-training questions only from source datasets. Fold membership hashes the original dataset and question ID with the frozen salt; IDs from different datasets remain distinct. Each member fits its normalizer and Ridge weights on four source folds. Development macro F1 over source datasets selects acceptance. Target gain-training and development questions are never used for fitting or selection. Target test records are used after fitting for quality and error reporting; the target scorer and initial pairs are unchanged. Input partition checks reject ID or normalized-question leakage.

## Proposal-search controls

`compare_search.py` generates four or eight alternatives per initial position for the eight- and sixteen-pair caps, respectively, and deduplicates them. Exhaustive search evaluates all distinct passage pairs except the fallback initial pair. Proposal features, gain ensemble, population disagreement, strict acceptance, reader, and token budgets are fixed. The report measures whether the eight-pair set contains an exhaustive risk-score maximizer, with a numerical tie tolerance of 1e-12. The optional oracle evaluates the initial pair and all pairs using observed answer F1; it is labeled as an offline result.

Reader generation failures are retained as empty answers with explicit errors and zero EM/F1 in both offline gain labeling and final evaluation. Cached formulation and policy comparisons propagate those errors rather than converting failed empty answers into matches to empty-normalized aliases.
