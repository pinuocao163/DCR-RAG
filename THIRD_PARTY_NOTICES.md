# Third-Party Notices

## Source acknowledgments

Data-adapter conventions, question–draft query formatting, passage prompts, deletion-necessity utilities, and text/score feature conventions draw on the [GainRAG source distribution](https://github.com/liunian-Jay/GainRAG) accompanying:

Yi Jiang, Sendong Zhao, Jianbo Li, Haochun Wang, and Bing Qin. *GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis*. arXiv:2505.18710, 2025. [Paper](https://arxiv.org/abs/2505.18710).

These acknowledgments cover the corresponding conventions in `birrag/data.py`, `prompts.py`, `teacher.py`, `models.py`, `training.py`, and `features.py`. Rights to external source materials remain with their holders, and applicable upstream terms govern their use.

The XLM-R scorer and Qwen wrappers extend Hugging Face Transformers interfaces. Transformers, PyTorch, scikit-learn, and other dependencies are installed separately; their notices and license texts are provided by their distributions.

## Model assets

The implementation loads the following external pretrained models:

- [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base).
- [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

Obtain complete model snapshots, including their accompanying licenses, from their providers. Model weights and tokenizer assets are stored locally under `models/` and are excluded from the code archive.

## Dataset assets

HotpotQA, 2WikiMultiHopQA, MuSiQue, Natural Questions, and TriviaQA remain subject to their respective dataset terms. The NQ/TriviaQA candidate format follows the E5 retrieval exports distributed with [RAG-Critic](https://github.com/RUC-NLPIR/RAG-Critic). Dataset files are obtained separately and are excluded from the code archive.
