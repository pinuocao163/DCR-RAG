import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .modeling import DraftPassageScorer
from .data import passage
from .prompts import SYSTEM, student_query


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_dtype(device, dtype):
    return getattr(torch, dtype) if str(device).startswith("cuda") else torch.float32


class CausalGenerator:
    def __init__(self, path, device="cuda:0", dtype="bfloat16", system=SYSTEM):
        self.device = torch.device(device)
        self.system = system
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "right"
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=model_dtype(device, dtype),
            local_files_only=True,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def prompt_ids(self, prompt, input_cap, system=None):
        messages = [
            {"role": "system", "content": self.system if system is None else system},
            {"role": "user", "content": prompt},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer.encode(rendered, add_special_tokens=False)[:input_cap]

    @torch.inference_mode()
    def generate(self, prompt, input_cap, output_cap):
        ids = self.prompt_ids(prompt, input_cap)
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        outputs = self.model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=output_cap,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(
            outputs[0, len(ids) :], skip_special_tokens=True
        ).strip()

    @torch.inference_mode()
    def mean_logprob(self, prompt, answer, input_cap):
        answer_ids = self.tokenizer.encode(answer.strip(), add_special_tokens=False)
        if not answer_ids:
            return 0.0
        prompt_ids = self.prompt_ids(prompt, input_cap)
        all_ids = torch.tensor(
            [prompt_ids + answer_ids], dtype=torch.long, device=self.device
        )
        logits = (
            self.model(
                input_ids=all_ids,
                attention_mask=torch.ones_like(all_ids),
                use_cache=False,
                logits_to_keep=len(answer_ids) + 1,
            )
            .logits[0, :-1]
            .float()
        )
        targets = torch.tensor(answer_ids, dtype=torch.long, device=self.device)
        return float(-F.cross_entropy(logits, targets, reduction="mean").cpu())


class VLLMGenerator:
    def __init__(self, path, config):
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = LLM(
            model=path,
            max_model_len=8192,
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.45),
            dtype=config.get("dtype", "bfloat16"),
            seed=config["seed"],
            tensor_parallel_size=1,
        )

    def generate(self, prompt, input_cap, output_cap):
        from vllm import SamplingParams

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )[:input_cap]
        outputs = self.model.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(temperature=0.0, max_tokens=output_cap),
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text.strip()


def pair_encoding(tokenizer, query, document, config):
    query_ids = tokenizer.encode(
        query, add_special_tokens=False, truncation=True, max_length=config["query_cap"]
    )
    document_ids = tokenizer.encode(
        document,
        add_special_tokens=False,
        truncation=True,
        max_length=config["document_cap"],
    )
    return tokenizer.prepare_for_model(
        query_ids,
        document_ids,
        truncation="only_second",
        max_length=config["joint_cap"],
        padding=False,
    )


class PassageScorer:
    def __init__(
        self, path, device="cuda:0", dtype="float32", token_config=None, semantic=False
    ):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.tokenizer.truncation_side = "right"
        model_class = (
            AutoModelForSequenceClassification if semantic else DraftPassageScorer
        )
        self.model = (
            model_class.from_pretrained(
                path, local_files_only=True, torch_dtype=model_dtype(device, dtype)
            )
            .to(self.device)
            .eval()
        )
        self.model.requires_grad_(False)
        self.config = token_config or {
            "query_cap": 200,
            "document_cap": 464,
            "joint_cap": 512,
        }
        self.semantic = semantic

    @torch.inference_mode()
    def score(self, row, batch_size=8, include_draft=True):
        query = row["question"] if self.semantic else student_query(row, include_draft)
        encoded = [
            pair_encoding(self.tokenizer, query, passage(item), self.config)
            for item in row["ctxs"]
        ]
        scores = []
        for start in range(0, len(encoded), batch_size):
            batch = self.tokenizer.pad(
                encoded[start : start + batch_size], padding=True, return_tensors="pt"
            ).to(self.device)
            scores.extend(self.model(**batch).logits.reshape(-1).float().cpu().tolist())
        if len(scores) != len(row["ctxs"]):
            raise ValueError("The scorer must output exactly one logit per passage")
        return scores
