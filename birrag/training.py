import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .data import passage
from .io import output_path, save_json
from .models import pair_encoding, seed_everything
from .modeling import DraftPassageScorer
from .prompts import student_query


def preference_loss(scores, positive_indices):
    scores = scores.float()
    positives = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)
    positives[positive_indices] = True
    if positives.sum() != 2 or (~positives).sum() == 0:
        raise ValueError(
            "Top-2 preferences require two positives and at least one negative"
        )
    return F.softplus(-(scores[positives, None] - scores[None, ~positives])).mean()


class QuestionDataset(Dataset):
    def __init__(self, rows, tokenizer, config, include_draft=True):
        self.examples = []
        for row in rows:
            if len(row["ctxs"]) < 3:
                raise ValueError(
                    "Scorer training requires at least three candidates per question"
                )
            positives = row["teacher_top2"]
            if len(set(positives)) != 2 or not all(
                0 <= i < len(row["ctxs"]) for i in positives
            ):
                raise ValueError("Invalid teacher Top-2 membership")
            query = student_query(row, include_draft)
            self.examples.append(
                {
                    "pairs": [
                        pair_encoding(tokenizer, query, passage(item), config)
                        for item in row["ctxs"]
                    ],
                    "positives": positives,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class GroupCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        sizes = [len(item["pairs"]) for item in features]
        pairs = [pair for item in features for pair in item["pairs"]]
        return {
            "pairs": self.tokenizer.pad(pairs, padding=True, return_tensors="pt"),
            "sizes": sizes,
            "positives": [item["positives"] for item in features],
        }


class PreferenceTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(**inputs["pairs"])
        scores = outputs.logits.reshape(-1)
        losses = []
        offset = 0
        for size, positives in zip(inputs["sizes"], inputs["positives"]):
            losses.append(preference_loss(scores[offset : offset + size], positives))
            offset += size
        loss = torch.stack(losses).mean()
        return (loss, outputs) if return_outputs else loss


def train_scorer(rows, config, destination, initial_model=None, include_draft=True):
    destination = output_path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    seed_everything(config["seed"])
    initial_model = initial_model or config["models"]["bge"]
    tokenizer = AutoTokenizer.from_pretrained(initial_model, local_files_only=True)
    tokenizer.truncation_side = "right"
    model = DraftPassageScorer.from_pretrained(initial_model, local_files_only=True)
    model.requires_grad_(True)
    dataset = QuestionDataset(rows, tokenizer, config["tokenization"], include_draft)
    arguments = TrainingArguments(
        output_dir=str(destination),
        overwrite_output_dir=False,
        learning_rate=training["learning_rate"],
        num_train_epochs=training["epochs"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=training["gradient_accumulation"],
        warmup_ratio=training["warmup_ratio"],
        weight_decay=training["weight_decay"],
        optim="adamw_torch",
        lr_scheduler_type="linear",
        fp16=training["fp16"],
        bf16=False,
        gradient_checkpointing=training["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=training["max_grad_norm"],
        seed=config["seed"],
        data_seed=config["seed"],
        save_strategy="no",
        eval_strategy="no",
        logging_steps=1,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        use_cpu=config.get("device") == "cpu",
    )
    trainer = PreferenceTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=GroupCollator(tokenizer),
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(destination)
    save_json(
        destination / "training_metadata.json",
        {
            "settings": training,
            "seed": config["seed"],
            "include_draft": include_draft,
            "question_ids": [row["id"] for row in rows],
            "train_metrics": result.metrics,
            "final_epoch": True,
            "initial_model": str(initial_model),
            "tokenization": config["tokenization"],
        },
    )
    return result.metrics
