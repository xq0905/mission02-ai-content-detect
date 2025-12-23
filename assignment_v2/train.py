import os
import json
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        score = item["score"]
        
        if score == 0.0:
            label = 0
        elif score == 0.5:
            label = 1
        else:  # score == 1.0
            label = 2
        
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=False, return_tensors=None)
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": label
        }
    


def read_labeled_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "score" in obj and obj["score"] is not None and (obj.get("text") is not None):
                items.append(obj)
    return items

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=torch.float, device=logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--model", default="/mnt/d/project/models/roberta-base-ai-text-detection-v1")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--lr-scheduler", type=str, default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-strategy", type=str, default="epoch")
    parser.add_argument("--save-strategy", type=str, default="epoch")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-weighted-loss", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_data = read_labeled_jsonl(os.path.join(args.input_dir, "train.jsonl"))
    val_data = read_labeled_jsonl(os.path.join(args.input_dir, "val.jsonl"))
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3, problem_type="single_label_classification", ignore_mismatched_sizes=True)

    train_ds = TextClassificationDataset(train_data, tokenizer, args.max_length)
    val_ds = TextClassificationDataset(val_data, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_model_dir, "logs"),
        logging_steps=50,
        evaluation_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.grad_accum_steps,
        label_smoothing_factor=args.label_smoothing,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    train_labels = []
    for item in train_data:
        s = item["score"]
        if s == 0.0:
            train_labels.append(0)
        elif s == 0.5:
            train_labels.append(1)
        else:
            train_labels.append(2)
    counts = np.bincount(train_labels, minlength=3)
    total = counts.sum()
    class_weights = (total / (counts + 1e-6)).tolist() if np.all(counts > 0) else None

    trainer = WeightedLossTrainer(
        class_weights=class_weights if args.use_weighted_loss else None,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        )],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Final eval:", trainer.evaluate())
    trainer.save_model(args.output_model_dir)
    tokenizer.save_pretrained(args.output_model_dir)

if __name__ == "__main__":
    main()