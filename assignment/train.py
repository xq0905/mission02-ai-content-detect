import os
import json
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import clean_text


class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        text = clean_text(item.get("generation") or "")
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=False, return_tensors=None)
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": 1 if item["label"] else 0
        }
    


def read_labeled_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "label" in obj and obj["label"] is not None and (obj.get("generation") is not None):
                items.append(obj)
    return items

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/mnt/d/project/Professor_Chen/task1027/xq/dataset")
    parser.add_argument("--output", default="/mnt/d/project/Professor_Chen/task1027/xq/roberta-base-ai-text-detection-v1/output_model")
    parser.add_argument("--model", default="/mnt/d/project/models/roberta-base-ai-text-detection-v1")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    args = parser.parse_args()

    train_data = read_labeled_jsonl(os.path.join(args.input, "train.jsonl"))
    val_data = read_labeled_jsonl(os.path.join(args.input, "val.jsonl"))
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, problem_type="single_label_classification")

    train_ds = TextClassificationDataset(train_data, tokenizer, args.max_length)
    val_ds = TextClassificationDataset(val_data, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output, "logs"),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
    )

    trainer = Trainer(
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
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

if __name__ == "__main__":
    main()