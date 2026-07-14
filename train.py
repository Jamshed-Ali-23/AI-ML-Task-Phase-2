import json
import numpy as np
import torch
import datasets

datasets.config.TORCHVISION_AVAILABLE = False

from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "bert-base-uncased"
SAVE_DIR = "news_bert_model"
SUBSET_TRAIN = 30000
SUBSET_TEST = 3000

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

raw = load_dataset("fancyzhx/ag_news")

if SUBSET_TRAIN:
    raw["train"] = raw["train"].shuffle(seed=42).select(range(SUBSET_TRAIN))
if SUBSET_TEST:
    raw["test"] = raw["test"].shuffle(seed=42).select(range(SUBSET_TEST))

label_names = raw["train"].features["label"].names
print("Labels:", label_names)

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)


def tokenize_batch(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)


tokenized = raw.map(tokenize_batch, batched=True)
tokenized = tokenized.remove_columns(["text"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_names))
model.to(device)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    metrics = trainer.evaluate()
    print("Final metrics:", metrics)

    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    with open(f"{SAVE_DIR}/label_names.json", "w") as f:
        json.dump(label_names, f)

    print(f"Model saved to {SAVE_DIR}")