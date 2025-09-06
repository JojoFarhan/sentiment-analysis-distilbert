# train_on_csv.py
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch

# Base pretrained model
MODEL_NAME = "distilbert-base-uncased"

# Labels for 3-class sentiment classification
LABELS = ["negative", "neutral", "positive"]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

@dataclass
class DS(torch.utils.data.Dataset):
    encodings: Dict
    labels: np.ndarray

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# def load_data(path: str):
#     """Load a CSV with columns: text,label"""
#     df = pd.read_csv(path)
#     df = df.dropna(subset=["text", "label"]).copy()
#     df["label"] = df["label"].str.lower().str.strip()
#     assert set(df["label"]).issubset(set(LABELS)), f"Labels must be in {LABELS}"
#     return df

def load_data(path: str):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"]).copy()

    # normalize labels
    df["label"] = df["label"].str.lower().str.strip()

    # keep only desired classes
    df = df[df["label"].isin(["negative", "neutral", "positive"])]

    return df



def main():
    # 1) Load your dataset
    df = load_data("data/train.csv")

    # 2) Split into train/validation
    train_df, val_df = train_test_split(
        df, test_size=0.1, random_state=42, stratify=df["label"]
    )

    # 3) Tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def encode_texts(texts):
        return tok(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=128,
        )

    train_enc = encode_texts(train_df["text"])
    val_enc = encode_texts(val_df["text"])
    y_train = train_df["label"].map(LABEL2ID).values
    y_val = val_df["label"].map(LABEL2ID).values

    # 4) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 5) Training arguments
    # args = TrainingArguments(
    #     output_dir="models/distilbert-sentiment",
    #     eval_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=32,
    #     num_train_epochs=3,
    #     weight_decay=0.01,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="eval_loss",
    #     logging_steps=50,
    #     fp16=torch.cuda.is_available(),
    # )
    args = TrainingArguments(
    output_dir="models/distilbert-sentiment",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir="./logs",            # <-- log dir
    logging_strategy="steps",        # <-- log by steps
    logging_steps=50,                # <-- print every 50 steps
    report_to="none",                # <-- turn off wandb
    fp16=torch.cuda.is_available(),
    )

    

    # 6) Datasets
    train_ds = DS(train_enc, y_train)
    val_ds = DS(val_enc, y_val)

    # 7) Metrics
    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score

        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    # 8) Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # 9) Train
    trainer.train()

    # 10) Save model + tokenizer
    save_dir = "models/distilbert-sentiment"
    trainer.save_model(save_dir)
    tok.save_pretrained(save_dir)
    print(f"✅ Model and tokenizer saved to {save_dir}")


if __name__ == "__main__":
    main()
