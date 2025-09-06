# model_infer.py
from transformers import pipeline
import re
import emoji

_url_re = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_mention_hashtag_re = re.compile(r"[@#]\w+")
_extra_ws_re = re.compile(r"\s+")

# Load a strong off‑the‑shelf DistilBERT sentiment model
# This is fine‑tuned on SST‑2; great baseline for English social text
_sai = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    return_all_scores=True,
)

LABEL_MAP = {
    # Map model labels to canonical output (keep consistent with UI)
    "NEGATIVE": "negative",
    "POSITIVE": "positive",
}


def clean_text(text: str) -> str:
    """Light normalization for social posts."""
    if not text:
        return ""
    # demojize (🙂 -> :slightly_smiling_face:), keeps sentiment cues as tokens
    text = emoji.demojize(text, language='en')
    text = _url_re.sub(" ", text)
    text = _mention_hashtag_re.sub(" ", text)
    text = text.replace("&amp;", "and")
    text = _extra_ws_re.sub(" ", text).strip()
    return text


def predict_sentiment(text: str):
    """Return label and probability. Adds a simple neutrality band.

    Neutral heuristic: if max(prob) < 0.60 -> label = neutral
    Adjust this threshold for your dataset.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {"label": "neutral", "score": 1.0, "raw": []}

    scores = _sai(cleaned)[0]  # list of dicts: {label: POSITIVE/NEGATIVE, score: float}
    # find the top class
    top = max(scores, key=lambda x: x["score"]) if scores else {"label": "POSITIVE", "score": 0.5}
    label = LABEL_MAP.get(top["label"], top["label"]).lower()

    # Neutral band: treat uncertain cases as neutral
    if top["score"] < 0.60:
        label = "neutral"

    return {"label": label, "score": float(top["score"]), "raw": scores, "cleaned": cleaned}


if __name__ == "__main__":
    test = "I didn't love it, but it's not bad either 🙂"
    print(predict_sentiment(test))