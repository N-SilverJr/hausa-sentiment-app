from pathlib import Path
import os
import re
import unicodedata

import gradio as gr
import joblib
import numpy as np
import torch
from torch import nn


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
CLASSICAL_PATH = ROOT / "models" / "classical_best.joblib"
RNN_PATH = ROOT / "models" / "rnn_best.pt"
LABELS = ["positive", "neutral", "negative"]
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

if not CLASSICAL_PATH.exists():
    raise FileNotFoundError(f"Classical model file not found: {CLASSICAL_PATH}")
if not RNN_PATH.exists():
    raise FileNotFoundError(f"RNN model file not found: {RNN_PATH}")

CLASSICAL_MODEL = joblib.load(CLASSICAL_PATH)
RNN_CHECKPOINT = torch.load(RNN_PATH, map_location="cpu", weights_only=False)


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, pooling: str):
        super().__init__()
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.classifier = nn.Linear(embedding_dim, 3)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        features = self.embedding(input_ids)
        valid = attention_mask.unsqueeze(-1)
        pooled = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.classifier(pooled)


RNN_CONFIG = RNN_CHECKPOINT["config"]
RNN_VOCAB = RNN_CHECKPOINT["vocab"]
RNN_MODEL = RNNClassifier(
    len(RNN_VOCAB),
    RNN_CONFIG["embedding_dim"],
    RNN_CONFIG["hidden_dim"],
    RNN_CONFIG["pooling"],
)
RNN_MODEL.load_state_dict(RNN_CHECKPOINT["state_dict"])
RNN_MODEL.eval()


def text_bow(text: str) -> str:
    """Match the preprocessing used to train the classical model."""
    if not isinstance(text, str):
        return ""
    cleaned = unicodedata.normalize("NFC", text)
    cleaned = CTRL_RE.sub(" ", cleaned)
    cleaned = URL_RE.sub("<url>", cleaned)
    cleaned = MENTION_RE.sub("@user", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace("@user", " ").replace("<url>", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def score_classical(text: str) -> tuple[str, dict[str, float]]:
    probabilities = CLASSICAL_MODEL.predict_proba([text_bow(text)])[0]
    order = CLASSICAL_MODEL.classes_.astype(int)
    scores = {
        LABELS[int(label_id)]: float(probability)
        for label_id, probability in zip(order, probabilities)
    }
    predicted_id = int(order[int(np.argmax(probabilities))])
    return LABELS[predicted_id], scores


def score_rnn(text: str) -> tuple[str, dict[str, float]]:
    tokens = text_bow(text).split()[:64]
    ids = [RNN_VOCAB.get(token, 1) for token in tokens]
    mask = [1] * len(ids)
    ids += [0] * (64 - len(ids))
    mask += [0] * (64 - len(mask))
    with torch.inference_mode():
        logits = RNN_MODEL(
            torch.tensor([ids], dtype=torch.long),
            torch.tensor([mask], dtype=torch.bool),
        )
        probabilities = torch.softmax(logits, dim=1)[0].numpy()
    scores = {label: float(probability) for label, probability in zip(LABELS, probabilities)}
    return LABELS[int(np.argmax(probabilities))], scores


def predict_both(text: str):
    if not text or not text.strip():
        raise gr.Error("Enter a Hausa text before submitting.")
    classical_label, classical_scores = score_classical(text)
    rnn_label, rnn_scores = score_rnn(text)
    classical_confidence = max(classical_scores.values())
    rnn_confidence = max(rnn_scores.values())
    if classical_label != rnn_label:
        warning = (
            "### Uncertain result\n"
            "The models disagree, so this text may be ambiguous or context-dependent. "
            f"The primary classical model predicts **{classical_label}** with "
            f"{classical_confidence:.0%} confidence."
        )
    elif min(classical_confidence, rnn_confidence) < 0.60:
        warning = (
            "### Low-confidence result\n"
            "Both models chose the same label, but at least one confidence score is below 60%. "
            "Treat this prediction as uncertain."
        )
    else:
        warning = (
            "### Result summary\n"
            f"Both models predict **{classical_label}**. The classical model is the primary "
            "model because it performed better on the held-out test set."
        )
    return classical_label, classical_scores, rnn_label, rnn_scores, warning


CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
.hero { padding: 28px 30px; border-radius: 18px; background: linear-gradient(120deg, #12372a, #1f6f50); color: white; margin-bottom: 18px; }
.hero h1 { margin: 0 0 8px; font-size: 2.2rem; }
.hero p { margin: 0; opacity: .9; font-size: 1.05rem; }
.model-card { border: 1px solid #dfe7e2; border-radius: 16px; padding: 8px; }
.footer-note { color: #52615a; font-size: .92rem; }
"""

with gr.Blocks(title="Hausa Sentiment Classifier", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        '<div class="hero"><h1>Hausa Sentiment Classifier</h1>'
        "<p>Compare two models for positive, neutral, and negative Hausa sentiment.</p></div>"
    )
    with gr.Row():
        with gr.Column(scale=2):
            text = gr.Textbox(
                label="Hausa text",
                placeholder="Rubuta saƙon Hausa a nan...",
                lines=5,
            )
            submit = gr.Button("Analyze sentiment", variant="primary", size="lg")
            gr.Examples(
                examples=[
                    ["Masha Allah Allah ya kara basira"],
                    ["Saura nawa nima"],
                    ["Mtsww karyar banza"],
                ],
                inputs=text,
                label="Try an example",
            )
        with gr.Column(scale=3):
            gr.Markdown("### Model comparison")
            warning = gr.Markdown()
            with gr.Row():
                with gr.Column(elem_classes=["model-card"]):
                    gr.Markdown("**Classical B4**  \nWord + character TF-IDF + LinearSVC  \nTest macro-F1: **0.781**")
                    classical_label = gr.Label(label="Prediction")
                    classical_scores = gr.Label(label="Probabilities", num_top_classes=3)
                with gr.Column(elem_classes=["model-card"]):
                    gr.Markdown("**RNN R1**  \nTrainable embeddings + mean pooling  \nTest macro-F1: **0.681**")
                    rnn_label = gr.Label(label="Prediction")
                    rnn_scores = gr.Label(label="Probabilities", num_top_classes=3)
    submit.click(
        predict_both,
        inputs=text,
        outputs=[classical_label, classical_scores, rnn_label, rnn_scores, warning],
    )
    text.submit(
        predict_both,
        inputs=text,
        outputs=[classical_label, classical_scores, rnn_label, rnn_scores, warning],
    )
    gr.Markdown(
        "The classical model is the recommended model because it scored higher on the held-out test set. "
        "Both models can struggle with sarcasm, negation, code-switching, short messages, and context-dependent expressions.",
        elem_classes=["footer-note"],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
