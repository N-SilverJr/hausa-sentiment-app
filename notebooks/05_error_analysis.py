from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODEL_PATH = ROOT / "models" / "classical_best.joblib"
LABELS = ["positive", "neutral", "negative"]

test = pd.read_csv(PROC / "afrisenti_hau_test.csv")
model = joblib.load(MODEL_PATH)
texts = test["text_bow"].fillna("").astype(str)
gold = test["label_id"].astype(int).to_numpy()
predictions = model.predict(texts).astype(int)
probabilities = model.predict_proba(texts) if hasattr(model, "predict_proba") else None
confidence = probabilities.max(axis=1) if probabilities is not None else np.full(len(test), np.nan)

cm = confusion_matrix(gold, predictions, labels=[0, 1, 2])
report = classification_report(
    gold,
    predictions,
    target_names=LABELS,
    labels=[0, 1, 2],
    output_dict=True,
)
print("Confusion matrix (rows=true, columns=predicted):")
print(cm)
print(pd.DataFrame(report).T.round(3))

error_pairs = pd.DataFrame(cm, index=LABELS, columns=LABELS)
off_diagonal = error_pairs.copy()
for label in LABELS:
    off_diagonal.loc[label, label] = 0
print("\nLargest error directions:")
print(off_diagonal.stack().sort_values(ascending=False).head(5))

rows = []
for index, row in test.iterrows():
    rows.append(
        {
            "index": int(index),
            "text": str(row["text"]),
            "text_bow": str(row["text_bow"]),
            "gold_id": int(gold[index]),
            "gold": LABELS[gold[index]],
            "pred_id": int(predictions[index]),
            "predicted": LABELS[predictions[index]],
            "confidence": (
                None if np.isnan(confidence[index]) else float(confidence[index])
            ),
            "correct": bool(gold[index] == predictions[index]),
            "n_tokens": int(row.get("n_tokens", len(str(row["text_bow"]).split()))),
            "n_emoji": int(row.get("n_emoji", 0)),
        }
    )

analysis = pd.DataFrame(rows)
correct_examples = (
    analysis[analysis["correct"]]
    .sort_values("confidence", ascending=False)
    .groupby("gold", sort=False)
    .head(3)
)
error_examples = (
    analysis[~analysis["correct"]]
    .sort_values("confidence", ascending=False)
    .head(12)
)

correct_examples.to_csv(RESULTS / "error_analysis_correct_examples.csv", index=False)
error_examples.to_csv(RESULTS / "error_analysis_error_examples.csv", index=False)
analysis.to_csv(RESULTS / "error_analysis_all_test_predictions.csv", index=False)

summary = {
    "model": "B4 word+char TF-IDF + LinearSVC",
    "test_size": int(len(test)),
    "errors": int((gold != predictions).sum()),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "largest_error_directions": [
        {"gold": str(gold_label), "predicted": str(pred_label), "count": int(count)}
        for (gold_label, pred_label), count in
        off_diagonal.stack().sort_values(ascending=False).head(5).items()
    ],
}
with open(RESULTS / "error_analysis_summary.json", "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print("\nSaved error-analysis artifacts to", RESULTS)
