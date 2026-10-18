# Hausa sentiment app

This Gradio app compares the selected B4 word+character TF-IDF + LinearSVC model with the R1 trainable-embedding mean-pooling model.

## Local run

From the repository root:

```powershell
python -m pip install -r app\requirements.txt
python app\app.py
```

The app expects these trained artifacts:

- `models/classical_best.joblib`
- `models/rnn_best.pt`

## Deployment

For Render:

1. Push the repository to GitHub, including `models/classical_best.joblib` and `models/rnn_best.pt`.
2. In Render, choose **New → Blueprint**, connect the GitHub repository, and apply the repository's `render.yaml`.
3. If you create a Web Service instead, use these settings:
   - Runtime: **Python**
   - Build command: `pip install -r app/requirements.txt`
   - Start command: `python app/app.py`
   - Health check path: `/`
   - Environment variable: `PYTHON_VERSION=3.13`
4. Choose the free instance if appropriate for the demonstration.

The repository also includes [`render.yaml`](../render.yaml), which can be used for Blueprint deployment.

The app returns each model's predicted label and the three class probabilities. It applies the same NFC, URL, mention, and whitespace normalization used before training. The classical model is recommended for real use because it achieved the higher test macro-F1.

### Input handling and scope

The classifier is trained for Hausa social-media text written in Latin script;
it is not a general language identifier. The interface therefore:

- rejects empty, link-only, mention-only, symbol-only, or emoji-only input;
- limits input to 2,000 characters because the models were trained on short messages;
- gives an advisory for non-Latin scripts; and
- gives an advisory when a longer input contains no common Hausa cues.

These language checks are intentionally warnings rather than hard blocks. Short
Hausa messages can omit recognizable cues, and a lightweight heuristic cannot
reliably distinguish every Hausa, English, or code-switched message. Predictions
for such inputs should be treated as out of domain and not as verified language
detection.
