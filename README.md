# Hausa Sentiment Classification (AfriSenti)

**Status:** classical model trained; deployment and report preparation in progress.

## Project type
Text classification for an African language (Hausa): positive / neutral / negative.

## Workflow
1. `notebooks/01_data_eda.ipynb` — download data, EDA, preprocess (Colab)
2. `notebooks/02_classical_baseline.ipynb` — TF-IDF baselines + experiments (Colab)
3. `notebooks/03_rnn.ipynb` — Word2Vec + BiLSTM (Colab, optional GPU)
4. `notebooks/04_transformer_lora.ipynb` — AfroXLMR + LoRA fine-tune (Colab **GPU**)
5. `app/` — Gradio deployment of the best model
6. `report/` — PDF research report (after results exist)

## Dataset
AfriSenti Hausa (Muhammad et al., EMNLP 2023). Official SemEval-2023 splits.

## How to run Step 1
1. Open Google Colab
2. Upload `notebooks/01_data_eda.ipynb` (or open from GitHub later)
3. Runtime → Run all
4. Download `data/processed/*.csv` and `results/data_summary.json` to your machine / Drive

## Step 1 status
Done. Processed CSVs and data_summary.json are in `data/processed/` and `results/`.

| Split | Rows |
|-------|-----:|
| train | 14 172 |
| dev | 2 677 |
| dev_noleak | 2 462 |
| test | 5 303 |
| leak rows removed | 215 |

## Step 2
Upload `notebooks/02_classical_baseline.ipynb` to Colab together with the four CSVs.

## Current results

The selected model is B4, word+character TF-IDF with calibrated LinearSVC:

| Model | Dev macro-F1 | Test macro-F1 |
|-------|-------------:|--------------:|
| B4 classical | 0.762 | **0.781** |
| R1 embedding mean-pooling | 0.682 | 0.681 |
| Transformer + LoRA | 0.662 | 0.601 |

Model selection uses `dev_noleak`; the official test set is used for final reporting.

## Run the app locally

```powershell
python -m pip install -r app\requirements.txt
python app\app.py
```

The app compares `models/classical_best.joblib` with `models/rnn_best.pt` side by side and returns each sentiment label with class probabilities. See [`app/README.md`](app/README.md) for deployment details.

## Render deployment

The app is configured for Render with [`render.yaml`](render.yaml). Push this project to GitHub, then either:

1. In Render, choose **New → Blueprint**, select the repository, and apply the detected `render.yaml`; or
2. Choose **New → Web Service** and use:
   - Runtime: **Python**
   - Build command: `pip install -r app/requirements.txt`
   - Start command: `python app/app.py`
   - Health check path: `/`

The Blueprint uses the free plan and Python 3.13. The service listens on Render's `PORT` environment variable, and the model files under `models/` must be committed to the repository.
