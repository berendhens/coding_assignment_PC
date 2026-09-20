# PointCaré Technical Challenge — Time-Series Feature Extraction

Semantic feature extraction from time-series data, built against the TSQA
dataset. The system classifies a time series along four independent
attributes (**Trend**, **Volatility**, **Seasonality**, **Outlier**), each
with 3 possible classes, and returns a confidence score per attribute
(Tier 2: multi-label classification with confidence scores).

This repository has two parts:
- **Reproducible training pipeline**: one command
    > fetches data, preprocesses it, trains the model, validates it, and saves everything needed to reproduce or serve the result.
- **Containerised Inference API**/
    > a FastAPI service, containerised with Docker, that loads a trained model and serves predictions over HTTP.

## 1. Reproducible Training Pipeline

### What it does

One command runs the full pipeline end to end:

```
fetch data $\rightarrow$ preprocess $\rightarrow$ train $\rightarrow$ validate $\rightarrow$ save model + metrics
```

Concretely, `python run.py`:
1. Downloads the [TSQA dataset](https://huggingface.co/datasets/ChengsenWang/TSQA)
   from HuggingFace (`fetch_data.py`)
2. Builds the label vocabulary (the fixed class list per attribute) from
   the raw data (`build_labels.py`)
3. Parses, filters, and normalises the time series, and maps labels to
   class indices (`preprocessing.py`)
4. Splits into train/validation sets, stratified by Task + Label
   (`split_data.py`)
5. Trains the model, tracking loss per epoch and keeping the
   best-validation-loss checkpoint (`train.py`)
6. Computes per-task accuracy, F1, and confidence on the validation set
   (`validate.py`)
7. Saves the best-validation-loss model weights, metrics, config, training history, and label
   vocabulary together in one timestamped folder (`save_run.py`)

### Setup

```bash
conda create -n pointcare python=3.12
conda activate pointcare
pip install -r requirements_train.txt
```

Verified to install and run correctly in a clean `python:3.12` Docker
container, which confirms no hidden dependencies in the local development environment.

### Run it

```bash
python run.py
```

Hyperparameters and paths are controlled by `config.yaml`. Edit this file to
change batch size, learning rate, epoch count, the random seed, or where
data/outputs are stored. No code changes needed for a different run.

### What gets produced

Each run creates `outputs/run_<timestamp>/` containing:

| File            | Contents                                                   |
|-----------------|--------------------------------------------------------------|
| `model.pt`      | Trained model weights (best validation-loss checkpoint)     |
| `metrics.json`  | Per-task accuracy, F1, mean confidence, random baseline     |
| `config.json`   | The exact resolved config that produced this run             |
| `history.json`  | Per-epoch train/val loss                                     |
| `labels.json`   | Label vocabulary (required to turn model output back into class names) |

### Model architecture

A shared 1D CNN backbone extracts a 128-dim feature vector from each
64-length time series. This feeds into **4 independent heads**
(`Dense(3) + softmax`), one per attribute:

```
Input (1, 64)
 $\rightarrow$ Conv1d(1 $\rightarrow$ 32, k=5) $\rightarrow$ ReLU $\rightarrow$ MaxPool
 $\rightarrow$ Conv1d(32 $\rightarrow$ 64, k=5) $\rightarrow$ ReLU $\rightarrow$ MaxPool
 $\rightarrow$ Conv1d(64→128, k=3) $\rightarrow$ ReLU $\rightarrow$ AdaptiveAvgPool(1)
 $\rightarrow$ 4x Linear(128 $\rightarrow$ 3), one per attribute (trend / volatility / seasonality / outliers)
```

**Why 4 separate heads instead of one 12-way classifier:** 
each attribute is an independent 3-way decision, not one mutually-exclusive 12-class
problem. Separate heads let the model express confidence per attribute
independently.

**Masked multi-task loss:** 
each row in TSQA only has ground truth for *one* attribute (its `Task` column). 
For example, a Trend-labeled series has no Volatility/Seasonality/Outlier label. 
During training, only the head matching a sample's task contributes to the loss 
for that sample. The other 3 heads' outputs are computed (since the backbone is shared),
but ignored. 

This lets one shared model learn from partially-labeled data
without needing fully-annotated multi-attribute examples. This was 
a deliberate design choice for real-world, sparsely-labeled clinical data. 

### Results

Validation set (20% stratified split), single training run:

| Attribute    | Accuracy | F1 (macro) | Mean Confidence | Random Baseline |
|--------------|----------|------------|------------------|------------------|
| Trend        | 0.998    | 0.998      | 0.99             | 0.30             |
| Outliers     | 0.997    | 0.996      | 0.99             | 0.39             |
| Seasonality  | 0.756    | 0.748      | 0.80             | 0.31             |
| Volatility   | 0.980    | 0.980      | 0.97             | 0.33             |
| **Macro avg**| **0.932**| **0.931**  | —                | —                |

**Observations:**
- All 4 heads clear the random baseline by a wide margin, and 3 of the 4
  attributes (Trend, Outliers, and Volatility) reach near-ceiling
  accuracy ($>97\%$) after 20 epochs.
- **Trend** and **Outliers** are the most directly learnable from local
  shape (a slope, or a sharp local discontinuity), so near-perfect
  performance here is expected.
- **Volatility** started weak in early training (52% accuracy, 0.48 mean
  confidence, at epoch 2) but converged to 98% by epoch 20. This trajectory,
  low accuracy early on, climbing with more training, suggests the model 
  needed more epochs to learn this attribute's signal. 
- **Seasonality is the remaining weak point** ($76\%$, versus
  $>97\%$ on the other three). Detecting periodicity over a full 64-length
  window is a more global pattern than a slope or a spike, and may
  benefit from a larger effective receptive field (bigger kernels or more
  conv layers) than this network currently has. 
- **Caveat that remains:** the validation set is used both for
  best-checkpoint selection during training and for these final reported
  metrics, which makes them somewhat optimistic. A strict held-out test
  set, would give a more honest read. This was chosen as simplification. 

### Known trade-offs and limitations

Other deliberate simplifications made:

- **Filtered to `Size == 64` only.** TSQA includes different lengths (64, 128, 256,
  512). Only the shortest is used, to get a fixed input size without
  padding/masking logic.
- **Per-sample z-score normalisation**, applied uniformly to all 4 tasks.
Other, more complex and individual, normalisation methods could be used.
- **Two-way split (train/validation) only**, as mentioned above.
 A stricter setup would hold out a third split touched only once, at the very end.
- **Best-checkpoint-only saved**, not full per-epoch checkpoints. Keeps
  storage light, at the cost of not being able to inspect intermediate
  training states after the fact.

## 2. Inference API

### What it does

A FastAPI service with two endpoints:
- `POST /predict`: takes one time series, returns a predicted class and
  confidence score for all 4 attributes.
- `GET /health`: liveness check

### Run it

```bash
docker compose up --build
```

No separate install step, dependencies are installed inside the
container from `requirements_api.txt` (deliberately minimal: no `datasets`
or `scikit-learn`, which the API doesn't need).

The model served is loaded from a specific training run's output folder
(`RUN_DIR` in `main.py`), currently pointing at
`outputs/run_2026-09-20_19-43-46/`.

### Example request

`POST /predict`, body:

```json
{
  "series": [-0.0728, 1.1188, 1.1404, -0.1161, "...", -0.7237]
}
```

A flat list of 64 raw (unnormalised) floats.

### Example response

```json
{
  "predictions": {
    "trend": { "predicted_class": "constant trend", "confidence": 1.0 },
    "volatility": { "predicted_class": "increased volatility", "confidence": 0.6465 },
    "seasonality": { "predicted_class": "fixed seasonal", "confidence": 0.7639 },
    "outliers": { "predicted_class": "no outlier", "confidence": 1.0 }
  }
}
```

One request, four independent predictions (one per attribute), each with
its own confidence score.

### Try it

A sample request is included as `sample.json`. This sample is the first row of the TSQA
dataset. This is a pragmatic, known choice for demonstrating the endpoint works
end to end. It may overlap with the training data, so treat it as a
smoke test rather than a held-out evaluation.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample.json
```

Expect `trend.predicted_class` to read `"constant trend"` with high
confidence, matching this sample's known ground truth.

### Error handling

Sending a series of the wrong length returns a `422` with a clear message:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"series": [1.0, 2.0, 3.0]}'
```

### Known constraints

- **Fixed input length (64).** The model was trained exclusively on
  `Size == 64` series. Other lengths are rejected rather than silently
  mispredicted.
- **CPU-only inference.** No GPU support configured, not needed for a
  small CNN at this scale.
- **Model path is hardcoded** (`RUN_DIR` in `main.py`) to one specific
  training run.

## Project structure

```
.
├── run.py                # training pipeline entrypoint
├── fetch_data.py          # HuggingFace dataset fetch
├── build_labels.py        # label vocabulary construction
├── preprocessing.py       # parsing, filtering, normalization, label mapping
├── normalize.py            # shared normalization fn (used by both training and API)
├── split_data.py           # stratified train/val split
├── model.py                 # CNN backbone + 4-head architecture, masked loss
├── train.py                  # training loop
├── validate.py                # per-task metrics computation
├── save_run.py                 # artifact bundling
├── use_config.py                # config loader
├── use_labels.py                 # label vocab load/lookup helpers
├── config.yaml                    # hyperparameters and paths
├── main.py                         # FastAPI inference service
├── normalize.py                     # (shared with preprocessing.py)
├── sample.json                       # example /predict request
├── Dockerfile
├── docker-compose.yaml
├── requirements-train.txt
├── requirements-api.txt
└── outputs/run_<timestamp>/            # per-run artifacts (model, metrics, config, history, labels)
```