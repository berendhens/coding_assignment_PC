from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Need packages to load in model
import torch
from model import ModelPC
from use_labels import load_label_vocab
from preprocessing import normalize_series

app = FastAPI()

# Define expected shape of an incoming request
class PredictRequest(BaseModel):
    """TODO: Explanation."""
    series: list[float]

class AttributePrediction(BaseModel):
    """TODO: Explanation."""
    predicted_class: str
    confidence: float

# Define the expected shape of output
class PredictResponse(BaseModel):
    """TODO: Explanation."""
    predictions: dict[str, AttributePrediction]

# Hardcode directory to use
RUN_DIR = "outputs/run_2026-09-20_19-43-46"
EXPECTED_LENGTH = 64  # model was trained on Size == 64 series only

# Load labels, model (and saved weights), turn on evaluation mode
labels = load_label_vocab(f"{RUN_DIR}/labels.json")
model = ModelPC(attributes=labels["attributes"])
model.load_state_dict(torch.load(f"{RUN_DIR}/model.pt", map_location="cpu"))
model.eval()

# Create the /predict endpoint + indicate how response should look like
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # `request` here is already a validated PredictRequest object —
    # FastAPI parsed the incoming JSON and checked it against the class above
    # before this line ever executes.
    if len(request.series) != EXPECTED_LENGTH:
        raise HTTPException(status_code=422, detail=f"Expected series of length {EXPECTED_LENGTH}, got {len(request.series)}")

    # Same normalization used during training
    arr = normalize_series(request.series)
    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # add batch dim: (1, seq_len)

    with torch.no_grad():
        outputs = model(x)  # dict[str, (1, 3)] raw logits, one entry per attribute

    predictions = {}
    # Run through all attributes, and get probabilites for ech class
    # This result in prediction for label + confidence
    for attr in labels["attributes"]:
        # Calculate probability for each class this attribute
        probs = torch.softmax(outputs[attr], dim=-1)
        # Take the index of class and its confidence
        confidence, idx = probs.max(dim=-1)
        predictions[attr] = AttributePrediction(
            predicted_class=labels["classes"][attr][idx.item()], # get through idx the predicted class label
            confidence=round(confidence.item(), 4), # give confidence with assigned label
        )

    return PredictResponse(predictions=predictions)

# Check if application is alive
@app.get("/health")
def health():
    return {"status": "ok"}