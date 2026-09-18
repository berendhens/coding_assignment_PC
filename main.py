from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define expected shape of an incoming request
class PredictRequest(BaseModel):
    values: list[float]
    # TODO: replace/extend this to match real input space


# Define the expected shape of output
class PredictResponse(BaseModel):
    prediction: str
    confidence: float

# Create the /predict endpoint + indicate how response should look like
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # `request` here is already a validated PredictRequest object —
    # FastAPI parsed the incoming JSON and checked it against the class above
    # before this line ever executes.

    # prediction = model.predict(request.values) TODO: Here comes final trained model.

    # Dummy logic just to prove the pipeline works end-to-end:
    dummy_prediction = "normal_pattern"
    dummy_confidence = 0.42

    return PredictResponse(prediction=dummy_prediction, confidence=dummy_confidence)

# Check if application is alive
@app.get("/health")
def health():
    return {"status": "ok"}