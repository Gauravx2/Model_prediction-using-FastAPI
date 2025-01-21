
import pandas as pd
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from models import generate_synthetic_data, train_model, predict

router = APIRouter()

# Global variables for storing the dataset
data = None
@router.get("/")
def read_root():
    return {"message": "Welcome to the Predictive Analysis API!"}
@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global data
    if file.filename.endswith('.csv'):
        data = pd.read_csv(file.file)
        return {"message": "Dataset uploaded successfully", "columns": list(data.columns)}
    return {"error": "Invalid file format. Please upload a CSV file."}

# Endpoint: Train Model
@router.post("/train")
def train_model_endpoint():
    global data, model
    if data is None:
        data = generate_synthetic_data()
    accuracy, f1 = train_model(data)
    return {"message": "Model trained successfully", "accuracy": accuracy, "f1_score": f1}
class PredictionInput(BaseModel):
    Temperature: float
    Run_Time: float

@router.post("/predict")
def predict_endpoint(input_data: PredictionInput):
    global data
    if data is None:
        return {"error": "No model trained. Use the /train endpoint first."}
    input_df = pd.DataFrame([input_data.dict()])
    prediction, confidence = predict(input_df)
    return {"Downtime": "Yes" if prediction[0] == 1 else "No", "Confidence": round(confidence, 2)}
