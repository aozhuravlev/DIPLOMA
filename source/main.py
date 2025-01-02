import dill
import polars as pl

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from pipeline import create_preprocessor


app = FastAPI()

# Load model
with open("../models/credit_risk_management_model.pkl", "rb") as file:
    model = dill.load(file)

# Response schema
class PredictionResponse(BaseModel):
    id: int
    predicted_default: int

@app.get("/status")
def status():
    return "I await your commands, my Lord."


@app.get("/version")
def version():
    return model["metadata"]


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    file_data = await file.read()
    parquet_file = pl.read_parquet(file_data)
    df = pl.DataFrame(parquet_file)

    preprocessor = create_preprocessor()
    preprocessed_data = preprocessor.fit_transform(df)
    y = model["model"].predict(preprocessed_data)[0]

    response = PredictionResponse(
        id=df["id"][0],
        predicted_default=y,
    )
    return response
