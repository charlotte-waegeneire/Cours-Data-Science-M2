from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


class TitanicInput(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    Embarked: str


app = FastAPI()
model = joblib.load("titanic_pipeline.pkl")


@app.post("/predict")
def predict(data: TitanicInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}
