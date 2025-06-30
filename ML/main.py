from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Optional


class TitanicInput(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    Sex: str = Field(
        ..., pattern="^(male|female)$", description="Sex (male or female)"
    )  # Changed regex to pattern
    Age: float = Field(..., ge=0, le=120, description="Age in years")
    Fare: float = Field(..., ge=0, description="Fare paid")
    Embarked: str = Field(
        ..., pattern="^(S|C|Q)$", description="Port of embarkation (S, C, or Q)"
    )  # Changed regex to pattern
    FamilySize: Optional[int] = Field(
        0, ge=0, le=20, description="Family size (SibSp + Parch)"
    )


class PredictionResponse(BaseModel):
    prediction: int
    survival_probability: float
    passenger_details: dict


app = FastAPI(
    title="Titanic Survival Predictor API",
    description="Predict passenger survival on the Titanic",
    version="1.0.0",
)

# Load the trained model
try:
    model = joblib.load("titanic_best_pipeline.pkl")  # Use the model with FamilySize
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Model file not found. Using basic model...")
    try:
        model = joblib.load("titanic_pipeline.pkl")  # Fallback to basic model
        print("✅ Basic model loaded successfully!")
    except FileNotFoundError:
        raise Exception("No model file found! Please train and save a model first.")


@app.get("/")
def read_root():
    return {
        "message": "Titanic Survival Predictor API",
        "status": "running",
        "endpoints": {"predict": "/predict", "docs": "/docs"},
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: TitanicInput):
    try:
        # Convert input to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        # Make prediction
        prediction = model.predict(df)[0]

        # Get prediction probabilities if available
        try:
            probabilities = model.predict_proba(df)[0]
            survival_prob = float(probabilities[1])  # Probability of survival
        except:
            # If model doesn't support predict_proba, use binary prediction
            survival_prob = float(prediction)

        return PredictionResponse(
            prediction=int(prediction),
            survival_probability=survival_prob,
            passenger_details=input_dict,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch")
def predict_batch(passengers: list[TitanicInput]):
    """Predict survival for multiple passengers"""
    try:
        # Convert all inputs to DataFrame
        input_data = [passenger.dict() for passenger in passengers]
        df = pd.DataFrame(input_data)

        # Make predictions
        predictions = model.predict(df)

        # Get probabilities if available
        try:
            probabilities = model.predict_proba(df)
            survival_probs = probabilities[:, 1].tolist()
        except:
            survival_probs = predictions.tolist()

        results = []
        for i, (passenger, pred, prob) in enumerate(
            zip(input_data, predictions, survival_probs)
        ):
            results.append(
                {
                    "passenger_id": i + 1,
                    "prediction": int(pred),
                    "survival_probability": float(prob),
                    "details": passenger,
                }
            )

        return {"predictions": results, "total_passengers": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model_info")
def get_model_info():
    """Get information about the loaded model"""
    try:
        # Try to get feature names from the model
        feature_names = getattr(model, "feature_names_in_", "Not available")

        return {
            "model_type": str(type(model)),
            "features": feature_names.tolist()
            if hasattr(feature_names, "tolist")
            else feature_names,
            "supports_probability": hasattr(model, "predict_proba"),
        }
    except Exception as e:
        return {"error": f"Could not retrieve model info: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
