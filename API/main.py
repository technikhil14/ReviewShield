from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
from utils import engineer_features, features_to_df

# Loading model
model = joblib.load("/app/Model/reviewshield_model.pkl")
features = joblib.load("/app/Model/features.pkl")

# App initialization
app = FastAPI(
    title="ReviewShield API",
    description="Fake Product Review Detection using ML",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Schemas
class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Review text")
    rating: float = Field(..., ge=1, le=5, description="Star rating 1-5")

class ReviewResponse(BaseModel):
    prediction: str
    confidence: float
    risk_level: str
    features_used: dict
    explanation: str
# Health Check
@app.get("/")
def root():
    return {"status": "ReviewShield API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# Helper Function
def build_explanation(feat: dict, prediction: str, fake_prob: float) -> str:
    reasons = []
    if feat["unique_word_ratio"] < 0.5:
        reasons.append("low vocabulary diversity (repetitive word usage)")
    if feat["review_length"] < 100:
        reasons.append("unusually short review length")
    if feat["avg_sentence_length"] > 150:
        reasons.append("abnormally long sentence structure")
    if feat["has_digits"] == 0:
        reasons.append("no specific details like numbers or model references")

    if prediction == "Fake":
        if reasons:
            return f"Flagged as fake due to: {', '.join(reasons)}."
        return "Flagged as fake based on combined text pattern analysis."
    else:
        return f"Appears genuine. Review shows natural writing patterns with {round(feat['unique_word_ratio']*100)}% unique vocabulary."

# Main Prediction Endpoint
@app.post("/predict", response_model=ReviewResponse)
def predict(review: ReviewRequest):
    try:
        feat = engineer_features(review.text, review.rating)
        df = features_to_df(feat)
        df = df[features]

        prob = model.predict_proba(df)[0]
        fake_prob = float(prob[1])
        genuine_prob = float(prob[0])

        prediction = "Fake" if fake_prob > 0.5 else "Genuine"
        confidence = round(fake_prob * 100 if prediction == "Fake" else genuine_prob * 100, 2)

        if fake_prob >= 0.75:
            risk_level = "High"
        elif fake_prob >= 0.5:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        explanation = build_explanation(feat, prediction, fake_prob)

        return ReviewResponse(
            prediction=prediction,
            confidence=confidence,
            risk_level=risk_level,
            features_used=feat,
            explanation=explanation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Bulk Prediction Endpoint
class BulkRequest(BaseModel):
    reviews: list[ReviewRequest]

@app.post("/predict/bulk")
def predict_bulk(bulk: BulkRequest):
    if len(bulk.reviews) > 100:
        raise HTTPException(status_code=400, detail="Max 100 reviews per bulk request")

    results = []
    for review in bulk.reviews:
        try:
            feat = engineer_features(review.text, review.rating)
            df = features_to_df(feat)
            df = df[features]

            prob = model.predict_proba(df)[0]
            fake_prob = float(prob[1])
            prediction = "Fake" if fake_prob > 0.5 else "Genuine"
            confidence = round(fake_prob * 100 if prediction == "Fake" else (1 - fake_prob) * 100, 2)

            results.append({
                "text_preview": review.text[:60] + "...",
                "prediction": prediction,
                "confidence": confidence,
                "fake_probability": round(fake_prob * 100, 2)
            })
        except Exception as e:
            results.append({
                "text_preview": review.text[:60] + "...",
                "prediction": "Error",
                "confidence": 0,
                "fake_probability": 0
            })

    fake_count = sum(1 for r in results if r["prediction"] == "Fake")

    return {
        "total_reviews": len(results),
        "fake_count": fake_count,
        "genuine_count": len(results) - fake_count,
        "fake_percentage": round(fake_count / len(results) * 100, 2),
        "results": results
    }