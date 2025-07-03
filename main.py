import joblib
import pathlib
import sys
import pandas as pd
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

project_root = pathlib.Path(__file__).parent
sys.path.append(str(project_root))

from src.utils import get_logger
from src.config import MODELS_DIR, ID_TO_LABEL
from src.feature_engineering import extract_url_features, extract_structural_features, extract_content_features
from predict import fetch_and_parse

logger = get_logger("api_logger")

try:
    model_path = MODELS_DIR / 'lgbm_final_model.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info(f"Loading model from {model_path}...")
    artifact = joblib.load(model_path)
    model = artifact['model']
    vectorizer = artifact['vectorizer']
    logger.info("Model and vectorizer loaded successfully.")

except Exception as e:
    logger.error(f"Failed to load model on startup: {e}", exc_info=True)
    sys.exit(1)


app = FastAPI(
    title="Website Classifier API",
    description="An API to classify a website as a personal blog or a corporate site.",
    version="1.0.0"
)

class PredictRequest(BaseModel):
    url: HttpUrl

class PredictResponse(BaseModel):
    url: str
    prediction: str
    confidence: float


def transform_for_prediction(site_data: dict) -> sp.csr_matrix:
    """ Replicates the feature engineering from training for a single data point. """
    df = pd.DataFrame([site_data])

    txt_features = vectorizer.transform(df["text_content"].fillna(""))

    url_features = extract_url_features(df["url"]).to_numpy(dtype="float32")
    structural_features = extract_structural_features(df["html_content"]).to_numpy(dtype="float32")
    content_features = extract_content_features(df["text_content"]).to_numpy(dtype="float32")

    return sp.hstack([
        txt_features,
        sp.csr_matrix(url_features),
        sp.csr_matrix(structural_features),
        sp.csr_matrix(content_features)
    ], format="csr")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Accepts a URL, fetches its content, and returns a classification.
    """
    url = str(request.url)
    logger.info(f"Received prediction request for URL: {url}")

    site_data = fetch_and_parse(url)
    if site_data is None:
        logger.error(f"Could not fetch or parse content for {url}")
        raise HTTPException(
            status_code=422,
            detail=f"Failed to fetch or process content from the URL: {url}"
        )

    try:
        features = transform_for_prediction(site_data)

        prob_for_class_1 = model.predict(features)[0]

        prediction_id = 1 if prob_for_class_1 > 0.5 else 0
        label = ID_TO_LABEL[prediction_id]

        if prediction_id == 1:
            confidence = prob_for_class_1
        else:
            confidence = 1 - prob_for_class_1

        logger.info(f"Prediction for {url}: {label.upper()} with {confidence:.2%} confidence.")

        return PredictResponse(
            url=url,
            prediction=label.upper(),
            confidence=float(confidence)
        )

    except Exception as e:
        logger.error(f"An error occurred during prediction for {url}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while making the prediction."
        )

@app.get("/health")
def health_check():
    return {"status": "ok"}
