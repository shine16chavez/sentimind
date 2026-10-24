from flask import Flask, request, jsonify, render_template
import pickle
import os
from typing import Optional, Dict, Any
import gzip
import re

app = Flask(__name__)



class HybridSentimentModel:
    def predict(self, text: str) -> dict:
        raise NotImplementedError("Model not loaded properly.")



import __main__
__main__.HybridSentimentModel = HybridSentimentModel


class DistressPredictionModel:
    def predict(self, text: str) -> dict:
        raise NotImplementedError("Prediction model not loaded properly.")


__main__.DistressPredictionModel = DistressPredictionModel

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hybrid_sentiment_model.pkl")
PREDICTION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "mental-health_model.pkl")

sentiment_model = None
prediction_model = None

def load_model():
    global sentiment_model, prediction_model
    try:
        with open(MODEL_PATH, "rb") as f:
            sentiment_model = pickle.load(f)
        print(f"Sentiment model loaded successfully: {type(sentiment_model)}")
    except FileNotFoundError:
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        print("Falling back to manual VADER+TextBlob ensemble mode.")
        sentiment_model = None
    except Exception as e:
        print(f"Error loading model: {e}")
        sentiment_model = None

    try:
        prediction_model = None

        try:
            with open(PREDICTION_MODEL_PATH, "rb") as f:
                prediction_model = pickle.load(f)
        except Exception:
            prediction_model = None

        if prediction_model is None:
            try:
                import joblib
                prediction_model = joblib.load(PREDICTION_MODEL_PATH)
            except Exception:
                prediction_model = None

        if prediction_model is None:
            try:
                with gzip.open(PREDICTION_MODEL_PATH, "rb") as f:
                    prediction_model = pickle.load(f)
            except Exception:
                prediction_model = None

        if prediction_model is None:
            raise RuntimeError(
                "Could not deserialize prediction model with pickle, joblib, or gzip+pickle."
            )

        print(f"Prediction model loaded successfully: {type(prediction_model)}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"REQUIRED file missing: {PREDICTION_MODEL_PATH}. "
            "Place your prediction model pickle at this path (mental-health_model.pkl)."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load REQUIRED prediction model: {e}") from e


def run_inference(text: str) -> dict:

    if sentiment_model is not None:
        try:
            result = sentiment_model.predict(text)

            label      = str(result.get("label", result.get("sentiment", "UNKNOWN"))).upper()
            confidence = float(result.get("confidence", result.get("score", 0.0)))
            if confidence <= 1.0:
                confidence = round(confidence * 100, 1)

            compound   = float(result.get("compound", 0.0))
            scores     = result.get("scores", {})

            return {
                "label":      label,
                "confidence": confidence,
                "compound":   compound,
                "scores":     scores,
                "source":     "HybridSentimentModel",
            }
        except Exception as e:
            print(f"Model.predict() failed: {e}. Falling back to manual VADER+TextBlob ensemble.")


    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        from textblob import TextBlob

        vader = SentimentIntensityAnalyzer()
        vader_scores = vader.polarity_scores(text)
        vader_compound = float(vader_scores.get("compound", 0.0))

        blob = TextBlob(text)
        tb_polarity = float(blob.sentiment.polarity)  # -1..1
        tb_subjectivity = float(blob.sentiment.subjectivity)

        # Manual weighted ensemble: prioritize VADER
        ensemble_compound = (0.65 * vader_compound) + (0.35 * tb_polarity)

        if ensemble_compound >= 0.05:
            label = "POSITIVE"
        elif ensemble_compound <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        confidence = round(50 + (min(abs(ensemble_compound), 1.0) * 50), 1)

        return {
            "label": label,
            "confidence": confidence,
            "compound": round(ensemble_compound, 4),
            "scores": {
                "vader": vader_scores,
                "textblob": {
                    "polarity": round(tb_polarity, 4),
                    "subjectivity": round(tb_subjectivity, 4),
                },
                "ensemble": {
                    "vader_weight": 0.65,
                    "textblob_weight": 0.35,
                    "compound": round(ensemble_compound, 4),
                },
            },
            "source": "Manual VADER+TextBlob ensemble (fallback)",
        }
    except ImportError:
        pass


    text_lower = text.lower()
    neg_words  = {"sad", "bad", "stress", "anxious", "depressed", "hate",
                  "lonely", "tired", "hopeless", "worthless", "fail"}
    pos_words  = {"happy", "great", "love", "good", "excited", "joy",
                  "amazing", "wonderful", "fantastic", "proud", "grateful"}

    neg_hits = sum(1 for w in neg_words if w in text_lower)
    pos_hits = sum(1 for w in pos_words if w in text_lower)

    if neg_hits > pos_hits:
        label, confidence, compound = "NEGATIVE", round(70 + neg_hits * 3, 1), -0.5
    elif pos_hits > neg_hits:
        label, confidence, compound = "POSITIVE", round(70 + pos_hits * 3, 1),  0.5
    else:
        label, confidence, compound = "NEUTRAL", 60.0, 0.0

    return {
        "label":      label,
        "confidence": min(confidence, 99.0),
        "compound":   compound,
        "scores":     {},
        "source":     "keyword heuristic (fallback)",
    }


def run_prediction(text: str, sentiment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
   
    if prediction_model is None:
        raise RuntimeError(
            f"Prediction model not loaded. Expected at {PREDICTION_MODEL_PATH}."
        )

    if hasattr(prediction_model, "predict") and callable(getattr(prediction_model, "predict")):
        result = prediction_model.predict(text)
        if isinstance(result, dict):
            raw_label = str(result.get("label", result.get("prediction", "UNKNOWN"))).upper()
            label = raw_label
            confidence = float(result.get("confidence", result.get("score", 0.0)))
            if confidence <= 1.0:
                confidence = round(confidence * 100, 1)
            description = str(result.get("description", "")).strip()
            return {
                "label": label,
                "confidence": confidence,
                "description": description,
                "source": "DistressPredictionModel",
                "raw_label": raw_label,
            }

    if isinstance(prediction_model, dict) and {"model", "vectorizer", "label_encoder"} <= set(prediction_model.keys()):
        clf = prediction_model["model"]
        vectorizer = prediction_model["vectorizer"]
        label_encoder = prediction_model["label_encoder"]

        normalized_text = text.lower()
        replacements = {
            r"\bstressed\b": "stress",
            r"\bstressful\b": "stress",
            r"\banxious\b": "anxiety",
            r"\bdepressed\b": "depression",
            r"\bdepressing\b": "depression",
            r"\bsuicidal thoughts?\b": "suicidal",
        }
        for pattern, replacement in replacements.items():
            normalized_text = re.sub(pattern, replacement, normalized_text)

        X = vectorizer.transform([normalized_text])

        proba = None
        if hasattr(clf, "predict_proba") and callable(getattr(clf, "predict_proba")):
            try:
                proba = clf.predict_proba(X)
            except Exception:
                proba = None

        y_pred = clf.predict(X)
        raw_label = y_pred[0]
        try:
            decoded = label_encoder.inverse_transform([raw_label])[0]
        except Exception:
            decoded = raw_label

        label = str(decoded).upper()

        confidence = 0.0
        if proba is not None:
            try:
                confidence = float(max(proba[0])) * 100.0
            except Exception:
                confidence = 0.0

        sent_label = str((sentiment or {}).get("label", "")).upper()
        sent_compound = float((sentiment or {}).get("compound", 0.0) or 0.0)
        stress_cues = any(
            cue in normalized_text
            for cue in ("stress", "overwhelmed", "burnout", "panic", "anxiety")
        )
        if label == "NORMAL" and stress_cues and (sent_label == "NEGATIVE" or sent_compound <= -0.3):
            label = "STRESS"
            confidence = max(confidence, 70.0)

        raw_label = label
        if raw_label in {"ANXIETY", "BIPOLAR", "DEPRESSION", "PERSONALITY DISORDER", "STRESS", "SUICIDAL"}:
            label = "NEGATIVE"
        elif raw_label == "NORMAL":
            label = "NEUTRAL"
        elif raw_label in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
            label = raw_label
        else:
            label = "NEUTRAL"

        return {
            "label": label,
            "confidence": round(confidence, 1),
            "description": "",
            "source": "sklearn bundle (model+vectorizer+label_encoder)",
            "raw_label": raw_label,
        }

    raise RuntimeError(
        "Expected an object with .predict(text)->dict, or a dict with keys: model, vectorizer, label_encoder."
    )



#  Routes

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "No text provided."}), 400

    text   = data["text"].strip()
    result = run_inference(text)

    label = result["label"]
    descriptions = {
        "POSITIVE": "The model detected a positive emotional state. Language patterns suggest well-being and optimism.",
        "NEGATIVE": "The model detected high-stress or distress indicators. Language patterns suggest emotional difficulty.",
        "NEUTRAL":  "The model detected a neutral emotional state. No strong positive or negative signals found.",
    }
    result["description"] = descriptions.get(label, "Sentiment analysis complete.")
    result["input"]       = text

    return jsonify(result)


@app.route("/api/infer", methods=["POST"])
def infer():
    data = request.get_json(silent=True)
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "No text provided."}), 400

    text = data["text"].strip()

    sentiment = run_inference(text)
    
    try:
        prediction = run_prediction(text, sentiment=sentiment)
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

    return jsonify({
        "input": text,
        "prediction": prediction, # Contains raw_label and confidence
        "sentiment": sentiment,   # Contains label and confidence
        "description": prediction.get("description", "Linguistic analysis complete.")
    })


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "sentiment_model_loaded": sentiment_model is not None,
        "sentiment_model_type":   str(type(sentiment_model).__name__) if sentiment_model else "none",
        "prediction_model_loaded": prediction_model is not None,
        "prediction_model_type":   str(type(prediction_model).__name__) if prediction_model else "none",
    })



#  Entry point
if __name__ == "__main__":
    load_model()
    app.run(host="127.0.0.1", port=5000)
