import os
import joblib
import numpy as np

HERE = os.path.dirname(__file__)
MODEL_FILE = os.path.join(HERE, "model.pkl")

_artefacts = None

def load_artefacts():
    global _artefacts
    if _artefacts is None:
        _artefacts = joblib.load(MODEL_FILE)
    return _artefacts

def preprocess_input(payload: dict):
    artefacts = load_artefacts()
    feature_names = artefacts.get("feature_names")

    if not feature_names:
        raise ValueError("feature_names not found in artefacts. Re-save model with feature_names list.")

    x = []
    for fname in feature_names:
        if fname not in payload:
            raise ValueError(f"Missing field: {fname}")
        x.append(float(payload[fname]))

    X = np.array([x])

    scaler = artefacts.get("scaler")
    if scaler is not None:
        X = scaler.transform(X)

    return X

def predict(payload: dict):
    artefacts = load_artefacts()
    model = artefacts["model"]
    X = preprocess_input(payload)
    pred = model.predict(X)
    result = {"prediction": str(pred[0])}

    if hasattr(model, "predict_proba"):
        result["probabilities"] = model.predict_proba(X)[0].tolist()

    return result
