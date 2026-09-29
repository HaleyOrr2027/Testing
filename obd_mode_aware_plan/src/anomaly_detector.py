import numpy as np


def score_feature_row(mode, feature_row, registry):
    """Score one completed feature window using the model for its mode."""

    if not registry.has_model(mode):
        return {
            "mode": mode,
            "checked": False,
            "reason": "no model available for this mode",
        }

    bundle = registry.get(mode)
    features = bundle["features"]

    X = feature_row[features].to_frame().T

    if X.isna().any(axis=None):
        return {
            "mode": mode,
            "checked": False,
            "reason": "missing model feature",
        }

    X_scaled = bundle["scaler"].transform(X)
    reconstructed = bundle["model"].predict(X_scaled, verbose=0)

    score = float(np.mean(np.square(X_scaled - reconstructed)))
    threshold = bundle["threshold"]

    return {
        "mode": mode,
        "checked": True,
        "anomaly": score > threshold,
        "anomaly_score": score,
        "threshold": threshold,
    }
