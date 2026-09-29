class ModelRegistry:
    """Stores the model bundle associated with each operating mode."""

    def __init__(self):
        self._models = {}

    def register(self, mode, model, scaler, features, threshold):
        self._models[mode] = {
            "model": model,
            "scaler": scaler,
            "features": list(features),
            "threshold": float(threshold),
        }

    def has_model(self, mode):
        return mode in self._models

    def get(self, mode):
        if mode not in self._models:
            raise KeyError(f"No anomaly model registered for mode: {mode}")
        return self._models[mode]
