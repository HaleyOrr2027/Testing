class ModeAwarePipeline:
    """Coordinates mode detection, window creation, and anomaly routing.

    Keep detailed preprocessing/window logic in their own modules. This
    class should mainly coordinate the pieces.
    """

    def __init__(self, model_registry):
        self.model_registry = model_registry

    def process_completed_window(self, mode, feature_row, score_function):
        if mode in {"ENGINE_OFF", "UNKNOWN_TRANSITION"}:
            return {
                "mode": mode,
                "checked": False,
                "reason": "mode is not scoreable",
            }

        return score_function(mode, feature_row, self.model_registry)
