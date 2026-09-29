This folder is a starter architecture for refactoring the current notebook/script.

plan.md
    Detailed project plan and design decisions.

src/config.py
    Thresholds and constants.

src/mode_detector.py
    Basic deterministic mode classification.

src/model_registry.py
    Keeps one anomaly model/scaler/threshold per mode.

src/anomaly_detector.py
    Routes a completed feature window to the correct model.

src/pipeline.py
    High-level orchestration.

These starter files are not a full replacement for the uploaded 3,600+ line
training script yet. The recommended first refactor is to preserve the existing
idle training behavior while moving its pieces into the architecture described
in plan.md.
