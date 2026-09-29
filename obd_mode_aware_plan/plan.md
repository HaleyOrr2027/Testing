# OBD Mode-Aware Anomaly Detection Plan

## Project Goal

Build a Raspberry Pi OBD-II monitoring system that:

1. Reads live vehicle data.
2. Determines the vehicle's current operating mode.
3. Compares the data only against normal behavior for that mode.
4. Produces an anomaly score and anomaly decision.
5. Runs continuously on a Raspberry Pi.

The key idea is:

`raw OBD data -> mode detection -> mode-specific feature window -> mode-specific anomaly model -> result`

A single anomaly model should NOT compare every driving condition against one definition of "normal." For example, high RPM and increasing throttle may be normal while accelerating but unusual during idle.

---

## What the Current Code Already Does

The current script already contains several useful pieces:

- Downloads and cleans the carOBD dataset.
- Builds 30-second windows.
- Calculates the mean and standard deviation of sensor values.
- Trains an autoencoder on stable idle data.
- Uses reconstruction error as an anomaly score.
- Selects an anomaly threshold from healthy calibration data.
- Estimates a time axis for new files.
- Smooths speed and estimates acceleration.
- Detects basic modes:
  - ENGINE_OFF
  - STARTUP
  - IDLE
  - ACCELERATING
  - DECELERATING
  - CRUISING
  - UNKNOWN
- Prevents very short mode changes from immediately changing the confirmed mode.
- Scores complete 30-second IDLE windows with the trained idle autoencoder.

The largest limitation is that only IDLE has an anomaly model.

---

## Why Modes Matter

Normal vehicle behavior is conditional on operating state.

Examples:

- 2500 RPM may be strange at idle but completely normal while accelerating.
- A dropping RPM may be normal during deceleration.
- High calculated engine load may be normal during acceleration or climbing a hill.
- Very low speed with engine RPM above zero may simply mean the vehicle is stopped at a traffic light.
- Coolant temperature behaves very differently during startup/warm-up than after the engine is fully warm.

Therefore anomaly detection should estimate:

`P(sensor behavior | operating mode)`

rather than treating all vehicle data as if it came from one distribution.

---

## Operating Modes to Support

### Phase 1 Modes

These modes are practical using commonly available OBD-II data.

| Mode | Meaning | Main signals |
|---|---|---|
| ENGINE_OFF | Engine is not running | RPM, engine run time |
| STARTUP_WARMUP | Engine recently started / temperature still stabilizing | RPM, engine run time, coolant temperature |
| IDLE | Engine on and vehicle approximately stopped | speed, RPM |
| ACCELERATING | Vehicle speed is increasing with meaningful positive acceleration | speed, acceleration, throttle/load |
| CRUISING | Vehicle is moving with approximately steady speed | speed, acceleration |
| COASTING_DECEL | Vehicle is slowing with little positive engine demand | speed, acceleration, throttle/load |
| BRAKING_DECEL | Strong or sustained deceleration | speed, acceleration |
| UNKNOWN_TRANSITION | Not enough information or mode is changing | confidence / missing data |

### Phase 2 Submodes

Once enough data exists, split broad modes when their normal behavior is clearly different.

Examples:

- LOW_SPEED_CRUISE
- HIGH_SPEED_CRUISE
- LIGHT_ACCELERATION
- HEAVY_ACCELERATION
- COLD_IDLE
- WARM_IDLE
- STOPPED_IN_GEAR vs PARK/NEUTRAL, if transmission/gear information is available
- HILL_LOAD / HIGH_LOAD_CRUISE, if load and/or road grade information is available

Do not create many submodes before enough normal examples exist for each one.

---

## Research Basis for the Mode Design

EPA MOVES is useful as a reference because it does not treat all moving operation as one condition.

MOVES separates braking and idle, and then separates moving operation using speed and Vehicle Specific Power (VSP). VSP is an estimate of vehicle power demand based mainly on speed, acceleration, road load, and grade.

This supports two design decisions for this project:

1. acceleration alone is not enough to describe operating condition;
2. speed/load context matters even inside "cruise" and "acceleration."

EPA also uses driving schedules containing idle, acceleration, deceleration, cruise, stop-and-go operation, and aggressive/high-acceleration operation.

References:
- https://www.epa.gov/moves/what-vehicle-specific-power-vsp
- https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules
- https://www.epa.gov/moves/there-way-model-emissions-single-vehicle-given-route-moves

---

## Recommended Architecture

Do not keep the full project in one Python file.

```text
OBD/
|
|-- plan.md
|-- requirements.txt
|-- models/
|   |-- idle/
|   |   |-- model.keras
|   |   |-- scaler.pkl
|   |   `-- metadata.json
|   |-- accelerating/
|   |-- cruising/
|   `-- decelerating/
|
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- labels/
|
|-- src/
|   |-- config.py
|   |-- obd_io.py
|   |-- preprocessing.py
|   |-- mode_detector.py
|   |-- feature_windows.py
|   |-- anomaly_detector.py
|   |-- model_registry.py
|   `-- pipeline.py
|
|-- training/
|   |-- prepare_dataset.py
|   |-- train_mode_model.py
|   |-- calibrate_thresholds.py
|   `-- evaluate.py
|
|-- scripts/
|   |-- run_file.py
|   `-- run_pi.py
|
`-- tests/
    |-- test_modes.py
    |-- test_windows.py
    `-- test_anomaly_routing.py
```

---

## Responsibility of Each File

### `src/config.py`

Contains constants only:

- speed thresholds
- acceleration thresholds
- persistence time
- startup/warm-up rules
- window length
- minimum samples
- sensor names

This prevents important thresholds from being scattered throughout the code.

### `src/obd_io.py`

Handles input only:

- CSV input during development
- Bluetooth/serial OBD input on Raspberry Pi later
- column normalization
- numeric conversion
- timestamps

The rest of the pipeline should not care whether the source was a CSV or live OBD adapter.

### `src/preprocessing.py`

Handles signal preparation:

- timestamp cleanup
- speed smoothing
- derivatives such as acceleration
- missing values
- optional derived features such as RPM change, throttle change, load change

### `src/mode_detector.py`

Responsible only for deciding the operating mode.

Input:
- recent sensor history

Output:
- mode
- mode confidence
- supporting values

Example:

```python
{
    "mode": "ACCELERATING",
    "confidence": 0.91,
    "acceleration_mps2": 0.82,
    "speed_kmh": 43.0
}
```

### `src/feature_windows.py`

Maintains rolling windows for anomaly models.

Important rule:

A model window should generally contain one stable mode.

Do not combine 15 seconds of cruising and 15 seconds of acceleration into one feature vector and call it either mode.

### `src/model_registry.py`

Loads the correct model, scaler, feature list, and threshold for each mode.

Example:

```python
models["IDLE"]
models["ACCELERATING"]
models["CRUISING"]
models["COASTING_DECEL"]
```

### `src/anomaly_detector.py`

Receives:

- mode
- feature vector

Then:

1. obtains the correct model for that mode;
2. applies that mode's scaler;
3. calculates reconstruction error;
4. compares it to that mode's threshold;
5. returns anomaly information.

### `src/pipeline.py`

Coordinates the runtime flow but does not contain the individual algorithms.

Example:

```text
read sample
    ->
preprocess
    ->
detect mode
    ->
update mode-specific window
    ->
if enough stable data:
    score with correct model
    ->
emit result
```

### `training/`

Training should be separate from Pi inference.

The Raspberry Pi should normally load already-trained models. It should not retrain a TensorFlow model every time the program starts.

---

## Mode Detection Strategy

### Do Not Start With Machine Learning for the Mode Detector

Start with a deterministic state/rule system.

Reasons:

- easy to inspect;
- easy to debug;
- works without manually labeled mode data;
- makes incorrect transitions obvious;
- can later generate labels for a learned classifier.

### Inputs

Minimum:

- VEHICLE_SPEED
- ENGINE_RPM
- ENGINE_RUN_TIME

Strongly recommended when available:

- THROTTLE_POSITION
- CALCULATED_ENGINE_LOAD
- COOLANT_TEMPERATURE
- MAF
- INTAKE_MANIFOLD_PRESSURE

### Derived Signals

Calculate:

- smoothed speed
- acceleration
- RPM change
- throttle change
- load change
- seconds since engine start

Later:

- approximate VSP or a simpler power-demand feature
- mode confidence

### Initial Rules

Pseudo-logic:

```text
if engine not running:
    ENGINE_OFF

elif recently started and engine/coolant is still stabilizing:
    STARTUP_WARMUP

elif speed approximately zero:
    IDLE

elif strong negative acceleration:
    BRAKING_DECEL

elif negative acceleration:
    COASTING_DECEL

elif positive acceleration:
    ACCELERATING

elif moving and acceleration approximately zero:
    CRUISING

else:
    UNKNOWN_TRANSITION
```

Use hysteresis/persistence so noisy speed readings do not rapidly switch:

`CRUISING -> ACCELERATING -> CRUISING -> ACCELERATING`

every second.

The existing code already implements a useful persistence concept. Keep that idea.

---

## Better Than One Acceleration Threshold

The current approach uses fixed acceleration cutoffs.

That is okay for the prototype, but it should eventually include engine-demand context.

Examples:

- positive acceleration + high throttle/load -> acceleration
- slight negative acceleration + near-closed throttle -> coasting
- strong sustained negative acceleration -> braking/deceleration
- nearly constant speed + high load -> possible hill/high-load cruise

This is closer to the reasoning behind EPA VSP operating bins.

---

## Mode Confidence

Do not force every sample into a confident mode.

Each mode decision should include confidence.

Example:

```text
MODE = CRUISING
CONFIDENCE = 0.94
```

or

```text
MODE = UNKNOWN_TRANSITION
CONFIDENCE = 0.42
```

If confidence is too low:

- do not update the anomaly window;
- do not produce a strong anomaly judgment.

This is safer than feeding uncertain transition data into the wrong normal model.

---

## Windowing

The current idle model uses 30-second mean/std windows.

Keep that for compatibility initially, but make the windowing configurable.

Potential later approach:

- IDLE: 20-30 seconds
- CRUISE: 10-20 seconds
- ACCELERATION: event-based or shorter rolling windows
- DECELERATION: event-based or shorter rolling windows

Acceleration events may not last 30 seconds, so requiring a full 30-second pure acceleration window could discard almost all useful data.

This is one of the main reasons the existing idle window design cannot simply be copied to every mode.

---

## Anomaly Models

### Recommended First Version

Train one autoencoder per mode:

```text
Idle Model
Acceleration Model
Cruise Model
Deceleration/Coasting Model
Startup Model (later)
```

Each model receives only healthy examples from its own mode.

Each model gets:

- its own feature list;
- its own scaler;
- its own reconstruction-error distribution;
- its own anomaly threshold.

Do NOT use the idle scaler or idle threshold for acceleration data.

---

## Dataset Problem

The current code trains on `idle*.csv`.

That is enough to build an idle prototype but not a full driving anomaly system.

Before training other modes, inspect all recordings in the dataset and determine:

- which files contain actual driving;
- how much time exists in each mode;
- which sensors are consistently available;
- whether the data is all healthy;
- whether vehicles/drivers differ across recordings.

Create a dataset inventory such as:

```csv
file,mode,seconds,vehicle_id,healthy
drive01.csv,ACCELERATING,84,car1,1
drive01.csv,CRUISING,210,car1,1
drive01.csv,COASTING_DECEL,52,car1,1
```

If there is not enough healthy data for a mode, do not pretend its anomaly model is reliable.

---

## Training Data Leakage

Always split by recording/session/vehicle before creating final training and test sets.

Do not allow nearby windows from the same drive to appear in both train and test.

The current idle code already splits by recording, which is the correct idea.

Later, if multiple vehicles are available, also test:

- same-vehicle unseen drives;
- completely unseen vehicle.

Those are different difficulty levels.

---

## Thresholds

Use separate thresholds per mode.

Example metadata:

```json
{
  "mode": "CRUISING",
  "window_seconds": 15,
  "threshold": 0.84,
  "features": [
    "ENGINE_RPM_mean",
    "ENGINE_RPM_std",
    "VEHICLE_SPEED_mean"
  ]
}
```

Threshold calibration should use healthy calibration data that was not used for fitting the model.

---

## Anomaly Output

Do not output only `True` or `False`.

Return something like:

```json
{
  "timestamp": 124.8,
  "mode": "CRUISING",
  "mode_confidence": 0.96,
  "anomaly": true,
  "anomaly_score": 1.42,
  "threshold": 0.88,
  "largest_feature_errors": [
    "COOLANT_TEMPERATURE_mean",
    "ENGINE_LOAD_mean",
    "ENGINE_RPM_std"
  ]
}
```

This makes debugging and eventual diagnosis much easier.

---

## Raspberry Pi Runtime

The Pi runtime should perform inference only.

```text
OBD adapter
   ->
sample collector
   ->
preprocessing / short history
   ->
mode detector
   ->
mode-specific rolling window
   ->
model registry
   ->
anomaly score
   ->
log / UI / warning
```

Save results locally in a simple log format such as CSV or JSONL.

TensorFlow may be unnecessarily heavy for the final Pi deployment. Once the model architecture is stable, evaluate:

- TensorFlow Lite
- ONNX Runtime
- a smaller scikit-learn model

Do this after the behavior is correct, not before.

---

## Testing Requirements

### Mode Tests

Create synthetic traces for:

- engine off -> startup -> idle
- idle -> acceleration
- acceleration -> cruise
- cruise -> deceleration
- deceleration -> stop -> idle
- noisy speed around zero
- brief acceleration spike that should NOT cause a mode change
- missing sensor values

### Routing Test

Confirm:

```text
IDLE -> idle model
ACCELERATING -> acceleration model
CRUISING -> cruise model
COASTING_DECEL -> deceleration model
```

A unit test should fail if acceleration data is ever sent to the idle model.

### Anomaly Tests

Start with controlled synthetic faults only as sanity tests:

- abnormal coolant temperature
- impossible sensor jump
- unusual RPM variance
- unexpected load for a stable mode

Synthetic faults are not a replacement for real fault data.

---

# Development Phases

## Phase 0 - Clean the Existing Prototype

- [ ] Move constants into `config.py`.
- [ ] Move CSV/time cleanup into `obd_io.py` / `preprocessing.py`.
- [ ] Move mode detection into `mode_detector.py`.
- [ ] Move model scoring into `anomaly_detector.py`.
- [ ] Move top-level flow into `pipeline.py`.
- [ ] Keep the current idle model working after refactor.

Definition of done:
Running the same idle test file after refactoring produces equivalent idle anomaly results.

---

## Phase 1 - Dataset Inventory

- [ ] Scan every CSV.
- [ ] Record sensors available in each file.
- [ ] Run the mode detector on every driving recording.
- [ ] Measure seconds/windows available for each mode.
- [ ] Plot/check a sample from each detected mode.
- [ ] Manually inspect incorrect mode transitions.
- [ ] Save a mode inventory CSV.

Definition of done:
We know how much usable healthy data exists for every target mode.

---

## Phase 2 - Improve Mode Detection

- [ ] Separate strong braking from normal coasting.
- [ ] Add throttle/load context.
- [ ] Add startup/warm-up logic using coolant temperature when available.
- [ ] Add confidence.
- [ ] Tune hysteresis/persistence.
- [ ] Add unit tests for transitions.

Definition of done:
Representative drives produce reasonable, stable mode labels without rapid mode flipping.

---

## Phase 3 - Build Mode-Specific Training Sets

- [ ] Extract stable mode segments.
- [ ] Do not let windows cross mode boundaries.
- [ ] Pick an appropriate window size for each mode.
- [ ] Split by recording before training.
- [ ] Save feature tables per mode.

Definition of done:
Each target mode has a clean healthy training/calibration/test dataset.

---

## Phase 4 - Train Mode-Specific Models

- [ ] Train idle model.
- [ ] Train cruise model.
- [ ] Train acceleration model.
- [ ] Train coasting/deceleration model.
- [ ] Calibrate threshold independently for each mode.
- [ ] Save model + scaler + metadata.

Definition of done:
The model registry can load and score every supported mode.

---

## Phase 5 - End-to-End Pipeline

- [ ] Input sample.
- [ ] Estimate mode.
- [ ] Accumulate correct mode window.
- [ ] Route to correct anomaly model.
- [ ] Output score/anomaly.
- [ ] Log decisions.

Definition of done:
One driving file can move through several modes and each stable section is scored by the correct model.

---

## Phase 6 - Raspberry Pi Integration

- [ ] Replace CSV source with live OBD source.
- [ ] Keep CSV replay mode for testing.
- [ ] Export lightweight models if necessary.
- [ ] Measure CPU usage, memory, and inference latency.
- [ ] Store timestamped output logs.
- [ ] Automatically reconnect to the OBD adapter.

Definition of done:
The Pi can run the detector continuously without model training and without losing data during normal operation.

---

## Phase 7 - Fault Data / Diagnosis

The current anomaly system answers:

> "Does this look unlike healthy behavior for the current operating mode?"

It does not yet answer:

> "What exact component is failing?"

For diagnosis:

- obtain real labeled fault data;
- record DTCs when available;
- collect pre-fault and post-fault behavior;
- analyze which features contribute most to anomaly score;
- later map anomaly patterns to likely fault classes.

---

# Immediate Next Tasks

1. Refactor the current working idle script without changing its behavior.
2. Inventory all dataset files instead of only `idle*.csv`.
3. Run the current mode detector over driving files.
4. Count how much data exists for each mode.
5. Manually inspect mode labels.
6. Improve `DECELERATING` into `COASTING_DECEL` and `BRAKING_DECEL`.
7. Add load/throttle information where available.
8. Build the first non-idle model, preferably CRUISING because it is likely to provide longer stable windows than acceleration.
9. Add acceleration/deceleration models after selecting shorter/event-based windows.
10. Deploy trained model artifacts to the Pi only after offline evaluation is reliable.

---

# Main Design Rule

**Detect the operating mode first. Compare the vehicle only to normal behavior for that mode.**

If the system is not confident about the mode, it should wait rather than confidently call normal behavior anomalous because it used the wrong model.
