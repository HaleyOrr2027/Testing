# Plan: Mode-Aware OBD Anomaly Detection Pipeline

## Goal

Build a Raspberry Pi OBD-II system that can:

1. read live vehicle data,
2. determine the current operating mode,
3. compare the current behavior only against normal behavior for that mode,
4. detect anomalies,
5. report the mode, anomaly score, and important contributing signals.

The final runtime pipeline should be:

```text
OBD-II data
    ↓
preprocessing
    ↓
mode detection
    ↓
mode-specific window
    ↓
correct mode model
    ↓
anomaly score
    ↓
normal / anomalous result
```

The main reason for this design is that normal behavior depends on the operating mode.

Example:

```text
3000 RPM while accelerating = possibly normal
3000 RPM while idling       = probably abnormal
```

The system therefore should not use one single definition of normal for every driving condition.

---

# 0. File Structure

```text
OBD/
│
├── plan.md
├── requirements.txt
├── README.md
│
├── config/
│   └── mode_config.json
│
├── data/
│   ├── raw/
│   │   └── carOBD/
│   │
│   ├── inventory/
│   │   ├── dataset_inventory.csv
│   │   ├── sensor_inventory.csv
│   │   └── mode_inventory.csv
│   │
│   ├── processed/
│   │   ├── idle_windows.csv
│   │   ├── cruise_windows.csv
│   │   ├── acceleration_windows.csv
│   │   └── deceleration_windows.csv
│   │
│   └── labeled/
│       └── labeled_drives/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── obd_io.py
│   ├── preprocessing.py
│   ├── mode_detector.py
│   ├── feature_windows.py
│   ├── anomaly_detector.py
│   ├── model_registry.py
│   ├── result_formatter.py
│   └── pipeline.py
│
├── training/
│   ├── 00_scan_dataset.py
│   ├── 01_label_modes.py
│   ├── 02_build_mode_windows.py
│   ├── 03_train_autoencoder.py
│   ├── 04_calibrate_thresholds.py
│   └── 05_evaluate_models.py
│
├── models/
│   ├── idle/
│   │   ├── model.keras
│   │   ├── scaler.pkl
│   │   └── metadata.json
│   │
│   ├── cruising/
│   │   ├── model.keras
│   │   ├── scaler.pkl
│   │   └── metadata.json
│   │
│   ├── accelerating/
│   │   ├── model.keras
│   │   ├── scaler.pkl
│   │   └── metadata.json
│   │
│   └── decelerating/
│       ├── model.keras
│       ├── scaler.pkl
│       └── metadata.json
│
├── scripts/
│   ├── run_file.py
│   └── run_pi.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_mode_detector.py
│   ├── test_feature_windows.py
│   ├── test_model_registry.py
│   └── test_pipeline.py
│
└── outputs/
    ├── mode_plots/
    ├── evaluation/
    └── pi_logs/
```

---

# 1. Phase 1 — Clean and Separate the Existing Code

## Purpose

Take the current large script and split it into reusable modules without changing the current idle anomaly behavior.

The first goal is not to improve the model yet.

The first goal is:

```text
same current behavior
+
better project structure
```

---

## Files

### `src/config.py`

Move constants into one file.

Examples:

```python
MODE_CONTEXT_SECONDS = 5.0
MODE_PERSIST_SECONDS = 2.0
STARTUP_GUARD_SECONDS = 30.0

ENGINE_ON_RPM_THRESHOLD = 100.0
STOPPED_SPEED_KMH = 0.5

ACCEL_THRESHOLD_MPS2 = 0.15
DECEL_THRESHOLD_MPS2 = -0.15
```

### Output

A single place to change project thresholds.

---

### `src/obd_io.py`

Move input and CSV cleanup code here.

Responsibilities:

- load CSV files,
- normalize column names,
- fix known dataset typos,
- remove unwanted unnamed columns,
- convert sensor columns to numeric values,
- later handle live OBD input.

### Output

```python
data = load_csv(file_path)
```

where `data` is a cleaned DataFrame.

---

### `src/preprocessing.py`

Move signal-processing code here.

Responsibilities:

- create/estimate time axis,
- smooth speed,
- calculate acceleration,
- detect engine-on state,
- calculate seconds since startup,
- prepare sensor values.

### Output

A DataFrame containing fields such as:

```text
TIME_SEC
SPEED_SMOOTH
ACCELERATION_MPS2
ENGINE_ON
SECONDS_SINCE_START
```

---

### `src/mode_detector.py`

Move current mode-detection logic here.

Initial modes:

```text
ENGINE_OFF
STARTUP
IDLE
ACCELERATING
DECELERATING
CRUISING
UNKNOWN
```

Keep the current persistence logic.

### Output

Each sample gets:

```text
PRELIMINARY_MODE
MODE
MODE_SEGMENT
```

---

### `src/feature_windows.py`

Move window-building code here.

Initial responsibility:

- reproduce current 30-second idle feature windows,
- calculate mean/std features,
- reject windows crossing mode boundaries.

### Output

A table like:

```text
ENGINE_SESSION
WINDOW
MODE
WINDOW_START
WINDOW_END
ENGINE_RPM_mean
ENGINE_RPM_std
...
```

---

### `src/anomaly_detector.py`

Move reconstruction-error scoring here.

Responsibilities:

- scale feature vectors,
- pass them through the autoencoder,
- calculate reconstruction error,
- compare against threshold.

### Output

```text
ANOMALY_SCORE
ANOMALY
```

---

### `src/pipeline.py`

Connect the pieces.

Initial flow:

```text
load file
    ↓
preprocess
    ↓
detect modes
    ↓
build idle windows
    ↓
score idle windows
```

### Output

The same kind of results the current script already produces.

---

## Phase 1 Deliverable

The original giant script is no longer the main project.

Instead:

```text
run_file.py
    ↓
pipeline.py
    ↓
separate src modules
```

### Phase 1 Definition of Done

- current idle model still works,
- same healthy idle files can still be scored,
- code is split into reusable files,
- training code and runtime code are no longer mixed together.

---

# 2. Phase 2 — Scan and Understand the Dataset

## Purpose

Before creating more anomaly models, determine what driving modes actually exist in the dataset.

The current model is based mainly on:

```text
idle*.csv
```

But the full dataset contains many more recordings.

We need to know:

- what files exist,
- what sensors each file contains,
- how long each recording is,
- which files contain driving,
- which modes appear,
- how much usable data exists for each mode.

---

## File

### `training/00_scan_dataset.py`

Scan all CSV files in the dataset.

For each file collect:

```text
filename
row_count
duration
available sensors
minimum speed
maximum speed
minimum RPM
maximum RPM
```

Also record whether important sensors exist:

```text
ENGINE_RPM
VEHICLE_SPEED
ENGINE_RUN_TIME
THROTTLE_POSITION
CALCULATED_ENGINE_LOAD
COOLANT_TEMPERATURE
MAF
MAP
```

---

## Outputs

### `data/inventory/dataset_inventory.csv`

Example:

```csv
file,rows,duration_sec,min_speed,max_speed,min_rpm,max_rpm
idle1.csv,800,240,0,0,710,1150
drive1.csv,1200,600,0,105,680,4100
```

---

### `data/inventory/sensor_inventory.csv`

Example:

```csv
file,ENGINE_RPM,VEHICLE_SPEED,THROTTLE_POSITION,ENGINE_LOAD
idle1.csv,1,1,1,1
drive1.csv,1,1,0,1
```

---

## Phase 2 Deliverable

A clear answer to:

> What usable data do we actually have?

---

# 3. Phase 3 — Build the Mode Detection System

## Purpose

Create a dedicated mode detector that labels all raw driving data before anomaly detection.

The anomaly detector should never decide the mode itself.

---

## File

### `src/mode_detector.py`

Expand the current mode logic.

Recommended first modes:

```text
ENGINE_OFF
STARTUP_WARMUP
IDLE
ACCELERATING
CRUISING
COASTING_DECEL
BRAKING_DECEL
UNKNOWN_TRANSITION
```

---

## Initial Inputs

Required:

```text
VEHICLE_SPEED
ENGINE_RPM
ENGINE_RUN_TIME
```

Recommended where available:

```text
THROTTLE_POSITION
CALCULATED_ENGINE_LOAD
COOLANT_TEMPERATURE
MAF
MAP
```

---

## Derived Inputs

Created in:

### `src/preprocessing.py`

Calculate:

```text
SPEED_SMOOTH
ACCELERATION_MPS2
RPM_CHANGE
THROTTLE_CHANGE
LOAD_CHANGE
SECONDS_SINCE_START
```

Later possibly:

```text
POWER_DEMAND
APPROX_VSP
```

---

## Initial Decision Logic

```text
if engine is off:
    ENGINE_OFF

elif recently started / still warming:
    STARTUP_WARMUP

elif vehicle is stopped:
    IDLE

elif strong negative acceleration:
    BRAKING_DECEL

elif moderate negative acceleration:
    COASTING_DECEL

elif positive acceleration:
    ACCELERATING

elif moving with near-zero acceleration:
    CRUISING

else:
    UNKNOWN_TRANSITION
```

---

## File

### `config/mode_config.json`

Store mode thresholds separately.

Example:

```json
{
  "stopped_speed_kmh": 0.5,
  "accel_threshold_mps2": 0.15,
  "decel_threshold_mps2": -0.15,
  "strong_decel_threshold_mps2": -1.0,
  "mode_persist_seconds": 2.0
}
```

This lets thresholds be changed without editing the algorithm.

---

## File

### `training/01_label_modes.py`

Run the mode detector over every driving file.

---

## Outputs

### `data/labeled/labeled_drives/<filename>.csv`

Example columns:

```text
TIME_SEC
ENGINE_RPM
VEHICLE_SPEED
ACCELERATION_MPS2
ENGINE_LOAD
MODE
MODE_CONFIDENCE
MODE_SEGMENT
```

---

### `data/inventory/mode_inventory.csv`

Example:

```csv
file,mode,seconds,samples,segments
drive1.csv,IDLE,120,300,4
drive1.csv,ACCELERATING,48,120,7
drive1.csv,CRUISING,280,700,5
drive1.csv,COASTING_DECEL,36,90,6
```

---

### `outputs/mode_plots/`

Create plots showing:

```text
speed
RPM
acceleration
mode
```

for representative drives.

These plots should be manually checked.

---

## Phase 3 Deliverable

A mode detector that can take a drive and produce stable operating-mode labels.

---

# 4. Phase 4 — Validate and Improve the Mode Detector

## Purpose

The detector should not be trusted just because it produces labels.

Compare labels against actual sensor behavior.

---

## File

### `tests/test_mode_detector.py`

Create synthetic tests such as:

```text
ENGINE_OFF → STARTUP → IDLE
IDLE → ACCELERATING
ACCELERATING → CRUISING
CRUISING → COASTING_DECEL
CRUISING → BRAKING_DECEL
DECELERATING → IDLE
```

Also test:

```text
brief speed spike
noisy zero-speed values
missing RPM
missing speed
short acceleration pulse
```

---

## Output

Passing unit tests for expected mode transitions.

---

## File

### `training/01_label_modes.py`

Use it again after thresholds are changed.

---

## Outputs

Updated:

```text
data/labeled/labeled_drives/
data/inventory/mode_inventory.csv
outputs/mode_plots/
```

---

## Phase 4 Deliverable

Mode detection that is stable enough to use for creating training data.

---

# 5. Phase 5 — Build Mode-Specific Training Windows

## Purpose

Convert labeled raw data into clean training examples.

A feature window should contain only one stable operating mode.

Never do:

```text
15 sec cruise
+
15 sec acceleration
=
one CRUISING window
```

---

## File

### `training/02_build_mode_windows.py`

Read:

```text
data/labeled/labeled_drives/
```

Create windows for each mode.

---

## Window Strategy

Initial idea:

```text
IDLE
20–30 second windows

CRUISING
10–20 second windows

ACCELERATING
short rolling or event-based windows

COASTING_DECEL
short rolling or event-based windows

BRAKING_DECEL
short event-based windows
```

Exact sizes should be based on the actual dataset.

Do not force every mode to use the current 30-second idle window.

---

## File

### `src/feature_windows.py`

Functions should include ideas such as:

```python
build_idle_windows(...)
build_cruise_windows(...)
build_acceleration_windows(...)
build_deceleration_windows(...)
```

or one generic function configured by mode.

---

## Outputs

### `data/processed/idle_windows.csv`

### `data/processed/cruise_windows.csv`

### `data/processed/acceleration_windows.csv`

### `data/processed/deceleration_windows.csv`

Each row should contain:

```text
FILE
MODE
SEGMENT
WINDOW_START
WINDOW_END
sensor_mean
sensor_std
...
```

---

## Phase 5 Deliverable

Clean healthy training datasets separated by operating mode.

---

# 6. Phase 6 — Train One Autoencoder Per Mode

## Purpose

Each operating mode learns its own normal behavior.

---

## File

### `training/03_train_autoencoder.py`

Inputs:

```text
mode name
processed mode dataset
feature list
```

Example usage:

```bash
python training/03_train_autoencoder.py --mode IDLE
python training/03_train_autoencoder.py --mode CRUISING
python training/03_train_autoencoder.py --mode ACCELERATING
```

---

## Training Split

Split by recording, not by random rows.

Example:

```text
training recordings
validation recordings
calibration recordings
testing recordings
```

Do not let windows from the same drive appear in both training and testing.

---

## Outputs Per Mode

Example:

### `models/cruising/model.keras`

Trained autoencoder.

### `models/cruising/scaler.pkl`

Scaler fitted only on cruising training data.

### `models/cruising/metadata.json`

Example:

```json
{
  "mode": "CRUISING",
  "window_seconds": 15,
  "features": [
    "ENGINE_RPM_mean",
    "ENGINE_RPM_std",
    "VEHICLE_SPEED_mean",
    "CALCULATED_ENGINE_LOAD_mean"
  ]
}
```

---

## Phase 6 Deliverable

One independently trained normal-behavior model for each supported mode.

---

# 7. Phase 7 — Calibrate Mode-Specific Anomaly Thresholds

## Purpose

Each model needs its own anomaly threshold.

Do not use the idle threshold for cruising or acceleration.

---

## File

### `training/04_calibrate_thresholds.py`

For each mode:

1. load healthy calibration data,
2. reconstruct it,
3. calculate reconstruction error,
4. choose threshold,
5. save threshold into metadata.

---

## Updated Output

Example:

### `models/cruising/metadata.json`

```json
{
  "mode": "CRUISING",
  "window_seconds": 15,
  "threshold": 0.84,
  "false_positive_target": 0.01,
  "features": [
    "ENGINE_RPM_mean",
    "ENGINE_RPM_std",
    "VEHICLE_SPEED_mean"
  ]
}
```

---

## Phase 7 Deliverable

Each mode has:

```text
model
scaler
feature list
threshold
```

---

# 8. Phase 8 — Create the Model Registry

## Purpose

Automatically choose the correct anomaly model based on the detected mode.

---

## File

### `src/model_registry.py`

Load all model bundles.

Conceptually:

```python
models["IDLE"]
models["CRUISING"]
models["ACCELERATING"]
models["COASTING_DECEL"]
models["BRAKING_DECEL"]
```

Each entry contains:

```text
model
scaler
features
threshold
window settings
```

---

## Output

A loaded registry that can answer:

```python
bundle = registry.get("CRUISING")
```

---

## Phase 8 Deliverable

No hard-coded call such as:

```python
score_idle_windows(...)
```

for every situation.

Instead:

```text
mode
 ↓
registry
 ↓
correct model
```

---

# 9. Phase 9 — Build the Complete Offline Pipeline

## Purpose

Test the complete system using recorded CSV files before running live on the Pi.

---

## File

### `src/pipeline.py`

Full sequence:

```text
raw sample
    ↓
preprocess
    ↓
mode detector
    ↓
mode confidence
    ↓
update mode-specific window
    ↓
window complete?
    ↓
model registry
    ↓
anomaly detector
    ↓
result
```

---

## File

### `src/anomaly_detector.py`

Input:

```text
mode
feature window
```

Output:

```text
anomaly score
threshold
anomaly True/False
```

---

## File

### `src/result_formatter.py`

Return useful results instead of only:

```text
True
False
```

Example:

```json
{
  "time": 124.8,
  "mode": "CRUISING",
  "mode_confidence": 0.95,
  "anomaly": true,
  "anomaly_score": 1.42,
  "threshold": 0.84
}
```

Later also include highest-error features.

---

## File

### `scripts/run_file.py`

Example:

```bash
python scripts/run_file.py data/raw/carOBD/drive1.csv
```

---

## Outputs

### Console output

Example:

```text
0.0–18.4 sec    IDLE            normal
18.4–25.7 sec   ACCELERATING    normal
25.7–85.1 sec   CRUISING        normal
85.1–99.3 sec   CRUISING        ANOMALY
99.3–106.4 sec  COASTING_DECEL  normal
```

### `outputs/evaluation/<filename>_results.csv`

Contains:

```text
time
mode
confidence
score
threshold
anomaly
```

---

## Phase 9 Deliverable

One recorded drive can move through multiple operating modes and each stable mode is scored by the correct model.

---

# 10. Phase 10 — Evaluate the Models

## Purpose

Measure whether the system works before deploying it.

---

## File

### `training/05_evaluate_models.py`

Evaluate each mode separately.

For healthy test data calculate:

```text
false positive rate
score distribution
number of windows tested
```

If real fault data becomes available:

```text
true positive rate
false negative rate
precision
recall
```

---

## Outputs

### `outputs/evaluation/model_summary.csv`

Example:

```csv
mode,test_windows,false_positives,false_positive_rate
IDLE,250,3,0.012
CRUISING,180,2,0.011
ACCELERATING,95,1,0.011
```

---

### `outputs/evaluation/<mode>_scores.csv`

Contains individual anomaly scores.

---

## Phase 10 Deliverable

Measured performance for every operating mode.

---

# 11. Phase 11 — Raspberry Pi Live Input

## Purpose

Replace recorded CSV input with live Bluetooth OBD-II data.

The mode/anomaly logic should stay the same.

---

## File

### `src/obd_io.py`

Add live input support.

Conceptually:

```python
read_live_sample()
```

returns the same sensor format used by CSV replay.

This is important because the rest of the system should not care whether data came from:

```text
CSV
```

or:

```text
Bluetooth OBD-II adapter
```

---

## File

### `scripts/run_pi.py`

Main Pi program.

Flow:

```text
connect OBD
    ↓
read sensors continuously
    ↓
pipeline.process(sample)
    ↓
print/log result
```

---

## Outputs

### `outputs/pi_logs/<date>.jsonl`

Example:

```json
{"time":"15:04:11","mode":"IDLE","score":0.21,"anomaly":false}
{"time":"15:04:32","mode":"ACCELERATING","score":0.56,"anomaly":false}
{"time":"15:05:12","mode":"CRUISING","score":1.42,"anomaly":true}
```

---

## Phase 11 Deliverable

The Raspberry Pi runs the already-trained models in real time.

The Pi should not retrain the models every time it starts.

---

# 12. Phase 12 — Optimize for Raspberry Pi

## Purpose

Make sure the system is lightweight enough for continuous operation.

---

## Possible Changes

Evaluate:

```text
TensorFlow Lite
ONNX Runtime
smaller neural network
reduced feature count
```

Only optimize after the full system is working correctly.

---

## Outputs

Record:

```text
CPU usage
memory usage
average inference time
samples per second
```

Store results in:

### `outputs/evaluation/pi_performance.csv`

---

## Phase 12 Deliverable

Continuous Pi inference with acceptable CPU, memory, and latency.

---

# 13. Phase 13 — Real Fault Data and Diagnosis

## Purpose

The current system answers:

> Does this behavior look abnormal for the current mode?

It does not automatically answer:

> Which component is broken?

That requires labeled fault information.

---

## Future Data

Collect:

```text
known healthy drives
known fault drives
DTC codes
maintenance/failure records
pre-failure recordings
post-repair recordings
```

---

## Future Output

Eventually:

```json
{
  "mode": "CRUISING",
  "anomaly": true,
  "score": 1.42,
  "largest_feature_errors": [
    "COOLANT_TEMPERATURE_mean",
    "ENGINE_LOAD_mean",
    "ENGINE_RPM_std"
  ],
  "possible_fault_class": "cooling_system"
}
```

Fault classification should be treated as a later stage after mode-aware anomaly detection works reliably.

---

# Final Runtime Architecture

```text
                OBD-II
                   │
                   ▼
              obd_io.py
                   │
                   ▼
           preprocessing.py
                   │
                   ▼
            mode_detector.py
                   │
                   ▼
           feature_windows.py
                   │
                   ▼
            model_registry.py
                   │
                   ▼
          anomaly_detector.py
                   │
                   ▼
          result_formatter.py
                   │
                   ▼
          NORMAL / ANOMALY
```

---

# Immediate Work Order

## First

Complete:

```text
Phase 1
Phase 2
Phase 3
```

before training new anomaly models.

That means:

1. split the existing code into files,
2. scan the full dataset,
3. label every driving file with modes,
4. inspect the mode labels,
5. determine how much usable data exists for each mode.

## Then

Continue with:

```text
Phase 5 → Phase 10
```

to build and evaluate the mode-specific anomaly models.

## Finally

Move to:

```text
Phase 11 → Phase 12
```

for Raspberry Pi deployment.

---

# Core Rule

```text
DETECT MODE FIRST
        ↓
USE THAT MODE'S NORMAL MODEL
        ↓
CHECK FOR ANOMALY
```

Never compare acceleration, cruising, deceleration, and idle against the same definition of normal.
