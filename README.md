# Smart Traffic Predictor

**AI-Based Smart Traffic Prediction System**

---

**Author:** Deepak Singh
**Email:** todeepaksingh.com@gmail.com
**Copyright:** © 2026 Deepak Singh. All rights reserved.

> This project is developed for portfolio and research demonstration purposes. Unauthorized commercial use, redistribution, or modification without explicit permission from the author is not permitted.

---

## Table of Contents

1. [About This Project](#about-this-project)
2. [How It Works](#how-it-works)
3. [Project Structure](#project-structure)
4. [Libraries and Techniques Used](#libraries-and-techniques-used)
5. [Installation Guide](#installation-guide)
6. [API Keys and URL Configuration](#api-keys-and-url-configuration)
7. [How to Use](#how-to-use)
8. [API Reference](#api-reference)
9. [Running Tests](#running-tests)
10. [Training Your Own Model](#training-your-own-model)
11. [Troubleshooting](#troubleshooting)

---

## About This Project

This project uses **Machine Learning** to predict traffic volume and congestion levels at road intersections. Given the time of day, weather conditions, day of week, and historical traffic data, it predicts:

- **Traffic volume** — how many vehicles per hour
- **Congestion level** — Low, Moderate, High, or Severe

### Real-world use cases

- City traffic management centers monitoring congestion in real time
- Navigation apps calculating estimated travel times
- Urban planners designing road infrastructure
- Smart city dashboards showing live traffic heat maps
- Event coordinators planning road closures

### Key results on test data

| Metric | Value |
|--------|-------|
| R² (volume accuracy) | **0.9855** (98.5% variance explained) |
| MAE (avg error) | **25.2 vehicles/hour** |
| Congestion Accuracy | **91.4%** |
| Congestion Macro-F1 | **0.841** |

---

## How It Works

The system uses a **dual-head model** — one head for regression (volume), one for classification (congestion level):

### Input Features (30 features)

**Temporal features:**
- Hour of day (0–23), day of week (0=Monday), month
- Is weekend, is holiday, is morning rush (7–9 AM), is evening rush (4–7 PM), is night
- Cyclic encoding: hour_sin, hour_cos, day_sin, day_cos (captures circular time)

**Location features:**
- Intersection ID, intersection type (residential / arterial / highway / downtown / commercial)
- Speed limit, number of lanes

**Weather features:**
- Weather code (clear / cloudy / rain / heavy rain / snow / fog)
- Temperature, precipitation (mm), visibility (km)

**Historical lag features:**
- Traffic volume 1, 2, and 3 hours ago
- Average speed 1, 2, and 3 hours ago
- Rolling mean volume over 3h, 6h, 12h windows
- Rolling mean speed over 3h window

### Model Architecture
```
Input Features (30)
        |
  StandardScaler
   /           \
Gradient          Gradient
Boosting          Boosting
Regressor      Classifier
   |                |
Predicted       Congestion
 Volume           Level
(vehicles/h)  (0-3: Low-Severe)
```

### Why Gradient Boosting?
- Handles non-linear relationships (e.g., rush-hour spikes are not linear)
- Built-in feature importance for interpretability
- Works well with mixed feature types (categorical + continuous + cyclic)
- No need for a GPU — fast training and inference on CPU

---

## Project Structure

```
smart-traffic-predictor/
|
+-- src/traffic_predictor/          # Main source package
|   +-- api/                        # REST API (FastAPI)
|   |   +-- main.py                 # App factory + startup
|   |   +-- routes/
|   |   |   +-- predict.py          # /predict, /predict/batch, /forecast
|   |   |   +-- health.py           # /health, /ready, /info
|   |   +-- schemas/
|   |       +-- prediction.py       # Pydantic request/response models
|   +-- data/
|   |   +-- dataset.py              # Dataset class + synthetic traffic generator
|   +-- features/
|   |   +-- feature_engineering.py  # Feature vector builder, scaler, congestion mapper
|   +-- models/
|   |   +-- base_model.py           # Abstract base class
|   |   +-- traditional/
|   |       +-- traffic_classifier.py  # Gradient Boosting / Random Forest / Ridge
|   +-- training/
|   |   +-- trainer.py              # Training loop, CV, evaluation, artifact saving
|   +-- evaluation/
|   |   +-- metrics.py              # MAE, RMSE, R2, accuracy, F1, plots
|   +-- utils/
|       +-- config.py               # Configuration management
|       +-- logger.py               # Structured logging (loguru)
|
+-- scripts/                        # Command-line tools
|   +-- train.py                    # Train a model
|   +-- evaluate.py                 # Evaluate a saved model
|   +-- predict.py                  # Make predictions from CLI
|
+-- tests/                          # Test suite (54 tests)
|   +-- unit/                       # Unit tests
|   |   +-- test_dataset.py
|   |   +-- test_features.py
|   |   +-- test_model.py
|   |   +-- test_metrics.py
|   +-- integration/
|       +-- test_api.py             # API endpoint tests
|
+-- models/saved/                   # Trained model files (.pkl)
+-- data/raw/                       # Raw data (CSV)
+-- reports/                        # Evaluation reports, confusion matrices, plots
|
+-- pyproject.toml                  # Package config (uv)
+-- requirements.txt                # Dependencies
+-- README.md                       # This file
```

---

## Libraries and Techniques Used

### Core ML
| Library | Version | Purpose |
|---------|---------|---------|
| **scikit-learn** | 1.3+ | Gradient Boosting, Random Forest, Ridge regression, StandardScaler, KFold CV |
| **numpy** | 1.24+ | Numerical arrays, matrix operations |
| **scipy** | 1.11+ | Statistical functions |
| **joblib** | 1.3+ | Saving/loading model files |

### Data Handling
| Library | Version | Purpose |
|---------|---------|---------|
| **polars** | 1.0+ | Fast CSV/Parquet file I/O |

### API / Web Service
| Library | Version | Purpose |
|---------|---------|---------|
| **FastAPI** | 0.104+ | REST API framework |
| **uvicorn** | 0.24+ | ASGI web server |
| **pydantic** | 2.5+ | Request/response validation |
| **pydantic-settings** | 2.1+ | Config from environment variables |
| **httpx** | 0.25+ | HTTP client for integration tests |

### Logging and Visualization
| Library | Version | Purpose |
|---------|---------|---------|
| **loguru** | 0.7+ | Structured logging |
| **matplotlib** | 3.8+ | Actual vs predicted volume plots |
| **seaborn** | 0.13+ | Confusion matrix heatmaps |

### Package Management
- **uv** — Fast Python package manager

### ML Techniques Applied
- **Gradient Boosting Regression** — Volume prediction (R²=0.98)
- **Gradient Boosting Classification** — 4-class congestion prediction
- **Dual-head model** — Regression + classification from the same feature input
- **Feature scaling** — StandardScaler for stable gradient descent
- **Cyclic time encoding** — sin/cos transforms for hour and day (avoids 23→0 discontinuity)
- **Lag features** — Captures temporal dependency (past volume/speed predicts current)
- **Rolling window averages** — 3h, 6h, 12h smoothed historical context
- **K-Fold cross-validation** — Robust performance estimation
- **BPR speed model** — Bureau of Public Roads formula for speed estimation from volume/capacity ratio
- **Volume-to-capacity ratio** — Physics-based congestion level derivation

---

## Installation Guide

### Prerequisites

- **Python 3.10 or higher**
- **Windows 10/11**, macOS, or Linux
- At least **1 GB free disk space**
- Internet connection (first-time setup)

### Step 1: Install uv

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Navigate to the project folder

```bash
cd smart-traffic-predictor
```

### Step 3: Create environment and install

```bash
uv venv
uv pip install -e .
```

---

## API Keys and URL Configuration

### No API keys required by default

This project runs fully offline — no external APIs, no authentication needed to get started.

### HuggingFace Token (optional — only for private datasets)

If you want to load a private traffic dataset from HuggingFace:

**How to get a token:**
1. Create a free account at [https://huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens → New token
3. Select "Read" role, copy the token (starts with `hf_...`)

**Set the token:**

Windows:
```powershell
set HF_TOKEN=hf_your_token_here
```

macOS / Linux:
```bash
export HF_TOKEN=hf_your_token_here
```

### Changing the API server URL

By default the API runs on `http://127.0.0.1:8001` (localhost only).

**To share with others on your network:**
```bash
uvicorn traffic_predictor.api.main:app --host 0.0.0.0 --port 8001
```

Others can then access it at `http://<your-ip>:8001`.

**For cloud deployment** (AWS / Azure / GCP / Render / Railway):
Update the base URL in any client apps from:
```
http://127.0.0.1:8001
```
to your server's public address:
```
https://traffic-api.yourserver.com
```

**Example `.env` file for a deployed server:**
```env
APP_HOST=0.0.0.0
APP_PORT=8001
APP__LOG_LEVEL=INFO
HF_TOKEN=hf_your_token_here
```

Load it before starting the server:

Windows:
```powershell
Get-Content .env | ForEach-Object { if ($_ -match "^(\w+)=(.+)$") { [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2]) } }
uvicorn traffic_predictor.api.main:app --host 0.0.0.0 --port 8001
```

macOS / Linux:
```bash
export $(cat .env | grep -v "#" | xargs)
uvicorn traffic_predictor.api.main:app --host 0.0.0.0 --port 8001
```

### CORS for web app integration

Edit `src/traffic_predictor/api/main.py` and add your frontend's URL:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-dashboard.com", "http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
```

### Quick Reference

| Feature | Needs Key/Config? |
|---------|-----------------|
| Local predictions (CLI) | No |
| Local API server | No |
| Public HuggingFace datasets | No |
| Private HuggingFace datasets | Yes — `HF_TOKEN` |
| Sharing API on local network | No — change `--host` to `0.0.0.0` |
| Cloud deployment | No key — update the URL in your clients |

---

## How to Use

### Step 1: Train the Model (required before first use)

**Windows:**
```powershell
.venv\Scripts\python.exe scripts\train.py
```

**macOS / Linux:**
```bash
.venv/bin/python scripts/train.py
```

This generates 50,000 synthetic traffic records across 20 intersections, trains a Gradient Boosting model with 5-fold cross-validation, and saves the model to `models/saved/`.

**Expected output:**
```
TRAINING COMPLETE
TEST MAE:     25.2
TEST R2:      0.9855
TEST Accuracy:0.9145
TEST F1:      0.8410
Model saved to models/saved/traffic_gradient_boosting.pkl
```

---

### Option A: Command Line Prediction

**Single hour prediction:**

Windows:
```powershell
.venv\Scripts\python.exe scripts\predict.py --hour 8 --dow 1 --month 6 --intersection 0 --itype 1 --weather 0 --speed-limit 50 --lanes 2
```

macOS / Linux:
```bash
.venv/bin/python scripts/predict.py --hour 8 --dow 1 --month 6 --intersection 0 --itype 1 --weather 0 --speed-limit 50 --lanes 2
```

**Expected output (8AM Tuesday, arterial road):**
```
Hour 08:00  |  Tuesday  |  Clear
  Volume:     1040 vehicles/hour
  Congestion: [RED] SEVERE
  Confidence: 69.2%
```

**Multi-hour forecast (next 6 hours):**
```bash
python scripts/predict.py --hour 8 --dow 1 --month 6 --intersection 0 --itype 1 --weather 0 --speed-limit 50 --lanes 2 --forecast 6
```

**Parameters explained:**

| Flag | Description | Example values |
|------|-------------|---------------|
| `--hour` | Hour of day | 0–23 |
| `--dow` | Day of week | 0=Monday, 5=Saturday, 6=Sunday |
| `--month` | Month | 1–12 |
| `--itype` | Intersection type | 0=residential, 1=arterial, 2=highway, 3=downtown, 4=commercial |
| `--weather` | Weather | 0=clear, 1=cloudy, 2=rain, 3=heavy rain, 4=snow, 5=fog |
| `--speed-limit` | Speed limit km/h | 30, 50, 70, 90, 110 |
| `--lanes` | Number of lanes | 2, 4, 6 |
| `--forecast` | Hours to predict | 1–24 |

---

### Option B: REST API

**Start the API server:**

Windows:
```powershell
.venv\Scripts\python.exe -m uvicorn traffic_predictor.api.main:app --host 127.0.0.1 --port 8001
```

macOS / Linux:
```bash
.venv/bin/python -m uvicorn traffic_predictor.api.main:app --host 127.0.0.1 --port 8001
```

**Interactive docs:** Open **http://127.0.0.1:8001/docs** in your browser.

---

## API Reference

### Health Check
```
GET http://127.0.0.1:8001/health
```
Response: `{"status": "healthy", "timestamp": "..."}`

---

### Single Prediction
```
POST http://127.0.0.1:8001/api/v1/predict
Content-Type: application/json

{
  "hour": 8,
  "day_of_week": 1,
  "month": 6,
  "intersection_id": 0,
  "intersection_type": 1,
  "weather_code": 0,
  "temperature_c": 22.0,
  "precipitation_mm": 0.0,
  "visibility_km": 10.0,
  "speed_limit_kmh": 50.0,
  "num_lanes": 2,
  "is_holiday": 0,
  "volume_lag1": 900.0,
  "volume_lag2": 850.0,
  "volume_lag3": 800.0,
  "speed_lag1": 30.0,
  "speed_lag2": 32.0,
  "speed_lag3": 35.0,
  "volume_rolling_mean_3h": 850.0,
  "volume_rolling_mean_6h": 700.0,
  "volume_rolling_mean_12h": 600.0,
  "speed_rolling_mean_3h": 32.0
}
```

**Response:**
```json
{
  "intersection_id": 0,
  "hour": 8,
  "day_of_week": 1,
  "predicted_volume": 1323.5,
  "predicted_speed_kmh": 22.1,
  "congestion_level": 3,
  "congestion_label": "severe",
  "congestion_probabilities": {
    "low": 0.0,
    "moderate": 0.01,
    "high": 0.0,
    "severe": 0.98
  },
  "confidence": 0.984,
  "model_name": "traffic_gradient_boosting"
}
```

---

### Batch Prediction (up to 100 time slots at once)
```
POST http://127.0.0.1:8001/api/v1/predict/batch
Content-Type: application/json

{
  "requests": [
    { "hour": 8, "day_of_week": 1, ... },
    { "hour": 17, "day_of_week": 1, ... }
  ]
}
```

---

### Multi-hour Forecast
```
POST http://127.0.0.1:8001/api/v1/forecast
Content-Type: application/json

{
  "intersection_id": 0,
  "intersection_type": 1,
  "start_hour": 7,
  "day_of_week": 1,
  "month": 6,
  "n_hours": 6,
  "weather_code": 0,
  "temperature_c": 20.0,
  "speed_limit_kmh": 50.0,
  "num_lanes": 2,
  "is_holiday": 0
}
```

**Response:**
```json
{
  "intersection_id": 0,
  "start_hour": 7,
  "n_hours": 6,
  "forecast": [
    {"hour": 7, "predicted_volume": 850.0, "congestion_label": "high", "congestion_level": 2, "confidence": 0.91},
    {"hour": 8, "predicted_volume": 1040.0, "congestion_label": "severe", "congestion_level": 3, "confidence": 0.69},
    {"hour": 9, "predicted_volume": 821.0, "congestion_label": "high", "congestion_level": 2, "confidence": 0.83},
    ...
  ]
}
```

---

### Model Info
```
GET http://127.0.0.1:8001/info
```

---

## Understanding the Results

### Congestion Levels

| Level | Label | V/C Ratio | What it means |
|-------|-------|-----------|---------------|
| 0 | **Low** (Green) | < 40% | Free flow, no delays |
| 1 | **Moderate** (Yellow) | 40–65% | Slight slowdown, minor delays |
| 2 | **High** (Orange) | 65–85% | Significant delays, reduce speed |
| 3 | **Severe** (Red) | > 85% | Heavy congestion, major delays |

V/C Ratio = Traffic Volume divided by Road Capacity.

### Volume / Capacity by Intersection Type

| Type | Typical Capacity (per lane) |
|------|-----------------------------|
| Residential | 400 vehicles/hour |
| Arterial | 1200 vehicles/hour |
| Highway on-ramp | 2500 vehicles/hour |
| Downtown | 1800 vehicles/hour |
| Commercial | 900 vehicles/hour |

---

## Running Tests

**Windows:**
```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

**macOS / Linux:**
```bash
.venv/bin/python -m pytest tests/ -v
```

**Expected result:** `54 passed`

---

## Training Your Own Model

### Using Synthetic Data (default)

```bash
# Default: 50000 records, Gradient Boosting
python scripts/train.py

# More records for better accuracy
python scripts/train.py --n-samples 100000

# Different model type
python scripts/train.py --model-type random_forest
python scripts/train.py --model-type ridge

# Skip cross-validation (faster training)
python scripts/train.py --no-cv

# More intersections (more variety)
python scripts/train.py --n-intersections 50
```

### Using Your Own CSV Data

Prepare a CSV file with these required columns:

| Column | Type | Description |
|--------|------|-------------|
| `hour` | int | Hour of day (0–23) |
| `day_of_week` | int | 0=Monday to 6=Sunday |
| `month` | int | 1–12 |
| `intersection_id` | int | Location identifier |
| `intersection_type` | int | 0=residential to 4=commercial |
| `weather_code` | int | 0=clear to 5=fog |
| `temperature_c` | float | Temperature |
| `precipitation_mm` | float | Rainfall/snowfall |
| `visibility_km` | float | Visibility |
| `speed_limit_kmh` | float | Posted speed limit |
| `num_lanes` | int | Lane count |
| `volume_lag1/2/3` | float | Past volumes |
| `speed_lag1/2/3` | float | Past speeds |
| `volume_rolling_mean_3h/6h/12h` | float | Rolling averages |
| `speed_rolling_mean_3h` | float | Speed rolling avg |
| `traffic_volume` | float | **Target: actual volume** |
| `congestion_level` | int | **Target: 0–3** |

```bash
python scripts/train.py --data path/to/your_traffic.csv
```

### Evaluate a Trained Model

```bash
python scripts/evaluate.py --model models/saved/traffic_gradient_boosting.pkl
```

---

## Troubleshooting

### "No model found" error when starting API
Run training first:
```bash
python scripts/train.py
```

### Training is slow
Use fewer samples or skip CV:
```bash
python scripts/train.py --n-samples 10000 --no-cv
```

### Installation fails on Windows with DLL errors
Use binary-only installation:
```bash
uv pip install -e . --no-build
```

### Port 8001 already in use
```bash
uvicorn traffic_predictor.api.main:app --port 8002
```

### Model predicts incorrectly for my real data
The default model is trained on synthetic data with idealized patterns. For better real-world accuracy, train with real traffic data from:
- **City open data portals** (many cities publish hourly traffic counts)
- **Uber Movement** — historical travel times
- **NOAA** — weather data to combine with counts
- **OpenTraffic** — open-source traffic data

---

## For Developers: Advanced Configuration

Override settings via environment variables:

```bash
# Windows
set APP__MODEL__DEFAULT_CLASSIFIER=random_forest
set APP__DATA__N_SYNTHETIC_RECORDS=100000
set APP__API__PORT=9000

# macOS / Linux
export APP__MODEL__DEFAULT_CLASSIFIER=random_forest
```

See `src/traffic_predictor/utils/config.py` for all available settings.

---

## Author

**Deepak Singh**
Email: todeepaksingh.com@gmail.com

© 2026 Deepak Singh. All rights reserved.
Built with Python, FastAPI, scikit-learn, and Gradient Boosting. Package management with [uv](https://docs.astral.sh/uv/).
