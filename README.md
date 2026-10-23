# AI-Based Motor & Bearing Condition Monitoring (research prototype)
Healthy/faulty + 4-class (Normal / Inner Race / Outer Race / Ball) diagnosis from labelled reference vibration data.

## Setup
    python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
    pip install -r requirements.txt

## Add CWRU data
Download the 12 kHz Drive-End .mat files from https://engineering.case.edu/bearingdatacenter/download-data-file
into `data/cwru/` (original names like 97.mat, 105.mat, 118.mat, 130.mat ... are recognised), or
sort them into folders named Normal/, InnerRace/, OuterRace/, Ball/.

## Train (real metrics, written to results/results.json)
    cd backend && python train.py --data ../data/cwru --fs 12000
(or press "Start training" on the Model Training page)

Training evaluates classical estimators plus 1D CNN, LSTM, and CNN-LSTM sequence
models. The sequence models are saved as `.pt` checkpoints under `models/`.
Reported metrics are measured on held-out reference recordings; they do not
guarantee performance on signals from different machines, sensors, or operating
conditions. Validate with representative labelled data before operational use.

## Run
    cd backend && uvicorn app:app --reload      # open http://127.0.0.1:8000
Signal analysis accepts recordings from 64 samples onward and reduces the
analysis window to fit shorter files. Set the sampling frequency to match the
uploaded signal.

## Deploy on Vercel

The Vercel deployment serves the frontend and FastAPI endpoints from one
project. Import the repository in Vercel and set its **Root Directory** to the
repository root (`./`), not `frontend`. `pyproject.toml` points Vercel to
`backend.app:app`; `frontend/config.js` leaves the API base empty so the UI
calls the same-origin API.

Vercel runs the API as a serverless function. Signal analysis and predictions
use the bundled models, but model training is disabled because background
training jobs and persistent writes are not supported by this deployment.
Training remains available when running the app locally. For local
development, use the commands in Setup and Run above.
