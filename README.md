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

## Deploy: Vercel frontend + Render API

The browser UI is a static site; model inference and training need a Python API.
Deploy them separately:

1. Push this project to a GitHub repository. The `.gitignore` excludes the
   virtual environment, local secrets, and the 140 MB reference-data directory;
   it retains the trained model artifacts and benchmark metrics needed to serve
   predictions.
2. In Vercel, import that repository and set the **Root Directory** to
   `frontend`. Vercel serves the static site from `frontend/vercel.json`.
3. In Render, create a Blueprint from the repository root and apply
   `render.yaml`. This creates a Python API service with a persistent disk for
   model checkpoints, results, and training data. Persistent disks require a
   paid Render web-service plan.
4. Set `CORS_ORIGINS` on Render to the exact deployed Vercel origin (for
   example, `https://your-project.vercel.app`) and redeploy the API.
5. Set `window.BEARING_API_BASE` in `frontend/config.js` to the Render service
   URL (for example, `https://bearing-monitor-api.onrender.com`), then push the
   change to GitHub so Vercel redeploys.
6. `TRAINING_TOKEN` is requested as a secret when creating the Render
   Blueprint. Use that token in the Model Training page. To populate the
   backend's persistent training-data directory, open the Render Shell and run:
       python backend/download_reference_data.py --output /var/data/cwru
   Alternatively, place your own labelled `.mat` recordings in class-named
   subfolders under `/var/data/cwru` (`Normal`, `InnerRace`, `OuterRace`, `Ball`).
   The training route reads backend-side labelled recordings; an individual
   user's diagnosis upload is used for prediction and is not added to training.

For local development, leave `window.BEARING_API_BASE` empty and run the API
using the command above. Do not put the Render training token in frontend
configuration, source control, or a Vercel environment variable exposed to
browser code.
