"""FastAPI backend. Run:  cd backend && uvicorn app:app --reload"""
import hmac, io, json, os, threading, time, html
from pathlib import Path
import numpy as np, joblib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from scipy.io import loadmat
from scipy import signal
import dsp, train

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BEARING_DATA_DIR", ROOT / "data" / "cwru"))
MODEL_DIR = Path(os.environ.get("BEARING_MODEL_DIR", ROOT / "models"))
RES = Path(os.environ.get("BEARING_RESULTS_PATH", ROOT / "results" / "results.json"))
app = FastAPI(title="Bearing Condition Monitor (research prototype)")
allowed_origins = [origin.strip() for origin in os.environ.get(
    "CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
LAST, TRAIN = {}, {"state": "idle", "log": []}

def results():
    return json.loads(RES.read_text()) if RES.exists() else None

def read_signal(name, raw):
    ext, rpm = Path(name).suffix.lower(), None
    try:
        if ext == ".mat":
            m = loadmat(io.BytesIO(raw)); keys = [k for k in m if not k.startswith("__")]
            k = next((k for k in keys if k.endswith("DE_time")), None) or next(
                (k for k in keys if np.size(m[k]) > 1000 and np.issubdtype(np.asarray(m[k]).dtype, np.number)), None)
            if not k: raise ValueError("no vibration array found")
            x = np.asarray(m[k]).ravel(); r = next((k for k in keys if k.endswith("RPM")), None)
            rpm = float(np.ravel(m[r])[0]) if r else None
        elif ext in (".csv", ".txt"):
            import pandas as pd
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python").apply(pd.to_numeric, errors="coerce")
            x = df.dropna(axis=1, how="all").iloc[:, 0].dropna().values
        elif ext == ".npz":
            z = np.load(io.BytesIO(raw)); x = z[z.files[0]].ravel()
        else: raise ValueError("unsupported type (use .mat .csv .txt .npz)")
    except Exception as e:
        raise HTTPException(400, f"Could not read '{name}': {e}")
    if len(x) < 64: raise HTTPException(400, "Signal too short (need at least 64 samples).")
    return x.astype(float), rpm

def predict(wf, windows):
    r = results()
    if not r: return {"available": False, "message": "Prediction unavailable — model has not been trained."}
    X = np.array([[f[n] for n in r["feature_names"]] for f in wf])
    best = r["best_model"]; t = time.time()
    if best in train.DEEP_MODELS:
        from deep_models import checkpoint_path, predict_proba
        P = predict_proba(checkpoint_path(MODEL_DIR, best), windows)
    else:
        pipe = joblib.load(MODEL_DIR / f"{best.replace(' ', '_')}.joblib")
        P = pipe.predict_proba(X)
    dt = time.time() - t
    mean = P.mean(0); i = int(mean.argmax()); votes = np.bincount(P.argmax(1), minlength=4)
    conf = float(mean[i])
    status = "HEALTHY" if i == 0 and conf >= .7 else "FAULT" if i != 0 and conf >= .7 else "WARNING"
    return {"available": True, "model": best, "condition": r["classes"][i], "status": status,
            "confidence": conf, "probabilities": dict(zip(r["classes"], map(float, mean))),
            "window_votes": dict(zip(r["classes"], map(int, votes))), "prediction_time_s": dt,
            "model_test_accuracy": r["models"][best]["accuracy"],
            "evaluation_note": r.get("evaluation_note", "Performance on other machines and operating conditions may differ. Validate with representative labelled data.")}

def thin(a, n=2500):
    return np.asarray(a)[::max(1, len(a) // n)].tolist()

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), fs: float = Form(12000), low: float = Form(10),
                  high: float = Form(5000), window: int = Form(2048), overlap: float = Form(50), order: int = Form(4)):
    x, rpm = read_signal(file.filename, await file.read())
    if not (0 < low < high < fs / 2): raise HTTPException(400, f"Need 0 < low < high < fs/2 ({fs/2:.0f} Hz).")
    if window < 64: raise HTTPException(400, "Window must be at least 64 samples.")
    window = min(window, len(x))
    y = dsp.preprocess(x, fs, low, high, order)          # original x is untouched
    W = dsp.windows(y, window, overlap / 100); wf = [dsp.features(w, fs) for w in W]
    seg = y[:2 ** 16]; fr, sp = dsp.spectrum(seg, fs); ef, es = dsp.envelope_spectrum(seg, fs)
    m = ef <= 1000
    nperseg = min(256, len(seg))
    f, t, Z = signal.spectrogram(seg, fs, nperseg=nperseg,
                                 noverlap=min(128, nperseg - 1))
    fi, ti = slice(None, None, max(1, len(f) // 80)), slice(None, None, max(1, len(t) // 120))
    out = {"file": {"name": file.filename, "samples": len(x), "fs": fs, "duration_s": len(x) / fs, "rpm": rpm},
           "params": dict(low=low, high=high, window=window, overlap=overlap, order=order, windows=len(W)),
           "raw": x[:4096].tolist(), "filtered": y[:4096].tolist(),
           "t": (np.arange(min(4096, len(x))) / fs).tolist(),
           "fft": {"f": thin(fr), "a": thin(sp)}, "env": {"f": ef[m].tolist(), "a": es[m].tolist()},
           "spec": {"f": f[fi].tolist(), "t": t[ti].tolist(), "z": (10 * np.log10(Z[fi][:, ti] + 1e-12)).tolist()},
           "stats": {k: float(v) for k, v in dsp.basic_stats(y).items()},
           "window_rms": [w["rms"] for w in wf][:400],
           "mean_features": {k: float(np.mean([w[k] for w in wf])) for k in wf[0]},
           "pred": predict(wf, W)}
    LAST.clear(); LAST.update(out); return out

@app.get("/api/dataset/info")
def dataset_info():
    r = results(); return {"loaded": bool(r), **({"dataset": r["dataset"], "split": r["split"]} if r else {})}

@app.get("/api/model-comparison")
def comparison(): return results() or {"models": {}}

@app.post("/api/train")
def start_training(fs: float = Form(12000), token: str = Form("")):
    expected_token = os.environ.get("TRAINING_TOKEN")
    if expected_token and not hmac.compare_digest(token, expected_token):
        raise HTTPException(403, "Invalid training access token.")
    if TRAIN["state"] == "running": raise HTTPException(409, "Training already running.")
    def job():
        TRAIN.update(state="running", log=[])
        try:
            train.run(DATA_DIR, fs, model_dir=MODEL_DIR, results_path=RES,
                      log=lambda s: TRAIN["log"].append(s))
            TRAIN["state"] = "done"
        except Exception as e: TRAIN["log"].append(f"ERROR: {e}"); TRAIN["state"] = "error"
    threading.Thread(target=job, daemon=True).start(); return {"started": True}

@app.get("/api/train/status")
def train_status(): return {"state": TRAIN["state"], "log": TRAIN["log"][-15:]}

@app.get("/api/report", response_class=HTMLResponse)
def report():
    r, a = results(), LAST
    def tab(rows, head):
        return "<table border=1 cellpadding=4 style='border-collapse:collapse'><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows) + "</table>"
    b = "<h1>AI-Based Motor &amp; Bearing Condition Monitoring — Report</h1><p><i>Academic research prototype; not a safety-certified diagnostic system.</i></p>"
    if r:
        d = r["dataset"]; b += f"<h2>Reference training benchmark</h2>{tab(d['windows_per_class'].items(), ['Class', 'Windows'])}<p>{d['recordings']} recordings, {d['windows']} windows. Split: {r['split']['method']}; {r['split']['windows']}</p><p>{html.escape(r.get('evaluation_note', 'Performance on other machines and operating conditions may differ. Validate with representative labelled data.'))}</p>"
        keys = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "fpr", "fnr", "train_time"]
        b += "<h2>Model comparison (held-out test recordings)</h2>" + tab([[n] + [f"{m[k]:.4f}" if m[k] is not None else "n/a" for k in keys] for n, m in r["models"].items()], ["Model", "Acc", "Prec", "Recall", "Spec", "F1", "AUC", "FPR", "FNR", "Train s"])
        b += f"<p><b>Best model (validation macro-F1): {r['best_model']}</b></p><h3>Confusion matrix — {r['best_model']}</h3>{tab(r['models'][r['best_model']]['confusion'], r['classes'])}<p>Not trained / unavailable: {', '.join(r['unavailable'])}</p>"
    else: b += "<p>Dataset not loaded / models not trained.</p>"
    if a:
        b += f"<h2>Analysed signal</h2>{tab(a['file'].items(), ['Field', 'Value'])}<h3>Preprocessing</h3>{tab(a['params'].items(), ['Parameter', 'Value'])}<h3>Features (mean over windows)</h3>{tab([[k, f'{v:.5g}'] for k, v in a['mean_features'].items()], ['Feature', 'Value'])}<h2>Diagnosis</h2><pre>{html.escape(json.dumps(a['pred'], indent=1))}</pre>"
    return HTMLResponse(b + "<p>Print this page (Ctrl+P) to save as PDF.</p>")

app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="ui")
