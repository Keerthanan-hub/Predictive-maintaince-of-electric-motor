"""Train and evaluate vibration models. Usage: python train.py --data ../data/cwru [--fs 12000]
Splits BY RECORDING (per class) so overlapping windows never cross train/val/test."""
import argparse, json, re, time
from pathlib import Path
import numpy as np, joblib
from scipy.io import loadmat
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
import dsp

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["Normal", "Inner Race", "Outer Race", "Ball"]
# CWRU 12k drive-end file numbers -> class (0 Normal, 1 IR, 2 OR, 3 Ball)
ID_MAP = {**{i: 0 for i in range(97, 101)},
          **{i: 1 for r in [(105, 109), (169, 173), (209, 213)] for i in range(*r)},
          **{i: 3 for r in [(118, 122), (185, 189), (222, 226)] for i in range(*r)},
          **{i: 2 for r in [(130, 134), (197, 201), (234, 238)] for i in range(*r)}}
FOLDER = [("normal", 0), ("inner", 1), ("outer", 2), ("ball", 3)]
DEEP_MODELS = ["1D CNN", "LSTM", "CNN-LSTM"]

def label_for(p):
    parts = " ".join(x.lower() for x in p.parts[-3:-1])
    for k, v in FOLDER:
        if k in parts: return v
    m = re.search(r"\d+", p.stem)
    return ID_MAP.get(int(m.group())) if m else None

def load_dataset(root, fs):
    recs = []
    for p in sorted(Path(root).rglob("*.mat")):
        y = label_for(p)
        if y is None: continue
        m = loadmat(p); k = next((k for k in m if k.endswith("DE_time")), None)
        if k: recs.append({"id": p.stem, "label": y, "x": m[k].ravel().astype(float), "fs": fs})
    return recs

def split_recordings(recs, seed=42):
    rng, split = np.random.default_rng(seed), {}
    for c in range(4):
        ids = [r["id"] for r in recs if r["label"] == c]
        if len(ids) < 3: raise ValueError(f"Class '{CLASSES[c]}' has {len(ids)} recordings; need >= 3.")
        rng.shuffle(ids); nt = max(1, round(.2 * len(ids)))
        for i, rid in enumerate(ids): split[rid] = "test" if i < nt else "val" if i < 2 * nt else "train"
    return split

def evaluate(y, pred, proba):
    k = len(CLASSES); cm = confusion_matrix(y, pred, labels=range(k))
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; tn = cm.sum() - tp - fp - fn
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(k), average="macro", zero_division=0)
    try: auc = float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=range(k)))
    except Exception: auc = None
    return {"accuracy": float(accuracy_score(y, pred)), "precision": float(p), "recall": float(r), "f1": float(f),
            "specificity": float(np.mean(tn / (tn + fp))), "fpr": float(np.mean(fp / (fp + tn))),
            "fnr": float(np.mean(fn / (fn + tp))), "roc_auc": auc, "confusion": cm.tolist(),
            "per_class": {CLASSES[i]: dict(TP=int(tp[i]), FP=int(fp[i]), TN=int(tn[i]), FN=int(fn[i])) for i in range(k)},
            "healthy_vs_faulty_accuracy": float(np.mean((y == 0) == (pred == 0)))}

def run(data_dir, fs=12000, size=2048, overlap=.5, log=print, model_dir=None, results_path=None):
    model_dir = Path(model_dir) if model_dir is not None else ROOT / "models"
    results_path = Path(results_path) if results_path is not None else ROOT / "results" / "results.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    recs = load_dataset(data_dir, fs)
    if not recs: raise ValueError(f"No labelled reference .mat files found in {data_dir}")
    split = split_recordings(recs); X, y, S, sequences = [], [], [], []
    for recording_number, r in enumerate(recs, start=1):
        W = dsp.windows(dsp.preprocess(r["x"], fs), size, overlap)
        for w in W:
            f = dsp.features(w, fs); X.append([f[n] for n in dsp.FEATURE_NAMES])
            sequences.append(w.astype(np.float32, copy=False))
        y += [r["label"]] * len(W); S += [split[r["id"]]] * len(W)
        log(f"Recording {recording_number}: {len(W)} windows -> {CLASSES[r['label']]} ({split[r['id']]})")
    X, y, S = np.array(X), np.array(y), np.array(S)
    tr, va, te = S == "train", S == "val", S == "test"
    models = {"SVM": SVC(probability=True, random_state=0), "KNN": KNeighborsClassifier(5),
              "Decision Tree": DecisionTreeClassifier(random_state=0),
              "Random Forest": RandomForestClassifier(300, random_state=0, n_jobs=-1),
              "Naive Bayes": GaussianNB(), "ANN (MLP)": MLPClassifier((64, 32), max_iter=500, random_state=0)}
    unavailable = []
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=200, eval_metric="mlogloss")
    except ImportError: unavailable.append("XGBoost")
    out = {}
    for name, m in models.items():
        log(f"Training {name}..."); pipe = make_pipeline(StandardScaler(), m)
        t = time.time(); pipe.fit(X[tr], y[tr]); tt = time.time() - t
        t = time.time(); proba = pipe.predict_proba(X[te]); it = (time.time() - t) / te.sum()
        res = evaluate(y[te], proba.argmax(1), proba)
        val_proba = pipe.predict_proba(X[va])
        val_metrics = evaluate(y[va], val_proba.argmax(1), val_proba)
        res.update(train_time=tt, infer_time_per_window=it,
                   val_accuracy=val_metrics["accuracy"], val_f1=val_metrics["f1"])
        out[name] = res; joblib.dump(pipe, model_dir / f"{name.replace(' ', '_')}.joblib")
    try:
        from deep_models import train_and_evaluate
    except ImportError as e:
        unavailable.extend(DEEP_MODELS)
        log(f"Deep-learning models unavailable: {e}")
    else:
        sequence_windows = np.stack(sequences)
        for i, name in enumerate(DEEP_MODELS):
            log(f"Training {name}...")
            trained = train_and_evaluate(
                name, sequence_windows, y, tr, va, te, CLASSES, model_dir,
                seed=42 + i, log=log,
            )
            res = evaluate(y[te], trained["probabilities"].argmax(1), trained["probabilities"])
            res.update(train_time=trained["train_time"],
                       infer_time_per_window=trained["infer_time_per_window"],
                       val_accuracy=trained["val_accuracy"], val_f1=trained["val_f1"])
            out[name] = res
    best = max(out, key=lambda n: (out[n]["val_f1"], out[n]["val_accuracy"]))
    imp = models["Random Forest"].feature_importances_ if hasattr(models["Random Forest"], "feature_importances_") else None
    F, _ = f_classif(X[tr], y[tr]); rng = np.random.default_rng(0)
    cstats = {n: {CLASSES[c]: X[y == c][rng.choice((y == c).sum(), min(150, (y == c).sum()), replace=False), i].tolist()
                  for c in range(4)} for i, n in enumerate(dsp.FEATURE_NAMES)}
    results = {"trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "classes": CLASSES, "feature_names": dsp.FEATURE_NAMES,
        "evaluation_note": "Metrics are measured on held-out reference recordings. Performance on other sensors, machines, sampling rates, and operating conditions may differ; validate with representative labelled data.",
        "dataset": {"recordings": len(recs), "windows": int(len(y)), "fs": fs, "window": size, "overlap": overlap,
            "windows_per_class": {CLASSES[c]: int((y == c).sum()) for c in range(4)},
            "recordings_per_class": {CLASSES[c]: sum(r["label"] == c for r in recs) for c in range(4)}},
        "split": {"method": "per-class split by recording (no window of one recording is in two sets)",
            "windows": {k: int(v.sum()) for k, v in zip(["train", "val", "test"], [tr, va, te])},
            "test_recording_count": sum(s == "test" for s in split.values())},
        "models": out, "unavailable": unavailable, "best_model": best,
        "rf_importance": dict(zip(dsp.FEATURE_NAMES, map(float, make_pipeline(StandardScaler(), RandomForestClassifier(300, random_state=0, n_jobs=-1)).fit(X[tr], y[tr])[-1].feature_importances_))),
        "anova_f": dict(zip(dsp.FEATURE_NAMES, map(float, np.nan_to_num(F)))), "feature_class_stats": cstats}
    results_path.write_text(json.dumps(results))
    log(f"Done. Best model by validation macro-F1: {best}"); return results

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--data", default=str(ROOT / "data" / "cwru"))
    a.add_argument("--fs", type=float, default=12000); a = a.parse_args(); run(a.data, a.fs)
