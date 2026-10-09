"""Small CPU-friendly sequence models for vibration-window classification."""

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ARCHITECTURES = ("1D CNN", "LSTM", "CNN-LSTM")
EPOCHS = 16
PATIENCE = 4
BATCH_SIZE = 64


class VibrationClassifier(nn.Module):
    def __init__(self, architecture, classes):
        super().__init__()
        if architecture == "1D CNN":
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=9, stride=4, padding=4),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Linear(64, classes)
        elif architecture == "LSTM":
            self.encoder = nn.LSTM(1, 48, batch_first=True, bidirectional=True)
            self.classifier = nn.Linear(96, classes)
        elif architecture == "CNN-LSTM":
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 24, kernel_size=11, stride=2, padding=5),
                nn.BatchNorm1d(24),
                nn.GELU(),
                nn.MaxPool1d(4),
                nn.Conv1d(24, 40, kernel_size=7, padding=3),
                nn.GELU(),
            )
            self.sequence = nn.LSTM(40, 48, batch_first=True, bidirectional=True)
            self.classifier = nn.Linear(96, classes)
        else:
            raise ValueError(f"Unknown sequence model: {architecture}")
        self.architecture = architecture

    def forward(self, x):
        if self.architecture == "1D CNN":
            return self.classifier(self.encoder(x).squeeze(-1))
        if self.architecture == "LSTM":
            sequence = x[:, :, ::8].transpose(1, 2)
            _, (hidden, _) = self.encoder(sequence)
            return self.classifier(torch.cat((hidden[-2], hidden[-1]), dim=1))
        encoded = self.encoder(x).transpose(1, 2)
        _, (hidden, _) = self.sequence(encoded)
        return self.classifier(torch.cat((hidden[-2], hidden[-1]), dim=1))


def checkpoint_path(model_dir, architecture):
    return Path(model_dir) / f"{architecture.replace(' ', '_')}.pt"


def _tensor(windows, mean, std):
    x = np.asarray(windows, dtype=np.float32)
    return torch.from_numpy((x - mean) / std).unsqueeze(1)


def predict_proba(model_path, windows, batch_size=BATCH_SIZE):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = VibrationClassifier(checkpoint["architecture"], len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = _tensor(windows, checkpoint["mean"], checkpoint["std"])
    probabilities = []
    with torch.inference_mode():
        for batch in x.split(batch_size):
            probabilities.append(torch.softmax(model(batch), dim=1).numpy())
    return np.concatenate(probabilities, axis=0)


def train_and_evaluate(architecture, windows, labels, train_mask, val_mask, test_mask,
                       classes, model_dir, seed=42, log=print):
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    x = np.asarray(windows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    mean = float(x[train_mask].mean())
    std = float(x[train_mask].std())
    if not np.isfinite(std) or std < 1e-8:
        raise ValueError("Training windows have no usable signal variation.")

    train_x = _tensor(x[train_mask], mean, std)
    train_y = torch.from_numpy(y[train_mask])
    val_x = _tensor(x[val_mask], mean, std)
    val_y = y[val_mask]
    test_x = _tensor(x[test_mask], mean, std)

    model = VibrationClassifier(architecture, len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)

    best_f1, best_accuracy, best_state, stale = -1.0, 0.0, None, 0
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            val_pred = torch.cat([
                model(batch).argmax(1) for batch in val_x.split(BATCH_SIZE)
            ]).numpy()
        val_f1 = float(f1_score(val_y, val_pred, labels=range(len(classes)),
                                average="macro", zero_division=0))
        val_accuracy = float(accuracy_score(val_y, val_pred))
        log(f"{architecture}: epoch {epoch + 1}/{EPOCHS}, validation macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_accuracy = val_f1, val_accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError(f"{architecture} did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    path = checkpoint_path(model_dir, architecture)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"architecture": architecture, "state_dict": best_state, "classes": classes,
                "mean": mean, "std": std, "window_size": int(x.shape[1])}, temporary)
    temporary.replace(path)

    model.eval()
    started_inference = time.perf_counter()
    with torch.inference_mode():
        probabilities = torch.cat([
            torch.softmax(model(batch), dim=1) for batch in test_x.split(BATCH_SIZE)
        ]).numpy()
    inference_time = (time.perf_counter() - started_inference) / int(test_mask.sum())

    return {
        "probabilities": probabilities,
        "train_time": time.perf_counter() - started,
        "infer_time_per_window": inference_time,
        "val_accuracy": best_accuracy,
        "val_f1": best_f1,
    }
