"""T4 - trainable multi-channel 1D-CNN spectral decoder.

The module now contains the complete training/evaluation path used by the
project: dataset loading, stratified train/validation/test splits, model
checkpointing, confusion matrices and a small inference API.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


TYRE_LABELS = ["passenger", "truck_hgv", "motorcycle", "otr_mining"]
TYRE_TO_ID = {name: i for i, name in enumerate(TYRE_LABELS)}
SURFACE_LABELS = ["tread", "sidewall"]
SURFACE_TO_ID = {name: i for i, name in enumerate(SURFACE_LABELS)}


class SpectraDataset(Dataset):
    def __init__(self, spectra: np.ndarray, tyre_y: np.ndarray, surface_y: np.ndarray, augment: bool = False):
        self.x = torch.as_tensor(spectra, dtype=torch.float32)
        self.y_tyre = torch.as_tensor(tyre_y, dtype=torch.long)
        self.y_surface = torch.as_tensor(surface_y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx].clone()
        if self.augment:
            # Sensor-domain augmentation: channel gain, baseline drift and noise.
            gain = 1.0 + 0.04 * torch.randn(1)
            drift = 0.025 * torch.randn(x.shape[0], 1)
            x = gain * x + drift + 0.05 * torch.randn_like(x)
        return x, self.y_tyre[idx], self.y_surface[idx]


class SpectralCNN(nn.Module):
    def __init__(self, n_bands: int = 4, n_channels: int = 128,
                 n_tyre_classes: int = 4, n_surface_classes: int = 2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(n_bands, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96), nn.GELU(),
            nn.Dropout(0.15),
            nn.AdaptiveAvgPool1d(1),
        )
        self.tyre_head = nn.Linear(96, n_tyre_classes)
        self.surface_head = nn.Linear(96, n_surface_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x).squeeze(-1)
        return self.tyre_head(feat), self.surface_head(feat)


@dataclass
class SplitData:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def load_dataset(dataset_dir: str | os.PathLike) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(dataset_dir)
    bundle = np.load(root / "spectra.npz")
    spectra = bundle["spectra"].astype(np.float32)
    ids = bundle["ids"].astype(str)
    rows = {}
    with open(root / "metadata.csv", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["fragment_id"]] = row
    tyre = np.array([TYRE_TO_ID[rows[i]["tyre_type"]] for i in ids], dtype=np.int64)
    surface = np.array([SURFACE_TO_ID[rows[i]["surface_class"]] for i in ids], dtype=np.int64)
    return spectra, tyre, surface, ids


def stratified_splits(tyre: np.ndarray, seed: int = 42) -> SplitData:
    idx = np.arange(len(tyre))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=seed, stratify=tyre)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed, stratify=tyre[temp_idx])
    return SplitData(train_idx, val_idx, test_idx)


def _loader(spectra, tyre, surface, idx, batch_size, shuffle, augment=False):
    return DataLoader(SpectraDataset(spectra[idx], tyre[idx], surface[idx], augment=augment),
                      batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _epoch(model, loader, criterion_t, criterion_s, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    yt, yp = [], []
    ys, ysp = [], []
    for x, y_t, y_s in loader:
        x, y_t, y_s = x.to(device), y_t.to(device), y_s.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        lt, ls = model(x)
        loss = criterion_t(lt, y_t) + 0.15 * criterion_s(ls, y_s)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += loss.item() * len(x)
        yt.extend(y_t.detach().cpu().numpy()); yp.extend(lt.argmax(1).detach().cpu().numpy())
        ys.extend(y_s.detach().cpu().numpy()); ysp.extend(ls.argmax(1).detach().cpu().numpy())
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "tyre_acc": accuracy_score(yt, yp),
        "surface_acc": accuracy_score(ys, ysp),
        "yt": np.array(yt), "yp": np.array(yp),
    }


def train_model(dataset_dir: str, out_dir: str, epochs: int = 60, batch_size: int = 32,
                lr: float = 1e-3, seed: int = 42) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    spectra, tyre, surface, ids = load_dataset(dataset_dir)
    # Per-channel standardisation fitted only on the training split.
    split = stratified_splits(tyre, seed)
    mean = spectra[split.train].mean(axis=(0, 2), keepdims=True)
    std = spectra[split.train].std(axis=(0, 2), keepdims=True) + 1e-6
    spectra = (spectra - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralCNN(n_bands=spectra.shape[1], n_channels=spectra.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=7, min_lr=1e-5)
    criterion_t = nn.CrossEntropyLoss(label_smoothing=0.03)
    criterion_s = nn.CrossEntropyLoss()
    train_loader = _loader(spectra, tyre, surface, split.train, batch_size, True, augment=True)
    val_loader = _loader(spectra, tyre, surface, split.val, batch_size, False)
    test_loader = _loader(spectra, tyre, surface, split.test, batch_size, False)

    best = -1.0; history = []
    os.makedirs(out_dir, exist_ok=True)
    for epoch in range(1, epochs + 1):
        tr = _epoch(model, train_loader, criterion_t, criterion_s, opt, device, True)
        va = _epoch(model, val_loader, criterion_t, criterion_s, opt, device, False)
        scheduler.step(va["tyre_acc"])
        history.append({"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"],
                        "train_acc": tr["tyre_acc"], "val_acc": va["tyre_acc"]})
        if va["tyre_acc"] > best:
            best = va["tyre_acc"]
            torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                        "labels": TYRE_LABELS, "n_bands": spectra.shape[1],
                        "n_channels": spectra.shape[2]}, Path(out_dir) / "spectral_cnn.pt")

    ckpt = torch.load(Path(out_dir) / "spectral_cnn.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    te = _epoch(model, test_loader, criterion_t, criterion_s, opt, device, False)
    report = classification_report(te["yt"], te["yp"], target_names=TYRE_LABELS, output_dict=True, zero_division=0)
    metrics = {
        "device": str(device), "epochs": epochs, "best_val_tyre_accuracy": best,
        "test_tyre_accuracy": te["tyre_acc"], "test_surface_accuracy": te["surface_acc"],
        "confusion_matrix": confusion_matrix(te["yt"], te["yp"]).tolist(), "classification_report": report,
        "train_size": len(split.train), "val_size": len(split.val), "test_size": len(split.test),
    }
    with open(Path(out_dir) / "training_metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    with open(Path(out_dir) / "training_history.json", "w") as f: json.dump(history, f, indent=2)
    return metrics


class SpectralInference:
    def __init__(self, checkpoint: str, device: str = "auto"):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
        self.device = dev
        self.mean = torch.as_tensor(ckpt["mean"], dtype=torch.float32, device=dev)
        self.std = torch.as_tensor(ckpt["std"], dtype=torch.float32, device=dev)
        self.model = SpectralCNN(ckpt["n_bands"], ckpt["n_channels"], len(ckpt["labels"])).to(dev)
        self.model.load_state_dict(ckpt["state_dict"]); self.model.eval()

    @torch.inference_mode()
    def predict(self, spectrum: np.ndarray) -> tuple[str, float, str, float]:
        x = torch.as_tensor(spectrum, dtype=torch.float32, device=self.device)
        if x.ndim == 2: x = x.unsqueeze(0)
        x = (x - self.mean) / self.std
        tyre_logits, surface_logits = self.model(x)
        tyre_probs = torch.softmax(tyre_logits, dim=1)[0]
        surface_probs = torch.softmax(surface_logits, dim=1)[0]
        tyre_idx = int(torch.argmax(tyre_probs))
        surface_idx = int(torch.argmax(surface_probs))
        return (
            TYRE_LABELS[tyre_idx], float(tyre_probs[tyre_idx]),
            SURFACE_LABELS[surface_idx], float(surface_probs[surface_idx]),
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    print(json.dumps(train_model(args.dataset, args.out, args.epochs, args.batch_size, args.lr), indent=2))
