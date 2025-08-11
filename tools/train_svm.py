# SPDX-License-Identifier: MIT
"""
Train a calibrated SVM on deep embeddings (or HOC features).

Input:  an .npz produced by tools/extract_embeddings.py with
        embeddings (N,D), labels (N,), snr (N,)

Output: joblib pipeline: StandardScaler -> SVC(probability=True) -> (optional) CalibratedClassifierCV
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
import joblib


def load_embeddings(path: str, snr_min: float | None, snr_max: float | None):
    data = np.load(path)
    X = data["embeddings"].astype(np.float32)
    y = data["labels"].astype(np.int64)
    snr = data["snr"].astype(np.float32)

    m = np.ones_like(snr, dtype=bool)
    if snr_min is not None:
        m &= snr >= snr_min
    if snr_max is not None:
        m &= snr <= snr_max
    X, y, snr = X[m], y[m], snr[m]
    return X, y, snr


def train_svm(args):
    X, y, snr = load_embeddings(args.input, args.snr_min, args.snr_max)
    print(f"Loaded {X.shape[0]} samples, dim={X.shape[1]}, classes={len(np.unique(y))}")

    # Split off a calibration/validation set (stratified)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.val_frac, random_state=42, stratify=y)

    base = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svc", SVC(
            kernel=args.kernel,
            C=args.C,
            gamma=("scale" if args.gamma is None else args.gamma),
            probability=True,   # enable prob outputs
            class_weight="balanced",
            random_state=42
        ))
    ])

    if args.calibrate:
        # Cross-validated calibration (sigmoid)
        cv = StratifiedKFold(n_splits=args.calibrate_folds, shuffle=True, random_state=42)
        clf = CalibratedClassifierCV(base, cv=cv, method="sigmoid")
    else:
        clf = base

    clf.fit(Xtr, ytr)

    # quick report
    p = clf.predict_proba(Xte)
    acc = accuracy_score(yte, np.argmax(p, axis=1))
    ll  = log_loss(yte, p)
    print(f"val acc={acc*100:.2f}% · logloss={ll:.4f} · kernel={args.kernel} C={args.C} gamma={args.gamma or 'scale'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out)
    print(f"saved SVM pipeline: {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="embeddings .npz from extract_embeddings.py")
    p.add_argument("--snr-min", type=float, default=None, help="optional lower SNR bound for training")
    p.add_argument("--snr-max", type=float, default=None, help="optional upper SNR bound for training")
    p.add_argument("--kernel", choices=["linear", "rbf"], default="rbf")
    p.add_argument("--C", type=float, default=4.0)
    p.add_argument("--gamma", type=float, default=None, help="rbf gamma; default sklearn 'scale'")
    p.add_argument("--calibrate", action="store_true", help="use sigmoid calibration (better fusion)")
    p.add_argument("--calibrate-folds", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--out", default="artifacts/svm_on_embed.joblib")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_svm(args)
