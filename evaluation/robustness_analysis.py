"""
Robustness analysis for radar signal detection.

Evaluates baseline threshold detector vs linear SVM
performance as a function of SNR.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from features.extract_features import extract_feature_matrix
from models.classifier import (
    threshold_detector,
    train_svm,
    predict_svm,
    evaluate_classifier
)

def load_dataset(path):
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    snrs = data["snrs_db"]
    labels = np.load(path + ".labels.npy", allow_pickle=True)
    return X, labels, snrs

def build_binary_labels(metadata_labels):
    """
    1 if at least one target is present, else 0
    """
    y = np.array([1 if len(m) > 0 else 0 for m in metadata_labels])
    return y

def evaluate_vs_snr(dataset_path, fs):
    X_signals, metadata, snrs = load_dataset(dataset_path)
    y = build_binary_labels(metadata)

    # Feature extraction
    X_feat = extract_feature_matrix(X_signals, fs)

    results = {}

    for snr in np.unique(snrs):
        idx = snrs == snr
        X_snr = X_feat[idx]
        y_snr = y[idx]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_snr, y_snr, test_size=0.3, random_state=0
        )

        # ----- Baseline -----
        energy_train = X_tr[:, 1]
        threshold = np.percentile(energy_train, 75)
        y_pred_base = threshold_detector(X_te[:, 1], threshold)
        acc_base = evaluate_classifier(y_te, y_pred_base)

        # ----- SVM -----
        svm = train_svm(X_tr, y_tr)
        y_pred_svm = predict_svm(svm, X_te)
        acc_svm = evaluate_classifier(y_te, y_pred_svm)

        results[snr] = {
            "baseline_accuracy": acc_base,
            "svm_accuracy": acc_svm
        }

    return results

if __name__ == "__main__":
    dataset_path = "data/simulated_dataset.npz"
    fs = 8000
    results = evaluate_vs_snr(dataset_path, fs)

    for snr, metrics in results.items():
        print(f"SNR = {snr} dB | "
              f"Baseline: {metrics['baseline_accuracy']:.2f} | "
              f"SVM: {metrics['svm_accuracy']:.2f}")
