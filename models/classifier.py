"""
Classification models for radar signal detection.

Includes:
- Simple threshold-based baseline detector
- Linear SVM classifier
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def threshold_detector(energy, threshold):
    """
    Simple energy-based detector.

    Parameters
    ----------
    energy : np.ndarray
        Signal energy values
    threshold : float
        Decision threshold

    Returns
    -------
    np.ndarray
        Binary predictions (0: no target, 1: target)
    """
    return (energy > threshold).astype(int)

def train_svm(X_train, y_train, C=1.0):
    """
    Train a linear SVM classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Feature matrix (N, n_features)
    y_train : np.ndarray
        Binary labels
    C : float
        Regularization parameter

    Returns
    -------
    sklearn.svm.SVC
        Trained SVM model
    """
    model = SVC(kernel="linear", C=C)
    model.fit(X_train, y_train)
    return model

def predict_svm(model, X):
    """
    Predict labels using trained SVM.

    Parameters
    ----------
    model : sklearn.svm.SVC
    X : np.ndarray

    Returns
    -------
    np.ndarray
        Predicted labels
    """
    return model.predict(X)

def evaluate_classifier(y_true, y_pred):
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray

    Returns
    -------
    float
        Accuracy score
    """
    return accuracy_score(y_true, y_pred)
