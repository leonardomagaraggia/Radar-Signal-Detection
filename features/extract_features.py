"""
Feature extraction for radar signals.

Uses FFT-based analysis to extract simple,
interpretable features suitable for classical
machine learning models.
"""

import numpy as np
from signal_processing.fft_analysis import (
    compute_fft,
    dominant_frequency,
    signal_energy
)

def spectral_bandwidth(freqs, magnitude, threshold_ratio=0.2):
    """
    Estimate spectral bandwidth.

    Bandwidth is defined as the frequency span
    where magnitude exceeds a fraction of the
    maximum magnitude.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency axis (Hz)
    magnitude : np.ndarray
        FFT magnitude
    threshold_ratio : float
        Fraction of max magnitude used as threshold

    Returns
    -------
    float
        Bandwidth in Hz
    """
    threshold = threshold_ratio * np.max(magnitude)
    active = freqs[magnitude >= threshold]

    if len(active) < 2:
        return 0.0

    return active[-1] - active[0]

def extract_features(signal, fs):
    """
    Extract feature vector from a single signal.

    Parameters
    ----------
    signal : np.ndarray
        Time-domain signal
    fs : float
        Sampling frequency (Hz)

    Returns
    -------
    dict
        Dictionary of extracted features
    """
    freqs, mag = compute_fft(signal, fs)

    features = {
        "dominant_frequency": dominant_frequency(freqs, mag),
        "energy": signal_energy(signal),
        "bandwidth": spectral_bandwidth(freqs, mag)
    }

    return features

def extract_feature_matrix(signals, fs):
    """
    Extract features from multiple signals.

    Parameters
    ----------
    signals : np.ndarray
        Array of signals (N, samples)
    fs : float

    Returns
    -------
    np.ndarray
        Feature matrix (N, n_features)
    """
    feature_list = []

    for sig in signals:
        feats = extract_features(sig, fs)
        feature_list.append(list(feats.values()))

    return np.array(feature_list)
