"""
FFT-based signal analysis utilities.

Provides basic frequency-domain analysis for
synthetic radar signals.
"""

import numpy as np

def compute_fft(signal, fs):
    """
    Compute single-sided FFT magnitude spectrum.

    Parameters
    ----------
    signal : np.ndarray
        Time-domain signal (1D)
    fs : float
        Sampling frequency (Hz)

    Returns
    -------
    freqs : np.ndarray
        Frequency axis (Hz)
    magnitude : np.ndarray
        Magnitude spectrum
    """
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_vals = fft_vals[:n // 2]
    magnitude = np.abs(fft_vals) / n

    freqs = np.fft.fftfreq(n, d=1/fs)
    freqs = freqs[:n // 2]

    return freqs, magnitude

def dominant_frequency(freqs, magnitude):
    """
    Return the dominant frequency component.

    Parameters
    ----------
    freqs : np.ndarray
    magnitude : np.ndarray

    Returns
    -------
    float
        Frequency (Hz) with maximum magnitude
    """
    idx = np.argmax(magnitude)
    return freqs[idx]

def signal_energy(signal):
    """
    Compute signal energy in time domain.
    """
    return np.sum(signal ** 2)

def spectral_energy(magnitude):
    """
    Compute spectral energy from FFT magnitude.
    """
    return np.sum(magnitude ** 2)
