"""
Simulazione segnali radar:
- waveform: chirp o sine
- target con Doppler
- rumore AWGN
- salva dataset in .npz + labels
"""
import os
import yaml
import argparse
import numpy as np
from scipy.signal import chirp
from pathlib import Path

def set_seed(seed):
    np.random.seed(seed)

def generate_time_vector(fs, duration):
    return np.arange(0, duration, 1.0/fs)

def base_waveform(t, cfg):
    wf = cfg["simulation"]["waveform"]
    amp = float(cfg["simulation"].get("amplitude", 1.0))
    if wf == "chirp":
        f0 = float(cfg["simulation"]["f0"])
        f1 = float(cfg["simulation"]["f1"])
        return amp * chirp(t, f0=f0, t1=t[-1], f1=f1, method='linear')
    elif wf == "sine":
        f0 = float(cfg["simulation"]["f0"])
        return amp * np.sin(2 * np.pi * f0 * t)
    else:
        raise ValueError(f"Unknown waveform: {wf}")

def apply_target(signal, t, doppler_hz, amp=1.0):
    # shift di frequenza semplificato
    return signal + amp * signal * np.cos(2 * np.pi * doppler_hz * t)

def add_awgn_to_snr(sig, snr_db):
    power_signal = np.mean(sig**2)
    snr_linear = 10 ** (snr_db / 10)
    power_noise = power_signal / snr_linear
    noise = np.sqrt(power_noise) * np.random.randn(*sig.shape)
    return sig + noise

def generate_example(cfg, t, base_s):
    n_targets = np.random.randint(cfg["dataset"]["min_targets"], cfg["dataset"]["max_targets"] + 1)
    doppler_min, doppler_max = cfg["targets"]["doppler_range"]
    amp_min, amp_max = cfg["targets"]["amplitude_range"]
    signal = np.copy(base_s)
    metadata_targets = []
    for _ in range(n_targets):
        dop = float(np.random.uniform(doppler_min, doppler_max))
        amp = float(np.random.uniform(amp_min, amp_max))
        signal = apply_target(signal, t, dop, amp)
        metadata_targets.append({"doppler_hz": dop, "amplitude": amp})

    # Aggiungi rumore di fondo casuale extra
    extra_noise = 0.02 * np.random.randn(*signal.shape)
    signal += extra_noise


    return signal, metadata_targets

def generate_dataset(cfg):
    fs = float(cfg["simulation"]["fs"])
    duration = float(cfg["simulation"]["duration"])
    t = generate_time_vector(fs, duration)
    base_s = base_waveform(t, cfg)
    n_signals = int(cfg["dataset"]["n_signals"])
    snr_list = cfg["noise"]["snr_db_list"]
    out_signals, out_labels, out_snrs = [], [], []
    for _ in range(n_signals):
        sig_clean, metadata = generate_example(cfg, t, base_s)
        snr_db = float(np.random.choice(snr_list))
        sig_noisy = add_awgn_to_snr(sig_clean, snr_db)
        out_signals.append(sig_noisy.astype(np.float32))
        out_labels.append(metadata)
        out_snrs.append(snr_db)
    X = np.stack(out_signals, axis=0)
    return {"X": X, "t": t, "labels": out_labels, "snrs_db": np.array(out_snrs), "config": cfg}

def save_dataset(data_dict, out_dir, filename):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fullpath = os.path.join(out_dir, filename)
    np.savez_compressed(fullpath, X=data_dict["X"], t=data_dict["t"], snrs_db=data_dict["snrs_db"])
    np.save(fullpath + ".labels.npy", np.array(data_dict["labels"], dtype=object), allow_pickle=True)
    cfg_path = fullpath + ".config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(data_dict["config"], f)
    print(f"Saved dataset: {fullpath} (+ labels, config)")

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def cli():
    parser = argparse.ArgumentParser(description="Generate synthetic radar signals")
    parser.add_argument("--config", "-c", default="config/config.yaml", help="YAML config path")
    parser.add_argument("--out", "-o", default=None, help="Output filename override")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.out:
        cfg["output"]["filename"] = args.out
    set_seed(int(cfg.get("seed", 0)))
    data = generate_dataset(cfg)
    save_dataset(data, cfg["output"]["out_dir"], cfg["output"]["filename"])

if __name__ == "__main__":
    cli()