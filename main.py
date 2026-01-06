"""
Main entry point for the Radar Signal Detection.

Generates a synthetic radar dataset based on parameters
defined in config/config.yaml.
"""

from data_generation.simulate_signals import (
    generate_dataset,
    save_dataset,
    load_yaml
)

def main():
    config_path = "config/config.yaml"
    cfg = load_yaml(config_path)

    dataset = generate_dataset(cfg)

    out_dir = cfg["output"]["out_dir"]
    filename = cfg["output"]["filename"]
    save_dataset(dataset, out_dir, filename)

if __name__ == "__main__":
    main()
