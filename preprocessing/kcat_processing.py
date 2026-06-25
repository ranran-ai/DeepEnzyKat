import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def project_root():
    return Path(__file__).resolve().parents[1]


def process_kcat(data_path, output_path=None, column_index=3):
    data_path = Path(data_path)
    output_path = Path(output_path) if output_path else project_root() / "data" / "test_label.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(data_path)
    kcat_values = data.iloc[:, column_index].to_numpy(dtype=np.float32)
    labels = kcat_values.reshape(-1, 1)
    np.save(output_path, labels)
    print(f"Labels saved to {output_path}")


def parse_args():
    root = project_root()
    parser = argparse.ArgumentParser(description="Label preprocessing")
    parser.add_argument("--input", type=str, default=str(root / "data" / "test.xlsx"))
    parser.add_argument("--output", type=str, default=str(root / "data" / "test_label.npy"))
    parser.add_argument("--column_index", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_kcat(args.input, args.output, args.column_index)
