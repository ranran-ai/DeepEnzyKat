import argparse
import ssl
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


ssl._create_default_https_context = ssl._create_unverified_context


def project_root():
    return Path(__file__).resolve().parents[1]


def encode_proteins(data_path, output_path=None, device_name=None, batch_size=1):
    data_path = Path(data_path)
    output_path = Path(output_path) if output_path else project_root() / "data" / "test_x2.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(data_path)
    sequences = data.iloc[:, 0].tolist()
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    data_for_esm = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
    all_embeddings = []

    for start_idx in tqdm(range(0, len(data_for_esm), batch_size), desc="Processing batches", unit="batch"):
        batch_chunk = data_for_esm[start_idx : start_idx + batch_size]
        _, _, batch_tokens = batch_converter(batch_chunk)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_reps = results["representations"][33]

        for i_seq, (_, seq_str) in enumerate(batch_chunk):
            real_len = len(seq_str)
            all_embeddings.append(token_reps[i_seq, 1 : real_len + 1, :].cpu())

    np.save(output_path, np.array([tensor.numpy() for tensor in all_embeddings], dtype=object), allow_pickle=True)
    print(f"Protein embeddings saved to {output_path}")


def parse_args():
    root = project_root()
    parser = argparse.ArgumentParser(description="Protein sequence encoding")
    parser.add_argument("--input", type=str, default=str(root / "data" / "test.xlsx"))
    parser.add_argument("--output", type=str, default=str(root / "data" / "test_x2.npy"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    encode_proteins(args.input, args.output, args.device, args.batch_size)
