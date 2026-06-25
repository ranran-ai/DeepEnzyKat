import argparse
import gc
import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import model_bimodal_regression_moe
from inference_data import DDPSafeBimodalDataset, my_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble inference")
    parser.add_argument("--model_paths", type=str, nargs="*", default=None, help="Model checkpoint paths")
    parser.add_argument("--model_dir", type=str, default="weights", help="Directory containing model checkpoints")
    parser.add_argument("--model_glob", type=str, default="*.pt", help="Checkpoint glob pattern")
    parser.add_argument("--model_sort", type=str, default="name", choices=["name", "mtime"], help="Model sort order")
    parser.add_argument("--test_data", type=str, default="data/test.xlsx", help="Input Excel file")
    parser.add_argument("--test_x2", type=str, default="data/test_x2.npy", help="Protein embedding file")
    parser.add_argument("--test_label", type=str, default="data/test_label.npy", help="Label file")
    parser.add_argument("--output_csv", type=str, default="ensemble_results.csv", help="Output CSV file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device name")
    parser.add_argument("--enable_cache", action="store_true", help="Enable graph cache")
    parser.add_argument("--augment_smiles", action="store_true", help="Enable SMILES augmentation")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--save_individual_preds", action="store_true", help="Save individual prediction columns")
    parser.add_argument("--no_save_individual_preds", dest="save_individual_preds", action="store_false")
    parser.set_defaults(save_individual_preds=True)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--hid_dim", type=int, default=512)
    parser.add_argument("--x1_dim", type=int, default=512)
    parser.add_argument("--x2_dim", type=int, default=1280)
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--node_output_dim", type=int, default=512)
    parser.add_argument("--edge_output_dim", type=int, default=512)
    parser.add_argument("--node_feat_dim", type=int, default=109)
    parser.add_argument("--edge_feat_dim", type=int, default=13)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--activation_type", type=str, default="leakyrelu")
    parser.add_argument("--dim_q", type=int, default=512)
    parser.add_argument("--dim_kv", type=int, default=1280)
    parser.add_argument("--dim_out", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--attn_dropout", type=float, default=0.2)
    parser.add_argument("--ffn_mult", type=int, default=2)
    return parser.parse_args()


def create_model(args):
    return model_bimodal_regression_moe.moe(
        num_experts=args.num_experts,
        drop_r=args.drop_rate,
        x1_dim=args.x1_dim,
        x2_dim=args.x2_dim,
        hid_dim=args.hid_dim,
        node_output_dim=args.node_output_dim,
        edge_output_dim=args.edge_output_dim,
        node_feat_dim=args.node_feat_dim,
        edge_feat_dim=args.edge_feat_dim,
        num_rounds=args.num_rounds,
        dropout_rate=args.dropout_rate,
        activation_type=args.activation_type,
        dim_q=args.dim_q,
        dim_kv=args.dim_kv,
        dim_out=args.dim_out,
        num_heads=args.num_heads,
        attn_dropout=args.attn_dropout,
        ffn_mult=args.ffn_mult,
    )


def load_checkpoint(ckpt_path, device):
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = state.get(key)
            if isinstance(value, dict):
                state = value
                break

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {ckpt_path}")

    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    return state


def load_state_dict_compat(model, ckpt_path, device):
    model.load_state_dict(load_checkpoint(ckpt_path, device), strict=True)


def build_model_paths(args):
    paths = []

    if args.model_dir:
        if not os.path.isdir(args.model_dir):
            raise NotADirectoryError(f"--model_dir is not a directory: {args.model_dir}")
        found = glob.glob(os.path.join(args.model_dir, args.model_glob))
        if args.model_sort == "mtime":
            found = sorted(found, key=lambda path: os.path.getmtime(path))
        else:
            found = sorted(found)
        paths.extend(found)

    if args.model_paths:
        paths.extend(args.model_paths)

    if len(paths) == 0:
        paths = ["best_model_ddp-42.pt", "best_model_ddp-0.pt", "best_model_ddp-1000.pt"]

    uniq = []
    seen = set()
    for path in paths:
        if path not in seen:
            uniq.append(path)
            seen.add(path)

    for path in uniq:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint does not exist: {path}")

    return uniq


def to_numpy_1d(x):
    if isinstance(x, np.ndarray):
        arr = x
    elif torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.array(x)
    return arr.reshape(-1)


def compute_metrics(y_true, y_pred):
    y_true = to_numpy_1d(y_true)
    y_pred = to_numpy_1d(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson_r = 0.0
    else:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])

    return float(mse), rmse, float(mae), float(r2), pearson_r


def pre_scan_kept_indices(dataset):
    kept_idx = []
    kept_labels = []

    for i in tqdm(range(len(dataset)), desc="Pre-scanning data"):
        graph, x2, label = dataset[i]
        if graph is None:
            continue
        kept_idx.append(i)
        kept_labels.append(float(to_numpy_1d(label)[0]))

    return kept_idx, np.array(kept_labels, dtype=np.float64)


def welford_update(mean, M2, k, x):
    k1 = k + 1
    delta = x - mean
    mean = mean + delta / k1
    delta2 = x - mean
    M2 = M2 + delta * delta2
    return mean, M2, k1


def predict_one_model(model_path, dataset_subset, device, args):
    print(f"\nLoading model: {os.path.basename(model_path)}")
    model = create_model(args)
    load_state_dict_compat(model, model_path, device)
    model.to(device)
    model.eval()

    preds_list = []

    if args.batch_size <= 1:
        with torch.inference_mode():
            for j in tqdm(range(len(dataset_subset)), desc="Inference"):
                graph, x2, label = dataset_subset[j]
                batched_graph, x2s, mask, labels = my_collate_fn([(graph, x2, label)])

                if labels is None or (hasattr(labels, "size") and labels.size(0) == 0):
                    continue

                batched_graph = batched_graph.to(device)
                x2s = x2s.to(device)
                if mask is not None and hasattr(mask, "to"):
                    mask = mask.to(device)

                out = model(batched_graph, x2s, mask)
                preds_list.append(float(to_numpy_1d(out)[0]))
    else:
        loader = DataLoader(
            dataset_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=my_collate_fn,
            drop_last=False,
        )
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Inference(bs={args.batch_size})"):
                batched_graph, x2s, mask, labels = batch
                if labels is None or (hasattr(labels, "size") and labels.size(0) == 0):
                    continue

                batched_graph = batched_graph.to(device)
                x2s = x2s.to(device)
                if mask is not None and hasattr(mask, "to"):
                    mask = mask.to(device)

                out = model(batched_graph, x2s, mask)
                preds_list.extend(to_numpy_1d(out).astype(np.float64).tolist())

    preds = np.array(preds_list, dtype=np.float64)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return preds


def safe_model_colname(model_path):
    name = os.path.basename(model_path)
    name = os.path.splitext(name)[0]
    name = name.replace("best_model_ddp_", "seed")
    return name


def print_metrics_table(model_names, individual_metrics, avg_metrics, ensemble_metrics, model_count):
    print("\nMetrics")
    print(f"{'Model':<35}{'MSE':>12}{'RMSE':>12}{'MAE':>12}{'R2':>10}{'r':>10}")
    print("-" * 91)
    for i, metrics in enumerate(individual_metrics):
        mse, rmse, mae, r2, pearson_r = metrics
        name = f"Model{i + 1} ({model_names[i]})"
        print(f"{name:<35}{mse:>12.5f}{rmse:>12.5f}{mae:>12.5f}{r2:>10.4f}{pearson_r:>10.4f}")
    print("-" * 91)
    print(
        f"{'Single-model average':<35}{avg_metrics[0]:>12.5f}{avg_metrics[1]:>12.5f}"
        f"{avg_metrics[2]:>12.5f}{avg_metrics[3]:>10.4f}{avg_metrics[4]:>10.4f}"
    )
    print(
        f"{f'Ensemble ({model_count} models)':<35}{ensemble_metrics[0]:>12.5f}{ensemble_metrics[1]:>12.5f}"
        f"{ensemble_metrics[2]:>12.5f}{ensemble_metrics[3]:>10.4f}{ensemble_metrics[4]:>10.4f}"
    )


def main(args):
    print("=" * 80)
    print("Ensemble inference")
    print("=" * 80)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_paths = build_model_paths(args)
    print(f"Models: {len(model_paths)}")
    if len(model_paths) <= 12:
        print("Model list:")
        for path in model_paths:
            print(f"  - {path}")
    else:
        print("Model list preview:")
        for path in model_paths[:5]:
            print(f"  - {path}")
        print(f"  ... and {len(model_paths) - 5} more")

    print("\nLoading test data")
    test_df = pd.read_excel(args.test_data)
    if "SMILES" not in test_df.columns:
        raise KeyError("The input Excel file must contain a SMILES column")

    test_smiles = test_df["SMILES"].tolist()
    test_x2 = np.load(args.test_x2, allow_pickle=True)
    test_label = np.load(args.test_label)
    print(f"Samples: {len(test_smiles)}")

    test_dataset = DDPSafeBimodalDataset(
        test_smiles,
        test_x2,
        test_label,
        enable_cache=args.enable_cache,
        augment_smiles=args.augment_smiles,
    )

    kept_idx, labels = pre_scan_kept_indices(test_dataset)
    if len(kept_idx) == 0:
        raise RuntimeError("No valid samples were found after pre-scan")

    print(f"Valid samples: {len(kept_idx)} / {len(test_dataset)}")
    dataset_subset = Subset(test_dataset, kept_idx)

    sample_count = len(kept_idx)
    mean = np.zeros(sample_count, dtype=np.float64)
    M2 = np.zeros(sample_count, dtype=np.float64)
    model_count = 0
    individual_metrics = []
    model_names = []
    individual_preds = [] if args.save_individual_preds else None

    print("\nStarting model inference")
    for i, model_path in enumerate(model_paths):
        print(f"\n[{i + 1}/{len(model_paths)}]")
        preds = predict_one_model(model_path, dataset_subset, device, args)

        if len(preds) != sample_count:
            raise RuntimeError(f"Prediction length mismatch: expected {sample_count}, got {len(preds)}")

        mean, M2, model_count = welford_update(mean, M2, model_count, preds)
        mse, rmse, mae, r2, pearson_r = compute_metrics(labels, preds)
        individual_metrics.append((mse, rmse, mae, r2, pearson_r))

        model_name = safe_model_colname(model_path)
        model_names.append(model_name)
        print(f"MSE={mse:.5f}, RMSE={rmse:.5f}, MAE={mae:.5f}, R2={r2:.4f}, r={pearson_r:.4f}")

        if args.save_individual_preds:
            individual_preds.append(preds.copy())

    ensemble_preds = mean
    if model_count > 1:
        pred_std = np.sqrt(M2 / (model_count - 1))
    else:
        pred_std = np.zeros_like(ensemble_preds)

    avg_metrics = np.mean(np.array(individual_metrics, dtype=np.float64), axis=0)
    ensemble_metrics = compute_metrics(labels, ensemble_preds)
    print_metrics_table(model_names, individual_metrics, avg_metrics, ensemble_metrics, model_count)

    print("\nImprovement")
    improvement_mse = avg_metrics[0] - ensemble_metrics[0]
    improvement_rmse = avg_metrics[1] - ensemble_metrics[1]
    improvement_mae = avg_metrics[2] - ensemble_metrics[2]
    improvement_r2 = ensemble_metrics[3] - avg_metrics[3]
    improvement_r = ensemble_metrics[4] - avg_metrics[4]
    print(f"MSE delta:  {improvement_mse:+.5f} ({improvement_mse / avg_metrics[0] * 100:+.2f}%)")
    print(f"RMSE delta: {improvement_rmse:+.5f} ({improvement_rmse / avg_metrics[1] * 100:+.2f}%)")
    print(f"MAE delta:  {improvement_mae:+.5f} ({improvement_mae / avg_metrics[2] * 100:+.2f}%)")
    print(f"R2 delta:   {improvement_r2:+.5f} ({improvement_r2 / avg_metrics[3] * 100:+.2f}%)")
    print(f"r delta:    {improvement_r:+.5f} ({improvement_r / avg_metrics[4] * 100:+.2f}%)")

    print("\nPrediction spread")
    print(f"Mean std: {pred_std.mean():.5f}")
    print(f"Max std:  {pred_std.max():.5f}")
    print(f"Min std:  {pred_std.min():.5f}")

    high_uncertainty_threshold = pred_std.mean() + 2 * pred_std.std()
    high_uncertainty_idx = np.where(pred_std > high_uncertainty_threshold)[0]
    if len(high_uncertainty_idx) > 0:
        preview = high_uncertainty_idx[:10].tolist()
        suffix = "..." if len(high_uncertainty_idx) > 10 else ""
        print(f"High-uncertainty samples: {len(high_uncertainty_idx)}")
        print(f"Threshold: {high_uncertainty_threshold:.5f}")
        print(f"Indices in valid subset: {preview}{suffix}")
    else:
        print("No high-uncertainty samples found")

    print("\nSaving results")
    kept_df = test_df.iloc[kept_idx].copy().reset_index(drop=True)
    kept_df.insert(0, "Orig_RowIdx", kept_idx)
    kept_df["True_Value"] = labels.astype(np.float64)

    if args.save_individual_preds:
        for model_name, preds in zip(model_names, individual_preds):
            kept_df[f"Pred_{model_name}"] = preds.astype(np.float64)

    kept_df["Ensemble_Avg"] = ensemble_preds.astype(np.float64)
    kept_df["Pred_Std"] = pred_std.astype(np.float64)
    kept_df["Ensemble_AbsError"] = np.abs(labels - ensemble_preds)
    kept_df["Ensemble_RelError(%)"] = np.abs(labels - ensemble_preds) / (np.abs(labels) + 1e-8) * 100.0

    kept_df.to_csv(args.output_csv, index=False, float_format="%.6f")
    print(f"Saved predictions to {args.output_csv}")

    summary_df = pd.DataFrame(
        {
            "Model": [f"Model{i + 1}_{model_names[i]}" for i in range(len(individual_metrics))]
            + ["Average", "Ensemble"],
            "MSE": [m[0] for m in individual_metrics] + [avg_metrics[0], ensemble_metrics[0]],
            "RMSE": [m[1] for m in individual_metrics] + [avg_metrics[1], ensemble_metrics[1]],
            "MAE": [m[2] for m in individual_metrics] + [avg_metrics[2], ensemble_metrics[2]],
            "R2": [m[3] for m in individual_metrics] + [avg_metrics[3], ensemble_metrics[3]],
            "Pearson_r": [m[4] for m in individual_metrics] + [avg_metrics[4], ensemble_metrics[4]],
        }
    )

    summary_csv = args.output_csv.replace(".csv", "_summary.csv")
    summary_df.to_csv(summary_csv, index=False, float_format="%.6f")
    print(f"Saved summary to {summary_csv}")
    print("Done")


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as exc:
        print(f"\nInference failed: {exc}")
        import traceback

        traceback.print_exc()
