# DeepEnzyKat
DeepEnzyKat integrated ESM2-derived protein embeddings and DMPNN-based molecular features, employing a cross-attention mechanism to capture their interaction patterns and a mixture-of-experts module to enhance accuracy across diverse enzyme families. Further combined with ensemble learning for robustness


## Model Overview

The inference pipeline uses two input modalities:

- Substrate modality: SMILES strings are converted into molecular graphs with RDKit and DGL.
- Protein modality: protein sequences are encoded into variable-length ESM-2 embeddings.

The model architecture contains:

- A D-MPNN encoder for substrate molecular graphs.
- A cross-attention fusion block for substrate-protein interaction features.
- Attention pooling over graph nodes and protein residues.
- A mixture-of-experts regression module for final kcat prediction.
- An ensemble inference script that averages predictions across multiple checkpoints and reports uncertainty from prediction spread.

## Repository Structure

```text
.
|-- ensemble_inference.py
|-- inference_data.py
|-- model_bimodal_regression_moe.py
|-- modules_cross_fusion.py
|-- substrate_features.py
|-- preprocessing
|   |-- protein_encoding.py
|   `-- kcat_processing.py
|-- data
`-- weights
```

`weights` should contain the released model checkpoint files.

`data` should contain the Excel input file and generated `.npy` files.

## Input Data Format

The default input file is:

```text
data/test.xlsx
```

The Excel file must contain a column named:

```text
SMILES
```

For preprocessing, the protein sequence encoder reads protein sequences from the first column of the Excel file. The label processor reads kcat values from column index `3` by default.

## Model Checkpoints

Place all model weight files in:

```text
weights/
```

By default, the inference script searches:

```text
weights/*.pt
```

You can also pass model paths manually:

```bash
python ensemble_inference.py --model_paths weights/model1.pt weights/model2.pt
```

## Environment

Install the required Python packages in an environment with PyTorch, DGL, RDKit, ESM, pandas, numpy, scikit-learn, and tqdm.

Example:

```bash
pip install numpy pandas scikit-learn tqdm
pip install fair-esm
```

Install PyTorch, DGL, and RDKit according to your CUDA and system configuration.

## Preprocessing

Generate protein embeddings:

```bash
python preprocessing/protein_encoding.py --input data/test.xlsx --output data/test_x2.npy
```

Generate label values:

```bash
python preprocessing/kcat_processing.py --input data/test.xlsx --output data/test_label.npy
```

After preprocessing, the expected files are:

```text
data/test.xlsx
data/test_x2.npy
data/test_label.npy
```

## Inference

Run ensemble inference:

```bash
python ensemble_inference.py
```

Default paths:

```text
Input Excel:      data/test.xlsx
Protein vectors: data/test_x2.npy
Labels:          data/test_label.npy
Weights:         weights/*.pt
Output:          ensemble_results.csv
```

Common options:

```bash
python ensemble_inference.py \
  --test_data data/test.xlsx \
  --test_x2 data/test_x2.npy \
  --test_label data/test_label.npy \
  --model_dir weights \
  --output_csv ensemble_results.csv \
  --device cuda:0
```

If no GPU is available, the script falls back to CPU.

## Output Files

The inference script produces:

```text
ensemble_results.csv
ensemble_results_summary.csv
```

`ensemble_results.csv` contains:

- Original input columns.
- Original row index.
- True label value.
- Individual model predictions, if enabled.
- Ensemble average prediction.
- Prediction standard deviation across models.
- Absolute error.
- Relative error percentage.

`ensemble_results_summary.csv` contains:

- MSE
- RMSE
- MAE
- R2
- Pearson correlation

for each individual model, the single-model average, and the ensemble.

## Notes

- Keep checkpoint files in `weights/`.
- Large checkpoint files may require Git LFS or external release hosting.
- The script is configured for the released 4-expert model by default.
- For best reproducibility, use the same input Excel file and generated `.npy` files used during evaluation.
