import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset
import dgl
from substrate_features import smiles_to_graph


class DDPSafeBimodalDataset(Dataset):
    def __init__(self, smiles_list, x2_array, label_array, enable_cache=False, augment_smiles=False):
        self.smiles_list = smiles_list
        self.x2_array = x2_array
        self.label_array = label_array
        self.enable_cache = enable_cache
        self.augment_smiles = augment_smiles
        self.data_cache = {}

    def __len__(self):
        return len(self.smiles_list)

    def _augment_smiles(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, doRandom=True)
        except Exception:
            pass
        return smiles

    def __getitem__(self, idx):
        if self.enable_cache and not self.augment_smiles and idx in self.data_cache:
            return self.data_cache[idx]

        smiles = self.smiles_list[idx]
        if self.augment_smiles:
            smiles = self._augment_smiles(smiles)

        substrate_graph = smiles_to_graph(smiles)
        x2 = torch.tensor(self.x2_array[idx], dtype=torch.float32)
        label = torch.tensor(self.label_array[idx], dtype=torch.float32)
        result = (substrate_graph, x2, label)

        if self.enable_cache and not self.augment_smiles:
            self.data_cache[idx] = result

        return result


def my_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None, torch.empty(0), torch.empty(0, dtype=torch.bool), torch.empty(0)

    substrate_graphs, x2s, labels = zip(*batch)
    batched_substrate_graph = dgl.batch(substrate_graphs)
    padded_np, mask_np = pad_and_mask([x.numpy() for x in x2s])
    padded_x2 = torch.from_numpy(padded_np).float()
    mask_x2 = torch.from_numpy(mask_np)
    labels = torch.stack(labels)
    return batched_substrate_graph, padded_x2, mask_x2, labels


def pad_and_mask(x2, lmax=None, pad_value=0):
    batch_size = len(x2)
    if lmax is None:
        lmax = max(t.shape[0] for t in x2)

    dim = x2[0].shape[1]
    padded_x2 = np.full((batch_size, lmax, dim), pad_value, dtype=x2[0].dtype)
    mask_x2 = np.zeros((batch_size, lmax), dtype=bool)

    for i, seq in enumerate(x2):
        length = min(seq.shape[0], lmax)
        padded_x2[i, :length, :] = seq[:length, :]
        mask_x2[i, :length] = True

    return padded_x2, mask_x2
