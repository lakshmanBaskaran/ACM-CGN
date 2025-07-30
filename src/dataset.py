# SPDX-License-Identifier: MIT
import os
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
from dsp_utils import estimate_snr
from tqdm import tqdm

# Load config once to get necessary parameters
with open("configs/config.yaml", "r", encoding="utf-8") as _f:
    CFG = yaml.safe_load(_f)

_FEAT_CFG = CFG.get("feature_cache", {})
_USE_FEATS = bool(_FEAT_CFG.get("seg_feats", True))
_USE_PCA = bool(_FEAT_CFG.get("use_pca", False))
_SEG_FEAT_DIM = int(CFG["model"]["seg_feat_dim"])
# ACCURACY FIX: Read the new dataset filtering section from the config
_DATASET_CFG = CFG.get("dataset", {})


class _ShardView:
    """A view into a single data shard that memory-maps all feature files."""

    def __init__(self, base: str, sid: int):
        self.base, self.sid = base, sid
        self._open()

    def _open(self):
        sid = self.sid

        def load_memmap(path):
            return np.load(path, mmap_mode="r") if os.path.exists(path) else None

        self.tm = load_memmap(f"{self.base}/X_tm_{sid}.npy")
        self.sp = load_memmap(f"{self.base}/X_spec_{sid}.npy")
        self.eng = load_memmap(f"{self.base}/X_eng_{sid}.npy")
        self.scat = load_memmap(f"{self.base}/X_scat_{sid}.npy")
        self.cw = load_memmap(f"{self.base}/X_cwt_{sid}.npy")
        self.dw = load_memmap(f"{self.base}/X_dwt_{sid}.npy")
        self.y = load_memmap(f"{self.base}/y_{sid}.npy")
        self.snr = load_memmap(f"{self.base}/X_snr_{sid}.npy")  # Keep loading for getitem
        self.fe = None

        if _USE_FEATS:
            pca_p = f"{self.base}/X_segfeat_pca_{sid}.npy"
            raw_p = f"{self.base}/X_segfeat_{sid}.npy"
            path_to_load = pca_p if _USE_PCA and os.path.exists(pca_p) else (raw_p if os.path.exists(raw_p) else None)
            if path_to_load:
                arr = load_memmap(path_to_load)
                if arr is not None and arr.shape[-1] == _SEG_FEAT_DIM:
                    self.fe = arr

        self.n = len(self.y) if self.y is not None else 0


class RadioMLDataset(Dataset):
    def __init__(self, proc_dir, split, augment=False):
        self.base = os.path.abspath(os.path.join(proc_dir, split))
        self.augment = augment
        self.view_cache = {}

        # ACCURACY FIX: Get filtering parameters from config
        self.min_snr_db = _DATASET_CFG.get("min_snr_db", -99)
        self.classes_to_exclude = set(_DATASET_CFG.get("classes_to_exclude", []))

        # This list will store tuples of (shard_id, local_index, snr_value) for valid samples
        self.valid_indices = []
        self.label_map = {}

        if os.path.exists(self.base):
            shard_ids = sorted({
                int(fn.split("_")[-1].split(".")[0])
                for fn in os.listdir(self.base) if fn.startswith("y_") and fn.endswith(".npy")
            })

            print(
                f"INFO: Filtering '{split}' dataset for SNR >= {self.min_snr_db} dB and excluding classes {list(self.classes_to_exclude)}...")
            pbar = tqdm(shard_ids, desc=f"  [Scanning {split} Shards]")
            for sid in pbar:
                # Temporarily open views to get data for filtering
                temp_view = _ShardView(self.base, sid)
                if temp_view.y is not None and temp_view.tm is not None:
                    labels = temp_view.y
                    tm_data = temp_view.tm
                    # Calculate SNR on the fly for every sample in the shard
                    for i in range(len(labels)):
                        if labels[i] not in self.classes_to_exclude:
                            snr_val = estimate_snr(tm_data[i])
                            if snr_val >= self.min_snr_db:
                                self.valid_indices.append((sid, i, snr_val))

            # Create a mapping from old labels to new, contiguous labels (0, 1, 2, ...)
            original_num_classes = 24
            all_labels = sorted(list(set(range(original_num_classes)) - self.classes_to_exclude))
            self.label_map = {old_label: new_label for new_label, old_label in enumerate(all_labels)}

        self.N = len(self.valid_indices)
        print(f"INFO: Found {self.N} samples matching the criteria.")

    def __len__(self):
        return self.N

    def _get_shard_view(self, shard_id):
        if shard_id not in self.view_cache:
            self.view_cache[shard_id] = _ShardView(self.base, shard_id)
        return self.view_cache[shard_id]

    def __getitem__(self, idx):
        shard_id, local_idx, snr_val = self.valid_indices[idx]
        v = self._get_shard_view(shard_id)

        original_label = int(v.y[local_idx])
        new_label = self.label_map[original_label]

        sample = {
            "tm": torch.from_numpy(v.tm[local_idx].copy()),
            "spec": torch.from_numpy(v.sp[local_idx].copy()),
            "eng_feats": torch.from_numpy(v.eng[local_idx].copy()),
            "y": torch.tensor(new_label, dtype=torch.int64),
            "snr": torch.tensor(snr_val, dtype=torch.float32)  # Use the on-the-fly calculated SNR
        }

        for key, attr in [("scat", v.scat), ("cwt", v.cw), ("dwt", v.dw), ("seg_feats", v.fe)]:
            sample[key] = torch.from_numpy(attr[local_idx].copy()) if attr is not None else torch.empty(0)

        return sample
