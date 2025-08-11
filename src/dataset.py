# SPDX-License-Identifier: MIT
import os
import math
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# Load save_dtype with UTF-8 decoding
with open("configs/config.yaml", "r", encoding="utf-8") as _f:
    _SAVE_DTYPE = yaml.safe_load(_f)["preprocess"]["save_dtype"]

class _ShardView:
    """Memory-maps one shard (handles optional CWT and DWT)."""
    def __init__(self, base, sid):
        self.base, self.sid = base, sid
        self._open()

    def _open(self):
        self.tm = np.load(f"{self.base}/X_tm_{self.sid}.npy",  mmap_mode="r")
        self.sp = np.load(f"{self.base}/X_spec_{self.sid}.npy", mmap_mode="r")
        cwt_path = f"{self.base}/X_cwt_{self.sid}.npy"
        self.has_cwt = _SAVE_DTYPE["cwt"] is not None and os.path.exists(cwt_path)
        self.cw = np.load(cwt_path, mmap_mode="r") if self.has_cwt else None
        dwt_path = f"{self.base}/X_dwt_{self.sid}.npy"
        self.has_dwt = _SAVE_DTYPE.get("dwt") is not None and os.path.exists(dwt_path)
        self.dw = np.load(dwt_path, mmap_mode="r") if self.has_dwt else None
        self.y  = np.load(f"{self.base}/y_{self.sid}.npy",    mmap_mode="r")
        self.n  = len(self.y)

    def __getstate__(self):
        return {"base": self.base, "sid": self.sid}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()

class RadioMLDataset(Dataset):
    """
    Stitches sharded *.npy files and returns samples as:
      { "tm":(T,4), "spec":(F,T,2), "cwt":(Scales,T)|empty,
        "dwt":(Levels,T)|empty, "snr":float, "y":int }.
    """
    def __init__(
        self,
        proc_dir,
        split,
        augment=False,
        seg_cfg=None,
        aug_cfg=None,
        min_snr_db: float = None,
        max_snr_db: float = None
    ):
        self.base = os.path.abspath(os.path.join(proc_dir, split))
        shards = sorted(
            int(f.split("_")[-1].split(".")[0])
            for f in os.listdir(self.base)
            if f.startswith("y_")
        )
        self.views = [_ShardView(self.base, s) for s in shards]
        self.offsets = np.cumsum([0] + [v.n for v in self.views])
        self.N_raw   = self.offsets[-1]

        self.augment, self.aug = augment, aug_cfg or {}
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        # Skip global SNR scan unless filtering is requested
        if self.min_snr_db is None and self.max_snr_db is None:
            self.valid_idxs = list(range(self.N_raw))
        else:
            self._build_index()
        self.N = len(self.valid_idxs)

    def _snr(self, x: np.ndarray) -> float:
        I, Q = x[:,0], x[:,1]
        power = (I*I + Q*Q).mean()
        noise_var = np.var(I - I.mean()) + np.var(Q - Q.mean()) + 1e-12
        return 10 * math.log10(power / noise_var)

    def _build_index(self):
        self.valid_idxs = []
        for gid in range(self.N_raw):
            shard_i = np.searchsorted(self.offsets, gid, side="right") - 1
            local   = gid - self.offsets[shard_i]
            tm      = self.views[shard_i].tm[local].astype("float32")
            snr_db  = self._snr(tm)
            if ((self.min_snr_db is None or snr_db >= self.min_snr_db)
            and  (self.max_snr_db is None or snr_db <= self.max_snr_db)):
                self.valid_idxs.append(gid)

    def __len__(self):
        return self.N

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if self.augment and np.random.rand() < self.aug.get("time_shift_prob", 0):
            x = np.roll(x, np.random.randint(1, x.shape[0]), axis=0)
        return x

    def __getitem__(self, idx: int):
        gid     = self.valid_idxs[idx]
        shard_i = np.searchsorted(self.offsets, gid, side="right") - 1
        local   = gid - self.offsets[shard_i]
        v       = self.views[shard_i]

        x4  = v.tm[local].astype("float32")
        x4  = self._augment(x4)
        snr = self._snr(x4)

        sample = {
            "tm":   torch.from_numpy(x4),
            "spec": torch.from_numpy(v.sp[local].astype("float32")),
            "snr":  torch.tensor(snr, dtype=torch.float32),
            "y":    torch.tensor(int(v.y[local]), dtype=torch.int64),
        }
        sample["cwt"] = torch.from_numpy(v.cw[local].astype("float32")) if v.has_cwt else torch.empty(0)
        sample["dwt"] = torch.from_numpy(v.dw[local].astype("float32")) if v.has_dwt else torch.empty(0)
        return sample
