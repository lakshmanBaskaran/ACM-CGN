# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo  # needed for @torch._dynamo.disable

# ─── small dtype shim (safetensors/timm) ─────────────────────────────────────
_dtype_map = {
    "uint8":  torch.uint8,  "int8":   torch.int8,
    "uint16": torch.int16,  "int16":  torch.int16,
    "uint32": torch.int32,  "int32":  torch.int32,
    "uint64": torch.int64,  "int64":  torch.int64,
}
for _name, _dtype in _dtype_map.items():
    if not hasattr(torch, _name):
        setattr(torch, _name, _dtype)

# minimal timm pieces already used elsewhere
from timm.layers.mlp import Mlp
from timm.layers.drop import DropPath

# ─── PyTorch Geometric (hard requirement now) ────────────────────────────────
try:
    from torch_geometric.nn import (
        GATv2Conv, TransformerConv, SAGPooling, global_mean_pool
    )
    HAS_PYG = True
except Exception as e:
    HAS_PYG = False
    _MISS = (
        "torch_geometric is required for this GNN version of the model. "
        "Install with:\n"
        "pip install --no-cache-dir torch-geometric==2.4.0 "
        "pyg-lib torch-scatter torch-sparse torch-cluster "
        "-f https://data.pyg.org/whl/torch-2.1.0+cu118.html"
    )
    raise RuntimeError(_MISS) from e


# ─────────────────── denoisers used ahead of segmentation ────────────────────
class DenoiserAutoEnc(nn.Module):
    """Time-domain residual autoencoder denoiser (1D)."""
    def __init__(self, in_ch=4, hidden=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, in_ch, 3, padding=1),
        )
    def forward(self, x):                 # x: (B, T, 4)
        z = x.permute(0, 2, 1)            # -> (B,4,T)
        h = self.enc(z)
        recon = self.dec(h)
        return (z - recon).permute(0, 2, 1)  # -> (B,T,4)

class Denoiser1D(nn.Module):
    """Simple 1D residual denoiser that works on (B,4,T)."""
    def __init__(self, ch=4, width=64, depth=5, p_drop=0.0):
        super().__init__()
        layers = [nn.Conv1d(ch, width, 3, padding=1)]
        for _ in range(depth - 2):
            layers += [nn.ReLU(), nn.Conv1d(width, width, 3, padding=1)]
        layers += [nn.ReLU(), nn.Conv1d(width, ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)
        self.do  = nn.Dropout(p_drop)
    def forward(self, x):  # x: (B,4,T)
        return x - self.do(self.net(x))

class Denoiser2D(nn.Module):
    """
    Light 2D residual denoiser on the (T x 4) plane.
    Treats the 4 IQ-variant channels as the short axis for spatial filtering.
    Input:  (B, T, 4)
    Output: (B, T, 4)
    """
    def __init__(self, width=16, p_drop=0.0):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, width,  (5, 3), padding=(2, 1)), nn.ReLU(),
            nn.Conv2d(width, width, (3, 3), padding=1),   nn.ReLU(),
            nn.Dropout2d(p_drop),
        )
        self.dec = nn.Conv2d(width, 1, (3, 3), padding=1)
    def forward(self, x_bt4):
        x = x_bt4.unsqueeze(1)        # (B,1,T,4)
        h = self.enc(x)
        recon = self.dec(h)           # (B,1,T,4)
        return (x - recon).squeeze(1) # (B,T,4)


# ─────────────────── small blocks reused in the encoders ─────────────────────
class MSCBlock(nn.Module):
    def __init__(self, in_c, out_c, p_drop=0.0):
        super().__init__()
        c1 = out_c // 3; c2 = out_c // 3; c3 = out_c - c1 - c2
        ks = (3, 5, 7)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_c, c1, ks[0], padding=ks[0] // 2),
            nn.Conv1d(in_c, c2, ks[1], padding=ks[1] // 2),
            nn.Conv1d(in_c, c3, ks[2], padding=ks[2] // 2),
        ])
        self.bn = nn.BatchNorm1d(out_c)
        self.do = nn.Dropout(p_drop)
    def forward(self, x):
        parts = [conv(x) for conv in self.convs]
        return F.relu(self.do(self.bn(torch.cat(parts, dim=1))))

class TinyTFBlock(nn.Module):
    """Tiny transformer block (LayerNorm → MHA → MLP) with DropPath."""
    def __init__(self, dim, num_heads=4, mlp_ratio=2., drop=0., drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = Mlp(dim, int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
    def forward(self, x):
        y = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.drop_path(y)
        return x + self.drop_path(self.mlp(self.norm2(x)))

class SegTransformer(nn.Module):
    """
    A small transformer that runs ACROSS segments (sequence length = S)
    to enrich (B, S, D) features before the GNN.
    """
    def __init__(self, dim, layers=2, heads=4, mlp_ratio=2.0, drop=0.0, drop_path=0.0):
        super().__init__()
        dps = [drop_path] * layers if isinstance(drop_path, (int, float)) else list(drop_path)
        if len(dps) < layers:
            dps = dps + [dps[-1]] * (layers - len(dps))
        self.blocks = nn.ModuleList([
            TinyTFBlock(dim, num_heads=heads, mlp_ratio=mlp_ratio, drop=drop, drop_path=dps[i])
            for i in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x_bsd):
        for blk in self.blocks:
            x_bsd = blk(x_bsd)
        return self.norm(x_bsd)


class TimeSegEncoder(nn.Module):
    """Shared time-domain segment encoder → (B,S,D)."""
    def __init__(self, feat_d, drop=0.0, drop_path=0.0, denoiser_type: str = "1d"):
        super().__init__()
        denoiser_type = (denoiser_type or "1d").lower()
        if denoiser_type == "2d":
            self.denoise_2d = Denoiser2D(width=16, p_drop=drop)
            self.denoise_1d = None
        elif denoiser_type == "none":
            self.denoise_2d = None
            self.denoise_1d = None
        else:
            self.denoise_2d = None
            self.denoise_1d = Denoiser1D(ch=4, p_drop=drop)

        self.msc = MSCBlock(4, feat_d, p_drop=drop)
        self.tf  = TinyTFBlock(feat_d, num_heads=4, mlp_ratio=2., drop=drop, drop_path=drop_path)

    def forward(self, tm_segments_bsd_l4, S):
        # tm_segments_bsd_l4: (B*S, L, 4)
        # apply denoiser (1D over (4,T) or 2D over (T,4)) as configured
        if self.denoise_1d is not None:
            x = tm_segments_bsd_l4.permute(0, 2, 1)          # (B*S,4,L)
            x = self.denoise_1d(x)                           # (B*S,4,L)
        elif self.denoise_2d is not None:
            x = self.denoise_2d(tm_segments_bsd_l4)          # (B*S,L,4)
            x = x.permute(0, 2, 1)                           # (B*S,4,L)
        else:
            x = tm_segments_bsd_l4.permute(0, 2, 1)          # (B*S,4,L)

        x = self.msc(x)                                      # (B*S, D, L)
        x = x.permute(0, 2, 1)                               # (B*S, L, D)
        x = self.tf(x)                                       # (B*S, L, D)
        seg_feat = x.mean(dim=1)                             # (B*S, D)
        B = seg_feat.size(0) // S
        return seg_feat.view(B, S, -1)                       # (B,S,D)


class SpecGlobalEncoder(nn.Module):
    def __init__(self, out_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, out_d, 3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
    def forward(self, spec_bf_t_2):   # (B, F, T, 2)
        x = spec_bf_t_2.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)  # (B,2,F,T)
        return self.net(x)            # (B, D)

class CWTGlobalEncoder(nn.Module):
    def __init__(self, out_d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, out_d, 3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
    def forward(self, cwt_full):      # (B, Scales, T) or empty
        if (not isinstance(cwt_full, torch.Tensor)) or cwt_full.numel() == 0:
            return None
        x = cwt_full.unsqueeze(1)     # (B,1,S,T)
        return self.net(x)            # (B, D)

class ExpertAdapter(nn.Module):
    """Per-expert light adapter: x + DropPath(MLP(LN(x)))."""
    def __init__(self, dim, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp  = Mlp(dim, dim, act_layer=nn.GELU, drop=drop)
        self.dp   = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    def forward(self, x):  # (B,S,D)
        return x + self.dp(self.mlp(self.norm(x)))


# ─────────────────────── Main classifier (PyG GNN) ───────────────────────────
class MultiBranchGNNClassifier(nn.Module):
    """
    Explicit GNN backbone built with PyG:
      - node = time segment
      - edges = per-sample segment graph (full or chain)
      - conv  = GATv2Conv or TransformerConv
      - pooling = SAGPooling → global_mean_pool

    The PyG portion is wrapped with @torch._dynamo.disable so the encoders/head
    can be compiled with torch.compile while the GNN runs eager (stable).
    """
    def __init__(self, cfg):
        super().__init__()
        seg, mdl = cfg["segmentation"], cfg["model"]
        self.S    = int(seg["num_segments"])
        self.L    = int(seg["segment_len"])
        self.step = int(self.L * (1 - seg["overlap"]))
        feat_d    = int(mdl["seg_feat_dim"])
        H         = int(mdl["gnn_hidden_dim"])
        heads     = int(mdl["gnn_heads"])
        drop      = float(mdl.get("dropout", 0.1))
        self.impl = mdl.get("gnn_impl", "gatv2").lower()          # "gatv2" | "transformer"
        # default to "chain" for stability (matches write-up)
        self.graph_mode = mdl.get("graph_connectivity", "chain")  # "full" | "chain"

        # config: denoiser selection & seg-transformer
        denoiser_type = (mdl.get("denoiser_type", "1d") or "1d").lower()
        segtf_cfg     = mdl.get("seg_transformer", {})
        self.use_seg_tf = bool(segtf_cfg.get("enabled", False))
        self.seg_tf = None
        if self.use_seg_tf:
            self.seg_tf = SegTransformer(
                dim=feat_d,
                layers=int(segtf_cfg.get("layers", 2)),
                heads=int(segtf_cfg.get("heads", 4)),
                mlp_ratio=float(segtf_cfg.get("mlp_ratio", 2.0)),
                drop=float(segtf_cfg.get("drop", drop)),
                drop_path=float(segtf_cfg.get("drop_path", 0.05)),
            )

        # AE denoise toggle (weights loaded in train.py if pre-trained)
        self.use_autoenc = bool(mdl.get("use_autoenc_denoise", False))
        self.autoenc = DenoiserAutoEnc(in_ch=4, hidden=32)
        if self.use_autoenc:
            for p in self.autoenc.parameters():
                p.requires_grad = False
            self.autoenc.eval()

        # shared encoders
        self.time_enc = TimeSegEncoder(
            feat_d, drop=drop, drop_path=0.05, denoiser_type=denoiser_type
        )
        self.spec_enc = SpecGlobalEncoder(feat_d)
        self.cwt_enc  = CWTGlobalEncoder(feat_d)

        # tiny expert adapters
        self.psk_adapt = ExpertAdapter(feat_d, drop=drop * 0.8, drop_path=0.05)
        self.qam_adapt = ExpertAdapter(feat_d, drop=drop * 0.6, drop_path=0.07)
        self.fsk_adapt = ExpertAdapter(feat_d, drop=drop * 0.4, drop_path=0.09)

        # gate: mean of experts + scaled snr
        self.gate = nn.Sequential(
            nn.Linear(3 * feat_d + 1, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )

        # embeddings
        self.snr_emb = nn.Linear(1, feat_d)
        pos_idx = torch.arange(self.S, dtype=torch.long)
        self.register_buffer("pos_idx", pos_idx, persistent=False)
        self.pos = nn.Embedding(self.S, feat_d)

        # GNN stack
        self.in_proj = nn.Linear(feat_d, H)      # for clean residual at first layer
        self.gnns    = nn.ModuleList()
        self.norms   = nn.ModuleList()
        Conv = GATv2Conv if self.impl.startswith("gat") else TransformerConv
        # first conv (D→H)
        self.gnns.append(Conv(H, H, heads=heads, concat=False, dropout=drop))
        self.norms.append(nn.LayerNorm(H))
        # remaining convs (H→H)
        for _ in range(1, int(mdl["gnn_layers"])):  # type: ignore[arg-type]
            self.gnns.append(Conv(H, H, heads=heads, concat=False, dropout=drop))
            self.norms.append(nn.LayerNorm(H))
        self.final_dim = H

        # pooling (SAG) – default ratio 0.4 (safer)
        self.pool = SAGPooling(self.final_dim, ratio=float(mdl.get("sag_pool_ratio", 0.4)))

        # precompute base edges per graph
        base = self._make_base_edge_index(self.S, self.graph_mode)   # (2, E)
        self.register_buffer("edge_cpu", base, persistent=False)
        self.register_buffer("edge_gpu", torch.empty(0), persistent=False)
        self.E = base.size(1)

        # head
        self.head = nn.Sequential(
            nn.Linear(self.final_dim, 128), nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, int(mdl["num_classes"]))
        )

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _make_base_edge_index(S: int, mode: str):
        if mode == "chain":
            row = torch.arange(S - 1)
            chain = torch.stack([torch.cat([row, row + 1]), torch.cat([row + 1, row])], dim=0)
            return chain  # (2, 2*(S-1))
        # full graph without self loops
        mat = torch.ones(S, S, dtype=torch.bool)
        mat.fill_diagonal_(False)
        src, dst = mat.nonzero(as_tuple=True)
        return torch.stack([src, dst], dim=0)    # (2, S*(S-1))

    def _edge_index(self, B, device):
        if device.type == "cuda" and self.edge_gpu.numel() == 0:
            self.edge_gpu = self.edge_cpu.to(device)
        base = self.edge_gpu if device.type == "cuda" else self.edge_cpu
        off  = torch.arange(B, device=device).repeat_interleave(self.E) * self.S
        return base.repeat(1, B) + off

    def _segment(self, x):
        B, T, C = x.shape
        full  = (self.S - 1) * self.step + self.L
        if T < full:
            pad = x.new_zeros(B, full - T, C)
            x   = torch.cat([x, pad], dim=1)
        win = x.unfold(1, self.L, self.step)  # (B, S_actual, L, C)
        S_actual = win.size(1)
        if S_actual != self.S:
            if S_actual > self.S:
                win = win[:, :self.S]
            else:
                pad_win = x.new_zeros(B, self.S - S_actual, self.L, C)
                win = torch.cat([win, pad_win], dim=1)
        return win.reshape(B * self.S, self.L, C)

    # ── PyG block kept eager for stability under torch.compile ───────────────
    @torch._dynamo.disable
    def _pyg_block(self, x, edge_index, batch_vec, B: int):
        # 5) GNN stack (residual + LN; dropout inside convs)
        for conv, ln in zip(self.gnns, self.norms):
            h = conv(x, edge_index)
            x = ln(F.gelu(h)) + x
        # 6) Pool with SAG, then global mean per graph
        x_p, edge_p, _, batch_out, _, _ = self.pool(x, edge_index, batch=batch_vec)
        g = global_mean_pool(x_p, batch_out, size=B)  # (B,H)
        return g

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, tm, spec, cwt, snr_db, return_embed: bool = False):
        # 1) optional AE denoise (residual) + segment + encode per segment
        if self.use_autoenc:
            with torch.no_grad():
                tm = self.autoenc(tm)                       # (B,T,4)
        B  = tm.size(0)
        tm_seg  = self._segment(tm)                         # (B*S, L, 4)
        seg_feat = self.time_enc(tm_seg, self.S)            # (B,S,D)

        # 2) add global spec + cwt encodings
        spec_vec = self.spec_enc(spec)                      # (B,D)
        seg_feat = seg_feat + spec_vec.unsqueeze(1)
        cwt_vec  = self.cwt_enc(cwt)                        # (B,D) or None
        if isinstance(cwt_vec, torch.Tensor):
            seg_feat = seg_feat + cwt_vec.unsqueeze(1)

        # 3) expert adapters + gating (scale SNR for stability)
        f1 = self.psk_adapt(seg_feat)
        f2 = self.qam_adapt(seg_feat)
        f3 = self.fsk_adapt(seg_feat)
        g1 = f1.mean(dim=1); g2 = f2.mean(dim=1); g3 = f3.mean(dim=1)
        snr_scaled = (snr_db / 30.0).to(g1.dtype).unsqueeze(1)
        w = F.softmax(self.gate(torch.cat([g1, g2, g3, snr_scaled], dim=1)), dim=1)  # (B,3)
        w = w.unsqueeze(1).unsqueeze(-1)                                             # (B,1,3,1)
        x = (torch.stack([f1, f2, f3], dim=2) * w).sum(dim=2)                        # (B,S,D)

        # 4) add positional + SNR embeddings (let seg-transformer see them)
        pos_emb = self.pos(self.pos_idx.to(tm.device)).unsqueeze(0).expand(B, -1, -1)
        snr_emb = self.snr_emb(snr_scaled).unsqueeze(1).expand(B, self.S, -1)
        x = x + pos_emb + snr_emb                                                   # (B,S,D)

        # 4b) optional transformer across segments
        if self.use_seg_tf and (self.seg_tf is not None):
            x = self.seg_tf(x)                                                      # (B,S,D)

        # 5) project & flatten to a batched graph
        x = self.in_proj(x)                                                         # (B,S,H)
        x = x.reshape(B * self.S, -1)                                               # (B*S,H)
        edge_index = self._edge_index(B, x.device)                                  # (2, E_total)
        batch_vec  = torch.arange(B, device=x.device).repeat_interleave(self.S)     # (B*S,)

        # 6) PyG GNN + pooling (kept eager)
        g = self._pyg_block(x, edge_index, batch_vec, B)                            # (B,H)

        # 7) classify
        logits = self.head(g)
        if return_embed:
            return logits, g
        return logits
