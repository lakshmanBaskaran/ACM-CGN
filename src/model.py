# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, DropPath
from torch_geometric.nn import SAGPooling, GATv2Conv, GraphNorm
from kymatio.torch import Scattering1D


class LinGNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index=None):
        return self.lin(x)

    def reset_parameters(self):
        self.lin.reset_parameters()


class DenoiserAutoEnc(nn.Module):
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

    def forward(self, x):
        z = x.permute(0, 2, 1)
        h = self.enc(z)
        recon = self.dec(h)
        return (z - recon).permute(0, 2, 1), recon.permute(0, 2, 1)


class SNREmbedding(nn.Module):
    def __init__(self, feat_d):
        super().__init__()
        self.lin = nn.Linear(1, feat_d)

    def forward(self, snr_db):
        return self.lin(snr_db.unsqueeze(1))


class MSCBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        ks = (3, 5, 7)
        base = out_c // len(ks)
        chs = [base] * len(ks)
        chs[-1] = out_c - sum(chs[:-1])
        self.convs = nn.ModuleList([
            nn.Conv1d(in_c, chs[i], ks[i], padding=ks[i] // 2)
            for i in range(len(ks))
        ])
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, x):
        x = torch.cat([c(x) for c in self.convs], dim=1)
        return F.relu(self.bn(x))


class TinyTFBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2., drop=0.1, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        y = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HybridEncoder(nn.Module):
    def __init__(self, cfg, feat_d):
        super().__init__()
        self.L = cfg["segmentation"]["segment_len"]
        self.denoise = MSCBlock(4, feat_d)
        self.tf = TinyTFBlock(dim=feat_d,
                              num_heads=cfg["model"]["tf_heads"],
                              mlp_ratio=cfg["model"].get("tf_mlp_ratio", 2.),
                              drop=cfg["model"]["dropout"],
                              drop_path=cfg["model"].get("drop_path", 0.1))

        # STABILITY FIX: Changed J=4 to J=3 to match preprocessing
        scat_obj = Scattering1D(J=3, shape=self.L)
        with torch.no_grad():
            dummy = torch.zeros(1, self.L)
            out = scat_obj(dummy)
            scat_ch = out.shape[1]
        self.scat_mlp = nn.Sequential(nn.Linear(scat_ch, feat_d), nn.ReLU())

        self.spec = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, feat_d, 3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.use_cwt = cfg["preprocess"].get("cwt") is not None and cfg["preprocess"]["save_dtype"].get("cwt")
        if self.use_cwt:
            self.cwt_conv = nn.Sequential(
                nn.Conv2d(1, feat_d, 3, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
            )

    def forward(self, tm, spec, cwt, scat):
        B, S, L, _ = tm.shape
        flat = tm.reshape(B * S, L, 4).permute(0, 2, 1)
        x = self.denoise(flat)
        x = x.permute(0, 2, 1)
        x = self.tf(x)
        tm_feat = x.mean(dim=1)
        C_s = scat.shape[-1]
        scat_flat = scat.view(B * S, C_s)
        scat_feat = self.scat_mlp(scat_flat)
        spec_flat = spec.permute(0, 3, 1, 2)
        sp_feat_tmp = self.spec(spec_flat)
        sp_feat = sp_feat_tmp.unsqueeze(1).expand(-1, S, -1).reshape(B * S, -1)
        out = tm_feat + scat_feat + sp_feat
        if self.use_cwt and cwt is not None and cwt.numel():
            cwt_flat = cwt.unsqueeze(1)
            cwt_feat_tmp = self.cwt_conv(cwt_flat)
            cwt_feat = cwt_feat_tmp.unsqueeze(1).expand(-1, S, -1).reshape(B * S, -1)
            out += cwt_feat
        return out.reshape(B, S, -1)


class PSKExpert(nn.Module):
    def __init__(self, cfg, feat_d):
        super().__init__();
        self.enc = HybridEncoder(cfg, feat_d)

    def forward(self, tm, spec, cwt, scat): return self.enc(tm, spec, cwt, scat)


class QAMExpert(PSKExpert): pass


class FSKExpert(PSKExpert): pass


class CombinedGNNClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        seg, mdl = cfg["segmentation"], cfg["model"]
        self.S, self.L = seg["num_segments"], seg["segment_len"]
        self.step = int(self.L * (1 - seg.get("overlap", 0.5)))
        feat_d, eng_d = mdl["seg_feat_dim"], mdl.get("eng_feat_dim", 0)
        idx = torch.arange(self.S)
        row, col = idx.repeat(self.S), idx.unsqueeze(1).repeat(1, self.S).view(-1)
        base_ei = torch.stack([row, col], dim=0)
        self.register_buffer("base_edge_index", base_ei)
        self.autoenc = DenoiserAutoEnc(in_ch=4, hidden=32)
        self.psk, self.qam, self.fsk = PSKExpert(cfg, feat_d), QAMExpert(cfg, feat_d), FSKExpert(cfg, feat_d)
        self.gate = nn.Sequential(nn.Linear(feat_d + 1, 64), nn.ReLU(), nn.Linear(64, 3))
        self.snr_emb, self.pos = SNREmbedding(feat_d), nn.Embedding(self.S, feat_d)
        if cfg.get("feature_cache", {}).get("seg_feats", False): self.seg_proj = nn.Identity()
        if eng_d > 0: self.eng_proj = nn.Linear(eng_d, feat_d)
        H, nh, Lg = mdl["gnn_hidden_dim"], mdl["gnn_heads"], mdl["gnn_layers"]
        use_gn = mdl.get("graphnorm", True)
        convs, norms = [], []
        in_c = feat_d
        for i in range(Lg):
            last = (i == Lg - 1)
            convs.append(GATv2Conv(in_c, H, heads=nh, concat=not last, dropout=mdl["dropout"]))
            out_c = H * nh if not last else H
            norms.append(GraphNorm(out_c) if use_gn else nn.Identity())
            in_c = out_c
        self.gnn_convs, self.gnn_norms = nn.ModuleList(convs), nn.ModuleList(norms)
        self.final_dim = in_c
        self.pool = SAGPooling(self.final_dim, ratio=0.5, GNN=LinGNN)
        self.head = nn.Sequential(nn.Linear(self.final_dim, 128), nn.ReLU(), nn.Linear(128, mdl["num_classes"]))

    def _segment(self, x):
        B, T, C = x.shape
        full = (self.S - 1) * self.step + self.L
        if T < full: x = torch.cat([x, x.new_zeros(B, full - T, C)], dim=1)
        return x.permute(0, 2, 1).unfold(2, self.L, self.step).permute(0, 2, 3, 1)

    def forward(self, tm, spec, cwt, scat, snr, seg_feats=None, eng_feats=None):
        B = tm.size(0)
        tm_res, recon = self.autoenc(tm)
        tm_seg = self._segment(tm_res)
        f1, f2, f3 = [e(tm_seg, spec, cwt, scat) for e in (self.psk, self.qam, self.fsk)]
        summary = f1.mean(dim=1)
        w_in = torch.cat([summary, snr.unsqueeze(1)], dim=1)
        w = F.softmax(self.gate(w_in), dim=1).view(B, 1, 3, 1)
        feats = (torch.stack([f1, f2, f3], dim=2) * w).sum(dim=2)
        h = feats + self.snr_emb(snr).unsqueeze(1) + self.pos.weight.unsqueeze(0)
        if seg_feats is not None and seg_feats.numel(): h = h + self.seg_proj(seg_feats)
        if eng_feats is not None and eng_feats.numel():
            e = self.eng_proj(eng_feats).unsqueeze(1).expand(-1, self.S, -1)
            h = h + e
        x = h.reshape(B * self.S, -1)
        batch = torch.arange(B, device=tm.device).repeat_interleave(self.S)
        Ei0, E0 = self.base_edge_index, self.base_edge_index.size(1)
        ei = Ei0.unsqueeze(2).repeat(1, 1, B).reshape(2, E0 * B)
        offset = (torch.arange(B, device=tm.device) * self.S).repeat_interleave(E0)
        edge_index = ei + offset.unsqueeze(0)
        for conv, norm in zip(self.gnn_convs, self.gnn_norms):
            x = F.relu(norm(conv(x, edge_index)), inplace=True)
        pooled, _, _, batch_out, _, _ = self.pool(x, edge_index, batch=batch)
        g = x.new_zeros(B, self.final_dim).index_add_(0, batch_out, pooled)
        return self.head(g), recon


MultiBranchGNNClassifier = CombinedGNNClassifier
