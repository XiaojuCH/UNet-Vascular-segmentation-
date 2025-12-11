import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def _check_tensor(name, t):
    if t is None:
        return
    if torch.isnan(t).any():
        raise RuntimeError(f"NaN detected in {name}")
    if torch.isinf(t).any():
        raise RuntimeError(f"Inf detected in {name}")

class _SimpleLN(nn.Module):
    """Channel-wise LayerNorm for (B,C,H,W)."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor):
        # x: (B,C,H,W)
        x = x.permute(0, 2, 3, 1)  # B,H,W,C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        return x

class SCSA(nn.Module):

    def __init__(self,
                 in_ch: int,
                 num_semantic: int = 32,
                 reduction: int = 4,
                 topk: int = 8,
                 prototype_mode: bool = False,
                 downsample: int = 1,
                 alpha_init: float = 0.5):
        super().__init__()
        self.in_ch = in_ch
        self.num_semantic = num_semantic
        self.reduction = max(1, reduction)
        self.topk = max(1, topk)
        self.prototype_mode = bool(prototype_mode)
        self.downsample = int(max(1, downsample))
        # small epsilon for stability
        self._eps = 1e-6

        # semantic assignment conv (1x1)
        self.sem_conv = nn.Conv2d(in_ch, num_semantic, kernel_size=1, bias=True)

        # projections
        d_inner = max(1, in_ch // self.reduction)
        self.q_proj = nn.Conv2d(in_ch, d_inner, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(in_ch, d_inner, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)

        # prototype placeholders
        if self.prototype_mode:
            self.prototypes = nn.Parameter(torch.randn(self.num_semantic, d_inner))
            self.prototype_value = nn.Parameter(torch.randn(self.num_semantic, in_ch))

        self.fuse = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.ln = _SimpleLN(in_ch)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.q_proj.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.k_proj.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.v_proj.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fuse.weight, nonlinearity='linear')
        if self.fuse.bias is not None:
            nn.init.zeros_(self.fuse.bias)
        if self.prototype_mode:
            nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
            nn.init.normal_(self.prototype_value, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, sem_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B,C,H,W)
        sem_map: optional (B,K,H,W) or (B,K,Hd,Wd) or (B,K,Nd) ; soft assignment per spatial pos
        returns: (B,C,H,W) same channels
        """
        # basic checks and setup
        if x is None:
            raise ValueError("input x is None")
        B, C, H, W = x.shape
        device = x.device
        base_dtype = x.dtype
        eps = self._eps

        # optionally downsample features to reduce computation
        if self.downsample > 1:
            x_att = F.avg_pool2d(x, kernel_size=self.downsample, stride=self.downsample)
            Hd, Wd = x_att.shape[2], x_att.shape[3]
            Nd = Hd * Wd
        else:
            x_att = x
            Hd, Wd = H, W
            Nd = H * W

        # semantic assignment (B, K, Nd)
        if sem_map is None:
            sem_logits = self.sem_conv(x_att)  # (B, K, Hd, Wd)
            sem_logits = sem_logits.view(B, self.num_semantic, -1)
            # compute softmax in FP32 for stability
            with torch.amp.autocast('cuda',enabled=False):
                sem32 = F.softmax(sem_logits.to(torch.float32), dim=1)
            sem = sem32.to(base_dtype)
            sem = torch.nan_to_num(sem, nan=0.0, posinf=1e-6, neginf=0.0)
        else:
            # accept various shapes; resize if needed
            if sem_map.dim() == 4:
                if sem_map.shape[2] != Hd or sem_map.shape[3] != Wd:
                    sem_map_resized = F.interpolate(sem_map, size=(Hd, Wd), mode='bilinear', align_corners=False)
                else:
                    sem_map_resized = sem_map
                sem = sem_map_resized.view(B, sem_map_resized.shape[1], -1)
            else:
                sem = sem_map  # assume already (B,K,Nd)
            sem = sem / (sem.sum(dim=1, keepdim=True) + 1e-8)
            sem = torch.nan_to_num(sem, nan=0.0, posinf=1e-6, neginf=0.0)
            if sem.dtype != base_dtype:
                sem = sem.to(base_dtype)

        # projections on x_att
        q = self.q_proj(x_att).view(B, -1, Nd)    # (B, d, Nd)
        k = self.k_proj(x_att).view(B, -1, Nd)    # (B, d, Nd)
        v = self.v_proj(x_att).view(B, C, Nd)     # (B, C, Nd)
        d = q.shape[1]
        scale = (d ** 0.5) + eps

        # ---------- Semantic Continuous Attention (SCA) ----------
        # region-level aggregates (use base dtype for region_v)
        # compute region_k and region_v using einsum; operate in base_dtype to match v
        region_k = torch.einsum('bkn,bdn->bkd', sem.to(base_dtype), k.to(base_dtype))  # (B,K,d)
        region_v = torch.einsum('bkn,bcn->bkc', sem.to(base_dtype), v.to(base_dtype))  # (B,K,C)

        # compute q·region_k in FP32 for numerical stability
        with torch.amp.autocast('cuda',enabled=False):
            q32 = q.to(torch.float32)
            region_k32 = region_k.to(torch.float32)
            scores_region32 = torch.einsum('bdn,bkd->bkn', q32, region_k32) / scale
            attn_region32 = F.softmax(scores_region32, dim=1)  # (B,K,Nd) in fp32
        attn_region = attn_region32.to(base_dtype)
        attn_region = torch.nan_to_num(attn_region, nan=0.0, posinf=1e6, neginf=-1e6)

        # continuous output
        cont_out = torch.einsum('bkn,bkc->bcn', attn_region, region_v)  # (B, C, Nd)
        cont_out = torch.nan_to_num(cont_out, nan=0.0, posinf=1e6, neginf=-1e6)

        # ---------- Semantic Sparse Attention (SSA) ----------
        # candidate pool per region: top-M positions by sem membership
        M = min(256, Nd)
        # top-M indices from sem (compute in fp32 for robustness)
        with torch.no_grad():
            top_vals, top_idx = torch.topk(sem.to(torch.float32), k=M, dim=2)  # (B, K, M)
        # pre-allocate pools in base_dtype
        k_pool = torch.zeros((B, self.num_semantic, d, M), device=device, dtype=base_dtype)
        v_pool = torch.zeros((B, self.num_semantic, C, M), device=device, dtype=base_dtype)

        # gather candidate keys/values (batch & region loops; K moderate so OK)
        for b in range(B):
            for kk in range(self.num_semantic):
                idxs = top_idx[b, kk]  # (M,)
                # ensure idxs on correct device and long
                idxs = idxs.to(device=device, dtype=torch.long)
                k_pool[b, kk] = k[b].index_select(dim=1, index=idxs).to(base_dtype)
                v_pool[b, kk] = v[b].index_select(dim=1, index=idxs).to(base_dtype)

        # hard assignment of pos -> region (for SSA candidate selection)
        hard_region = torch.argmax(sem.to(base_dtype), dim=1)  # (B, Nd)
        sparse_out = torch.zeros_like(cont_out, device=device, dtype=base_dtype)  # (B, C, Nd)

        # compute sparse attention per batch and per region
        # Use FP32 to compute sims/topk/softmax for stability, then cast att back to base_dtype
        for b in range(B):
            q_b = q[b]  # (d, Nd)  might be base_dtype
            k_pool_b = k_pool[b]  # (K, d, M)
            v_pool_b = v_pool[b]  # (K, C, M)
            hr_b = hard_region[b]  # (Nd,)

            for kk in range(self.num_semantic):
                pos_idx = torch.where(hr_b == kk)[0]  # 1D indices
                n_pos = pos_idx.numel()
                if n_pos == 0:
                    continue

                # queries for these positions
                q_sel = q_b[:, pos_idx]  # (d, n_pos)

                # candidate keys/values for region kk
                kk_k = k_pool_b[kk]  # (d, M) in base_dtype
                kk_v = v_pool_b[kk]  # (C, M) in base_dtype

                # sims: compute in FP32 for stability
                with torch.amp.autocast('cuda',enabled=False):
                    sims32 = (q_sel.transpose(0, 1).to(torch.float32) @ kk_k.to(torch.float32)) / scale  # (n_pos, M)
                    k_num = min(self.topk, sims32.shape[1])
                    topv32, topi = torch.topk(sims32, k=k_num, dim=1)  # (n_pos, k_num)
                    att32 = F.softmax(topv32, dim=1)  # (n_pos, k_num) in fp32

                # convert att to base_dtype (match kk_v dtype)
                att = att32.to(base_dtype)  # (n_pos, k_num)
                # ensure topi indices on correct device/dtype
                topi = topi.to(device=device, dtype=torch.long)

                # gather corresponding v vectors: kk_v (C, M) -> v_selected (C, n_pos, k_num)
                # advanced indexing: use topi as indices for second dim
                # topi shape (n_pos, k_num) -> kk_v[:, topi] yields (C, n_pos, k_num)
                v_selected = kk_v[:, topi]  # (C, n_pos, k_num)

                # att: (n_pos, k_num) -> expand to (1, n_pos, k_num)
                att_exp = att.unsqueeze(0)  # (1, n_pos, k_num)

                # weighted sum over k_num -> out_pos (C, n_pos)
                out_pos = (v_selected * att_exp).sum(dim=2)  # (C, n_pos)

                # final dtype guard: ensure out_pos matches sparse_out dtype
                if out_pos.dtype != sparse_out.dtype:
                    out_pos = out_pos.to(sparse_out.dtype)

                # assign
                sparse_out[b, :, pos_idx] = out_pos

        sparse_out = torch.nan_to_num(sparse_out, nan=0.0, posinf=1e6, neginf=-1e6)

        # optional prototype augmentation
        if self.prototype_mode:
            proto = self.prototypes.unsqueeze(0).expand(B, -1, -1)  # (B,K,d)
            with torch.amp.autocast('cuda',enabled=False):
                proto_scores32 = torch.einsum('bdn,bkd->bkn', q.to(torch.float32), proto.to(torch.float32)) / scale
                proto_att32 = F.softmax(proto_scores32, dim=1)
            proto_att = proto_att32.to(base_dtype)
            pval = self.prototype_value.unsqueeze(0).expand(B, -1, -1).to(base_dtype)  # (B,K,C)
            proto_out = torch.einsum('bkn,bkc->bcn', proto_att, pval)
            cont_out = cont_out + proto_out

        # combine
        alpha = torch.sigmoid(self.alpha)  # (0,1)
        combined = alpha * cont_out + (1.0 - alpha) * sparse_out  # (B,C,Nd)
        combined = torch.nan_to_num(combined, nan=0.0, posinf=1e6, neginf=-1e6)

        # unflatten + fuse
        combined = combined.view(B, C, Hd, Wd)  # (B,C,Hd,Wd)
        fused = self.fuse(combined)  # (B,C,Hd,Wd)
        if self.downsample > 1:
            fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)

        out = x + fused
        out = self.ln(out)
        return out

# -----------------------
# quick unit test (run this file directly to verify forward)
if __name__ == "__main__":
    import os
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # simulate typical convnext feature (B, C, H, W)
    B, C, H, W = 2, 192, 28, 28
    x = torch.randn(B, C, H, W, device=device, dtype=torch.float32)
    scsa = SCSA(in_ch=C, num_semantic=16, reduction=4, topk=4, downsample=1).to(device)

    # run with autocast to simulate AMP behavior
    try:
        with torch.amp.autocast('cuda',enabled=True):
            with torch.no_grad():
                y = scsa(x)
        print("Forward OK, out shape:", y.shape, "dtype:", y.dtype)
    except Exception as e:
        print("Forward failed:", e)
        raise
