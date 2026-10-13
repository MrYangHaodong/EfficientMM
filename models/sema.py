"""SeMA — sequential multimodal fusion through a shared bottleneck.

Every modality has its own branch (input projection -> per-modality Transformer encoder
-> bottleneck-fusion Transformer). Modalities are processed ONE AT A TIME in any order:

    encode(m_1) -> fuse(m_1) -> readout p_1
    encode(m_2) -> fuse(m_2) -> readout p_2
    ...

Only the K bottleneck tokens carry information between modalities. After each modality
the running bottleneck is updated (running mean over the modalities seen so far) and a
prediction is read out from the average of all fused tokens seen so far, so the model
gives a prediction after every prefix of any acquisition order. A modality that is
never acquired is never encoded.

Training uses random modality orders (``seq_random_order``); evaluation and ARC use the
order they are given.
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.self_attn = self.attn          # alias kept so released checkpoints load strictly
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, hidden_size), nn.Dropout(dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = self.dropout1(x) + residual
        return self.mlp(self.ln2(x)) + x


class ModalityBranch(nn.Module):
    """One modality's projector + encoder + bottleneck-fusion Transformer."""

    def __init__(self, feat_dim: int, d_model: int, nhead: int, num_enc_layers: int,
                 num_fusion_layers: int, mlp_ratio: float = 1.0, dropout: float = 0.1):
        super().__init__()
        mlp_dim = int(d_model * mlp_ratio)
        self.projector = nn.Linear(feat_dim, d_model)
        self.in_norm = nn.LayerNorm(d_model)
        self.encoder = nn.ModuleList([TransformerBlock(d_model, nhead, mlp_dim, dropout)
                                      for _ in range(num_enc_layers)])
        self.enc_norm = nn.LayerNorm(d_model)
        self.fusion = nn.ModuleList([TransformerBlock(d_model, nhead, mlp_dim, dropout)
                                     for _ in range(num_fusion_layers)])
        self.fusion_norm = nn.LayerNorm(d_model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, feat_dim] -> [B, T, d_model]."""
        x = self.in_norm(self.projector(x))
        for blk in self.encoder:
            x = blk(x)
        return self.enc_norm(x)

    def fuse(self, tokens: torch.Tensor, bottleneck: torch.Tensor):
        """Attend jointly over [bottleneck ; tokens]; -> (new bottleneck [B, K, D], tokens [B, T, D])."""
        K = bottleneck.shape[1]
        bn, tok = bottleneck, tokens
        for blk in self.fusion:
            h = blk(torch.cat([bn, tok], dim=1))
            bn, tok = h[:, :K, :], h[:, K:, :]
        return self.fusion_norm(bn), self.fusion_norm(tok)


class SeMA(nn.Module):
    def __init__(self, modalities: List[str], feat_dims: Dict[str, int], num_classes: int,
                 d_model: int = 128, nhead: int = 8, num_enc_layers: int = 3,
                 num_fusion_layers: int = 3, n_bottlenecks: int = 16, mlp_ratio: float = 1.0,
                 dropout: float = 0.1, seq_random_order: bool = False):
        super().__init__()
        self.modalities = list(modalities)
        self.d_model = d_model
        self.seq_random_order = seq_random_order
        self.branches = nn.ModuleDict({
            m: ModalityBranch(feat_dims[m], d_model, nhead, num_enc_layers, num_fusion_layers, mlp_ratio, dropout)
            for m in self.modalities})
        self.bottleneck = nn.Parameter(torch.randn(1, n_bottlenecks, d_model) * 0.02)
        self.final_norm = nn.LayerNorm(d_model)
        self.regressor = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                       nn.Linear(d_model, num_classes))

    @classmethod
    def from_config(cls, cfg: dict, feat_dims: Dict[str, int]) -> 'SeMA':
        """Build from a ``results_fold{N}.json`` written by training/train_sema.py."""
        return cls(cfg['modalities'], feat_dims, cfg['num_classes'], d_model=cfg['d_model'],
                   nhead=cfg['nhead'], num_enc_layers=cfg['num_enc_layers'],
                   num_fusion_layers=cfg['num_fusion_layers'], n_bottlenecks=cfg['n_bottlenecks'])

    def load_state_dict(self, state_dict, *args, **kwargs):
        # checkpoints trained with an older branch definition carry unused '.pos.' buffers
        return super().load_state_dict({k: v for k, v in state_dict.items() if '.pos.' not in k},
                                       *args, **kwargs)

    # ---- incremental interface (one modality at a time) ----
    def init_state(self, batch_size: int, device) -> dict:
        return {'bn': self.bottleneck.expand(batch_size, -1, -1).contiguous(),
                'tok_sum': torch.zeros(batch_size, self.d_model, device=device, dtype=self.bottleneck.dtype),
                'tok_count': 0, 'step': 0}

    def acquire(self, state: dict, m: str, x: torch.Tensor) -> torch.Tensor:
        """Encode and fuse modality ``m`` into ``state`` (updated in place); -> logits [B, C]."""
        branch = self.branches[m]
        new_bn, out_tok = branch.fuse(branch.encode(x), state['bn'])
        state['bn'] = self._agg_bottleneck(state['step'], state['bn'], new_bn)
        state['tok_sum'] = state['tok_sum'] + out_tok.sum(dim=1)
        state['tok_count'] += out_tok.shape[1]
        state['step'] += 1
        return self.regressor(self.final_norm(state['tok_sum'] / max(state['tok_count'], 1)))

    def _agg_bottleneck(self, step: int, running: torch.Tensor, new_bn: torch.Tensor) -> torch.Tensor:
        """Running mean of the per-modality bottlenecks."""
        if step == 0:
            return new_bn
        return (step * running + new_bn) / (step + 1)

    # ---- full forward ----
    def forward(self, inputs: Dict[str, torch.Tensor], order: Optional[List[str]] = None,
                return_prefix: bool = False):
        """inputs: {modality: [B, T, feat_dim]} (only the present modalities are used).
        order: processing order; default = canonical order (random during training).
        Returns logits [B, C], or (logits, prefix_logits [B, S, C], S) with return_prefix."""
        active = [m for m in (order or self.modalities) if m in inputs]
        if not active:
            raise ValueError('no modality present in inputs')
        x0 = inputs[active[0]]
        if self.seq_random_order and self.training:
            active = [active[i] for i in torch.randperm(len(active), device=x0.device).tolist()]
        state = self.init_state(x0.shape[0], x0.device)
        prefix = [self.acquire(state, m, inputs[m]) for m in active]
        if return_prefix:
            return prefix[-1], torch.stack(prefix, dim=1), len(prefix)
        return prefix[-1]
