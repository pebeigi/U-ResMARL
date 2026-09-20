"""Dense port of CtRL-Sim's MIT RelativeSocialAttentionLayer.

Same edge-conditioned keys/values, destination softmax, gated aggregate and
parameter names. Dense tensors avoid PyG/compiled scatter on Windows.
"""
import torch
from torch import nn


class RelativeSocialAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout, dim_feedforward):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        for name in ('lin_q_node', 'lin_k_node', 'lin_k_edge', 'lin_v_node',
                     'lin_v_edge', 'lin_self', 'lin_ih', 'lin_hh', 'out_proj'):
            setattr(self, name, nn.Linear(d_model, d_model))
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model),
                                 nn.Dropout(dropout))

    def forward(self, x, edge_attr, src_key_padding_mask):
        x = x + self._mha_block(x, src_key_padding_mask, edge_attr)
        return self.norm2(x + self.mlp(self.norm1(x)))

    def _mha_block(self, x, mask, edge):
        n, b, d = x.shape
        h, k = self.nhead, d // self.nhead
        nodes = x.transpose(0, 1)
        edges = edge.reshape(b, n, n, d)
        q = self.lin_q_node(nodes).reshape(b, n, h, k)[:, :, None]
        keys = self.lin_k_node(nodes).reshape(b, n, h, k)[:, None]
        keys = keys + self.lin_k_edge(edges).reshape(b, n, n, h, k)
        values = self.lin_v_node(nodes).reshape(b, n, h, k)[:, None]
        values = values + self.lin_v_edge(edges).reshape(b, n, n, h, k)
        logits = (q * keys).sum(-1) / k**.5
        valid = (~mask[:, :, None] & ~mask[:, None, :])[..., None]
        logits = logits.masked_fill(~valid, -1e9)
        weights = self.attn_drop(logits.softmax(dim=2)) * valid
        messages = (weights[..., None] * values).sum(2).reshape(b, n, d)
        gate = torch.sigmoid(self.lin_ih(messages) + self.lin_hh(nodes))
        aggregate = messages + gate * (self.lin_self(nodes) - messages)
        return self.proj_drop(self.out_proj(aggregate).transpose(0, 1))
