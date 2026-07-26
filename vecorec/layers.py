"""
CompGCN layer implemented with PyTorch Geometric (replaces DGL version).
Functionally equivalent to the original MyCompGCN.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class MyCompGCN(MessagePassing):
    def __init__(self, indim, outdim, nr, dropout, opn, **kwargs):
        super(MyCompGCN, self).__init__(aggr='add')
        self.indim  = indim
        self.outdim = outdim
        self.opn    = opn

        self.W_R   = nn.Parameter(torch.Tensor(nr, indim, outdim))
        self.W_rel = nn.Parameter(torch.Tensor(indim, outdim))
        self.bias  = nn.Parameter(torch.Tensor(1, indim))
        self.dropout = nn.Dropout(dropout)
        self.bn      = nn.BatchNorm1d(outdim)
        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.W_R)
        nn.init.xavier_normal_(self.W_rel)
        nn.init.zeros_(self.bias)

    def comp(self, h, r):
        if self.opn == 'mult':
            return h * r
        elif self.opn == 'sub':
            return h - r
        elif self.opn == 'rotate':
            d = h.shape[-1]
            h_re, h_im = torch.split(h, d // 2, -1)
            r_re, r_im = torch.split(r, d // 2, -1)
            return torch.cat([h_re * r_re - h_im * r_im,
                              h_re * r_im + h_im * r_re], dim=-1)
        else:
            raise KeyError(f'Unknown composition operator: {self.opn}')

    def forward(self, node_feat, edge_index, edge_type, edge_norm, rel_emb):
        # Use object.__setattr__ to bypass nn.Module's parameter registry.
        # When called during training, rel_emb is R.weight (nn.Parameter); if we
        # do self._rel_emb = rel_emb directly, PyTorch registers it as a param,
        # then eval's clone()-based tensor assignment raises TypeError.
        object.__setattr__(self, '_rel_emb', rel_emb)
        out = self.propagate(edge_index, x=node_feat,
                             edge_type=edge_type, edge_norm=edge_norm)
        out = self.dropout(out)
        out = self.bn(out)
        out = torch.tanh(out)
        rel_emb_new = torch.matmul(rel_emb, self.W_rel)
        return out, rel_emb_new

    def message(self, x_j, edge_type, edge_norm):
        rel  = self._rel_emb[edge_type]
        comp = self.comp(x_j, rel)
        # 原写法 torch.bmm(comp.unsqueeze(1), self.W_R[edge_type]) 会将所有边的
        # 权重矩阵展开为 [E, indim, outdim]（330k 边 × 64 × 64 ≈ 5 GB），导致 OOM。
        # 改为按关系类型分组做 matmul，每次只用一个 [indim, outdim] 矩阵。
        msg = torch.zeros(comp.shape[0], self.outdim,
                          device=comp.device, dtype=comp.dtype)
        for r in range(self.W_R.shape[0]):
            mask = (edge_type == r)
            if mask.any():
                msg[mask] = comp[mask] @ self.W_R[r]
        msg = msg * edge_norm.unsqueeze(-1)
        return msg
