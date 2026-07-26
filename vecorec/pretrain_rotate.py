"""
RotatE KG Pretraining for VeCo
- 用 RotatE 替换 TuckER 预训练，提供不同几何结构的初始嵌入
- 输出：ER_beijing_RotatE_64.npz (与TuckER格式相同，直接可替换)
"""
import os, sys, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import Data


class RotatEModel(nn.Module):
    def __init__(self, n_ent, n_rel, edim, gamma=12.0):
        super().__init__()
        self.edim = edim
        self.gamma = gamma
        # RotatE: entity in complex space (edim/2 complex = edim real)
        self.E = nn.Embedding(n_ent, edim)
        # relation: phase only (edim/2 phases → real values in [0, 2π])
        self.R = nn.Embedding(n_rel, edim // 2)
        nn.init.uniform_(self.E.weight, -6/edim**0.5, 6/edim**0.5)
        nn.init.uniform_(self.R.weight, -np.pi, np.pi)

    def score(self, h_idx, r_idx, t_idx):
        """RotatE score: γ - ||h ∘ r - t||"""
        h = self.E(h_idx)  # [B, edim]
        r_phase = self.R(r_idx)  # [B, edim//2]
        t = self.E(t_idx)  # [B, edim]

        edim2 = self.edim // 2
        h_re, h_im = h[:, :edim2], h[:, edim2:]
        t_re, t_im = t[:, :edim2], t[:, edim2:]

        # rotation: (h_re + i*h_im) * (cos(r) + i*sin(r))
        cos_r, sin_r = torch.cos(r_phase), torch.sin(r_phase)
        rot_re = h_re * cos_r - h_im * sin_r
        rot_im = h_re * sin_r + h_im * cos_r

        # distance
        diff_re = rot_re - t_re
        diff_im = rot_im - t_im
        dist = (diff_re**2 + diff_im**2 + 1e-8).sqrt().sum(dim=-1)  # [B]
        return self.gamma - dist

    def forward(self, h, r, t_pos, t_neg):
        """Self-adversarial negative sampling loss"""
        pos_score = self.score(h, r, t_pos)  # [B]

        # neg: t_neg shape [B, K]
        B, K = t_neg.shape
        h_exp = h.unsqueeze(1).expand(B, K)
        r_exp = r.unsqueeze(1).expand(B, K)
        neg_score = self.score(h_exp.reshape(-1), r_exp.reshape(-1),
                               t_neg.reshape(-1)).reshape(B, K)  # [B, K]

        # Self-adversarial weighting
        with torch.no_grad():
            adv_w = F.softmax(neg_score * 0.5, dim=-1)

        pos_loss = -F.logsigmoid(pos_score - 0.0).mean()
        neg_loss = -(adv_w * F.logsigmoid(-neg_score)).sum(dim=-1).mean()
        loss = (pos_loss + neg_loss) / 2
        return loss


class KGDataset(Dataset):
    def __init__(self, triples, n_ent, n_neg=64):
        self.triples = triples
        self.n_ent = n_ent
        self.n_neg = n_neg
        # build filter set for clean negatives
        self.pos_set = set([(h, r, t) for h, r, t in triples])

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        # random negative tail sampling
        negs = np.random.randint(0, self.n_ent, self.n_neg)
        return h, r, t, negs


def collate_fn(batch):
    h = torch.tensor([x[0] for x in batch], dtype=torch.long)
    r = torch.tensor([x[1] for x in batch], dtype=torch.long)
    t = torch.tensor([x[2] for x in batch], dtype=torch.long)
    negs = torch.tensor(np.stack([x[3] for x in batch]), dtype=torch.long)
    return h, r, t, negs


def train_rotate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, f'../data/{args.dataset}/')
    out_dir  = os.path.join(root, '../pretrain_emb/')

    print(f'Loading data from {data_dir}...')
    d = Data(data_dir=data_dir, kg_name='kg')
    n_ent, n_rel = d.num_ents, d.num_rels
    triples = [(h, r, t) for h, r, t in d.kg_data]

    print(f'KG: {n_ent} ents, {n_rel} rels, {len(triples)} triples')
    print(f'Training RotatE edim={args.edim}, gamma={args.gamma}...')

    model = RotatEModel(n_ent, n_rel, args.edim, gamma=args.gamma).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    dataset = KGDataset(triples, n_ent, n_neg=args.n_neg)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, collate_fn=collate_fn)

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for h, r, t, negs in tqdm(loader, leave=False, desc=f'Epoch {epoch}'):
            h, r, t, negs = h.to(device), r.to(device), t.to(device), negs.to(device)
            loss = model(h, r, t, negs)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch:3d}: loss={avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Save embeddings in same format as TuckER
    E_np = model.E.weight.detach().cpu().numpy()   # [n_ent, edim]
    # For R, expand phase to full edim by duplicating (cos, sin)
    R_phase = model.R.weight.detach().cpu().numpy()  # [n_rel, edim//2]
    R_cos = np.cos(R_phase); R_sin = np.sin(R_phase)
    R_np = np.concatenate([R_cos, R_sin], axis=1)    # [n_rel, edim]

    out_path = os.path.join(out_dir, f'ER_{args.dataset}_RotatE_{args.edim}.npz')
    np.savez(out_path, E_pretrain=E_np, R_pretrain=R_np)
    print(f'\nSaved: {out_path}')
    print(f'E_pretrain: {E_np.shape}, R_pretrain: {R_np.shape}')


if __name__ == '__main__':
    pa = argparse.ArgumentParser()
    pa.add_argument('--dataset',    default='beijing')
    pa.add_argument('--edim',       type=int,   default=64)
    pa.add_argument('--gamma',      type=float, default=12.0)
    pa.add_argument('--lr',         type=float, default=1e-3)
    pa.add_argument('--epochs',     type=int,   default=200)
    pa.add_argument('--batch_size', type=int,   default=1024)
    pa.add_argument('--n_neg',      type=int,   default=64)
    pa.add_argument('--seed',       type=int,   default=42)
    args = pa.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    train_rotate(args)
