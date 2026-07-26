import numpy as np
import torch


def get_ndcg(y_pred, relat_score, k):
    relat_score_sort = -np.sort(-relat_score)
    idcg = (2**relat_score_sort - 1) / np.log2(np.arange(1, len(relat_score_sort)+1) + 1)
    idcg = np.sum(idcg[0:k])
    if idcg == 0:
        return 0.0
    y_pred_rank = np.argsort(-y_pred)
    dcg = (2**relat_score[y_pred_rank] - 1) / np.log2(np.arange(1, len(y_pred_rank)+1) + 1)
    return np.sum(dcg[0:k]) / idcg


def get_hit(y_pred, relat_score, k):
    y_pred_rank = np.argsort(-y_pred)
    for i in y_pred_rank[0:k]:
        if relat_score[i] > 0.0:
            return 1.0
    return 0.0


def get_pre_rec_ap(y_pred, relat_score, k):
    y_pred_rank = np.argsort(-y_pred)
    y_true_rank = np.argsort(-relat_score)
    n_nonzero = len(relat_score.nonzero()[0])
    if n_nonzero == 0:
        return 0.0, 0.0, 0.0
    n = len(set(y_true_rank[0:n_nonzero]).intersection(y_pred_rank[0:k]))
    pre = n / k
    rec = n / min(n_nonzero, k)
    pre_k = 0
    for j in range(k):
        pre_k += len(set(y_true_rank[0:n_nonzero]).intersection(
            y_pred_rank[0:j+1])) / (j+1) * int(relat_score[y_pred_rank[j]] > 0)
    ap = pre_k / min(n_nonzero, k)
    return pre, rec, ap


def build_graph_pyg(d, device=None):
    """
    Build PyG-compatible graph tensors from Data object.
    图张量始终在 CPU 上返回，进 CompGCN forward 时再按需移到 device，
    避免在共享 GPU 服务器上启动时就占用显存。

    Returns:
        edge_index : LongTensor [2, E]  (CPU)
        edge_type  : LongTensor [E]     (CPU)
        edge_norm  : FloatTensor [E]    (CPU)
    """
    kg_data = torch.tensor(d.kg_data, dtype=torch.long)
    src   = kg_data[:, 0]
    dst   = kg_data[:, 2]
    etype = kg_data[:, 1]

    num_rels  = len(d.rel2id)
    num_nodes = d.num_ents

    # etype_norm[e] = 1 / #(edges of same type arriving at dst[e])
    dst_rel = dst * num_rels + etype          # unique (node, rel) key
    count   = torch.zeros(num_nodes * num_rels, dtype=torch.float)
    count.scatter_add_(0, dst_rel, torch.ones(len(dst_rel), dtype=torch.float))
    edge_norm = 1.0 / count[dst_rel].clamp(min=1.0)

    # 保留在 CPU：不在此处 .to(device)
    edge_index = torch.stack([src, dst], dim=0)   # [2, E]
    # device 参数仅为向后兼容保留，不再使用
    return edge_index, etype, edge_norm
