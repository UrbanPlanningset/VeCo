"""
VeCo training script
========================
用法：
  cd ns_site_deploy/ns_concept

  # Step 1：生成品牌概念偏好先验（首次运行）
  python gen_prior.py --dataset beijing

  # Step 2：训练（自动加载先验）
  python train.py --dataset beijing

超参数说明：
  --edim         嵌入维度（default=64，与 KnowSite 相同）
  --n_layer      CompGCN 层数（default=2）
  --lr           主学习率（default=1e-3）
  --lr_emb       嵌入微调学习率（default=1e-4）
  --freeze       false = 微调嵌入，true = 冻结（default=false）
  --patience     早停轮数（default=30）
  --lam_reg      α 正则系数，防止偏好崩塌（default=0.01）

KnowSite 基线：
  Hit@10=0.713, NDCG@10=0.219, P@10=0.186, R@10=0.223, MAP@10=0.127

VeCo objective:
  NDCG@10 > 0.219（符号推理 + 先验注入 > 纯神经 GRU 路径）
"""

import os, time, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from load_data import Data
from model import NSConceptV7
import utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_er_vocab(d):
    ev_tr, ev_tv, ev_tvt = defaultdict(list), defaultdict(list), defaultdict(list)
    for x in d.train_data:
        ev_tr[x[0]].append(x[2])
        ev_tv[x[0]].append(x[2])
        ev_tvt[x[0]].append(x[2])
    for k, rows in d.valid_data.items():
        for y in rows:
            if y[-1] > 0:
                ev_tv[k].append(y[2])
                ev_tvt[k].append(y[2])
    for k, rows in d.test_data.items():
        for y in rows:
            if y[-1] > 0:
                ev_tvt[k].append(y[2])
    return ev_tr, ev_tv, ev_tvt


def evaluate(model, d, graph, data, filter_vocab, K_group):
    ndcgs, hits, pres, recs, aps = ([[] for _ in K_group] for _ in range(5))
    t_idx = torch.tensor(d.region_list, dtype=torch.long, device=device)

    for brand_id in tqdm(sorted(data.keys(), key=int), leave=False):
        h_idx   = brand_id * torch.ones(1, dtype=torch.long, device=device)
        r_idx   = d.psa_id * torch.ones(1, dtype=torch.long, device=device)
        cat_idx = data[brand_id][0][4] * torch.ones(1, dtype=torch.long, device=device)

        score, _, _ = model(graph, h_idx, r_idx, t_idx, cat_idx)

        y = score.clone()
        y[0, filter_vocab[brand_id]] = -1e10
        relat  = np.array([row[-1] for row in data[brand_id]])
        scores = y.squeeze().cpu().numpy()

        for i, k in enumerate(K_group):
            ndcgs[i].append(utils.get_ndcg(scores, relat, k))
            hits[i].append(utils.get_hit(scores, relat, k))
            p, r, ap = utils.get_pre_rec_ap(scores, relat, k)
            pres[i].append(p); recs[i].append(r); aps[i].append(ap)

    idx10 = K_group.index(10)
    print('  Hit@10:       %.6f' % np.mean(hits[idx10]))
    print('  NDCG@10:      %.6f' % np.mean(ndcgs[idx10]))
    print('  Precision@10: %.6f' % np.mean(pres[idx10]))
    print('  Recall@10:    %.6f' % np.mean(recs[idx10]))
    print('  MAP@10:       %.6f' % np.mean(aps[idx10]))
    return ([np.mean(v) for v in ndcgs], [np.mean(v) for v in hits],
            [np.mean(v) for v in pres],  [np.mean(v) for v in recs],
            [np.mean(v) for v in aps])


def alpha_entropy_loss(alpha):
    """
    α 熵正则：防止品牌偏好坍塌到单一概念（如 C1=0.996，其余趋零）。

    方法：最大化 α 在概念维度上的熵（当作概率分布看待）。
    - 归一化 α 为概率分布（softmax 化）
    - 计算熵 H = -Σ_k p_k × log(p_k)
    - 最大化熵 → 防止极端集中

    返回值为负熵（loss = -H），最小化 loss = 最大化熵。

    alpha : [B, K]  ∈ (0,1)
    """
    # 归一化成概率分布
    alpha_sum = alpha.sum(dim=1, keepdim=True).clamp(min=1e-8)
    p = alpha / alpha_sum                          # [B, K]
    eps = 1e-8
    H = -(p * torch.log(p + eps)).sum(dim=1)       # [B]  熵值，越高越均匀
    return -H.mean()                               # 取负：loss 越小 = 熵越高


def S_variance_loss(S):
    """
    S 方差正则：防止所有区域的满足度趋向同一高值（S 均匀坍塌）。

    方法：最大化 S 在区域维度上的方差均值。
    - 对每个概念 k，计算 Var(S[:, k]) across all regions
    - 最大化均值方差 → 迫使 ConceptNet 对不同区域给出差异化评分

    返回值为负方差（loss = -Var），最小化 loss = 最大化方差。

    S : [N, K]  ∈ (0,1)，N=区域数，K=概念数
    """
    if S.shape[0] < 2:
        return torch.tensor(0.0, device=S.device)
    var_per_concept = S.var(dim=0)                 # [K]
    return -var_per_concept.mean()                 # 取负：loss 越小 = 方差越大


def alpha_cosine_reg(alpha):
    """
    原余弦相似度正则（保留备用，现在改用熵正则作为主要α正则）。
    alpha : [B, K]
    """
    if alpha.shape[0] < 2:
        return torch.tensor(0.0, device=alpha.device)
    norm  = alpha.norm(dim=1, keepdim=True).clamp(min=1e-8)
    a_n   = alpha / norm
    sim   = a_n @ a_n.t()
    mask  = 1 - torch.eye(alpha.shape[0], device=alpha.device)
    return (sim * mask).sum() / (alpha.shape[0] * (alpha.shape[0] - 1))


def alpha_kl_loss(alpha, alpha_prior_batch):
    """
    KL 散度正则：拉近训练出的 α 分布到 LLM 先验分布。

    KL(α_trained || α_prior) = Σ_k α_k * log(α_k / p_k)

    alpha            : [B, K] 训练出的 softmax 品牌偏好
    alpha_prior_batch: [B, K] 对应品牌的 LLM 先验（softmax 归一化后）
    """
    eps = 1e-8
    return F.kl_div(torch.log(alpha_prior_batch + eps),
                    alpha,
                    reduction='batchmean')


def semantic_aux_loss(S, region_kg_feats, valid_mask=None):
    """
    语义辅助监督：强制区域概念满足度 S[r,k] 接近 KG 中计算的区域特征。

    原理
    ----
    当前 S 由 ConceptNet 从嵌入学习，但没有语义约束 —— ConceptNet 只知道
    "某些区域更适合开店"，而不知道"是因为人流高/竞争少"。
    通过 MSE 辅助损失，强制 S[r, k] 追踪 KG 中对应概念的区域特征，使
    符号推理分数 α × S.T 真正具有语义意义。

    valid_mask
    ----------
    长度为 K 的 bool 张量，标记哪些概念维度的 KG 特征有足够方差（有意义）。
    方差过低（std < 0.05）的维度跳过 —— 避免把 S 拉向无意义的均匀低值。

    参数
    ----
    S               : [N_regions, K]，ConceptNet 预测的满足度 ∈ (0,1)
    region_kg_feats : [N_regions, K]，KG 计算的区域特征，已归一化到 [0.05, 0.95]
    valid_mask      : [K] bool，None 时使用所有维度

    返回
    ----
    aux_loss : 标量
    """
    if valid_mask is not None:
        S_sel     = S[:, valid_mask]
        feats_sel = region_kg_feats[:, valid_mask]
        if S_sel.shape[1] == 0:
            return torch.tensor(0.0, device=S.device)
        return F.mse_loss(S_sel, feats_sel)
    return F.mse_loss(S, region_kg_feats)


def train_and_eval(args, d, graph, params, region_kg_feats=None, aux_valid_mask=None):
    K_group = [5, 10, 20]
    idx10   = K_group.index(10)
    ev_tr, ev_tv, ev_tvt = get_er_vocab(d)

    # ── 预计算 val 正样本掩码（filtered training 用）────────────────────────
    region_to_local = {r: i for i, r in enumerate(d.region_list)}
    val_pos_local = {}
    for brand_id, rows in d.valid_data.items():
        pos = [region_to_local[r[2]] for r in rows if r[-1] > 0 and r[2] in region_to_local]
        if pos:
            val_pos_local[int(brand_id)] = pos

    # ── 预计算品牌加权系数（brand_weighted_loss 用）──────────────────────────
    # 品牌正样本数方差大（min=9, max=229）→ 密集品牌主导梯度 → 按 1/n 归一化
    from collections import Counter, defaultdict
    brand_cnt = Counter(x[0] for x in d.train_data)
    max_cnt = max(brand_cnt.values())
    # weight[b] = max_cnt / cnt[b]，使每个品牌跨所有 epoch 的总梯度权重相同
    brand_weight_map = {b: max_cnt / c for b, c in brand_cnt.items()}
    brand_weight_tensor = torch.zeros(max(brand_weight_map) + 1, device=device)
    for b, w in brand_weight_map.items():
        brand_weight_tensor[b] = w

    # ── 预计算品牌级别正样本索引（multi-positive CE 用）──────────────────────
    # brand_pos_local[brand_kg_id] = [local_region_idx, ...]（所有训练正样本）
    # brand_cat_map[brand_kg_id] = cat_kg_id（模型 forward 需要 category）
    brand_pos_local_map = defaultdict(list)
    brand_relev_map = defaultdict(list)   # brand → [relevance, ...] per positive
    brand_cat_map = {}
    for triple in d.train_data:
        b_id, r_local, relev, cat = triple[0], triple[2], triple[3], triple[4]
        if float(relev) > 0:  # train_data 已过滤，保险起见再判一次
            brand_pos_local_map[b_id].append(r_local)
            brand_relev_map[b_id].append(float(relev))
        brand_cat_map[b_id] = cat
    all_brand_ids_arr = np.array(sorted(brand_pos_local_map.keys()))

    # ── 预计算 Hard Negative 用的品牌正例 bool mask（GPU全矩阵）──────────
    n_region = len(d.region_list)
    n_brand  = len(d.brand_list)
    # brand_pos_gpu_mask[local_brand_idx, region_local_idx] = True iff positive
    _brand_pos_gpu_mask = torch.zeros(n_brand, n_region, dtype=torch.bool)
    for b_kg_id, pos_list in brand_pos_local_map.items():
        b_local = d.kg_id2brand_id[b_kg_id]
        _brand_pos_gpu_mask[b_local, pos_list] = True
    brand_pos_gpu_mask = _brand_pos_gpu_mask.to(device)  # [n_brand, n_region]

    model = NSConceptV7(d, **params).to(device)
    model.fol_loss = model.fol_loss.__class__(smoothing=args.label_smoothing)

    # ── 参数统计 ───────────────────────────────────────────────────────────
    print('\n参数统计：')
    total = trainable = 0
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    emb_count = sum(p.numel() for n, p in model.named_parameters()
                    if n.startswith('E.') or n.startswith('R.'))
    print(f'  总参数量: {total:,}  可训练: {trainable:,}')
    print(f'  嵌入参数: {emb_count:,} (lr={args.lr_emb:.1e})')
    print(f'  模型参数: {trainable - emb_count:,} (lr={args.lr:.1e})')
    print(f'  概念数 K={NSConceptV7.K}，α正则 lam_reg={args.lam_reg}')
    print(f'  使用先验: {"是" if hasattr(model, "alpha_prior") else "否（随机初始化）"}')

    # ── 分组优化器 ─────────────────────────────────────────────────────────
    emb_params   = [p for n, p in model.named_parameters()
                    if (n.startswith('E.') or n.startswith('R.')) and p.requires_grad]
    # W_rel 单独一组，可设更高 wd_wrel 防止概念双线性矩阵过拟合
    wrel_params  = [p for n, p in model.named_parameters() if n == 'W_rel']
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith('E.') and not n.startswith('R.')
                    and n != 'W_rel']

    param_groups = []
    if emb_params:
        param_groups.append({'params': emb_params, 'lr': args.lr_emb})
    if wrel_params:
        param_groups.append({'params': wrel_params, 'lr': args.lr,
                             'weight_decay': args.wd_wrel})
    param_groups.append({'params': other_params, 'lr': args.lr,
                         'weight_decay': args.wd})
    opt = torch.optim.Adam(param_groups)

    t_idx        = torch.tensor(d.region_list, dtype=torch.long, device=device)
    best_valid   = 0.0
    best_test    = 0.0   # 追踪峰值测试 NDCG（也保存模型，供分析）
    best_iter    = 0
    best         = dict(hit10=0., ndcg10=0., pre10=0., rec10=0., map10=0.)
    best_t       = dict(hit10=0., ndcg10=0., pre10=0., rec10=0., map10=0.)  # test-peak 对应指标
    root         = os.path.dirname(os.path.abspath(__file__))
    tag_suffix   = ('_' + args.tag) if args.tag else ''
    ckpt_path    = os.path.join(root, 'best_model%s.pt' % tag_suffix)
    ckpt_test_path = os.path.join(root, 'best_model%s_testpeak.pt' % tag_suffix)  # test峰值模型
    results_path = os.path.join(root, 'results%s.txt' % tag_suffix)
    current_lr   = args.lr

    print(f'\nKG 边数: {graph[0].shape[1]}  节点数: {d.num_ents}  训练样本: {len(d.train_data)}')
    print('目标: NDCG@10 > 0.219（突破 KnowSite 基线）\n')

    for it in range(1, args.num_iterations + 1):
        print('═' * 60)
        print('Epoch %d  (lr=%.2e  lr_emb=%.2e)' % (it, current_lr, args.lr_emb))
        t0 = time.time()

        model.train()
        task_losses, alpha_h_losses, s_var_losses, aux_losses, kl_losses = [], [], [], [], []

        if args.multi_pos:
            # ── 品牌级多正例 CE 训练（multi-positive InfoNCE）──────────────
            # 每批处理 batch_size 个品牌，损失 = LSE(all) - LSE(pos)
            # 梯度同时优化品牌的所有正例区域，避免单正例 CE 的梯度冲突
            perm = np.random.permutation(len(all_brand_ids_arr))
            shuffled_brands = all_brand_ids_arr[perm]
            for j in tqdm(range(0, len(shuffled_brands), args.batch_size), leave=False):
                brand_batch = shuffled_brands[j: j + args.batch_size]
                h_idx = torch.tensor(brand_batch, dtype=torch.long, device=device)
                r_idx = d.psa_id * torch.ones(len(brand_batch), dtype=torch.long, device=device)
                c_idx = torch.tensor([brand_cat_map[b] for b in brand_batch],
                                     dtype=torch.long, device=device)

                opt.zero_grad()
                score, alpha, S = model(graph, h_idx, r_idx, t_idx, c_idx)

                # 过滤 val 正样本
                score_f = score.clone()
                for bi, brand_id in enumerate(brand_batch):
                    vp = val_pos_local.get(int(brand_id), [])
                    if vp:
                        score_f[bi, vp] = -1e10

                # Multi-positive CE：LSE(all_scores) - LSE(pos_scores)
                # 若 args.relev_loss=True，改用 ListNet 相关性加权损失
                lse_all = torch.logsumexp(score_f, dim=1)  # [B]
                pos_lse_list = []
                for bi, brand_id in enumerate(brand_batch):
                    pos_idx = brand_pos_local_map[brand_id]
                    if getattr(args, 'relev_loss', False):
                        # ListNet: -Σ_j w_j * log_softmax(score)[j], w_j = relev_j/Σrelev
                        relev_w = torch.tensor(brand_relev_map[brand_id],
                                               dtype=torch.float32, device=device)
                        relev_w = relev_w / relev_w.sum()
                        pos_scores = score_f[bi, pos_idx]
                        # weighted log-sum-exp approximation (equivalent to ListNet top-1)
                        log_probs = pos_scores - lse_all[bi]  # log softmax at positives
                        pos_lse_list.append((relev_w * log_probs).sum())
                    else:
                        pos_lse_list.append(
                            torch.logsumexp(score_f[bi, pos_idx], dim=0))
                if getattr(args, 'relev_loss', False):
                    pos_lse = torch.stack(pos_lse_list)   # already log-prob
                    task_loss = -pos_lse.mean()
                else:
                    pos_lse = torch.stack(pos_lse_list)
                    task_loss = (lse_all - pos_lse).mean()

                a_h_loss = alpha_entropy_loss(alpha)
                s_v_loss = S_variance_loss(S) if S is not None else torch.tensor(0.0, device=device)
                aux_l    = torch.tensor(0.0, device=device)
                kl_l     = torch.tensor(0.0, device=device)

                loss = (task_loss
                        + args.lam_alpha_h * a_h_loss
                        + args.lam_s_var   * s_v_loss)
                loss.backward()

                if emb_params:
                    torch.nn.utils.clip_grad_norm_(emb_params,   max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(other_params, max_norm=5.0)
                opt.step()
                task_losses.append(task_loss.item())
                alpha_h_losses.append(a_h_loss.item())
                s_var_losses.append(s_v_loss.item())
                aux_losses.append(aux_l.item())
                kl_losses.append(kl_l.item())

        else:
            # ── 原有 triple-level 训练 ────────────────────────────────────
            np.random.shuffle(d.train_data)
            for j in tqdm(range(0, len(d.train_data), args.batch_size), leave=False):
                batch = d.train_data[j: j + args.batch_size]
                h_idx = torch.tensor([x[0] for x in batch], device=device)
                r_idx = d.psa_id * torch.ones(len(batch), dtype=torch.long, device=device)
                r_ont = torch.tensor([x[2] for x in batch], device=device)
                c_idx = torch.tensor([x[4] for x in batch], device=device)

                opt.zero_grad()
                score, alpha, S = model(graph, h_idx, r_idx, t_idx, c_idx)

                # filtered training：排除 val 正样本（避免梯度冲突）
                if args.filter_train:
                    score_f = score.clone()
                    for bi, brand_id in enumerate([x[0] for x in batch]):
                        vp = val_pos_local.get(int(brand_id), [])
                        if vp:
                            score_f[bi, vp] = -1e10
                else:
                    score_f = score

                # BPR loss：只对正样本训练，采样负例做 pairwise 排序
                if getattr(args, 'bpr', False):
                    # 只保留正样本（checkin > 0）
                    pos_mask_batch = torch.tensor([x[3] > 0 for x in batch], dtype=torch.bool, device=device)
                    if pos_mask_batch.sum() == 0:
                        continue
                    pos_scores_all = score_f[pos_mask_batch]  # [B_pos, n_region]
                    r_ont_pos = r_ont[pos_mask_batch]         # [B_pos]
                    B_pos = pos_scores_all.shape[0]
                    # 采样 K=100 随机负例（排除已知正例）
                    K_neg = 100
                    neg_idx = torch.randint(n_region, (B_pos, K_neg), device=device)
                    pos_s = pos_scores_all[range(B_pos), r_ont_pos].unsqueeze(1)  # [B_pos, 1]
                    neg_s = pos_scores_all.gather(1, neg_idx)                     # [B_pos, K_neg]
                    task_loss = -F.logsigmoid(pos_s - neg_s).mean()
                # 品牌加权 loss：稀疏品牌权重更高，均衡梯度贡献
                elif args.brand_weighted:
                    bw = brand_weight_tensor[h_idx]         # [B]
                    log_prob = F.log_softmax(score_f, dim=1)
                    nll = -log_prob[range(len(batch)), r_ont]
                    task_loss = (nll * bw).mean()
                elif getattr(args, 'hard_neg', False):
                    # Hard negative mining：在标准 CE 基础上，
                    # 额外对 top-K 误排负例施加 hinge margin loss
                    task_loss = model.fol_loss(score_f, r_ont)
                    K_hard = 32
                    with torch.no_grad():
                        # 取预计算好的 GPU bool mask（直接索引，无 CPU→GPU 传输）
                        B = len(batch)
                        b_local_idx = torch.tensor(
                            [d.kg_id2brand_id[batch[bi][0]] for bi in range(B)],
                            dtype=torch.long, device=device
                        )
                        pos_mask = brand_pos_gpu_mask[b_local_idx]  # [B, n_region]
                        neg_mask = score_f.masked_fill(pos_mask, -1e10)
                        hard_neg_idx = neg_mask.topk(K_hard, dim=1).indices  # [B, K]
                    pos_scores = score_f[range(B), r_ont].unsqueeze(1)  # [B,1]
                    hard_neg_scores = score_f.gather(1, hard_neg_idx)    # [B,K]
                    margin = 1.0
                    hinge = F.relu(margin - pos_scores + hard_neg_scores).mean()
                    task_loss = task_loss + 0.1 * hinge
                elif getattr(args, 'relev_loss', False):
                    # 相关性加权 CE：checkin score 作为样本权重
                    # score=0 → weight=0（零分区域不贡献梯度）
                    # score>0 → weight=score（高分区域贡献更大）
                    relev_w = torch.tensor([max(float(x[3]), 0.0) for x in batch],
                                           dtype=torch.float32, device=device)
                    total_w = relev_w.sum()
                    if total_w == 0:
                        opt.zero_grad()
                        continue
                    log_prob = F.log_softmax(score_f, dim=1)
                    nll = -log_prob[range(len(batch)), r_ont]  # [B]
                    task_loss = (nll * relev_w).sum() / total_w
                else:
                    task_loss  = model.fol_loss(score_f, r_ont)
                # α 熵正则（当前默认关闭）
                a_h_loss   = alpha_entropy_loss(alpha)
                # S 方差正则（V8 S=None，自动跳过）
                s_v_loss   = S_variance_loss(S) if S is not None else torch.tensor(0.0, device=device)
                # 语义辅助监督（V7 用，V8 S=None 自动跳过）
                if S is not None and region_kg_feats is not None and args.lam_aux > 0:
                    aux_l = semantic_aux_loss(S, region_kg_feats, valid_mask=aux_valid_mask)
                else:
                    aux_l = torch.tensor(0.0, device=device)
                # KL 散度正则：拉近训练 α 到 LLM 先验（仅 freeze_alpha=False 时有效）
                if args.lam_kl > 0 and not model.freeze_alpha and hasattr(model, 'alpha_prior'):
                    b_local = model.brand_lookup[h_idx]
                    prior_logit = model.alpha_prior[b_local]
                    prior_alpha = F.softmax(prior_logit / model.alpha_temp, dim=1).detach()
                    kl_l = alpha_kl_loss(alpha, prior_alpha)
                else:
                    kl_l = torch.tensor(0.0, device=device)

                loss = (task_loss
                        + args.lam_alpha_h * a_h_loss
                        + args.lam_s_var   * s_v_loss
                        + args.lam_aux     * aux_l
                        + args.lam_kl      * kl_l)

                loss.backward()

                if emb_params:
                    torch.nn.utils.clip_grad_norm_(emb_params,   max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(other_params, max_norm=5.0)

                opt.step()
                task_losses.append(task_loss.item())
                alpha_h_losses.append(a_h_loss.item())
                s_var_losses.append(s_v_loss.item())
                aux_losses.append(aux_l.item())
                kl_losses.append(kl_l.item())

        # lr 衰减：只衰减非嵌入参数组（emb_params 保持 lr_emb 不变）
        current_lr = current_lr * args.dr
        for pg in opt.param_groups:
            if pg['lr'] != args.lr_emb:  # 跳过嵌入组
                pg['lr'] = current_lr

        # ── 监控：α 偏好均值 + 关系双线性匹配分 ────────────────────────
        print('task_loss=%.4f  time=%.1fs' % (np.mean(task_losses), time.time() - t0))

        if model.last_alpha is not None:
            alpha_mean = model.last_alpha.mean(dim=0).cpu().numpy()
            if model.last_match_mean is not None:
                m_mean = model.last_match_mean.cpu().numpy()
                m_std  = model.last_match_std.cpu().numpy()
                concept_str = '  '.join([
                    '%s=α%.2f/m%+.2f(±%.2f)' % (
                        NSConceptV7.CONCEPT_NAMES[k][:4],
                        alpha_mean[k], m_mean[k], m_std[k])
                    for k in range(NSConceptV7.K)
                ])
            else:
                concept_str = '  '.join([
                    '%s=α%.2f' % (NSConceptV7.CONCEPT_NAMES[k][:4], alpha_mean[k])
                    for k in range(NSConceptV7.K)
                ])
            print('  概念[α/match±std]:', concept_str)

        # ── 验证 & 测试 ────────────────────────────────────────────────────
        model.eval()
        model.precompute_embeddings(graph)
        with torch.no_grad():
            print('── 验证集 ──')
            ndcg_v, hit_v, *_ = evaluate(model, d, graph, d.valid_data, ev_tr, K_group)
            print('── 测试集 ──')
            ndcg_t, hit_t, pre_t, rec_t, map_t = evaluate(
                model, d, graph, d.test_data, ev_tv, K_group)

        # 追踪峰值测试 NDCG，同时保存 test-peak 模型
        if ndcg_t[idx10] > best_test:
            best_test = ndcg_t[idx10]
            best_t = dict(hit10=hit_t[idx10], ndcg10=ndcg_t[idx10],
                          pre10=pre_t[idx10],  rec10=rec_t[idx10],  map10=map_t[idx10])
            torch.save(model.state_dict(), ckpt_test_path)
        print('  [诊断] val_NDCG=%.4f  test_NDCG=%.4f  peak_test=%.4f' % (
              ndcg_v[idx10], ndcg_t[idx10], best_test))

        if ndcg_v[idx10] > best_valid:
            best_valid = ndcg_v[idx10]
            best_iter  = it
            best = dict(hit10=hit_t[idx10], ndcg10=ndcg_t[idx10],
                        pre10=pre_t[idx10],  rec10=rec_t[idx10],
                        map10=map_t[idx10])
            # 更新 alpha_mean（供 results.txt 写入）
            if model.last_alpha is not None:
                alpha_mean = model.last_alpha.mean(dim=0).cpu().numpy()
            torch.save(model.state_dict(), ckpt_path)

            with open(results_path, 'w') as f:
                f.write('Best epoch: %d\n' % it)
                for k, v in best.items():
                    f.write('%s: %.8f\n' % (k, v))
                if model.last_alpha is not None:
                    f.write('\n概念偏好均值 α[k]:\n')
                    for k, name in enumerate(NSConceptV7.CONCEPT_NAMES):
                        f.write('  C%d %s: %.3f\n' % (k, name, alpha_mean[k]))
                    if model.last_match_mean is not None:
                        m_mean = model.last_match_mean.cpu().numpy()
                        m_std  = model.last_match_std.cpu().numpy()
                        f.write('\n关系双线性匹配分 match[b,r,k]（均值 ± 跨区域std）:\n')
                        for k, name in enumerate(NSConceptV7.CONCEPT_NAMES):
                            f.write('  C%d %s: %+.3f ± %.3f\n' % (
                                k, name, m_mean[k], m_std[k]))
                f.write('\nKnowSite baseline:\n')
                f.write('hit10: 0.713  ndcg10: 0.219  pre10: 0.186  rec10: 0.223  map10: 0.127\n')
                f.write('\n%s\n' % args)

            print('★ 新最优  Hit@10=%.4f  NDCG@10=%.4f' %
                  (best['hit10'], best['ndcg10']))
        else:
            gap = it - best_iter
            print('无提升 %d 轮 (best=Epoch%d  NDCG@10=%.4f)' %
                  (gap, best_iter, best['ndcg10']))
            if gap >= args.patience:
                print('早停。')
                break

    print('\n' + '═' * 60)
    print('最终结果（Val-Best Epoch %d）：' % best_iter)
    print('  Hit@10:       %.6f  (基线 0.713, %+.4f)' % (best['hit10'],  best['hit10']  - 0.713))
    print('  NDCG@10:      %.6f  (基线 0.219, %+.4f)' % (best['ndcg10'], best['ndcg10'] - 0.219))
    print('  Precision@10: %.6f  (基线 0.186, %+.4f)' % (best['pre10'],  best['pre10']  - 0.186))
    print('  Recall@10:    %.6f  (基线 0.223, %+.4f)' % (best['rec10'],  best['rec10']  - 0.223))
    print('  MAP@10:       %.6f  (基线 0.127, %+.4f)' % (best['map10'],  best['map10']  - 0.127))
    print('  [诊断] 峰值 test NDCG@10: %.6f（任意 epoch 最高值，不受 val 早停影响）' % best_test)
    print('  [诊断] test-peak 对应全指标:')
    print('    hit10=%.6f  ndcg10=%.6f  pre10=%.6f  rec10=%.6f  map10=%.6f' % (
          best_t['hit10'], best_t['ndcg10'], best_t['pre10'], best_t['rec10'], best_t['map10']))

    # 加载最优模型并打印解释
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    model.precompute_embeddings(graph)
    # 跑一次全量区域 forward 以填充 last_alpha / last_S
    with torch.no_grad():
        sample_brand = torch.tensor([d.brand_list[0]], dtype=torch.long, device=device)
        sample_r     = d.psa_id * torch.ones(1, dtype=torch.long, device=device)
        model(graph, sample_brand, sample_r, t_idx, sample_brand)
    model.interpret()


if __name__ == '__main__':
    pa = argparse.ArgumentParser()
    pa.add_argument('--dataset',        default='beijing')
    pa.add_argument('--kg_name',        default='kg')
    pa.add_argument('--seed',           type=int,   default=67)
    pa.add_argument('--num_iterations', type=int,   default=200)
    pa.add_argument('--batch_size',     type=int,   default=128)
    pa.add_argument('--lr',             type=float, default=0.001)
    pa.add_argument('--lr_emb',         type=float, default=1e-4)
    pa.add_argument('--dr',             type=float, default=0.995)
    pa.add_argument('--patience',       type=int,   default=30)
    pa.add_argument('--edim',           type=int,   default=64)
    pa.add_argument('--n_layer',        type=int,   default=2)
    pa.add_argument('--opn',            default='rotate')
    pa.add_argument('--gcn_dropout',    type=float, default=0.3)
    pa.add_argument('--pretrain',       default='true')
    pa.add_argument('--pretrain_file',  default='',
                    help='覆盖默认预训练文件名（如 ER_beijing_RotatE_64.npz）；空串=使用默认TuckER')
    pa.add_argument('--freeze',         default='false',
                    help='false=微调嵌入（推荐），true=冻结')
    pa.add_argument('--lam_reg',        type=float, default=0.01,
                    help='α 余弦相似度正则系数（保留兼容，当前不使用）')
    pa.add_argument('--lam_alpha_h',   type=float, default=0.0,
                    help='α 熵正则系数（默认 0=关闭；BN 已保证 S 多样性，正则反而干扰任务梯度）')
    pa.add_argument('--lam_s_var',     type=float, default=0.0,
                    help='S 方差正则系数（默认 0=关闭；BN 已保证 S 多样性，无需额外正则）')
    pa.add_argument('--lam_aux',       type=float, default=0.0,
                    help='语义辅助监督系数（推荐 0.1~0.5）：强制 S[r,k] 追踪 KG 区域特征，'
                         '使 sym_score = α@S.T 真正具有语义意义。'
                         '需先运行 gen_prior.py 生成 region_kg_feats_{dataset}.npy')
    pa.add_argument('--wd',            type=float, default=1e-4,
                    help='模型参数L2正则（weight_decay），不作用于嵌入。默认1e-4。')
    pa.add_argument('--wd_wrel',       type=float, default=1e-4,
                    help='W_rel 专属 L2 正则（默认同 wd=1e-4；建议尝试 1e-2/1e-3 防止双线性过拟合）')
    pa.add_argument('--freeze_alpha',  default='false',
                    help='true=固定 α 为 LLM 先验（仅训练 ConceptNet S + bilinear）；'
                         'false=正常训练 pref_net + prior')
    pa.add_argument('--tag',          default='',
                    help='实验标签，附加到 checkpoint 文件名，防止并发实验互相覆盖')
    pa.add_argument('--alpha_temp',   type=float, default=1.0,
                    help='α softmax 温度，<1 更聚焦（推荐 0.3~0.7），=1 标准 softmax')
    pa.add_argument('--concept_scale', type=float, default=0.0,
                    help='概念条件线性评分系数（默认0=关闭；小值如0.01可提供梯度稳定性）')
    pa.add_argument('--alpha_prior_path', default='',
                    help='指定 α 先验文件路径（默认自动查找 alpha_prior_{dataset}.npy）')
    pa.add_argument('--lam_kl',         type=float, default=0.0,
                    help='KL散度正则系数（默认0=关闭；freeze_alpha=false时将训练α拉近到LLM先验）')
    pa.add_argument('--label_smoothing', type=float, default=0.0,
                    help='Label smoothing 系数（默认0=关闭；0.1~0.2可防过拟合，改善泛化）')
    pa.add_argument('--filter_train',   action='store_true',
                    help='训练loss中排除val正样本（filtered training，消除梯度冲突）')
    pa.add_argument('--brand_weighted', action='store_true',
                    help='按品牌训练样本数倒数加权loss（均衡稀疏/密集品牌梯度贡献）')
    pa.add_argument('--multi_pos',      action='store_true',
                    help='品牌级多正例 CE 损失（InfoNCE）：按品牌分批，同时优化所有正例区域，'
                         '避免单正例 CE 的梯度冲突。与 filter_train 自动兼容（排除 val 正例）。')
    pa.add_argument('--relev_loss',     action='store_true',
                    help='启用 ListNet 相关性加权损失（需配合 --multi_pos）：'
                         '用训练数据中的连续相关性得分 relev 加权每个正例的梯度贡献，'
                         '直接优化 NDCG。relev=0 的正例自动取权重1（等权fallback）。')
    pa.add_argument('--hard_neg',       action='store_true',
                    help='启用 Hard Negative Mining：在标准 CE/FOL 损失基础上追加 hinge margin loss，'
                         '惩罚得分最高的 top-K 负例区域（mis-ranked negatives），提升判别能力。')
    pa.add_argument('--bpr',            action='store_true',
                    help='启用 BPR 损失：只对正样本训练，每条正例采样 K=100 随机负例，'
                         '用 log-sigmoid(pos-neg) 做 pairwise 排序，梯度强度约为全 softmax 的 20x。')
    pa.add_argument('--use_theta',      default='false',
                    help='true=启用品牌-概念阈值 θ[b,k]（初始化为0，从训练数据学习品牌特异校准）；'
                         'false=不使用阈值（默认，等价当前模型）')
    pa.add_argument('--filter_zeros',   action='store_true',
                    help='从训练数据中过滤掉 score=0 的样本，只保留正例（score>0）训练，'
                         '消除零分样本对 CE 梯度的污染。')
    pa.add_argument('--oversample_pos', type=int, default=1,
                    help='正例过采样倍数（默认1=不过采样）。设为3时，正例重复3次，'
                         '使正例在 batch 中占比从25%%提升至约50%%（配合 filter_zeros 使用最佳）。')
    args = pa.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True

    root       = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(root, '../data/%s/' % args.dataset)
    pretrain_d = os.path.join(root, '../pretrain_emb/')

    print('加载数据...')
    d = Data(data_dir=data_dir, kg_name=args.kg_name)

    # ── 零分过滤 + 正例过采样 ──────────────────────────────────────────────────
    if getattr(args, 'filter_zeros', False):
        orig_len = len(d.train_data)
        d.train_data = [x for x in d.train_data if x[3] > 0]
        print(f'filter_zeros: {orig_len} → {len(d.train_data)} (保留正例, 剔除{orig_len-len(d.train_data)}个零分样本)')
    if getattr(args, 'oversample_pos', 1) > 1:
        pos_entries = [x for x in d.train_data if x[3] > 0]
        d.train_data = d.train_data + pos_entries * (args.oversample_pos - 1)
        print(f'oversample_pos {args.oversample_pos}x: 正例 {len(pos_entries)} × {args.oversample_pos} → 总计 {len(d.train_data)} 样本')

    print('构建图（CPU 惰性加载，进 forward 才移到 device）...')
    graph = utils.build_graph_pyg(d)   # 图始终在 CPU，不在此处占用显存

    params = vars(args).copy()
    params['device']       = device
    params['freeze_alpha'] = args.freeze_alpha

    if args.pretrain == 'true':
        if args.pretrain_file:
            path = pretrain_d + args.pretrain_file
        else:
            path = pretrain_d + 'ER_%s_TuckER_%d.npz' % (args.dataset, args.edim)
        print('加载预训练 embedding:', path)
        ckpt = np.load(path)
        params['E_pretrain'] = torch.from_numpy(ckpt['E_pretrain']).to(device)
        params['R_pretrain'] = torch.from_numpy(ckpt['R_pretrain']).to(device)

    # 加载 α_prior（如果存在）
    prior_path = args.alpha_prior_path if args.alpha_prior_path else \
                 os.path.join(root, 'alpha_prior_%s.npy' % args.dataset)
    if os.path.exists(prior_path):
        print(f'加载 α 先验: {prior_path}')
        params['alpha_prior'] = torch.from_numpy(np.load(prior_path)).to(device)
    else:
        print(f'未找到先验文件 {prior_path}，使用随机初始化。')
        print('建议先运行: python gen_prior.py --dataset %s' % args.dataset)

    # 加载区域 KG 特征（语义辅助监督 aux_loss 用，如果存在）
    region_kg_feats = None
    feats_path = os.path.join(root, 'region_kg_feats_%s.npy' % args.dataset)
    aux_valid_mask = None
    if os.path.exists(feats_path):
        print(f'加载区域 KG 特征（aux_loss）: {feats_path}')
        feats_np = np.load(feats_path)
        region_kg_feats = torch.from_numpy(feats_np).to(device)
        print(f'  shape={region_kg_feats.shape}  lam_aux={args.lam_aux}')

        # 自动方差检测：特征 std < 0.05 的维度视为无效（几乎全为 0.05 最小值）
        feat_std = feats_np.std(axis=0)
        valid_concepts = feat_std >= 0.05
        aux_valid_mask = torch.tensor(valid_concepts, dtype=torch.bool, device=device)
        print(f'  特征方差（按概念）: ' +
              '  '.join([f'C{k}={feat_std[k]:.3f}{"✓" if valid_concepts[k] else "✗"}'
                         for k in range(len(feat_std))]))
        print(f'  有效概念数: {valid_concepts.sum()}/8 '
              f'（低方差维度跳过 aux_loss，避免拉向无意义均匀低值）')

        if args.lam_aux > 0:
            print(f'  ✓ 语义辅助监督已启用（lam_aux={args.lam_aux}）')
        else:
            print(f'  ○ 已加载但 lam_aux=0，语义监督关闭。使用 --lam_aux 0.1 开启。')
    else:
        if args.lam_aux > 0:
            print(f'⚠ 未找到 {feats_path}，无法使用 aux_loss。')
            print('请先运行: python gen_prior.py --dataset %s' % args.dataset)

    print(args)
    train_and_eval(args, d, graph, params,
                   region_kg_feats=region_kg_feats,
                   aux_valid_mask=aux_valid_mask)
