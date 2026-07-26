"""
VeCo: Brand-Adaptive Neuro-Symbolic Commercial Site Selection
=====================================================================================

设计哲学
--------
商业选址本质上是"品牌需求 ∩ 区域属性"的匹配问题。这要求两类完全不同的知识：
  - 区域属性（客观，可从 KG 数据感知）：朝阳区人流 0.85，商业集聚 0.91
  - 品牌需求（主观，需领域知识注入）：麦当劳要高人流，星巴克要商业氛围

纯神经方法（KnowSite）用 GRU 隐式学习品牌偏好，但每品牌平均仅 ~37 条门店记录，
数据量远不足以从零学好个性化策略，且结果完全不可解释。

VeCo explicitly separates two forms of knowledge:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Layer 1：神经感知（CompGCN）                                         │
  │   KG → 5 类关系专属区域嵌入（h_flow / h_comp / h_func / h_brand / h_cat）│
  │                                                                     │
  │ Layer 2：概念空间抽象                                                 │
  │   区域侧：S[r,k] = sigmoid(MLP_k(h_concept_k[r])) ∈ (0,1)          │
  │           "朝阳区在人流概念上的满足度 = 0.85"                         │
  │   品牌侧：α[b,k] = sigmoid(pref_net(ctx[b]) + α_prior[b,k]) ∈ (0,1) │
  │           "麦当劳对人流概念的偏好权重 = 0.90"（先验从 KG/LLM 注入）   │
  │                                                                     │
  │ Layer 3：符号推理（乘积 t-范数 / 软逻辑 AND）                         │
  │   sym(b,r) = Σ_k α[b,k] × log(S[r,k] + ε)                         │
  │   语义：区域 r 必须同时满足品牌 b 的所有关键需求                       │
  │   当 S[r,k]→0 且 α[b,k] 高 → 大惩罚（必要条件不满足）                │
  │                                                                     │
  │ 最终分数：score(b,r) = sym(b,r) + λ × bilinear(b,r)                 │
  └─────────────────────────────────────────────────────────────────────┘

8 个语义概念（与 KnowSite 8 条路径对齐）
  C0 TrafficFlow       → h_flow[r]               (FlowTransition/OD/BorderBy)
  C1 CompetitionField  → h_comp[r]               (Competitive/LocateAt)
  C2 FunctionalFit     → h_func[r]               (SimilarFunction/BAServe/NearBy)
  C3 CommercialCluster → x[r]                    (基础区域嵌入捕获商业区信息)
  C4 BrandAffinity     → h_brand[r]              (RelatedBrand/BrandOf)
  C5 CategoryConsumer  → h_cat[r]                (Brand2Cat/CatOf/BelongTo)
  C6 SpatialOpportunity → (h_flow[r]+x[r])/2     (人流×空间复合，扩张机会)
  C7 LifestyleMatch    → (h_func[r]+h_cat[r])/2  (功能×品类复合，生活方式)

为什么能超越 KnowSite（NDCG 0.219）
  1. α_prior 注入了训练数据里没有的品牌战略知识（KG统计/LLM）
  2. 乘积 t-范数比 MHA 加权平均更适合"必要条件"语义
  3. S[r,k] 天然稠密（从嵌入学，不依赖稀疏路径实例）
  4. 可解释：打印 α[b] 和 S[r] 直接解释推荐原因
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

from layers import MyCompGCN


# ── 损失函数 ───────────────────────────────────────────────────────────────

class FOLLoss(nn.Module):
    """交叉熵排序损失，支持 label smoothing 正则化。

    smoothing=0.0: 标准 cross-entropy（默认）
    smoothing>0.0: label smoothing，防止过度自信，提升泛化
    """
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, scores, target_idx):
        log_prob = F.log_softmax(scores, dim=1)
        nll = -log_prob[range(scores.shape[0]), target_idx]
        if self.smoothing > 0:
            smooth = -log_prob.mean(dim=1)
            return ((1 - self.smoothing) * nll + self.smoothing * smooth).mean()
        return nll.mean()


# ── 概念满足度网络（区域侧）────────────────────────────────────────────────

class ConceptNet(nn.Module):
    """
    单个语义概念的区域满足度网络。

    输入: 区域的关系专属嵌入（edim 维）
    输出: 区域在该概念上的满足度 ∈ (0,1)

    BatchNorm1d 加在第一层激活后：
      - 强制中间特征在区域间有零均值、单位方差
      - 防止所有区域的满足度同时收敛到相同高值（S 均匀坍塌）
      - 让 Sigmoid 工作在其最有判别力的线性区而非饱和区
    """
    def __init__(self, edim):
        super().__init__()
        hidden = edim // 2
        self.net = nn.Sequential(
            nn.Linear(edim, hidden),
            nn.BatchNorm1d(hidden),      # ← 新增：强制方差，防止 S 均匀高值
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)   # [N]


# ── 主模型 ────────────────────────────────────────────────────────────────

class NSConceptV7(nn.Module):

    K = 6   # V15：去掉C4(BrandAffinity)/C5(CategoryConsumer)，二者区域嵌入退化为零信号

    CONCEPT_NAMES = [
        'TrafficFlow',       # C0
        'CompetitionField',  # C1
        'FunctionalFit',     # C2
        'CommercialCluster', # C3
        'SpatialOpportunity',# C4 (原C6)
        'LifestyleMatch',    # C5 (原C7)
    ]

    # alpha_prior 原始列索引（K=8 中去掉 C4=4, C5=5 后的剩余列）
    PRIOR_COLS = [0, 1, 2, 3, 6, 7]

    def __init__(self, d, **kwargs):
        super().__init__()
        edim         = kwargs['edim']
        self.edim    = edim
        self.device  = kwargs['device']
        self.n_layer = kwargs['n_layer']

        # ── Embeddings ────────────────────────────────────────────────────
        if kwargs.get('pretrain', 'false') == 'true':
            freeze = kwargs.get('freeze', 'true') == 'true'
            self.E = nn.Embedding.from_pretrained(kwargs['E_pretrain'], freeze=freeze)
            self.R = nn.Embedding.from_pretrained(kwargs['R_pretrain'], freeze=freeze)
        else:
            self.E = nn.Embedding(len(d.ent2id), edim)
            self.R = nn.Embedding(len(d.rel2id), edim)
            xavier_normal_(self.E.weight)
            xavier_normal_(self.R.weight)

        # ── CompGCN（与 Round1 相同）────────────────────────────────────
        self.gcn_layers = nn.ModuleList([
            MyCompGCN(indim=edim, outdim=edim, nr=len(d.rel2id),
                      dropout=kwargs['gcn_dropout'], opn=kwargs['opn'])
            for _ in range(self.n_layer)
        ])

        # ── 5 类关系分组（与之前版本相同）─────────────────────────────────
        r = d.rel2id
        self.flow_rel_ids  = [r['rel_od'], r['rel_od_rev'], r['rel_borderby']]
        self.comp_rel_ids  = [r['rel_competitive'],
                              r['rel_locateat'],     r['rel_locateat_rev'],
                              r['rel_placestoreat'], r['rel_placestoreat_rev']]
        self.func_rel_ids  = [r['rel_baserve'], r['rel_baserve_rev'],
                              r['rel_simpoi'],  r['rel_nearby']]
        self.brand_rel_ids = [r['rel_relatedbrand'],
                              r['rel_brandof'], r['rel_brandof_rev']]
        self.cat_rel_ids   = [
            r['rel_brand2cat1'], r['rel_brand2cat1_rev'],
            r['rel_brand2cat2'], r['rel_brand2cat2_rev'],
            r['rel_brand2cat3'], r['rel_brand2cat3_rev'],
            r['rel_1_catof'],    r['rel_1_catof_rev'],
            r['rel_2_catof'],    r['rel_2_catof_rev'],
            r['rel_belongto'],   r['rel_belongto_rev'],
        ]

        # ── 关系专属融合投影 ───────────────────────────────────────────────
        self.proj_flow  = nn.Linear(2 * edim, edim)
        self.proj_comp  = nn.Linear(2 * edim, edim)
        self.proj_func  = nn.Linear(2 * edim, edim)
        self.proj_brand = nn.Linear(2 * edim, edim)
        self.proj_cat   = nn.Linear(2 * edim, edim)

        # ── Layer 2（V18）：恢复 W_rel 概念专属双线性投影 ──────────────────
        # match[b,r,k] = tanh((b_emb @ W_rel[k]) · h_rel_k[r] / scale / 3)
        # W_rel[k] ∈ R^{edim×edim}：将品牌嵌入投影到第k个概念的语义空间
        # 允许同一品牌在不同概念维度上以不同方式表达自己的偏好
        # tanh(./3) 控制量级（V8教训），softmax-α 控制概念选择（V17教训）
        self.W_rel = nn.Parameter(torch.empty(self.K, edim, edim))
        for k in range(self.K):
            xavier_normal_(self.W_rel[k])

        # ── 双线性残差项：确保模型不低于纯双线性基线 ─────────────────────
        # score = bilinear_scale * bilinear_score + ns_score
        # bilinear_scale 可学习，让模型自动平衡两部分量级
        self.W_bilinear = nn.Parameter(torch.empty(edim, edim))
        nn.init.xavier_normal_(self.W_bilinear)
        self.bilinear_scale = nn.Parameter(torch.tensor(1.0))

        # ── V20：概念条件线性评分（与 t-范数共享 W_rel）────────────────────
        self.concept_scale = float(kwargs.get('concept_scale', 0.0))

        # ── V21：OR 语义模糊逻辑评分 ─────────────────────────────────────────
        # sym_AND = Σ_k α_k·log(S_k)       ∈(-∞,0]  严格-任一概念不满足均被惩罚
        # sym_OR  = -Σ_k α_k·log(1-S_k)    ∈[0,+∞)  宽松-任一概念满足均获奖励
        # 神经-符号意义：AND 对应"充分条件"（必须满足所有概念），
        #              OR 对应"必要条件"（满足任一概念即可入围）
        # or_scale 可学习：数据自动决定各品牌需要多宽松的选址逻辑
        self.or_scale = nn.Parameter(torch.tensor(0.0))

        # ── V22：品牌-概念阈值 θ[b,k]（Brand-specific concept thresholds）──────
        # S_k(b,r) = sigmoid(z_k(b,r) - θ[b,k])
        # θ[b,k] 提升/降低品牌 b 对概念 k 的"入场门槛"
        # - θ>0: 品牌 b 对概念 k 要求高（只有 z_k>θ 才能 S_k 接近 1）
        # - θ<0: 品牌 b 对概念 k 要求低（z_k 较低时 S_k 也接近 1）
        # 初始化为 0（等价于当前无阈值模型），训练中从数据学习品牌特异校准
        # 参数规模: N_brands × K = 398 × 6 = 2388，正则化由 wd 控制
        n_brands = len(d.brand_list)
        use_theta = kwargs.get('use_theta', 'false') == 'true'
        if use_theta:
            self.theta = nn.Parameter(torch.zeros(n_brands, self.K))
            print(f'  [VeCo] use_theta=True：品牌概念阈值 θ[{n_brands},{self.K}]，初始化为零（等价当前模型）')
        else:
            self.register_buffer('theta', torch.zeros(n_brands, self.K))  # 固定为零（非参数）
        self.use_theta = use_theta

        # ── Layer 2：品牌偏好网络（品牌侧）────────────────────────────────
        # 输入: 品牌上下文 = concat(x[b], h_cat[b])
        # 输出: K 个概念的偏好 logit（加到 α_prior 上后经 sigmoid 得到 α）
        self.brand_ctx_mlp = nn.Linear(2 * edim, edim)
        self.pref_net      = nn.Sequential(
            nn.Linear(edim, edim // 2),
            nn.ReLU(),
            nn.Linear(edim // 2, self.K),
        )
        for m in list(self.brand_ctx_mlp.modules()) + list(self.pref_net.modules()):
            if isinstance(m, nn.Linear):
                xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # ── V10：区域概念激活网络 ──────────────────────────────────────────
        # 输入: 区域的K个关系嵌入均值 [edim]
        # 输出: 区域在各概念上的激活logit [K]
        # 与品牌侧brand_logit相加后softmax，得到区域感知的α[B,N,K]
        self.region_concept_net = nn.Sequential(
            nn.Linear(edim, edim // 2),
            nn.ReLU(),
            nn.Linear(edim // 2, self.K),
        )
        for m in self.region_concept_net.modules():
            if isinstance(m, nn.Linear):
                xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # ── freeze_alpha 模式：固定 α = sigmoid(prior)，不经过 pref_net ─────
        # 用于"LLM 先验 + 仅训练 S"消融实验，直接测试 LLM α 的质量
        self.freeze_alpha = kwargs.get('freeze_alpha', 'false') == 'true'
        if self.freeze_alpha:
            print('  [VeCo] freeze_alpha=True：α 固定为 LLM 先验，仅训练 ConceptNet S + bilinear')

        # ── α 温度：控制 softmax 的锐度，越小越聚焦于核心概念 ─────────────
        # T=1.0（默认）：标准 softmax；T=0.5：更聚焦；T=2.0：更均匀
        self.alpha_temp = kwargs.get('alpha_temp', 1.0)
        print(f'  [VeCo] alpha_temp={self.alpha_temp}')

        # ── α_prior（可选，KG统计/LLM 初始化，logit 空间）───────────────
        # shape [N_brands, K]，在 gen_prior.py 中生成并以 .npy 形式保存
        alpha_prior = kwargs.get('alpha_prior', None)
        if alpha_prior is not None:
            # V15：若先验是 K=8 格式，截取本模型需要的列
            if alpha_prior.shape[1] != self.K:
                cols = torch.tensor(self.PRIOR_COLS, dtype=torch.long)
                alpha_prior = alpha_prior[:, cols]
            self.register_buffer('alpha_prior', alpha_prior)
            print(f'  [VeCo] 已加载 α 先验，shape={alpha_prior.shape}')
        else:
            print('  [VeCo] 未使用 α 先验（随机初始化）')

        # 品牌/区域 KG-id → 本地 id 查找表
        max_ent      = d.num_ents
        brand_lookup = torch.full((max_ent,), -1, dtype=torch.long)
        for kg_id, loc_id in d.kg_id2brand_id.items():
            brand_lookup[kg_id] = loc_id
        self.register_buffer('brand_lookup', brand_lookup)

        self.fol_loss = FOLLoss()

        # ── Eval 缓存 ─────────────────────────────────────────────────────
        self.register_buffer('x_cache',       None)
        self.register_buffer('h_flow_cache',  None)
        self.register_buffer('h_comp_cache',  None)
        self.register_buffer('h_func_cache',  None)
        self.register_buffer('h_brand_cache', None)
        self.register_buffer('h_cat_cache',   None)

        # 最近一批的 α 和 match 统计，供监控/解释输出
        self.last_alpha       = None   # [B, K]
        self.last_match_mean  = None   # [K]  各概念匹配分均值（跨品牌跨区域）
        self.last_match_std   = None   # [K]  各概念匹配分跨区域标准差（均值品牌）

    # ── 工具：关系聚合 ───────────────────────────────────────────────────

    def _relation_aggregate(self, x, edge_index, edge_type, edge_norm, rel_ids):
        dev  = edge_type.device
        mask = torch.zeros(edge_type.shape[0], dtype=torch.bool, device=dev)
        for rid in rel_ids:
            mask |= (edge_type == rid)
        if not mask.any():
            return torch.zeros_like(x)
        src  = edge_index[0][mask]
        dst  = edge_index[1][mask]
        norm = edge_norm[mask]
        msg  = x[src] * norm.unsqueeze(1)
        aggr = torch.zeros_like(x)
        aggr.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        return aggr

    def _fuse(self, x_base, x_rel, proj):
        return F.relu(proj(torch.cat([x_base, x_rel], dim=1)))

    def _to_device_graph(self, graph):
        """
        将图张量惰性移动到与模型参数相同的 device，并缓存结果。
        第一次调用时做 CPU→GPU 拷贝，后续直接返回缓存，避免每次 forward 重复拷贝。
        """
        if not hasattr(self, '_cached_graph') or self._cached_graph is None:
            dev = self.E.weight.device
            ei, et, en = graph
            self._cached_graph = (ei.to(dev), et.to(dev), en.to(dev))
        return self._cached_graph

    def _run_gcn(self, graph):
        edge_index, edge_type, edge_norm = self._to_device_graph(graph)
        x = self.E.weight
        r = self.R.weight
        for layer in self.gcn_layers:
            x, r = layer(x, edge_index, edge_type, edge_norm, r)
        return x

    def _get_fused_embeddings(self, graph):
        graph = self._to_device_graph(graph)
        edge_index, edge_type, edge_norm = graph
        x       = self._run_gcn(graph)
        h_flow  = self._fuse(x, self._relation_aggregate(x, edge_index, edge_type, edge_norm, self.flow_rel_ids),  self.proj_flow)
        h_comp  = self._fuse(x, self._relation_aggregate(x, edge_index, edge_type, edge_norm, self.comp_rel_ids),  self.proj_comp)
        h_func  = self._fuse(x, self._relation_aggregate(x, edge_index, edge_type, edge_norm, self.func_rel_ids),  self.proj_func)
        h_brand = self._fuse(x, self._relation_aggregate(x, edge_index, edge_type, edge_norm, self.brand_rel_ids), self.proj_brand)
        h_cat   = self._fuse(x, self._relation_aggregate(x, edge_index, edge_type, edge_norm, self.cat_rel_ids),   self.proj_cat)
        return x, h_flow, h_comp, h_func, h_brand, h_cat

    @torch.no_grad()
    def precompute_embeddings(self, graph):
        # 确保图在正确 device
        edge_index, edge_type, edge_norm = self._to_device_graph(graph)
        graph_dev = (edge_index, edge_type, edge_norm)
        x = self._run_gcn(graph_dev).detach()
        self.x_cache       = x
        self.h_flow_cache  = self._fuse(x, self._relation_aggregate(x, edge_index, edge_type, edge_norm, self.flow_rel_ids),  self.proj_flow).detach()
        self.h_comp_cache  = self._fuse(x, self._relation_aggregate(x, edge_index, edge_type, edge_norm, self.comp_rel_ids),  self.proj_comp).detach()
        self.h_func_cache  = self._fuse(x, self._relation_aggregate(x, edge_index, edge_type, edge_norm, self.func_rel_ids),  self.proj_func).detach()
        self.h_brand_cache = self._fuse(x, self._relation_aggregate(x, edge_index, edge_type, edge_norm, self.brand_rel_ids), self.proj_brand).detach()
        self.h_cat_cache   = self._fuse(x, self._relation_aggregate(x, edge_index, edge_type, edge_norm, self.cat_rel_ids),   self.proj_cat).detach()

    # ── Layer 2（V8）：关系专属区域嵌入 & 品牌-区域双线性匹配 ────────────

    def _get_rel_inputs(self, x, h_flow, h_comp, h_func, h_brand, h_cat, region_idx):
        """
        提取 8 个概念对应的区域关系专属嵌入列表。

        V15：K=6，去掉退化为零信号的 C4(h_brand) 和 C5(h_cat)。

          C0 TrafficFlow       : h_flow[r]
          C1 CompetitionField  : h_comp[r]
          C2 FunctionalFit     : h_func[r]
          C3 CommercialCluster : x[r]
          C4 SpatialOpportunity: (h_flow[r] + x[r]) / 2    (原C6)
          C5 LifestyleMatch    : (h_func[r] + h_cat[r]) / 2 (原C7)

        返回 list of [N, edim]，长度 K=6
        """
        r_x    = x[region_idx]
        r_flow = h_flow[region_idx]
        r_comp = h_comp[region_idx]
        r_func = h_func[region_idx]
        r_cat  = h_cat[region_idx]
        return [
            r_flow,                          # C0
            r_comp,                          # C1
            r_func,                          # C2
            r_x,                             # C3
            (r_flow + r_x) * 0.5,            # C4 (原C6)
            (r_func + r_cat) * 0.5,          # C5 (原C7)
        ]

    def _compute_S(self, b_emb, h_rels, theta_b=None):
        """
        V22：品牌-区域概念满足度矩阵 S[b,r,k] ∈ (0,1) 及原始 logit Z[b,r,k]

        z_k(b,r)  = (b_emb[b] @ W_rel[k]) · h_rels[k][r] / sqrt(edim)  ∈ (-∞, +∞)
        S_k(b,r)  = sigmoid(z_k - θ[b,k])                               ∈ (0, 1)
        θ[b,k]（品牌概念阈值）：提升(>0) 或降低(<0) 品牌 b 对概念 k 的"入场门槛"

        参数
        ----
        b_emb   : [B, edim]
        h_rels  : list of [N, edim]，K 个概念区域嵌入
        theta_b : [B, K] 品牌特异阈值（None = 不使用阈值，等价于 θ=0）

        返回
        ----
        S_matrix : [B, N, K]，概念满足度，∈ (0,1)
        Z_matrix : [B, N, K]，原始双线性得分，∈ (-∞, +∞)
        """
        scale = self.edim ** 0.5
        S_list, Z_list = [], []
        for k in range(self.K):
            q_k   = b_emb @ self.W_rel[k]          # [B, edim]，品牌在概念k的投影
            z_k   = q_k @ h_rels[k].t() / scale    # [B, N]，原始双线性得分
            if theta_b is not None:
                z_shifted = z_k - theta_b[:, k:k+1]   # [B, N] - [B, 1] = [B, N]
            else:
                z_shifted = z_k
            s_k   = torch.sigmoid(z_shifted)        # ∈ (0,1)，概念满足度（含阈值）
            S_list.append(s_k)
            Z_list.append(z_k)                      # Z 保留原始分（不减 θ）

        return torch.stack(S_list, dim=2), torch.stack(Z_list, dim=2)  # [B, N, K]

    # ── Layer 2：计算品牌概念偏好 α[B, K] ────────────────────────────────

    def _compute_alpha(self, x, h_cat, brand_idx):
        """
        品牌偏好向量 α[B, K]，每个元素 ∈ (0,1)。

        两种模式：
        A. freeze_alpha=True（消融实验）：
           α = softmax(alpha_prior)，完全由 LLM 先验决定，不经过 pref_net。
           此时只训练 bilinear，测试"LLM 先验 + 无 ConceptNet"的上限。

        B. freeze_alpha=False（正常训练）：
           pref_logit = pref_net(concat(x[b], h_cat[b])) + alpha_prior[b]
           α[b,k] = softmax(pref_logit[b,k])  ∈ (0,1)，sum_k α[b,k] = 1
           pref_net 学习对 LLM 先验的品牌特异调整，softmax 强制竞争选择。
        """
        b_local = self.brand_lookup[brand_idx]   # [B]  品牌本地 id

        if self.freeze_alpha:
            # ── 固定模式：完全用 LLM 先验 ────────────────────────────────
            if hasattr(self, 'alpha_prior'):
                alpha = F.softmax(self.alpha_prior[b_local], dim=1)  # [B, K]，sum=1
            else:
                # 无先验时退化为均匀分布
                alpha = torch.full((len(brand_idx), self.K), 1.0 / self.K,
                                   device=x.device, dtype=x.dtype)
            return alpha

        # ── 正常模式：pref_net + prior ─────────────────────────────────
        b_ctx      = F.relu(self.brand_ctx_mlp(
                        torch.cat([x[brand_idx], h_cat[brand_idx]], dim=1)))  # [B, edim]
        pref_logit = self.pref_net(b_ctx)   # [B, K]

        if hasattr(self, 'alpha_prior'):
            pref_logit = pref_logit + self.alpha_prior[b_local]

        return pref_logit  # [B, K] 原始logit；softmax在forward中执行（V11：品牌级别）

    # ── 前向传播（V19：乘积 t-范数 NS）──────────────────────────────────

    def forward(self, graph, h_idx, r_idx, t_idx, cat_idx):
        """
        V21 评分公式（AND + OR 混合模糊逻辑）：

          S[b,r,k]    = sigmoid(x[b] @ W_rel[k] @ h_rel_k[r].T / sqrt(d))      ∈(0,1)
          sym_AND     = Σ_k α_k·log(S_k+ε)          ∈(-∞,0]   [充分条件：全概念满足]
          sym_OR      = -Σ_k α_k·log(1-S_k+ε)       ∈[0,+∞)   [必要条件：任一满足]
          sym         = sym_AND + or_scale·sym_OR     [混合逻辑，or_scale 可学习]
          score[b,r]  = sym + concept_scale·neu + bilinear_scale·bilinear

        神经-符号设计：
          α 来自 LLM 先验（符号知识）；OR/AND 组合来自模糊逻辑（符号推理）；
          or_scale 由数据学习（神经部分），反映品牌选址的"宽容度"：
          - or_scale→0：便利店等需要"高流量且商业集聚"（AND）
          - or_scale→∞：奢侈品等满足任一关键特征即可（OR主导）
        """
        # 获取 CompGCN 嵌入
        if self.training:
            x, h_flow, h_comp, h_func, h_brand, h_cat = \
                self._get_fused_embeddings(graph)
        else:
            x, h_flow = self.x_cache, self.h_flow_cache
            h_comp, h_func = self.h_comp_cache, self.h_func_cache
            h_brand, h_cat = self.h_brand_cache, self.h_cat_cache

        # 品牌嵌入 & K 个概念的区域关系嵌入
        b_emb  = x[h_idx]                                                              # [B, edim]
        h_rels = self._get_rel_inputs(x, h_flow, h_comp, h_func, h_brand, h_cat, t_idx)  # list[K] of [N, edim]

        # V22：品牌本地 id（供 alpha/theta 查表）
        b_local = self.brand_lookup[h_idx]   # [B]

        # V22：品牌-概念阈值（use_theta=True 时为可学习参数，否则为零张量）
        theta_b = self.theta[b_local]        # [B, K]  θ[b,k]

        # V22：概念满足度矩阵 S[b,r,k] ∈ (0,1) + 原始双线性 Z[b,r,k]（含阈值偏移）
        S_matrix, Z_matrix = self._compute_S(b_emb, h_rels, theta_b=theta_b)           # [B, N, K]

        # 品牌概念偏好 α（softmax，sum=1，竞争选择）
        brand_logit = self._compute_alpha(x, h_cat, h_idx)  # [B, K] 原始logit
        if self.freeze_alpha:
            logit = self.alpha_prior[b_local]                                # [B, K] 原始logit
            alpha = F.softmax(logit / self.alpha_temp, dim=1)               # 温度缩放后softmax
        else:
            alpha = F.softmax(brand_logit / self.alpha_temp, dim=1)         # [B, K]，sum=1
        alpha_3d = alpha.unsqueeze(1)                        # [B, 1, K]

        # ── V21 符号模糊逻辑：AND + OR 混合 ─────────────────────────────────
        EPS = 1e-6
        # AND（乘积 t-范数）：所有概念都须满足 → 严格充分条件
        # sym_AND[b,r] = Σ_k α_k·log(S_k+ε) ∈(-∞,0]，好区域→接近0
        log_S    = torch.log(S_matrix + EPS)                  # [B, N, K]
        sym_and  = (alpha_3d * log_S).sum(dim=2)              # [B, N]

        # OR（概率求和 t-余范数）：任一概念满足即获奖励 → 宽松必要条件
        # sym_OR[b,r] = -Σ_k α_k·log(1-S_k+ε) ∈[0,+∞)，高S→高OR分
        # 直觉：不要求某个品牌在所有概念上都强——高traffic OR 高商业集聚 均可
        log_1mS  = torch.log(1.0 - S_matrix + EPS)           # [B, N, K]
        sym_or   = -(alpha_3d * log_1mS).sum(dim=2)          # [B, N]，∈[0,+∞)

        sym_score = sym_and + self.or_scale * sym_or          # 混合 AND+OR

        # 概念条件线性（V20 梯度稳定项）
        neu_score = (alpha_3d * Z_matrix).sum(dim=2)         # [B, N]

        # 全局双线性残差项（神经部分）
        bilinear_score = (b_emb @ self.W_bilinear) @ x[t_idx].t() / (self.edim ** 0.5)

        score = sym_score + self.concept_scale * neu_score + self.bilinear_scale * bilinear_score

        # 缓存监控信息（S 对应原来的 match）
        self.last_alpha = alpha.detach()
        with torch.no_grad():
            s = S_matrix.detach()                      # [B, N, K]
            self.last_match_mean = s.mean(dim=(0, 1))  # [K]，各概念平均满足度
            self.last_match_std  = s.std(dim=1).mean(dim=0)  # [K] per-brand std, avg

        return score, alpha, None   # 第三项 None 兼容 train.py 的 (score, alpha, S) 解包

    # ── 解释输出 ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def interpret(self, brand_names=None, region_names=None, top_k=3):
        """打印品牌概念偏好和各概念匹配分统计，供人工验证。"""
        if self.last_alpha is None:
            print('请先运行 forward() 生成缓存。')
            return

        print('\n─ 品牌概念偏好 α[b,k]（批次均值）─')
        alpha_mean = self.last_alpha.mean(dim=0).cpu().numpy()
        for k, name in enumerate(self.CONCEPT_NAMES):
            bar = '█' * int(alpha_mean[k] * 20) + '░' * (20 - int(alpha_mean[k] * 20))
            print(f'  C{k} {name:20s}: {alpha_mean[k]:.3f} [{bar}]')

        if self.last_match_mean is not None:
            print('\n─ 关系双线性匹配分 match[b,r,k]（均值 ± 跨区域std）─')
            m_mean = self.last_match_mean.cpu().numpy()
            m_std  = self.last_match_std.cpu().numpy()
            for k, name in enumerate(self.CONCEPT_NAMES):
                print(f'  C{k} {name:20s}: {m_mean[k]:+.3f} ± {m_std[k]:.3f}')

        # 品牌最关注的概念
        top_concepts = alpha_mean.argsort()[::-1][:top_k]
        print(f'\n品牌最关注的 Top-{top_k} 概念：')
        for k in top_concepts:
            print(f'  → {self.CONCEPT_NAMES[k]} (α={alpha_mean[k]:.3f})')
