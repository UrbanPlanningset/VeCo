"""
gen_prior.py — 品牌概念偏好先验生成
====================================

从 KG 结构统计出每个品牌对 8 个语义概念的偏好先验 α_prior[N_brands, 8]。

8 个语义概念（与 KnowSite 的 8 条关系路径对齐）：
  C0 TrafficFlow       人流强度（FlowTransition/OD/BorderBy）
  C1 CompetitionField  竞争密度（品牌容忍竞争的能力）
  C2 FunctionalFit     区域功能契合（SimilarFunction/NearBy）
  C3 CommercialCluster 商业集聚（商业区覆盖程度）
  C4 BrandAffinity     友商协同（相关品牌聚集效应）
  C5 CategoryConsumer  目标客群密度（品类消费者分布）
  C6 SpatialOpportunity 空间扩张机会（市场渗透潜力）
  C7 LifestyleMatch    生活方式契合（功能+品类复合）

先验计算逻辑：
  - 竞争关系多的品牌 → 高 CompetitionField（主动进入竞争市场）
  - 关联品牌多的品牌 → 高 BrandAffinity（依赖友商生态）
  - 现有门店少的品牌 → 高 SpatialOpportunity（处于扩张初期）
  - 大品类品牌         → 高 CategoryConsumer（品类消费基础广）
  - 默认值             → 基于行业通识（人流、商业氛围普遍重要）

可扩展：将 prior_llm 部分替换为实际 LLM API 调用（见末尾注释）

用法：
  cd ns_site_deploy/ns_concept
  python gen_prior.py --dataset beijing
  → 生成 alpha_prior_beijing.npy
"""

import os
import argparse
import numpy as np
from collections import defaultdict


def _build_region_kgfeatures(d):
    """
    计算每个区域在 8 个概念维度上的 KG 邻域特征（原始计数）。

    返回 feat_matrix [N_regions, 8]（已按列归一化到 [0,1]）
    以及 region_id_to_local 映射。

    修复说明（v2）
    ─────────────
    C4 BrandAffinity 和 C5 CategoryConsumer 原来直接查区域节点的边，
    但 rel_relatedbrand / rel_brand2cat 只连接品牌节点，区域节点上无此类边，
    导致所有区域的 C4/C5 特征 ≈ 0（被 clip 到 0.05）。

    修复：通过"区域内已有品牌"作中转，计算二阶特征：
      C4：区域内品牌的关联品牌数之和（品牌生态丰富度）
      C5：区域内品牌覆盖的品类数（品类多样性）
      C7：C2（功能联系数）+ C5（品类覆盖数）的复合
    """
    r          = d.rel2id
    region_list = d.region_list
    region_set  = set(region_list)
    N_r         = len(region_list)
    region_id_to_local = {kg_id: i for i, kg_id in enumerate(region_list)}

    # 构建邻接表（双向）
    adj = defaultdict(lambda: defaultdict(set))
    for h_id, rel_id, t_id in d.kg_data:
        adj[rel_id][h_id].add(t_id)

    def get(rel_name):
        return r.get(rel_name, -1)

    feats = np.zeros((N_r, 8), dtype=np.float32)

    for i, reg in enumerate(region_list):
        # C0 TrafficFlow：OD / BorderBy 连接数（人流/通勤）
        feats[i, 0] = (len(adj[get('rel_od')][reg]) +
                       len(adj[get('rel_od_rev')][reg]) +
                       len(adj[get('rel_borderby')][reg]))

        # C1 CompetitionField：该区域内已有多少品牌（竞争密度代理）
        brands_in_reg = (adj[get('rel_locateat_rev')][reg] |
                         adj[get('rel_placestoreat_rev')][reg])
        feats[i, 1] = len(brands_in_reg)

        # C2 FunctionalFit：功能型 POI 连接数（BAServe/SimilarPOI/NearBy）
        feats[i, 2] = (len(adj[get('rel_baserve_rev')][reg]) +
                       len(adj[get('rel_simpoi')][reg]) +
                       len(adj[get('rel_nearby')][reg]))

        # C3 CommercialCluster：商业服务覆盖（baserve 正向连接）
        feats[i, 3] = len(adj[get('rel_baserve')][reg])

        # C4 BrandAffinity（二阶）：
        #   区域内品牌的关联品牌总数 —— 品牌生态丰富度
        #   rel_relatedbrand 连接品牌↔品牌，不直接连区域，须经品牌中转
        c4 = 0
        for b in brands_in_reg:
            c4 += len(adj[get('rel_relatedbrand')][b])
        feats[i, 4] = c4

        # C5 CategoryConsumer（二阶）：
        #   区域内品牌覆盖的不同品类数 —— 消费场景多样性
        #   rel_brand2cat / rel_belongto 连接品牌→品类，须经品牌中转
        cats_in_reg = set()
        for b in brands_in_reg:
            cats_in_reg.update(adj[get('rel_brand2cat1')][b])
            cats_in_reg.update(adj[get('rel_brand2cat2')][b])
            cats_in_reg.update(adj[get('rel_brand2cat3')][b])
            cats_in_reg.update(adj[get('rel_belongto')][b])
        feats[i, 5] = len(cats_in_reg)

        # C6 SpatialOpportunity：空间扩张机会（逆竞争密度：竞争少=机会大）
        feats[i, 6] = feats[i, 1]   # 先存竞争数，后面取逆

        # C7 LifestyleMatch：功能联系数 + 品类多样性（修复后）
        feats[i, 7] = feats[i, 2] + feats[i, 5]

    # C6 取逆（门店少的区域 → 扩张机会大）
    feats[:, 6] = feats[:, 6].max() - feats[:, 6]

    # 列归一化到 [0.05, 0.95]（保留区间内，避免 logit 发散）
    col_max = feats.max(axis=0)
    col_max[col_max == 0] = 1.0
    feats = feats / col_max
    feats = np.clip(feats, 0.05, 0.95)

    return feats, region_id_to_local


def compute_kg_prior(d):
    """
    数据驱动的品牌概念偏好先验。

    核心思想：品牌过去在哪种 KG 环境的区域开店，就说明品牌偏好那种区域特征。
    对每个品牌，取其训练集门店所在区域的 KG 特征均值作为偏好先验。
    这比手工启发式规则更可靠：直接从行为数据反推偏好。

    对于训练样本极少（< 3 条）的品牌，回退到全区域背景均值。

    参数
    ----
    d : Data 对象

    返回
    ----
    prior_logit : np.ndarray, shape [N_brands, 8]，logit 空间，值域约 [-1, 1]
    """
    r          = d.rel2id
    N_b        = len(d.brand_list)
    region_set = set(d.region_list)

    # ── Step1：计算每个区域的 8 维 KG 特征 ──────────────────────────────────
    feat_matrix, region_id_to_local = _build_region_kgfeatures(d)
    background = feat_matrix.mean(axis=0)          # [8] 全区域背景均值

    # ── Step2：品牌 → 训练门店区域映射 ───────────────────────────────────────
    brand_to_stores = defaultdict(set)
    for rec in d.train_data:
        b_kg    = rec[0]
        r_kg    = rec[1]   # rec[1]=region_kg_id, rec[2]=region_ont_id(local index)
        if r_kg in region_set:
            brand_to_stores[b_kg].add(r_kg)

    # ── Step3：为每个品牌计算偏好先验 ─────────────────────────────────────────
    prior = np.zeros((N_b, 8), dtype=np.float32)

    for b_local, b_kg in enumerate(d.brand_list):
        stores = brand_to_stores.get(b_kg, set())
        store_locals = [region_id_to_local[s] for s in stores
                        if s in region_id_to_local]

        if len(store_locals) >= 3:
            # 足够样本：用品牌实际门店位置的 KG 特征均值
            brand_profile = feat_matrix[store_locals].mean(axis=0)  # [8]
        elif len(store_locals) >= 1:
            # 少量样本：品牌实际 + 背景均值加权混合（避免过拟合少量门店）
            brand_profile = 0.6 * feat_matrix[store_locals].mean(axis=0) + 0.4 * background
        else:
            # 无样本：使用背景均值（中性先验）
            brand_profile = background.copy()

        prior[b_local] = brand_profile   # 已在 [0.05, 0.95] 内

    # ── Step4：转 logit 空间，压缩幅度（先验为"提示"，不强制）──────────────
    # prior 已在 [0.05, 0.95]，logit 范围约 [-2.9, 2.9]
    # 乘以 0.4 压缩到 [-1.2, 1.2]，给 pref_net 留足可学习空间
    prior_logit = 0.4 * np.log(prior / (1.0 - prior))

    return prior_logit.astype(np.float32)


# ── LLM 先验生成（DeepSeek API，兼容 OpenAI 格式）──────────────────────────

def _load_brand_names(data_dir, d):
    """
    从 train/valid/test.txt 解析品牌实体名 → 可读名称的映射。

    train.txt 每行格式：
        Brand_1:7-eleven=aka=seven eleven=aka=711便利店\t...
    实体名为冒号前的部分（Brand_1），可读名为第一个别名（7-eleven）。

    返回 dict: brand_kg_id (int) → primary_name (str)
    """
    name_map = {}   # entity_str → primary_name
    for fname in ['train.txt', 'valid.txt', 'test.txt']:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if not parts:
                    continue
                head = parts[0]          # "Brand_1:7-eleven=aka=711便利店"
                colon_idx = head.find(':')
                if colon_idx < 0:
                    continue
                ent_str  = head[:colon_idx]          # "Brand_1"
                aka_part = head[colon_idx + 1:]       # "7-eleven=aka=..."
                primary  = aka_part.split('=aka=')[0] # "7-eleven"
                name_map[ent_str] = primary

    # 转换为 kg_id → name
    id2name = {}
    for ent_str, primary in name_map.items():
        if ent_str in d.ent2id:
            id2name[d.ent2id[ent_str]] = primary
    return id2name


def _build_prompt(names_batch):
    CONCEPT_DESC = (
        'C0 人流强度(0-1)：选址时对区域日均客流/通勤人流的重视程度\n'
        'C1 竞争适应(0-1)：愿意进入竞争品牌密集商圈的倾向（高=不怕竞争）\n'
        'C2 功能契合(0-1)：对周边餐饮/零售/办公等业态互补环境的需求\n'
        'C3 商业集聚(0-1)：对购物中心/商业街等成熟商业配套的依赖程度\n'
        'C4 友商协同(0-1)：依赖关联品牌聚集带来协同客流的程度\n'
        'C5 目标客群(0-1)：对目标消费群体在区域内密度的重视程度\n'
        'C6 扩张机会(0-1)：倾向于进入尚未饱和的新兴/成长型区域\n'
        'C7 生活方式(0-1)：区域整体生活氛围与品牌调性的契合重要性'
    )
    names_list = '\n'.join(f'- {n}' for n in names_batch)
    example    = '{"%s": [0.9, 0.6, 0.5, 0.8, 0.4, 0.7, 0.3, 0.6]}' % names_batch[0]
    return (
        f'你是资深商业地产和品牌扩张策略专家，熟悉中国连锁商业品牌。\n'
        f'请对以下每个品牌，在8个北京商业选址评估维度上打分（0.0=不重视，1.0=极为重视）：\n\n'
        f'{CONCEPT_DESC}\n\n'
        f'品牌列表（中英文均有，如不熟悉请根据品牌类型判断）：\n{names_list}\n\n'
        f'输出格式：严格的 JSON 对象，键为品牌名，值为长度8的浮点数组。示例：\n'
        f'{example}\n\n'
        f'要求：只输出 JSON，不要解释，不要 markdown 代码块，不要其他内容。'
    )


def _parse_llm_response(content, names_batch):
    import json
    if '```' in content:
        parts   = content.split('```')
        content = parts[1] if len(parts) > 1 else content
        if content.startswith('json'):
            content = content[4:]
    # 尝试找到第一个 { ... } 块
    start = content.find('{')
    end   = content.rfind('}')
    if start >= 0 and end > start:
        content = content[start:end+1]
    data   = json.loads(content.strip())
    result = {}
    for name in names_batch:
        if name in data:
            vals = [float(v) for v in list(data[name])[:8]]
            if len(vals) == 8:
                result[name] = vals
    return result


def _deepseek_batch(api_key, names_batch, model='deepseek-reasoner'):
    """调用 DeepSeek API，返回 {name: [8 floats]} 字典。"""
    import requests, json, time

    prompt  = _build_prompt(names_batch)
    url     = 'https://api.deepseek.com/chat/completions'
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.1,
        'max_tokens': 2048,
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            content = resp.json()['choices'][0]['message']['content'].strip()
            return _parse_llm_response(content, names_batch)
        except Exception as e:
            wait = 2 ** attempt
            print(f'    [重试 {attempt+1}/3] 错误: {e}，等待 {wait}s...')
            time.sleep(wait)
    return {}


def _ollama_batch(names_batch, model='qwen3:14b'):
    """
    调用本地 Ollama，返回 {name: [8 floats]} 字典。
    支持 qwen3:14b / deepseek-r1:70b 等本地模型。
    使用 Ollama 原生 API（/api/chat）。
    """
    import requests, json, time

    prompt = _build_prompt(names_batch)
    url    = 'http://localhost:11434/api/chat'
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'options': {'temperature': 0.1},
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            content = resp.json()['message']['content'].strip()
            # qwen3 有时输出 <think>...</think> 标签，去掉
            if '<think>' in content:
                end_think = content.find('</think>')
                if end_think >= 0:
                    content = content[end_think + 8:].strip()
            return _parse_llm_response(content, names_batch)
        except Exception as e:
            wait = 2 ** attempt
            print(f'    [重试 {attempt+1}/3] 错误: {e}，等待 {wait}s...')
            time.sleep(wait)
    return {}


def compute_llm_prior(d, data_dir, api_key='', model='qwen3:14b',
                      batch_size=10, cache_path=None, use_ollama=False):
    """
    调用 DeepSeek LLM 为每个品牌生成概念偏好先验。

    原理
    ----
    品牌战略知识（"麦当劳偏好高人流低端商圈"）存在于 LLM 预训练语料中，
    KG 结构统计无法表达这类语义。LLM 先验直接注入品牌战略知识，
    使 α[b,k] 在训练开始时就具有语义意义，解决数据稀疏下的偏好学习困境。

    流程
    ----
    1. 从 train/valid/test.txt 解析品牌可读名称
    2. 每批 batch_size 个品牌调用 DeepSeek API
    3. 解析 JSON 响应，逐步缓存到本地（防止中断丢失）
    4. 转为 logit 空间（×0.5 压缩，先验为"提示"而非"强制"）

    参数
    ----
    d          : Data 对象
    api_key    : DeepSeek API key
    data_dir   : 数据目录（用于读取品牌名）
    model      : model name, default deepseek-reasoner
    batch_size : 每次 API 调用的品牌数（建议 15-25）
    cache_path : JSON 缓存文件路径（中断后可续跑）
    """
    import json, time

    id2name = _load_brand_names(data_dir, d)
    N_b     = len(d.brand_list)

    # 加载已有缓存
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f'  [缓存] 已加载 {len(cache)} 个品牌的先验结果')

    prior = np.full((N_b, 8), 0.5, dtype=np.float32)   # 默认中性

    # 找出还没有缓存的品牌
    pending_indices = [i for i in range(N_b)
                       if id2name.get(d.brand_list[i], '') not in cache]
    total_batches   = (len(pending_indices) + batch_size - 1) // batch_size

    print(f'  共 {N_b} 个品牌，待查询 {len(pending_indices)} 个，'
          f'分 {total_batches} 批（每批 {batch_size} 个）')

    for batch_i in range(0, len(pending_indices), batch_size):
        batch_idx   = pending_indices[batch_i: batch_i + batch_size]
        batch_names = [id2name.get(d.brand_list[i], f'unknown_brand_{i}')
                       for i in batch_idx]

        print(f'  批次 {batch_i // batch_size + 1}/{total_batches}：'
              f'{batch_names[0]} ... {batch_names[-1]}')

        if use_ollama:
            result = _ollama_batch(batch_names, model=model)
        else:
            result = _deepseek_batch(api_key, batch_names, model=model)

        # 存入缓存
        cache.update(result)
        if cache_path:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

        hit = len(result)
        miss = len(batch_names) - hit
        print(f'    ✓ {hit} 个成功，{miss} 个失败（将用中性值）')

        if not use_ollama:
            time.sleep(0.5)  # 只对云端 API 限速

    # 将缓存结果填入 prior
    for b_local, b_kg in enumerate(d.brand_list):
        name = id2name.get(b_kg, '')
        if name in cache:
            vals = cache[name]
            prior[b_local] = np.clip(vals, 0.05, 0.95)

    # 统计覆盖率
    covered = sum(1 for i, b_kg in enumerate(d.brand_list)
                  if id2name.get(b_kg, '') in cache)
    print(f'\n  LLM 先验覆盖率: {covered}/{N_b} ({100*covered/N_b:.1f}%)')

    # 转 logit 空间，0.5× 压缩（先验为提示，非强制）
    prior_clipped = np.clip(prior, 0.05, 0.95)
    return (0.5 * np.log(prior_clipped / (1.0 - prior_clipped))).astype(np.float32)


# ── 主程序 ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from load_data import Data

    pa = argparse.ArgumentParser()
    pa.add_argument('--dataset',  default='beijing')
    pa.add_argument('--kg_name',  default='kg')
    pa.add_argument('--llm_key',  default='',
                    help='DeepSeek API key。提供时使用 LLM 先验；否则使用 KG 统计先验。')
    pa.add_argument('--llm_model', default='deepseek-reasoner',
                    help='LLM model name (default: deepseek-reasoner)')
    pa.add_argument('--batch_size', type=int, default=20,
                    help='每次 API 调用的品牌数（默认 20）')
    pa.add_argument('--ollama', action='store_true',
                    help='使用本地 Ollama 而非 DeepSeek API')
    pa.add_argument('--ollama_model', default='qwen3:14b',
                    help='Ollama 模型名（默认 qwen3:14b）')
    args = pa.parse_args()

    root     = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, '../data/%s/' % args.dataset)

    print('加载数据...')
    d = Data(data_dir=data_dir, kg_name=args.kg_name)

    if args.llm_key or args.ollama:
        use_ollama = args.ollama
        model = args.ollama_model if use_ollama else args.llm_model
        print(f'\n使用 {"Ollama 本地" if use_ollama else "DeepSeek"} LLM 先验（模型: {model}）...')
        cache_path  = os.path.join(root, 'llm_prior_cache_%s.json' % args.dataset)
        prior_logit = compute_llm_prior(
            d, api_key=args.llm_key, data_dir=data_dir,
            model=model, batch_size=args.batch_size,
            cache_path=cache_path, use_ollama=use_ollama,
        )
        print(f'  缓存已保存至: {cache_path}（下次可直接复用，无需重新调用 API）')
    else:
        print('\n使用 KG 统计先验（未提供 --llm_key 或 --ollama）...')
        prior_logit = compute_kg_prior(d)

    out_path = os.path.join(root, 'alpha_prior_%s.npy' % args.dataset)
    np.save(out_path, prior_logit)
    print(f'\n已保存：{out_path}  shape={prior_logit.shape}')

    # 打印统计（转回概率空间方便查看）
    prior_prob = 1.0 / (1.0 + np.exp(-prior_logit))
    concept_names = ['TrafficFlow', 'Competition', 'FuncFit', 'Commercial',
                     'BrandAffin', 'CatConsumer', 'SpatialOpp', 'Lifestyle']
    print('\n各概念先验偏好均值（品牌平均）：')
    for k, name in enumerate(concept_names):
        print(f'  C{k} {name:15s}: mean={prior_prob[:,k].mean():.3f}'
              f'  std={prior_prob[:,k].std():.3f}'
              f'  min={prior_prob[:,k].min():.3f}'
              f'  max={prior_prob[:,k].max():.3f}')

    # ── 同时保存区域 KG 特征（用于 S 语义辅助监督 aux_loss）─────────────────
    # shape: [N_regions, 8]，每列已归一化到 [0.05, 0.95]
    # 与 S[r,k] 的概念对齐：C0=TrafficFlow, C1=Competition, ..., C7=Lifestyle
    feats, _ = _build_region_kgfeatures(d)
    feats_path = os.path.join(root, 'region_kg_feats_%s.npy' % args.dataset)
    np.save(feats_path, feats)
    print(f'\n已保存区域 KG 特征（aux_loss 用）：{feats_path}  shape={feats.shape}')
    print('\n区域 KG 特征均值（各概念）：')
    for k, name in enumerate(concept_names):
        print(f'  C{k} {name:15s}: mean={feats[:,k].mean():.3f}'
              f'  std={feats[:,k].std():.3f}'
              f'  min={feats[:,k].min():.3f}'
              f'  max={feats[:,k].max():.3f}')
