"""多样性分析：alpha 指数、beta 距离、排序与多元统计检验。

所有算法不依赖 skbio / statsmodels，仅用 numpy + scipy + sklearn 实现。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import kruskal, mannwhitneyu

from .preprocess import as_numeric_matrix, clr_transform

# --------------------------------------------------------------------------
# Alpha 多样性
# --------------------------------------------------------------------------
def alpha_diversity(table: pd.DataFrame) -> pd.DataFrame:
    """计算每个样本的 alpha 多样性指数。"""
    table = as_numeric_matrix(table)
    counts = table.to_numpy(dtype=float).T  # 样本 × 特征
    samples = list(table.columns)
    lib = counts.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.divide(counts, lib[:, None], out=np.zeros_like(counts), where=lib[:, None] > 0)
        # Shannon
        plogp = np.where(rel > 0, rel * np.log(rel), 0.0)
        shannon = -plogp.sum(axis=1)
        # Simpson (1 - D) 与 Inverse Simpson
        simpson_d = (rel ** 2).sum(axis=1)
        simpson = 1.0 - simpson_d
        inv_simpson = np.divide(1.0, simpson_d, out=np.zeros_like(simpson_d), where=simpson_d > 0)
        # 丰富度
        observed = (counts > 0).sum(axis=1).astype(float)
        # Pielou 均匀度
        evenness = np.divide(shannon, np.log(observed), out=np.zeros_like(shannon), where=observed > 1)
        # Chao1
        f1 = (counts == 1).sum(axis=1).astype(float)
        f2 = (counts == 2).sum(axis=1).astype(float)
        chao1 = observed + np.divide(f1 ** 2, 2.0 * f2, out=np.zeros_like(f1), where=f2 > 0)
        chao1 = np.where((f2 == 0) & (f1 > 0), observed + f1 * (f1 - 1) / 2.0, chao1)
        # Goods 覆盖度
        goods = 1.0 - np.divide(f1, lib, out=np.zeros_like(f1), where=lib > 0)

    return pd.DataFrame(
        {
            "Observed": observed,
            "Chao1": chao1,
            "Shannon": shannon,
            "Simpson": simpson,
            "InvSimpson": inv_simpson,
            "Pielou": evenness,
            "GoodsCoverage": goods,
            "LibrarySize": lib,
        },
        index=samples,
    )


ALPHA_METRICS = ["Observed", "Chao1", "Shannon", "Simpson", "InvSimpson", "Pielou"]

ALPHA_LABELS = {
    "Observed": "Observed richness",
    "Chao1": "Chao1",
    "Shannon": "Shannon index",
    "Simpson": "Simpson index",
    "InvSimpson": "Inverse Simpson",
    "Pielou": "Pielou evenness",
    "GoodsCoverage": "Goods coverage",
    "LibrarySize": "Library size",
}


def group_test(values: pd.Series, groups: pd.Series) -> dict:
    """按分组做 Mann-Whitney U（两组）或 Kruskal-Wallis（多组）检验。"""
    frame = pd.DataFrame({"value": values, "group": groups}).dropna()
    frame["group"] = frame["group"].astype(str)
    cats = [g for g in pd.unique(frame["group"]) if (frame["group"] == g).sum() >= 1]
    if len(cats) < 2:
        return {"test": "n/a", "statistic": np.nan, "p": np.nan, "n_groups": len(cats)}
    samples = [frame.loc[frame["group"] == g, "value"].to_numpy() for g in cats]
    if any(len(s) < 2 for s in samples):
        return {"test": "n/a", "statistic": np.nan, "p": np.nan, "n_groups": len(cats)}
    try:
        if len(cats) == 2:
            stat, p = mannwhitneyu(samples[0], samples[1], alternative="two-sided")
            test = "Mann-Whitney U"
        else:
            stat, p = kruskal(*samples)
            test = "Kruskal-Wallis"
    except ValueError:
        return {"test": "n/a", "statistic": np.nan, "p": np.nan, "n_groups": len(cats)}
    return {"test": test, "statistic": float(stat), "p": float(p), "n_groups": len(cats)}


# --------------------------------------------------------------------------
# Beta 多样性
# --------------------------------------------------------------------------
BETA_METRICS = ["braycurtis", "jaccard", "aitchison", "euclidean"]

BETA_LABELS = {
    "braycurtis": "Bray-Curtis",
    "jaccard": "Jaccard",
    "aitchison": "Aitchison",
    "euclidean": "Euclidean",
}


def distance_matrix(table: pd.DataFrame, metric: str = "braycurtis") -> pd.DataFrame:
    """计算样本间距离矩阵（对称方阵）。"""
    table = as_numeric_matrix(table)
    samples = list(table.columns)

    if metric == "aitchison":
        mat = clr_transform(table).to_numpy(dtype=float).T
        from scipy.spatial.distance import pdist

        d = pdist(mat, metric="euclidean")
    elif metric == "jaccard":
        binary = (table.to_numpy(dtype=float).T > 0).astype(float)
        from scipy.spatial.distance import pdist

        d = pdist(binary, metric="jaccard")
    else:
        from scipy.spatial.distance import pdist

        d = pdist(table.to_numpy(dtype=float).T, metric=metric)

    d = np.nan_to_num(d, nan=0.0, posinf=1.0, neginf=0.0)
    if d.size == 0:
        return pd.DataFrame(np.zeros((len(samples), len(samples))), index=samples, columns=samples)
    return pd.DataFrame(squareform(d), index=samples, columns=samples)


def pcoa(distance: pd.DataFrame, n_axes: int = 4) -> tuple[pd.DataFrame, np.ndarray]:
    """经典多维标度（PCoA）。返回 (坐标表, 各轴解释度)。"""
    d = distance.to_numpy(dtype=float)
    n = d.shape[0]
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)

    a = -0.5 * (d ** 2)
    row_mean = a.mean(axis=1, keepdims=True)
    col_mean = a.mean(axis=0, keepdims=True)
    grand = a.mean()
    b = a - row_mean - col_mean + grand
    b = (b + b.T) / 2.0

    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    pos = eigvals > 1e-12
    eigvals, eigvecs = eigvals[pos], eigvecs[:, pos]
    if eigvals.size == 0:
        return pd.DataFrame(np.zeros((n, 2)), index=distance.index), np.array([0.0, 0.0])

    n_axes = min(n_axes, eigvals.size)
    coords = eigvecs[:, :n_axes] * np.sqrt(eigvals[:n_axes])
    explained = eigvals[:n_axes] / eigvals.sum()

    axis_names = [f"PCo{i + 1}" for i in range(n_axes)]
    return pd.DataFrame(coords, index=distance.index, columns=axis_names), explained


def nmds(distance: pd.DataFrame, n_axes: int = 2, seed: int = 42) -> tuple[pd.DataFrame, float]:
    """非度量多维标度（NMDS），返回 (坐标表, stress)。"""
    from sklearn.manifold import MDS

    n = distance.shape[0]
    if n < 3:
        coords = pd.DataFrame(
            np.zeros((n, n_axes)), index=distance.index, columns=[f"NMDS{i + 1}" for i in range(n_axes)]
        )
        return coords, float("nan")

    model = MDS(
        n_components=n_axes,
        metric=False,
        dissimilarity="precomputed",
        random_state=seed,
        n_init=8,
        max_iter=500,
        # 归一化 stress 落在 0~1，才能用通用的 0.05/0.1/0.2 阈值判断排序质量
        normalized_stress=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        emb = model.fit_transform(distance.to_numpy(dtype=float))
    cols = [f"NMDS{i + 1}" for i in range(n_axes)]
    return pd.DataFrame(emb, index=distance.index, columns=cols), float(model.stress_)


# --------------------------------------------------------------------------
# 多元统计检验
# --------------------------------------------------------------------------
def _ss_within(d: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """返回 (组内平方和, 组间平方和)。d 为距离矩阵，已取上三角平方。"""
    n = d.shape[0]
    d2 = d ** 2
    total = d2[np.triu_indices(n, k=1)].sum() / n

    within = 0.0
    groups = pd.unique(labels)
    for g in groups:
        idx = np.where(labels == g)[0]
        ng = idx.size
        if ng < 2:
            continue
        sub = d2[np.ix_(idx, idx)]
        within += sub[np.triu_indices(ng, k=1)].sum() / ng
    return within, total - within


def permanova(
    distance: pd.DataFrame,
    groups: pd.Series,
    permutations: int = 999,
    seed: int = 42,
) -> dict:
    """PERMANOVA (Anderson 2001)，等价于 adonis2 的默认设定。"""
    common = [s for s in distance.index if s in groups.index]
    if len(common) < 3:
        return {"test": "PERMANOVA", "R2": np.nan, "F": np.nan, "p": np.nan, "permutations": 0}
    d = distance.loc[common, common].to_numpy(dtype=float)
    labels = groups.loc[common].astype(str).to_numpy()

    a = pd.unique(labels).size
    n = len(common)
    if a < 2 or n - a < 1:
        return {"test": "PERMANOVA", "R2": np.nan, "F": np.nan, "p": np.nan, "permutations": 0}

    within, between = _ss_within(d, labels)
    df_a, df_w = a - 1, n - a
    if within <= 0:
        f_stat, r2 = np.inf, 1.0
    else:
        f_stat = (between / df_a) / (within / df_w)
        r2 = between / (between + within)

    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(permutations):
        shuffled = rng.permutation(labels)
        w, b = _ss_within(d, shuffled)
        f_perm = (b / df_a) / (w / df_w) if w > 0 else np.inf
        if f_perm >= f_stat:
            ge += 1
    p = (ge + 1) / (permutations + 1)

    return {
        "test": "PERMANOVA",
        "R2": float(r2),
        "F": float(f_stat),
        "p": float(p),
        "permutations": int(permutations),
        "df_between": int(df_a),
        "df_within": int(df_w),
    }


def anosim(
    distance: pd.DataFrame, groups: pd.Series, permutations: int = 999, seed: int = 42
) -> dict:
    """ANOSIM 相似性分析。"""
    common = [s for s in distance.index if s in groups.index]
    if len(common) < 3:
        return {"test": "ANOSIM", "R": np.nan, "p": np.nan}
    d = distance.loc[common, common].to_numpy(dtype=float)
    labels = groups.loc[common].astype(str).to_numpy()
    n = len(common)
    if pd.unique(labels).size < 2:
        return {"test": "ANOSIM", "R": np.nan, "p": np.nan}

    iu = np.triu_indices(n, k=1)
    flat = d[iu]
    ranks = pd.Series(flat).rank().to_numpy()
    same = labels[iu[0]] == labels[iu[1]]

    def r_stat(rank_arr: np.ndarray, same_mask: np.ndarray) -> float:
        m = rank_arr.size
        rw = rank_arr[same_mask].mean() if same_mask.any() else 0.0
        rb = rank_arr[~same_mask].mean() if (~same_mask).any() else 0.0
        return (rb - rw) / (m / 2.0)

    r_obs = r_stat(ranks, same)

    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(permutations):
        perm = rng.permutation(labels)
        same_p = perm[iu[0]] == perm[iu[1]]
        if r_stat(ranks, same_p) >= r_obs:
            ge += 1
    return {
        "test": "ANOSIM",
        "R": float(r_obs),
        "p": float((ge + 1) / (permutations + 1)),
        "permutations": int(permutations),
    }


def pairwise_permanova(
    distance: pd.DataFrame,
    groups: pd.Series,
    permutations: int = 999,
    seed: int = 42,
    correct: str = "bh",
) -> pd.DataFrame:
    """两两分组间的 PERMANOVA，并做多重检验校正。"""
    from .stats import bh_fdr

    common = [s for s in distance.index if s in groups.index]
    labs = groups.loc[common].astype(str)
    cats = sorted(pd.unique(labs))
    rows = []
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            a, b = cats[i], cats[j]
            idx = labs[labs.isin([a, b])].index
            res = permanova(distance.loc[idx, idx], labs.loc[idx], permutations=permutations, seed=seed)
            rows.append(
                {
                    "group1": a,
                    "group2": b,
                    "n1": int((labs.loc[idx] == a).sum()),
                    "n2": int((labs.loc[idx] == b).sum()),
                    "R2": res["R2"],
                    "F": res["F"],
                    "p": res["p"],
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty and correct:
        out["p_adj"] = bh_fdr(out["p"].to_numpy())
    return out


def distance_boxplot_data(distance: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """把距离矩阵摊平成 (组内/组间, 距离) 长表，用于箱线图。"""
    common = [s for s in distance.index if s in groups.index]
    d = distance.loc[common, common].to_numpy(dtype=float)
    labels = groups.loc[common].astype(str).to_numpy()
    n = len(common)
    iu = np.triu_indices(n, k=1)
    same = labels[iu[0]] == labels[iu[1]]
    pair = np.where(
        same,
        "Within " + pd.Series(labels[iu[0]]),
        pd.Series(labels[iu[0]]) + " vs " + pd.Series(labels[iu[1]]),
    )
    return pd.DataFrame({"pair": pair, "distance": d[iu], "same_group": same})
