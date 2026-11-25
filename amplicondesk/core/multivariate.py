"""多元统计检验：MRPP、PERMDISP、Mantel、envfit、Procrustes 与两两分组封装。

本模块只依赖 numpy / scipy / scikit-learn / pandas，不依赖 R 或 statsmodels。
复用 ``amplicondesk.core.diversity`` 中已有的 ``pcoa`` / ``permanova`` / ``anosim``
以及 ``amplicondesk.core.stats`` 中的 ``bh_fdr``，不重复实现。

约定
----
* 丰度表：行 = 特征，列 = 样本。
* 距离矩阵：样本 × 样本 的对称 ``DataFrame``（对角为 0）。
* ``groups`` 可以是 ``pd.Series``（index = 样本名）、``dict``、``list`` 或 ``ndarray``。
* 置换检验统一使用 ``np.random.default_rng(seed)``，
  p 值统一为 ``(count + 1) / (permutations + 1)``。

退化情形的统一策略
------------------
* 结构性错误（距离矩阵非方阵、含 NaN/Inf、含负值、组数 < 2、某组样本 < 2 等）
  一律抛中文 ``ValueError``，便于调用方尽早失败。
* 数值性退化（例如置换中某个中间量无法定义）返回 ``p = NaN``，但不抛异常。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as _sps
from scipy.spatial.distance import pdist, squareform

try:  # 正常包内导入
    from .diversity import anosim, distance_matrix, pcoa, permanova
    from .stats import bh_fdr
except ImportError:  # pragma: no cover - 兼容 ``python amplicondesk/core/multivariate.py``
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from amplicondesk.core.diversity import anosim, distance_matrix, pcoa, permanova
    from amplicondesk.core.stats import bh_fdr

__all__ = [
    "mrpp",
    "permdisp",
    "mantel",
    "mantel_heatmap",
    "envfit",
    "procrustes",
    "procrustes_permutation_test",
    "pairwise_test",
]


# --------------------------------------------------------------------------
# 公共校验与工具
# --------------------------------------------------------------------------
def _as_square(distance: pd.DataFrame, name: str = "distance") -> pd.DataFrame:
    """校验并规范化距离方阵（对称、零对角、无 NaN/Inf、非负）。

    Args:
        distance: 待校验的距离矩阵。
        name: 出错信息中使用的变量名。

    Returns:
        规范化后的距离矩阵（浮点、对称、对角为 0）。

    Raises:
        ValueError: 矩阵不是方阵、含 NaN/Inf、含负值、或行列标签不一致。
    """
    if not isinstance(distance, pd.DataFrame):
        raise ValueError(f"{name} 必须是 pandas DataFrame（样本 × 样本 的距离方阵）。")
    if distance.shape[0] != distance.shape[1]:
        raise ValueError(
            f"{name} 必须是方阵（样本 × 样本），当前为 {distance.shape[0]} 行 × {distance.shape[1]} 列。"
        )
    if distance.shape[0] < 2:
        raise ValueError(f"{name} 至少需要 2 个样本，当前只有 {distance.shape[0]} 个。")
    if list(distance.index) != list(distance.columns):
        raise ValueError(f"{name} 的行标签与列标签必须完全一致（且顺序相同）。")

    arr = distance.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} 含有 NaN 或 Inf，请先清洗距离矩阵。")
    if (arr < -1e-9).any():
        raise ValueError(f"{name} 含有负的距离值，这不是合法的距离矩阵。")

    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 0.0)
    return pd.DataFrame(arr, index=distance.index, columns=distance.columns)


def _as_labels(groups, samples: Sequence) -> pd.Series:
    """把分组信息对齐到给定的样本顺序，并丢弃没有分组的样本。

    Args:
        groups: Series / dict / list / ndarray 形式的分组变量。
        samples: 目标样本顺序（通常是距离矩阵的行标签）。

    Returns:
        index 为样本名、值为字符串的 ``pd.Series``，只保留有分组的样本。

    Raises:
        ValueError: 分组长度与样本数不匹配，或没有任何样本能匹配上分组。
    """
    if isinstance(groups, pd.Series):
        labels = groups.copy()
    elif isinstance(groups, dict):
        labels = pd.Series(groups, dtype=object)
    else:
        values = list(groups)
        if len(values) != len(samples):
            raise ValueError(
                f"分组变量长度（{len(values)}）与样本数（{len(samples)}）不一致，"
                "请传入带样本名的 pandas Series 或与样本顺序一致的列表。"
            )
        labels = pd.Series(values, index=list(samples), dtype=object)

    labels = labels[~labels.isna()]
    common = [s for s in samples if s in labels.index]
    if not common:
        raise ValueError("分组变量与距离矩阵没有任何共同的样本名，请检查数据对齐。")
    labels = labels.loc[common].astype(str)
    if labels.nunique() < 2:
        raise ValueError(f"分组变量只有 {labels.nunique()} 个组，至少需要 2 个组才能做多元检验。")
    return labels


def _as_coords(coords: pd.DataFrame, name: str, min_samples: int = 3) -> pd.DataFrame:
    """校验排坐标表（样本 × 轴）。"""
    if not isinstance(coords, pd.DataFrame):
        raise ValueError(f"{name} 必须是 pandas DataFrame（样本 × 排序轴）。")
    if coords.shape[0] < min_samples:
        raise ValueError(f"{name} 至少需要 {min_samples} 个样本，当前只有 {coords.shape[0]} 个。")
    if coords.shape[1] < 1:
        raise ValueError(f"{name} 至少需要 1 个排序轴。")
    arr = coords.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} 含有 NaN 或 Inf，请先清洗排序坐标。")
    if np.allclose(arr, arr[0], atol=1e-12):
        raise ValueError(f"{name} 所有样本的坐标完全相同（没有方差），无法做拟合或 Procrustes 分析。")
    return coords.astype(float)


def _align_env(env: pd.DataFrame, samples: Sequence[str], name: str = "env") -> pd.DataFrame:
    """把环境/分组变量表按样本名对齐到给定顺序，并做基础校验。"""
    if isinstance(env, pd.Series):
        env = env.to_frame(name=env.name if env.name is not None else "env")
    if not isinstance(env, pd.DataFrame):
        raise ValueError(f"{name} 必须是 pandas DataFrame（样本 × 变量）。")
    common = [s for s in samples if s in env.index]
    if len(common) < 3:
        raise ValueError(
            f"{name} 与距离矩阵/排序坐标的共同样本不足（{len(common)} 个），至少需要 3 个。"
        )
    out = env.loc[common].copy()
    if out.shape[1] == 0:
        raise ValueError(f"{name} 至少需要 1 个变量。")
    return out


def _label_groups(labels: Sequence) -> tuple[list[str], np.ndarray[str]]:
    """返回 (组名列表, 每组样本的下标数组)。"""
    values = np.asarray([str(v) for v in labels])
    cats = sorted(pd.unique(values).tolist())
    idx_list = [np.where(values == g)[0] for g in cats]
    return cats, idx_list


def _require_group_sizes(labels: Sequence, min_size: int = 2, context: str = "") -> None:
    """要求每个组的样本量都不小于 min_size，否则抛中文 ValueError。"""
    counts = pd.Series([str(v) for v in labels]).value_counts()
    bad = counts[counts < min_size]
    if not bad.empty:
        detail = "、".join(f"{g}({c} 个)" for g, c in bad.items())
        prefix = f"{context}：" if context else ""
        raise ValueError(
            f"{prefix}以下分组的样本数小于 {min_size}，无法计算组内统计量 —— {detail}。"
        )


def _p_value(count: int, permutations: int) -> float:
    """统一置换 p 值：(count + 1) / (permutations + 1)。"""
    return float((count + 1) / (permutations + 1))


def _corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    """计算两向量相关系数，method ∈ {pearson, spearman, kendall}。"""
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    if method == "pearson":
        return float(_sps.pearsonr(x, y)[0])
    if method == "spearman":
        return float(_sps.spearmanr(x, y)[0])
    if method == "kendall":
        return float(_sps.kendalltau(x, y)[0])
    raise ValueError(f"未知的相关系数方法：{method!r}，可选 pearson / spearman / kendall。")


def _r2_ols(y: np.ndarray, x: np.ndarray) -> float:
    """多元回归的决定度 R²（y ~ x，含截距）。"""
    design = np.column_stack([np.ones(x.shape[0]), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 0:
        return float("nan")
    return float(1.0 - float((resid ** 2).sum()) / sst)


def _geometric_median(points: np.ndarray, tol: float = 1e-9, max_iter: int = 500) -> np.ndarray:
    """Weiszfeld 迭代求几何中位数（PERMDISP 的 center="median"）。"""
    points = np.asarray(points, dtype=float)
    if points.shape[0] == 1:
        return points[0].copy()
    est = points.mean(axis=0)
    for _ in range(max_iter):
        diff = points - est
        dist = np.linalg.norm(diff, axis=1)
        zero = dist < 1e-12
        if zero.any():
            return points[np.argmax(zero)].copy()
        w = 1.0 / dist
        new = (points * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(new - est) < tol:
            est = new
            break
        est = new
    return est


def _environ_distance(values: pd.Series) -> pd.DataFrame:
    """数值型环境变量的 |样本 - 样本| 相异矩阵。"""
    v = values.to_numpy(dtype=float)
    d = np.abs(v[:, None] - v[None, :])
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=values.index, columns=values.index)


def _categorical_distance(values: pd.Series) -> pd.DataFrame:
    """分类变量的 0/1 相异矩阵：同组记 0，不同组记 1。"""
    codes = pd.factorize(values.astype(str))[0]
    d = (codes[:, None] != codes[None, :]).astype(float)
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=values.index, columns=values.index)


# --------------------------------------------------------------------------
# MRPP
# --------------------------------------------------------------------------
def mrpp(
    distance: pd.DataFrame,
    groups: pd.Series,
    permutations: int = 999,
    seed: int = 42,
    weighting: str = "n",
) -> dict:
    """多元响应置换过程 MRPP（Multi-Response Permutation Procedure, Mielke）。

    统计量（Mielke & Berry 2001, *Permutation Methods*, 第 3 章；
    vegan::mrpp 的 ``delta``）：

    .. math::

        \\delta = \\sum_{g} \\xi_g \\, \\bar{d}_g, \\qquad
        \\bar{d}_g = \\binom{n_g}{2}^{-1} \\sum_{i<j,\\, g_i=g_j=g} d_{ij}

    其中权重 :math:`\\xi_g` 由 ``weighting`` 决定：

    * ``"n"``（默认）：:math:`\\xi_g = n_g`
    * ``"n2"``：:math:`\\xi_g = n_g (n_g - 1)`

    期望值来自"组内距离均值等于全体平均距离"的超几何恒等式：
    :math:`E[\\bar{d}_g] = \\bar{d}_{\\text{all}}`（对标签的所有置换取平均），于是

    .. math:: E[\\delta] = \\bar{d}_{\\text{all}} \\sum_g \\xi_g

    机会校正组内一致度 :math:`A = 1 - \\delta / E[\\delta]`（A 越接近 1，
    组内越聚集）。置换检验统计量为

    .. math:: z = \\frac{\\delta - E[\\delta]}{\\sqrt{\\operatorname{Var}[\\delta]}}

    其中 :math:`\\operatorname{Var}[\\delta]` 用置换分布的标准差估计（等价于 vegan
    用置换分布做的标准化），因此置换 p 值与直接对 :math:`\\delta` 排序等价。

    Args:
        distance: 样本 × 样本 距离方阵。
        groups: 样本分组（index 需与距离矩阵行标签一致）。
        permutations: 置换次数，默认 999。
        seed: 随机种子，默认 42。
        weighting: ``"n"``（按组样本数加权）或 ``"n2"``（按组样本数平方加权）。

    Returns:
        dict，包含 ``test``/``delta``/``expected_delta``/
        ``chance_corrected_within_group_agreement``/``statistic``/``p``/
        ``permutations``/``n_groups``。

    Raises:
        ValueError: 距离矩阵非法、组数 < 2、某组样本 < 2、或 weighting 取值非法。
    """
    if weighting not in ("n", "n2"):
        raise ValueError(f"weighting 只能是 'n' 或 'n2'，收到 {weighting!r}。")
    if permutations < 1:
        raise ValueError(f"permutations 必须是正整数，收到 {permutations}。")

    dist = _as_square(distance)
    labels = _as_labels(groups, list(dist.index))
    idx = list(labels.index)
    d = dist.loc[idx, idx].to_numpy(dtype=float)
    labs = labels.to_numpy()
    n = d.shape[0]

    cats, idx_list = _label_groups(labs)
    _require_group_sizes(labs, min_size=2, context="MRPP")

    iu = np.triu_indices(n, k=1)
    all_mean = float(d[iu].mean())
    sizes = np.array([ix.size for ix in idx_list], dtype=float)
    # 权重：n 或 n * (n - 1)
    if weighting == "n":
        weights = sizes.copy()
    else:
        weights = sizes * (sizes - 1.0)

    def delta_of(label_arr: np.ndarray) -> float:
        """按给定标签计算加权组内平均距离 delta。"""
        total = 0.0
        for w, g in zip(weights, cats):
            sel = np.where(label_arr == g)[0]
            if sel.size < 2:
                continue
            sub = d[np.ix_(sel, sel)]
            k = sel.size
            within_mean = sub[np.triu_indices(k, k=1)].sum() / (k * (k - 1) / 2.0)
            total += w * within_mean
        return float(total)

    delta_obs = delta_of(labs)
    expected_delta = all_mean * float(weights.sum())
    a_stat = float(1.0 - delta_obs / expected_delta) if expected_delta > 0 else float("nan")

    rng = np.random.default_rng(seed)
    perm_delta = np.empty(permutations, dtype=float)
    for i in range(permutations):
        perm_delta[i] = delta_of(rng.permutation(labs))

    sd = float(perm_delta.std(ddof=1)) if permutations > 1 else float("nan")
    statistic = float((delta_obs - expected_delta) / sd) if sd > 0 else float("nan")
    p = _p_value(int((perm_delta <= delta_obs).sum()), permutations)

    return {
        "test": "MRPP",
        "delta": float(delta_obs),
        "expected_delta": float(expected_delta),
        "chance_corrected_within_group_agreement": a_stat,
        "statistic": statistic,
        "p": p,
        "permutations": int(permutations),
        "n_groups": int(len(cats)),
        "weighting": weighting,
    }


# --------------------------------------------------------------------------
# PERMDISP
# --------------------------------------------------------------------------
def permdisp(
    distance: pd.DataFrame,
    groups: pd.Series,
    permutations: int = 999,
    seed: int = 42,
    center: str = "centroid",
) -> dict:
    """多元离散度齐性检验 PERMDISP（Anderson 2006, *Biometrics* 62:245-253）。

    步骤：

    1. 对距离矩阵做经典 PCoA，取全部正特征值对应主轴（保持欧氏几何）；
    2. 每个组在主轴空间求中心（``centroid`` 为算术质心，``median`` 为
       Weiszfeld 几何中位数），计算各样本到本组中心的欧氏距离 :math:`z_i`；
    3. 对 :math:`z_i` 做单因素方差分析（这是 PERMDISP 的 F 统计量，
       :math:`F = \\frac{SS_{between}/(a-1)}{SS_{within}/(N-a)}`）；
    4. 置换组标签得到 F 的零分布，p = ``(count + 1) / (permutations + 1)``，
       其中 count 为置换 F ≥ 观测 F 的次数。

    Args:
        distance: 样本 × 样本 距离方阵。
        groups: 样本分组。
        permutations: 置换次数，默认 999。
        seed: 随机种子，默认 42。
        center: ``"centroid"`` 或 ``"median"``。

    Returns:
        dict，包含 ``test``/``F``/``p``/``permutations``/
        ``distances``（DataFrame，index = 样本名，columns = ["Group", "distance"]）/
        ``group_means``（dict 组名 -> 平均距离）/``n_groups``。

    Raises:
        ValueError: 距离矩阵非法、样本数不足、center 取值非法、组数 < 2 或某组样本 < 2。
    """
    if center not in ("centroid", "median"):
        raise ValueError(f"center 只能是 'centroid' 或 'median'，收到 {center!r}。")
    if permutations < 1:
        raise ValueError(f"permutations 必须是正整数，收到 {permutations}。")

    dist = _as_square(distance)
    labels = _as_labels(groups, list(dist.index))
    idx = list(labels.index)
    d = dist.loc[idx, idx].to_numpy(dtype=float)
    labs = labels.to_numpy()
    n = d.shape[0]

    cats, idx_list = _label_groups(labs)
    _require_group_sizes(labs, min_size=2, context="PERMDISP")
    if n < 3:
        raise ValueError(f"PERMDISP 至少需要 3 个样本，当前只有 {n} 个。")

    coords, _ = pcoa(dist.loc[idx, idx], n_axes=max(n - 1, 1))
    pts = coords.to_numpy(dtype=float)
    if pts.shape[1] == 0:
        raise ValueError("PCoA 没有得到任何有效主轴（距离矩阵可能全部为 0），无法做 PERMDISP。")

    def group_distances(points: np.ndarray, label_arr: np.ndarray) -> np.ndarray:
        """返回每个样本到本组中心的欧氏距离。"""
        z = np.empty(points.shape[0], dtype=float)
        for g in cats:
            sel = np.where(label_arr == g)[0]
            sub = points[sel]
            c = sub.mean(axis=0) if center == "centroid" else _geometric_median(sub)
            z[sel] = np.linalg.norm(sub - c, axis=1)
        return z

    def f_stat(z: np.ndarray, label_arr: np.ndarray) -> float:
        """对 z 做单因素方差分析的 F 统计量。"""
        grand = z.mean()
        ss_between = 0.0
        ss_within = 0.0
        for g in cats:
            sel = np.where(label_arr == g)[0]
            if sel.size == 0:
                continue
            m = z[sel].mean()
            ss_between += sel.size * (m - grand) ** 2
            ss_within += float(((z[sel] - m) ** 2).sum())
        df_b, df_w = len(cats) - 1, n - len(cats)
        if df_b < 1 or df_w < 1:
            return float("nan")
        if ss_within <= 0:
            return float("inf") if ss_between > 0 else float("nan")
        return float((ss_between / df_b) / (ss_within / df_w))

    z_obs = group_distances(pts, labs)
    f_obs = f_stat(z_obs, labs)

    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(permutations):
        f_perm = f_stat(z_obs, rng.permutation(labs))
        if np.isnan(f_perm):
            continue
        if f_perm >= f_obs:
            ge += 1

    distances = pd.DataFrame({"Group": labs, "distance": z_obs}, index=idx)
    group_means = {g: float(z_obs[np.where(labs == g)[0]].mean()) for g in cats}

    return {
        "test": "PERMDISP",
        "F": float(f_obs),
        "p": _p_value(ge, permutations),
        "permutations": int(permutations),
        "distances": distances,
        "group_means": group_means,
        "n_groups": int(len(cats)),
        "center": center,
    }


# --------------------------------------------------------------------------
# Mantel
# --------------------------------------------------------------------------
def mantel(
    dist1: pd.DataFrame,
    dist2: pd.DataFrame,
    method: str = "pearson",
    permutations: int = 999,
    seed: int = 42,
    alternative: str = "two-sided",
) -> dict:
    """Mantel 检验：两个距离矩阵之间的相关性（Mantel 1967）。

    取两个矩阵的上三角（不含对角）展开为向量后计算相关系数。置换时保持
    ``dist2`` 不动，对 ``dist1`` 做**行列同步重排**（即重排样本标签，
    保持其内部的样本结构），从而在"无关联"零假设下求 p 值。

    Args:
        dist1: 距离方阵 1（被重排的那个）。
        dist2: 距离方阵 2。
        method: ``pearson`` / ``spearman`` / ``kendall``。
        permutations: 置换次数，默认 999。
        seed: 随机种子，默认 42。
        alternative: ``two-sided`` / ``greater`` / ``less``。

    Returns:
        dict，包含 ``test``/``method``/``r``/``p``/``permutations``/``n``，
        其中 ``n`` 为上三角元素个数。

    Raises:
        ValueError: 矩阵非法、两者标签或维度不一致、方法名非法。
    """
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError(f"method 只能是 pearson / spearman / kendall，收到 {method!r}。")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"alternative 只能是 two-sided / greater / less，收到 {alternative!r}。")
    if permutations < 1:
        raise ValueError(f"permutations 必须是正整数，收到 {permutations}。")

    a = _as_square(dist1, "dist1")
    b = _as_square(dist2, "dist2")
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"dist1 与 dist2 的样本数不一致（{a.shape[0]} vs {b.shape[0]}），无法做 Mantel 检验。"
        )
    common = [s for s in a.index if s in b.index]
    if len(common) != a.shape[0]:
        raise ValueError("dist1 与 dist2 的样本名不完全一致，请先把两个矩阵对齐到同一批样本。")
    b = b.loc[a.index, a.index]

    n = a.shape[0]
    iu = np.triu_indices(n, k=1)
    x = a.to_numpy(dtype=float)[iu]
    y = b.to_numpy(dtype=float)[iu]
    r_obs = _corr(x, y, method)
    if not np.isfinite(r_obs):
        return {
            "test": "Mantel",
            "method": method,
            "r": float("nan"),
            "p": float("nan"),
            "permutations": int(permutations),
            "n": int(x.size),
        }

    rng = np.random.default_rng(seed)
    perm_r = np.empty(permutations, dtype=float)
    arr = a.to_numpy(dtype=float)
    for i in range(permutations):
        order = rng.permutation(n)
        permuted = arr[np.ix_(order, order)][iu]
        perm_r[i] = _corr(permuted, y, method)

    finite = np.isfinite(perm_r)
    vals = perm_r[finite]
    if alternative == "greater":
        count = int((vals >= r_obs).sum())
    elif alternative == "less":
        count = int((vals <= r_obs).sum())
    else:
        count = int((np.abs(vals) >= abs(r_obs)).sum())
    denom = int(finite.sum()) if finite.any() else permutations

    return {
        "test": "Mantel",
        "method": method,
        "r": float(r_obs),
        "p": _p_value(count, max(denom, 1)),
        "permutations": int(permutations),
        "n": int(x.size),
    }


def mantel_heatmap(
    distance: pd.DataFrame,
    env: pd.DataFrame,
    permutations: int = 999,
    seed: int = 42,
    method: str = "pearson",
    correct: str = "bh",
) -> pd.DataFrame:
    """距离矩阵 vs 多个环境/分组变量的 Mantel 检验表（vegan::mantel 的批量版）。

    对每个变量构造一个相异矩阵再与 ``distance`` 做 Mantel 检验：

    * 数值型变量：:math:`D_{ij} = |x_i - x_j|`；
    * 分类变量：哑变量式的"是否同组"0/1 相异矩阵（同组 0，不同组 1）。

    Args:
        distance: 样本 × 样本 距离方阵。
        env: 样本 × 变量 的环境/分组表（index 为样本名）。
        permutations: 每个变量的置换次数，默认 999。
        seed: 随机种子，默认 42。
        method: 相关系数方法，同 :func:`mantel`。
        correct: 多重检验校正，``"bh"`` 或 None/``""`` 表示不校正。

    Returns:
        DataFrame，列：``variable, type, r, p, p_adj, n``。

    Raises:
        ValueError: 距离矩阵非法或没有共同样本。
    """
    dist = _as_square(distance)
    env_aligned = _align_env(env, list(dist.index))
    samples = list(env_aligned.index)
    dist = dist.loc[samples, samples]

    rows = []
    for col in env_aligned.columns:
        series = env_aligned[col]
        values = series.dropna()
        if values.nunique() < 2 or values.size < 3:
            rows.append(
                {
                    "variable": col,
                    "type": "categorical" if values.dtype == object else "numeric",
                    "r": np.nan,
                    "p": np.nan,
                    "n": int(values.size),
                }
            )
            continue
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            var_type = "numeric"
            var_dist = _environ_distance(values)
        else:
            var_type = "categorical"
            var_dist = _categorical_distance(values)
        sub = dist.loc[values.index, values.index]
        res = mantel(sub, var_dist, method=method, permutations=permutations, seed=seed)
        rows.append(
            {
                "variable": col,
                "type": var_type,
                "r": res["r"],
                "p": res["p"],
                "n": res["n"],
            }
        )

    out = pd.DataFrame(rows, columns=["variable", "type", "r", "p", "n"])
    if correct and not out.empty:
        out["p_adj"] = bh_fdr(out["p"].to_numpy(dtype=float))
    else:
        out["p_adj"] = out["p"]
    return out[["variable", "type", "r", "p", "p_adj", "n"]]


# --------------------------------------------------------------------------
# envfit
# --------------------------------------------------------------------------
def envfit(
    ordination_coords: pd.DataFrame,
    env: pd.DataFrame,
    permutations: int = 999,
    seed: int = 42,
    correct: str = "bh",
) -> pd.DataFrame:
    """把环境变量/分组变量以箭头形式拟合到排序图上（等价 vegan::envfit）。

    * 数值变量：把该变量对各排序轴做多元线性回归，拟合优度取决定度
      :math:`R^2`；箭头方向为变量与各轴的 Pearson 相关系数向量
      （等价于标准化回归向量，只差每个轴方向上的一个公共缩放）。
    * 分类变量：把"分组"当作解释项，:math:`R^2 = SS_{between} / SS_{total}`
      （组间平方和占总平方和的比例，即 PERMANOVA 在坐标上的版本），
      箭头方向取各组质心相对总体质心的加权平均方向向量。

    置换检验：固定坐标，重排该变量的取值，重新计算 R²，
    p = ``(count + 1) / (permutations + 1)``（count 为置换 R² ≥ 观测 R² 的次数）。

    Args:
        ordination_coords: 样本 × 轴 的排序坐标表。
        env: 样本 × 变量 的环境/分组表。
        permutations: 每个变量的置换次数，默认 999。
        seed: 随机种子，默认 42。
        correct: 多重检验校正，``"bh"`` 或 None/``""``。

    Returns:
        DataFrame，列：``variable, type, axis1, axis2, r2, p, p_adj``。

    Raises:
        ValueError: 坐标表非法或与 env 没有足够的共同样本。
    """
    coords = _as_coords(ordination_coords, "ordination_coords", min_samples=3)
    env_aligned = _align_env(env, list(coords.index))
    samples = list(env_aligned.index)
    coords = coords.loc[samples]
    mat = coords.to_numpy(dtype=float)
    n = mat.shape[0]

    rng = np.random.default_rng(seed)
    rows = []
    for col in env_aligned.columns:
        series = env_aligned[col]
        values = series.dropna()
        if values.nunique() < 2 or values.size < 3:
            rows.append(
                {
                    "variable": col,
                    "type": "categorical" if values.dtype == object else "numeric",
                    "axis1": np.nan,
                    "axis2": np.nan,
                    "r2": np.nan,
                    "p": np.nan,
                }
            )
            continue
        sel = [samples.index(s) for s in values.index]
        pts = mat[sel]

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            var_type = "numeric"
            vec = np.asarray(values.to_numpy(dtype=float))

            def r2_stat(y: np.ndarray) -> float:
                """数值变量的 R²：y ~ 全部排序轴。"""
                return _r2_ols(y, pts)

            arrow = np.array([_corr(vec, pts[:, k], "pearson") for k in range(pts.shape[1])])
            r2_obs = r2_stat(vec)
            label_arr = None
        else:
            var_type = "categorical"
            label_arr = values.astype(str).to_numpy()
            cats = sorted(pd.unique(label_arr).tolist())
            sizes_by_cat = {g: int((label_arr == g).sum()) for g in cats}
            grand = pts.mean(axis=0)
            sst = float(((pts - grand) ** 2).sum())

            def r2_stat(label: np.ndarray) -> float:
                """分类变量的 R² = 组间平方和 / 总平方和（按当前标签重新划分样本）。"""
                if sst <= 0:
                    return float("nan")
                # 按传入标签（可能是置换后的）重建每一组的样本下标
                masks = [np.where(label == g)[0] for g in cats]
                masks = [m for m in masks if m.size > 0]
                if len(masks) < 2:
                    return float("nan")
                ssb = 0.0
                for m in masks:
                    c = pts[m].mean(axis=0)
                    ssb += m.size * float(((c - grand) ** 2).sum())
                return float(ssb / sst)

            r2_obs = r2_stat(label_arr)
            dir_vec = np.zeros(pts.shape[1], dtype=float)
            for g in cats:
                m = np.where(label_arr == g)[0]
                dir_vec += sizes_by_cat[g] * (pts[m].mean(axis=0) - grand)
            norm = np.linalg.norm(dir_vec)
            arrow = dir_vec / norm if norm > 0 else dir_vec

        ge = 0
        for _ in range(permutations):
            # 数值变量：置换取值；分类变量：置换组标签（等价于重新划分样本）
            shuffled = rng.permutation(vec if var_type == "numeric" else label_arr)
            r2_perm = r2_stat(np.asarray(shuffled, dtype=float) if var_type == "numeric" else shuffled)
            if np.isfinite(r2_perm) and r2_perm >= r2_obs:
                ge += 1

        rows.append(
            {
                "variable": col,
                "type": var_type,
                "axis1": float(arrow[0]) if arrow.size >= 1 else np.nan,
                "axis2": float(arrow[1]) if arrow.size >= 2 else np.nan,
                "r2": float(r2_obs),
                "p": _p_value(ge, permutations),
            }
        )

    out = pd.DataFrame(rows, columns=["variable", "type", "axis1", "axis2", "r2", "p"])
    if correct and not out.empty:
        out["p_adj"] = bh_fdr(out["p"].to_numpy(dtype=float))
    else:
        out["p_adj"] = out["p"]
    return out


# --------------------------------------------------------------------------
# Procrustes
# --------------------------------------------------------------------------
def _procrustes_fit(
    coords1: np.ndarray, coords2: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Procrustes 最优相似变换（平移 + 缩放 + 旋转/反射）。

    求解 ``min_{s,R,t} ||X - s Y R - t||``，其中 R 为正交阵。令 X、Y 先按列中心化，
    记 :math:`A = X_c^T Y_c = U S V^T`，则最优旋转为 :math:`R = UV^T`、
    最优缩放为 :math:`s = \\operatorname{tr}(S) / \\|Y_c\\|_F^2`（Gower 1975）。

    Returns:
        ``(m2, r, coords1_fitted, coords2_aligned)``，其中

        .. math:: m2 = 1 - \\frac{\\operatorname{tr}(X_c^T X_c Y_c^T Y_c)}{\\|X_c\\|_F^2 \\|Y_c\\|_F^2}
    """
    x = np.asarray(coords1, dtype=float)
    y = np.asarray(coords2, dtype=float)
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)

    ssx = float((xc ** 2).sum())
    ssy = float((yc ** 2).sum())
    if ssx <= 0 or ssy <= 0:
        raise ValueError("Procrustes 需要两个坐标表都有非零方差（先去中心化后不能全为 0）。")

    u, s, vt = np.linalg.svd(xc.T @ yc)
    rot = u @ vt
    scale = float(s.sum()) / ssy if ssy > 0 else 1.0

    # 把 coords2 对齐到 coords1 的坐标框架（平移 + 缩放 + 旋转/反射）
    y_aligned = yc @ rot.T * scale + x.mean(axis=0)
    fitted = (x + y_aligned) / 2.0
    resid = x - y_aligned
    m2 = float((resid ** 2).sum() / ssx)
    m2 = float(np.clip(m2, 0.0, 1.0))
    r = float(np.sqrt(max(0.0, 1.0 - m2)))
    return m2, r, fitted, y_aligned


def procrustes_permutation_test(
    coords1: pd.DataFrame | np.ndarray,
    coords2: pd.DataFrame | np.ndarray,
    permutations: int = 999,
    seed: int = 42,
) -> dict:
    """Procrustes 置换检验（重排 ``coords2`` 的行，破坏样本对应关系）。

    p = ``(count + 1) / (permutations + 1)``，count 为置换 m2 ≤ 观测 m2 的次数
    （m2 越小说明拟合越好，因此是左尾检验）。

    Args:
        coords1: 坐标表 1。
        coords2: 坐标表 2（置换时重排其行）。
        permutations: 置换次数。
        seed: 随机种子。

    Returns:
        dict：``m2`` / ``r`` / ``p`` / ``permutations`` / ``n``。
    """
    a = coords1.to_numpy(dtype=float) if isinstance(coords1, pd.DataFrame) else np.asarray(coords1, float)
    b = coords2.to_numpy(dtype=float) if isinstance(coords2, pd.DataFrame) else np.asarray(coords2, float)
    if a.shape != b.shape:
        raise ValueError(f"两个坐标表的形状必须一致，当前为 {a.shape} 与 {b.shape}。")
    if permutations < 1:
        raise ValueError(f"permutations 必须是正整数，收到 {permutations}。")

    m2_obs, r_obs, _, _ = _procrustes_fit(a, b)
    rng = np.random.default_rng(seed)
    le = 0
    n = b.shape[0]
    for _ in range(permutations):
        m2_perm, _, _, _ = _procrustes_fit(a, b[rng.permutation(n)])
        if m2_perm <= m2_obs:
            le += 1
    return {
        "m2": float(m2_obs),
        "r": float(r_obs),
        "p": _p_value(le, permutations),
        "permutations": int(permutations),
        "n": int(n),
    }


def procrustes(
    coords1: pd.DataFrame,
    coords2: pd.DataFrame,
    permutations: int = 999,
    seed: int = 42,
) -> dict:
    """Procrustes 分析：两个排序结果的一致性检验（Gower 1975；vegan::procrustes）。

    对 ``coords2`` 施加最优相似变换（平移 + 缩放 + 旋转/反射）去逼近 ``coords1``，
    残差统计量为 :math:`m_2 = 1 - SS(\\text{残差}) / SS(\\text{coords1 中心化})`，
    一致性相关系数为 :math:`r = \\sqrt{1 - m_2}`。

    Args:
        coords1: 坐标表 1（样本 × 轴）。
        coords2: 坐标表 2（同样的样本 × 轴）。
        permutations: 置换次数，默认 999。
        seed: 随机种子，默认 42。

    Returns:
        dict，包含 ``test``/``m2``/``r``/``p``/``permutations``/``n``/
        ``coords1_fitted``/``coords2_aligned``（两者都是 DataFrame）。

    Raises:
        ValueError: 坐标表非法、形状不一致或样本名不一致。
    """
    c1 = _as_coords(coords1, "coords1", min_samples=3)
    c2 = _as_coords(coords2, "coords2", min_samples=3)
    if c1.shape != c2.shape:
        raise ValueError(
            f"coords1 与 coords2 的形状必须一致，当前为 {c1.shape} 与 {c2.shape}；"
            "请先取两者共有的排序轴。"
        )
    common = [s for s in c1.index if s in c2.index]
    if len(common) != c1.shape[0]:
        raise ValueError("coords1 与 coords2 的样本名不完全一致，请先对齐样本。")
    c2 = c2.loc[c1.index]

    stats_res = procrustes_permutation_test(c1, c2, permutations=permutations, seed=seed)
    _, _, fitted, aligned = _procrustes_fit(
        c1.to_numpy(dtype=float), c2.to_numpy(dtype=float)
    )
    cols = [f"Dim{i + 1}" for i in range(c1.shape[1])]
    return {
        "test": "Procrustes",
        "m2": stats_res["m2"],
        "r": stats_res["r"],
        "p": stats_res["p"],
        "permutations": int(permutations),
        "n": int(c1.shape[0]),
        "coords1_fitted": pd.DataFrame(fitted, index=c1.index, columns=cols),
        "coords2_aligned": pd.DataFrame(aligned, index=c1.index, columns=cols),
    }


# --------------------------------------------------------------------------
# 两两分组检验封装
# --------------------------------------------------------------------------
_METHOD_STAT = {
    "mrpp": ("delta", "MRPP"),
    "permdisp": ("F", "PERMDISP"),
    "permanova": ("R2", "PERMANOVA"),
    "anosim": ("R", "ANOSIM"),
}


def pairwise_test(
    distance: pd.DataFrame,
    groups: pd.Series,
    method: str = "mrpp",
    permutations: int = 999,
    seed: int = 42,
    correct: str = "bh",
) -> pd.DataFrame:
    """两两分组检验的通用封装（MRPP / PERMDISP / PERMANOVA / ANOSIM）。

    ``statistic`` 取各方法的主统计量：MRPP → ``delta``，PERMDISP → ``F``，
    PERMANOVA → ``R2``，ANOSIM → ``R``。``p_adj`` 为 BH-FDR 校正后的 p 值。

    Args:
        distance: 样本 × 样本 距离方阵。
        groups: 样本分组。
        method: ``"mrpp"`` / ``"permdisp"`` / ``"permanova"`` / ``"anosim"``。
        permutations: 每个配对的置换次数，默认 999。
        seed: 随机种子，默认 42。
        correct: 多重检验校正，``"bh"`` 或 None/``""``。

    Returns:
        DataFrame，列：``group1, group2, n1, n2, statistic, p, p_adj``。

    Raises:
        ValueError: method 非法、距离矩阵非法或组数 < 2。
    """
    if method not in _METHOD_STAT:
        raise ValueError(
            f"未知 method：{method!r}，可选 {sorted(_METHOD_STAT)}。"
        )
    dist = _as_square(distance)
    labels = _as_labels(groups, list(dist.index))
    idx = list(labels.index)
    dist = dist.loc[idx, idx]
    key, _ = _METHOD_STAT[method]

    cats, _ = _label_groups(labels.to_numpy())
    rows = []
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            a, b = cats[i], cats[j]
            sel = [s for s in idx if labels[s] in (a, b)]
            sub_dist = dist.loc[sel, sel]
            sub_groups = labels.loc[sel]
            try:
                if method == "mrpp":
                    res = mrpp(sub_dist, sub_groups, permutations=permutations, seed=seed)
                elif method == "permdisp":
                    res = permdisp(sub_dist, sub_groups, permutations=permutations, seed=seed)
                elif method == "permanova":
                    res = permanova(sub_dist, sub_groups, permutations=permutations, seed=seed)
                else:
                    res = anosim(sub_dist, sub_groups, permutations=permutations, seed=seed)
                statistic = float(res.get(key, np.nan))
                p = float(res.get("p", np.nan))
            except ValueError:
                statistic, p = float("nan"), float("nan")
            rows.append(
                {
                    "group1": a,
                    "group2": b,
                    "n1": int((sub_groups == a).sum()),
                    "n2": int((sub_groups == b).sum()),
                    "statistic": statistic,
                    "p": p,
                }
            )

    out = pd.DataFrame(rows, columns=["group1", "group2", "n1", "n2", "statistic", "p"])
    if correct and not out.empty:
        out["p_adj"] = bh_fdr(out["p"].to_numpy(dtype=float))
    else:
        out["p_adj"] = out["p"]
    return out


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------
def _sim_random_distance(n_per_group: int = 12, n_features: int = 40, seed: int = 0):
    """合成"分组与数据无关"的距离矩阵：随机丰度表 → Bray-Curtis。"""
    rng = np.random.default_rng(seed)
    n = n_per_group * 3
    table = pd.DataFrame(
        rng.lognormal(mean=0.0, sigma=1.0, size=(n_features, n)).round(3),
        index=[f"F{i}" for i in range(n_features)],
        columns=[f"S{i}" for i in range(n)],
    )
    dist = distance_matrix(table, "braycurtis")
    groups = pd.Series(
        np.repeat(["A", "B", "C"], n_per_group), index=table.columns, name="Group"
    )
    return dist, groups


def _sim_clustered_points(n_per_group: int = 20, dim: int = 3, spread: float = 0.35, seed: int = 0):
    """三组各自围绕不同质心的低维高斯云（组间可分）。"""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 2.5, 0.0]])[:, :dim]
    pts, labs = [], []
    for gi, c in enumerate(centers):
        pts.append(rng.normal(loc=c, scale=spread, size=(n_per_group, dim)))
        labs.extend([f"G{gi + 1}"] * n_per_group)
    arr = np.vstack(pts)
    names = [f"S{i}" for i in range(arr.shape[0])]
    coords = pd.DataFrame(arr, index=names, columns=[f"Axis{i + 1}" for i in range(dim)])
    dist = pd.DataFrame(squareform(pdist(arr)), index=names, columns=names)
    return coords, dist, pd.Series(labs, index=names, name="Group")


def _sim_equal_dispersion(n_per_group: int = 15, dim: int = 3, seed: int = 0):
    """三组离散度相同的随机点（PERMDISP 零假设）。"""
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(n_per_group * 3, dim))
    names = [f"S{i}" for i in range(arr.shape[0])]
    dist = pd.DataFrame(squareform(pdist(arr)), index=names, columns=names)
    groups = pd.Series(np.repeat(["A", "B", "C"], n_per_group), index=names, name="Group")
    return dist, groups


def _sim_unequal_dispersion(n_per_group: int = 15, dim: int = 3, seed: int = 0):
    """一组离散度明显更大的随机点（PERMDISP 备择假设）。"""
    rng = np.random.default_rng(seed)
    scales = [0.5, 0.5, 2.2]
    pts = [rng.normal(scale=s, size=(n_per_group, dim)) for s in scales]
    arr = np.vstack(pts)
    names = [f"S{i}" for i in range(arr.shape[0])]
    dist = pd.DataFrame(squareform(pdist(arr)), index=names, columns=names)
    groups = pd.Series(np.repeat(["A", "B", "C"], n_per_group), index=names, name="Group")
    return dist, groups


def _check(condition: bool, message: str) -> bool:
    """打印单条自检结果并返回是否通过。"""
    print(f"  [{'PASS' if condition else 'FAIL'}] {message}")
    return bool(condition)


def _self_test() -> int:
    """运行全部自检，返回失败条目数。"""
    failures = 0
    n_sim = 200

    # ---------------- MRPP 第一类错误控制 ----------------
    print("\n== MRPP 第一类错误控制（200 次独立模拟，分组与数据无关）==")
    hits = 0
    for i in range(n_sim):
        dist, groups = _sim_random_distance(seed=1000 + i)
        if mrpp(dist, groups, permutations=199, seed=i)["p"] < 0.05:
            hits += 1
    rate = hits / n_sim
    print(f"  p<0.05 比例 = {rate:.3f}（{hits}/{n_sim}）")
    failures += not _check(0.02 <= rate <= 0.10, "第一类错误率落在 [0.02, 0.10]")

    # ---------------- MRPP 检验效能 ----------------
    print("\n== MRPP 检验效能（三组各 20 个样本，围绕不同质心的高斯云）==")
    _, dist, groups = _sim_clustered_points(n_per_group=20, spread=0.35, seed=7)
    res = mrpp(dist, groups, permutations=999, seed=7)
    print(
        f"  delta={res['delta']:.4f}  expected_delta={res['expected_delta']:.4f}  "
        f"A={res['chance_corrected_within_group_agreement']:.4f}  "
        f"z={res['statistic']:.3f}  p={res['p']:.6f}"
    )
    failures += not _check(
        res["p"] < 0.01 and res["chance_corrected_within_group_agreement"] > 0.1,
        "p < 0.01 且 A > 0.1",
    )

    # ---------------- PERMDISP 第一类错误控制 ----------------
    print("\n== PERMDISP 第一类错误控制（200 次独立模拟，三组离散度相同）==")
    hits = 0
    for i in range(n_sim):
        dist, groups = _sim_equal_dispersion(seed=2000 + i)
        if permdisp(dist, groups, permutations=199, seed=i)["p"] < 0.05:
            hits += 1
    rate = hits / n_sim
    print(f"  p<0.05 比例 = {rate:.3f}（{hits}/{n_sim}）")
    failures += not _check(0.02 <= rate <= 0.10, "第一类错误率落在 [0.02, 0.10]")

    # ---------------- PERMDISP 效能 ----------------
    print("\n== PERMDISP 效能（第三组离散度明显更大）==")
    dist, groups = _sim_unequal_dispersion(seed=11)
    res = permdisp(dist, groups, permutations=999, seed=11)
    print(
        f"  F={res['F']:.3f}  p={res['p']:.6f}  "
        f"组平均距离={ {k: round(v, 3) for k, v in res['group_means'].items()} }"
    )
    failures += not _check(res["p"] < 0.01, "p < 0.01")
    print(f"  distances 表形状 = {res['distances'].shape}，列 = {list(res['distances'].columns)}")
    failures += not _check(
        list(res["distances"].columns) == ["Group", "distance"]
        and res["distances"].shape[0] == dist.shape[0],
        "distances 结构正确（index=样本名, columns=['Group','distance']）",
    )

    # ---------------- Mantel ----------------
    print("\n== Mantel：dist2 = dist1 + 小幅噪声 ==")
    _, dist1, groups = _sim_clustered_points(n_per_group=20, spread=0.5, seed=21)
    rng = np.random.default_rng(99)
    noisy = dist1.to_numpy() + rng.uniform(0, 0.02, size=dist1.shape)
    noisy = (noisy + noisy.T) / 2
    np.fill_diagonal(noisy, 0.0)
    dist_b = pd.DataFrame(noisy, index=dist1.index, columns=dist1.columns)
    res = mantel(dist1, dist_b, method="pearson", permutations=999, seed=21)
    print(f"  r={res['r']:.4f}  p={res['p']:.6f}  n={res['n']}")
    failures += not _check(res["r"] > 0.8 and res["p"] < 0.01, "r > 0.8 且 p < 0.01")
    res_sp = mantel(dist1, dist_b, method="spearman", permutations=199, seed=21)
    res_kt = mantel(dist1, dist_b, method="kendall", permutations=199, seed=21)
    print(f"  spearman r={res_sp['r']:.4f} p={res_sp['p']:.6f}；kendall r={res_kt['r']:.4f} p={res_kt['p']:.6f}")

    print("\n== Mantel 第一类错误控制（200 次独立模拟，两个独立随机矩阵）==")
    hits = 0
    for i in range(n_sim):
        a, _ = _sim_equal_dispersion(n_per_group=12, seed=3000 + i)
        b_rng = np.random.default_rng(9000 + i)
        b = b_rng.uniform(size=a.shape)
        b = (b + b.T) / 2
        np.fill_diagonal(b, 0.0)
        b_dist = pd.DataFrame(b, index=a.index, columns=a.columns)
        if mantel(a, b_dist, permutations=199, seed=i)["p"] < 0.05:
            hits += 1
    rate = hits / n_sim
    print(f"  p<0.05 比例 = {rate:.3f}（{hits}/{n_sim}）")
    failures += not _check(0.02 <= rate <= 0.10, "第一类错误率落在 [0.02, 0.10]")

    # ---------------- mantel_heatmap ----------------
    print("\n== mantel_heatmap（数值变量 + 分类变量）==")
    _, dist, groups = _sim_clustered_points(n_per_group=20, spread=0.4, seed=31)
    rng = np.random.default_rng(31)
    driver = pd.Series(
        np.where(groups.to_numpy() == "G1", 0.0, np.where(groups.to_numpy() == "G2", 3.0, 6.0))
        + rng.normal(scale=0.1, size=dist.shape[0]),
        index=dist.index,
        name="driver",
    )
    noise_var = pd.Series(rng.normal(size=dist.shape[0]), index=dist.index, name="noise")
    env = pd.concat([driver, noise_var, groups.rename("Group")], axis=1)
    table = mantel_heatmap(dist, env, permutations=999, seed=31)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    failures += not _check(
        list(table.columns) == ["variable", "type", "r", "p", "p_adj", "n"],
        "列名与顺序正确",
    )
    drv = table.set_index("variable")
    failures += not _check(
        drv.loc["Group", "type"] == "categorical" and drv.loc["driver", "type"] == "numeric",
        "类型识别正确（Group 为 categorical）",
    )
    failures += not _check(
        drv.loc["driver", "r"] > drv.loc["noise", "r"]
        and drv.loc["driver", "p_adj"] < 0.01,
        "驱动变量的 Mantel r 更高且 p_adj < 0.01",
    )

    # ---------------- envfit ----------------
    print("\n== envfit（排序坐标由 driver 驱动）==")
    coords, _, groups = _sim_clustered_points(n_per_group=30, dim=2, spread=0.4, seed=41)
    rng = np.random.default_rng(41)
    driver = pd.Series(
        np.where(groups.to_numpy() == "G1", 0.0, np.where(groups.to_numpy() == "G2", 3.0, 6.0))
        + rng.normal(scale=0.1, size=coords.shape[0]),
        index=coords.index,
        name="driver",
    )
    noise_var = pd.Series(rng.normal(size=coords.shape[0]), index=coords.index, name="noise")
    env = pd.concat([driver, noise_var, groups.rename("Group")], axis=1)
    ef = envfit(coords, env, permutations=999, seed=41)
    print(ef.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    failures += not _check(
        list(ef.columns) == ["variable", "type", "axis1", "axis2", "r2", "p", "p_adj"],
        "列名与顺序正确",
    )
    f = ef.set_index("variable")
    failures += not _check(
        f.loc["driver", "r2"] > f.loc["noise", "r2"] and f.loc["driver", "p_adj"] < 0.01,
        "driver 的 r2 高于 noise 且 p_adj < 0.01",
    )
    failures += not _check(
        f.loc["Group", "p_adj"] < 0.01, "分类变量 Group 的 p_adj < 0.01"
    )

    # ---------------- Procrustes ----------------
    print("\n== Procrustes（旋转+缩放 vs 独立随机坐标）==")
    base, _, _ = _sim_clustered_points(n_per_group=15, dim=2, spread=0.6, seed=51)
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    transformed = base.to_numpy() @ rot.T * 2.5 + np.array([5.0, -3.0])
    same = pd.DataFrame(transformed, index=base.index, columns=base.columns)
    res_same = procrustes(base, same, permutations=999, seed=51)
    print(
        f"  同源坐标：m2={res_same['m2']:.3e}  r={res_same['r']:.6f}  p={res_same['p']:.6f}"
    )
    failures += not _check(res_same["m2"] < 1e-6 and res_same["r"] > 0.999, "m2 ≈ 0 且 r ≈ 1")

    other = pd.DataFrame(
        np.random.default_rng(52).normal(size=base.shape),
        index=base.index,
        columns=base.columns,
    )
    res_other = procrustes(base, other, permutations=999, seed=52)
    print(f"  独立坐标：m2={res_other['m2']:.4f}  r={res_other['r']:.4f}  p={res_other['p']:.6f}")
    failures += not _check(
        res_other["m2"] > res_same["m2"] + 0.1, "独立坐标的 m2 明显更大"
    )
    failures += not _check(
        isinstance(res_same["coords1_fitted"], pd.DataFrame)
        and list(res_same["coords1_fitted"].index) == list(base.index),
        "coords1_fitted / coords2_aligned 为 DataFrame 且样本名对齐",
    )

    # ---------------- pairwise_test ----------------
    print("\n== pairwise_test（三组，MRPP + BH 校正）==")
    _, dist, groups = _sim_clustered_points(n_per_group=15, spread=0.5, seed=61)
    pw = pairwise_test(dist, groups, method="mrpp", permutations=499, seed=61)
    print(pw.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    failures += not _check(len(pw) == 3, "输出 3 行（3 个配对）")
    failures += not _check(
        list(pw.columns) == ["group1", "group2", "n1", "n2", "statistic", "p", "p_adj"],
        "列名与顺序正确",
    )
    monotone = bool((pw["p_adj"] >= pw["p"] - 1e-12).all())
    order_ok = bool((pw["p_adj"].to_numpy() >= pw["p"].to_numpy() - 1e-12).all())
    print(f"  p_adj >= p 全部成立 = {monotone}；BH 单调性检查 = {order_ok}")
    failures += not _check(monotone, "p_adj ≥ 原始 p")
    p_sorted = pw.sort_values("p")["p_adj"].to_numpy()
    failures += not _check(
        bool((np.diff(p_sorted) >= -1e-12).all()), "p_adj 随 p 单调不减"
    )
    for m in ("permdisp", "permanova", "anosim"):
        pw_m = pairwise_test(dist, groups, method=m, permutations=299, seed=61)
        print(f"  method={m:9s} statistic 列 = {pw_m['statistic'].round(4).tolist()}")
    failures += not _check(
        pairwise_test(dist, groups, method="permanova", permutations=299, seed=61)[
            "statistic"
        ].notna().all(),
        "四种方法的 statistic 均可计算",
    )

    # ---------------- 退化情形 ----------------
    print("\n== 退化情形（应抛出中文 ValueError）==")
    cases = [
        ("非方阵距离矩阵", lambda: mrpp(dist.iloc[:5, :3], groups, permutations=9)),
        (
            "含 NaN 的距离矩阵",
            lambda: mrpp(
                dist.mask(dist.to_numpy() == dist.to_numpy()[0, 1]), groups, permutations=9
            ),
        ),
        (
            "某组样本 < 2",
            lambda: mrpp(
                dist,
                pd.Series(
                    ["A"] * 2 + ["B"] * (dist.shape[0] - 3) + ["C"], index=dist.index
                ),
                permutations=9,
            ),
        ),
        ("只有 1 个组", lambda: mrpp(dist, pd.Series(["A"] * dist.shape[0], index=dist.index), permutations=9)),
        ("center 参数非法", lambda: permdisp(dist, groups, center="mode", permutations=9)),
        ("weighting 非法", lambda: mrpp(dist, groups, weighting="n3", permutations=9)),
        (
            "Procrustes 轴数不一致",
            lambda: procrustes(base, base.iloc[:, :1], permutations=9),
        ),
        ("pairwise_test method 非法", lambda: pairwise_test(dist, groups, method="foo", permutations=9)),
    ]
    for name, fn in cases:
        try:
            fn()
            print(f"  [FAIL] {name}：未抛出 ValueError")
            failures += 1
        except ValueError as exc:
            msg = str(exc)
            has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in msg)
            print(f"  [{'PASS' if has_cjk else 'FAIL'}] {name}：{msg}")
            failures += 0 if has_cjk else 1

    print(f"\n===== 自检汇总：{'全部通过' if failures == 0 else f'{failures} 项失败'} =====")
    return failures


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(1 if _self_test() else 0)
