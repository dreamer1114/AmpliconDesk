"""排序分析：PCA、RDA、dbRDA、CCA、CAP、t-SNE、UMAP 与 PLS-DA。

所有算法只依赖 numpy + scipy + scikit-learn，UMAP 为惰性导入的可选依赖，
不依赖 R 与 statsmodels。表约定统一为 **行 = 特征，列 = 样本**。

主要接口：
    pca_ordination / rda / dbrda / cca / cap /
    tsne_embedding / umap_embedding / plsda / scree_table
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

try:  # 正常的包内导入
    from .diversity import BETA_LABELS, BETA_METRICS, distance_matrix, pcoa
    from .preprocess import as_numeric_matrix, clr_transform, to_relative
    from .stats import bh_fdr  # noqa: F401  (沿用项目的多重检验校正，供下游调用)
except ImportError:  # pragma: no cover - 支持 `python amplicondesk/core/ordination.py` 直接运行
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from amplicondesk.core.diversity import BETA_LABELS, BETA_METRICS, distance_matrix, pcoa
    from amplicondesk.core.preprocess import as_numeric_matrix, clr_transform, to_relative
    from amplicondesk.core.stats import bh_fdr  # noqa: F401

# 变换方法注册表（transform 参数的可选值）
TRANSFORMS = ("clr", "hellinger", "logcpm", "relative", "none")

_EPS = 1e-12


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def hellinger(table: pd.DataFrame) -> pd.DataFrame:
    """Hellinger 变换：先转相对丰度，再开平方。"""
    rel = to_relative(table)
    return np.sqrt(rel.clip(lower=0.0))


def _apply_transform(table: pd.DataFrame, transform: str) -> tuple[pd.DataFrame, str]:
    """按名称把丰度表变换成用于欧氏距离的矩阵。

    Args:
        table: 丰度表（行 = 特征，列 = 样本）。
        transform: clr / hellinger / logcpm / relative / none。

    Returns:
        (变换后的表, 实际使用的变换名)。

    Raises:
        ValueError: transform 不是支持的方法。
    """
    name = str(transform or "none").lower()
    mat = as_numeric_matrix(table)
    if name == "clr":
        return clr_transform(mat), name
    if name == "hellinger":
        return hellinger(mat), name
    if name == "logcpm":
        # log2(CPM + 1)：先转相对丰度再乘 1e6，避免重复除以库大小
        return np.log2(to_relative(mat) * 1e6 + 1.0), name
    if name == "relative":
        return to_relative(mat), name
    if name == "none":
        return mat.astype(float), name
    raise ValueError(f"未知的变换方法：{transform}，可选 {list(TRANSFORMS)}。")


def _weighted_stats(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """加权均值与加权标准差（标准差为 0 时返回 1，保证可除）。"""
    mean = (w[:, None] * x).sum(axis=0)
    var = (w[:, None] * (x - mean) ** 2).sum(axis=0)
    sd = np.sqrt(np.maximum(var, 0.0))
    return mean, np.where(sd > _EPS, sd, 1.0)


def _weighted_correlation(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """按样本权重计算列间 Pearson 相关系数矩阵，形状 (a 列数, b 列数)。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / max(w.sum(), _EPS)
    ac = a - (w[:, None] * a).sum(axis=0)
    bc = b - (w[:, None] * b).sum(axis=0)
    cov = (w[:, None] * ac).T @ bc
    va = (w[:, None] * ac ** 2).sum(axis=0)
    vb = (w[:, None] * bc ** 2).sum(axis=0)
    denom = np.sqrt(np.outer(np.maximum(va, 0.0), np.maximum(vb, 0.0)))
    out = np.zeros_like(cov)
    np.divide(cov, denom, out=out, where=denom > 1e-15)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)


def _axis_names(prefix: str, k: int) -> list[str]:
    return [f"{prefix}{i + 1}" for i in range(k)]


def _metric_choices() -> str:
    """把可用的 beta 距离指标格式化成中文提示串。"""
    return "、".join(f"{m}（{BETA_LABELS.get(m, m)}）" for m in BETA_METRICS)


def _standardize(x: np.ndarray, scale: bool = True) -> np.ndarray:
    """按列中心化（可选标准化），零方差列保持为 0。"""
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    if scale:
        sd = x.std(axis=0, ddof=1) if x.shape[0] > 1 else np.ones(x.shape[1])
        sd = np.where(np.isfinite(sd) & (sd > _EPS), sd, 1.0)
        x = x / sd
    return x


def _prepare_env(env: pd.DataFrame | pd.Series, samples: list) -> tuple[np.ndarray, list, list]:
    """对齐并清洗环境/分组变量矩阵。

    Args:
        env: DataFrame 或 Series（样本 × 变量）。
        samples: 丰度表的样本名顺序。

    Returns:
        (数值矩阵, 变量名, 参与分析的样本名)。

    Raises:
        ValueError: 共有样本太少或所有环境变量都是常数。
    """
    if env is None:
        raise ValueError("缺少环境变量矩阵 env（应为 样本 × 变量的数值表）。")
    frame = env.to_frame() if isinstance(env, pd.Series) else pd.DataFrame(env)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    common = [s for s in samples if s in frame.index]
    if len(common) < 4:
        raise ValueError("env 与丰度表的共有样本少于 4 个，无法进行约束排序。")
    sub = frame.loc[common]
    sub = sub.fillna(sub.mean(axis=0)).fillna(0.0)
    keep = [c for c in sub.columns if sub[c].nunique(dropna=False) > 1]
    if not keep:
        raise ValueError("env 中没有任何存在变异的环境变量，无法建立约束。")
    return sub[keep].to_numpy(dtype=float), list(keep), common


# --------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------
def _prepare_groups(groups, samples: list) -> pd.Series:
    """把分组标签统一成以样本名为 index 的字符串 Series。

    允许传入 Series（按样本名对齐）或与样本顺序一致的列表/数组。

    Raises:
        ValueError: 分组标签数量与样本数不一致。
    """
    if isinstance(groups, pd.Series):
        return groups.astype(str)
    arr = np.asarray(groups).astype(str).ravel()
    if arr.size != len(samples):
        raise ValueError(f"分组标签数量（{arr.size}）与样本数（{len(samples)}）不一致。")
    return pd.Series(arr, index=list(samples))


def _pca_core(
    x: np.ndarray,
    n_components: int,
    sample_names: list,
    feature_names: list,
    scale: bool = True,
    prefix: str = "PC",
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """SVD 主成分分析内核。

    Args:
        x: (样本 × 特征) 数值矩阵。
        n_components: 保留的主成分数。
        sample_names / feature_names: 输出的行列名。
        scale: 是否按特征标准差标准化。
        prefix: 轴名前缀。

    Returns:
        (coords, explained, loadings)；loadings 为与主成分的相关系数（biplot 箭头）。

    Raises:
        ValueError: 样本数 < 3 或矩阵没有有效变异。
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n < 3:
        raise ValueError(f"样本数不足（当前 {n} 个），至少需要 3 个样本才能做排序分析。")
    xc = _standardize(x, scale)
    total_var = float((xc ** 2).sum())
    if total_var <= _EPS:
        raise ValueError("数据没有有效变异（所有特征均为常数），无法进行排序分析。")

    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    s2 = s ** 2
    tol = max(1e-14, 1e-10 * (float(s[0]) if s.size else 0.0))
    pos = s > tol
    k = int(min(int(n_components), int(pos.sum())))
    total = float(s2.sum())

    if k < 1:  # 数值退化：保留 1 条零轴，保证输出结构稳定
        coords = pd.DataFrame(np.zeros((n, 1)), index=sample_names, columns=_axis_names(prefix, 1))
        loadings = pd.DataFrame(np.zeros((len(feature_names), 1)), index=feature_names,
                                columns=_axis_names(prefix, 1))
        return coords, np.zeros(1), loadings

    scores = u[:, :k] * s[:k]
    explained = s2[:k] / total

    # 载荷 = 特征与主成分得分的相关系数（相关系数尺度，便于画 biplot 箭头）
    sd_feat = xc.std(axis=0, ddof=1)
    sd_feat = np.where(sd_feat > _EPS, sd_feat, 1.0)
    sd_score = scores.std(axis=0, ddof=1)
    sd_score = np.where(sd_score > _EPS, sd_score, 1.0)
    load = (xc.T @ scores) / max(n - 1, 1)
    load = load / sd_feat[:, None] / sd_score[None, :]
    load = np.nan_to_num(load, nan=0.0, posinf=0.0, neginf=0.0)

    coords = pd.DataFrame(scores, index=sample_names, columns=_axis_names(prefix, k))
    loadings = pd.DataFrame(load, index=feature_names, columns=_axis_names(prefix, k))
    return coords, explained, loadings


def pca_ordination(
    table: pd.DataFrame,
    n_components: int = 4,
    transform: str = "clr",
    scale: bool = True,
) -> dict:
    """样本 PCA（主成分分析）。

    Args:
        table: 丰度表（行 = 特征，列 = 样本）。
        n_components: 保留的主成分个数。
        transform: clr / hellinger / logcpm / relative / none。
        scale: 是否按特征标准化（对 CLR 数据建议 True）。

    Returns:
        dict:
          coords    — 样本 × PC1..PCn 的坐标表；
          explained — 各主成分解释度（和为 1）；
          loadings  — 特征 × PC1..PCn 的相关系数载荷（biplot 箭头）；
          transform — 实际使用的变换名。
    """
    mat, name = _apply_transform(table, transform)
    x = mat.to_numpy(dtype=float).T
    coords, explained, loadings = _pca_core(
        x, n_components, list(mat.columns), list(mat.index), scale=scale, prefix="PC"
    )
    return {
        "coords": coords,
        "explained": explained,
        "loadings": loadings,
        "transform": name,
        "n_components": int(coords.shape[1]),
    }


# --------------------------------------------------------------------------
# 约束排序内核（RDA / dbRDA / CCA 共用）
# --------------------------------------------------------------------------
def _constrained_core(
    x: np.ndarray,
    env: np.ndarray,
    n_components: int,
    *,
    scale: bool = True,
    weights: np.ndarray | None = None,
    center: bool = True,
    prefix: str = "RDA",
    sample_names: list | None = None,
    feature_names: list | None = None,
    env_names: list | None = None,
) -> dict:
    """加权约束排序内核：多元线性回归 + 拟合值 PCA + 残差投影。

    做法（McArdle & Anderson 风格的加权版）：
      1. 用样本权重对工作矩阵（站点 × 变量）加权中心化/标准化；
      2. 对（同样加权中心化的）环境变量做最小二乘回归，得到拟合值；
      3. 对拟合值做 SVD 得到约束轴，残差投影到约束轴上；
      4. 约束惯量 / 总惯量来自加权平方和分解。

    Args:
        x: (样本 × 变量) 工作矩阵（已变换的特征或 PCoA 坐标）。
        env: (样本 × q) 环境变量矩阵，与 x 行一一对应。
        n_components: 保留的约束轴数。
        scale: 是否对 x 的列做标准化。
        weights: 样本权重（转置后和归一），默认等权。
        center: 是否对 x 做加权中心化（CCA 的卡方残差已中心化，用 False）。
        prefix: 轴名前缀。
        sample_names / feature_names / env_names: 输出的行列名。

    Returns:
        dict，含 coords / sites_constrained / sites_residual / explained /
        env_biplot / feature_biplot / constrained_inertia / total_inertia /
        n_components。
    """
    x = np.asarray(x, dtype=float)
    env = np.asarray(env, dtype=float)
    n = x.shape[0]
    if sample_names is None:
        sample_names = [f"S{i + 1}" for i in range(n)]
    if feature_names is None:
        feature_names = [f"V{i + 1}" for i in range(x.shape[1])]
    if env_names is None:
        env_names = [f"E{i + 1}" for i in range(env.shape[1])]

    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.clip(np.asarray(weights, dtype=float).ravel(), 0.0, None)
        if w.size != n or w.sum() <= _EPS:
            raise ValueError("样本权重与样本数不匹配或总和为 0，无法进行约束排序。")
        w = w / w.sum()

    sw = np.sqrt(w)[:, None]
    if center:
        xm, xsd = _weighted_stats(x, w)
        xc = x - xm
    else:
        xc = x
        _, xsd = _weighted_stats(x, w)
    if scale:
        xc = xc / xsd

    a = sw * xc                                # 加权响应矩阵
    total_inertia = float((a ** 2).sum())

    em, esd = _weighted_stats(env, w)
    ec = (env - em) / np.where(esd > _EPS, esd, 1.0)   # 环境变量标准化不影响拟合子空间
    z = sw * ec                                # 加权设计矩阵

    try:
        coef, *_ = np.linalg.lstsq(z, a, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover - 极端病态时回退伪逆
        coef = np.linalg.pinv(z) @ a
    coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
    fitted = z @ coef
    residual = a - fitted
    constrained_inertia = float((fitted ** 2).sum())

    # 拟合值 SVD → 约束轴；特征值 ≤ 0 的轴安全截断
    if constrained_inertia > _EPS:
        u, s, vt = np.linalg.svd(fitted, full_matrices=False)
        tol = max(1e-14, 1e-10 * float(s[0]))
        k = int(min(int(n_components), int((s > tol).sum())))
    else:
        u, s, vt, k = np.zeros((n, 0)), np.zeros(0), np.zeros((0, x.shape[1])), 0
    if k < 1:  # 数值退化：保留 1 条零轴，保证输出结构稳定
        k = 1
        u, s, vt = np.zeros((n, 1)), np.zeros(1), np.zeros((1, x.shape[1]))

    # 站点坐标为平方根特征值标度（加权范数 = sqrt(特征值)）
    site = u[:, :k] * s[:k] / np.sqrt(np.maximum(w, _EPS))[:, None]
    eig = (s[:k] ** 2) / max(total_inertia, _EPS)
    explained = np.clip(np.nan_to_num(eig, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    cols = _axis_names(prefix, k)
    coords = pd.DataFrame(site, index=sample_names, columns=cols)
    residual_sites = pd.DataFrame(residual @ vt[:k].T, index=sample_names, columns=cols)

    env_corr = _weighted_correlation(env, site, w)
    feat_corr = _weighted_correlation(x, site, w)
    env_biplot = pd.DataFrame(env_corr, index=env_names, columns=cols)
    feature_biplot = pd.DataFrame(feat_corr, index=feature_names, columns=cols)

    return {
        "coords": coords,
        "sites_constrained": coords.copy(),
        "sites_residual": residual_sites,
        "explained": explained,
        "env_biplot": env_biplot,
        "feature_biplot": feature_biplot,
        "constrained_inertia": constrained_inertia,
        "total_inertia": total_inertia,
        "n_components": int(k),
        "eigenvalues": (s[:k] ** 2),
    }


def rda(
    table: pd.DataFrame,
    env: pd.DataFrame,
    n_components: int = 4,
    scale: bool = True,
) -> dict:
    """冗余分析（RDA，约束排序）。

    先对丰度表做 CLR 变换（成分数据的标准做法），再对样本矩阵做多元线性回归
    （解释变量为 env），对拟合值做 PCA 得到约束轴，最后把残差投影到约束轴上。
    总惯量为变换后数据的加权平方和。

    Args:
        table: 丰度表（行 = 特征，列 = 样本）。
        env: 环境/分组变量表（样本 × 变量，数值型）。
        n_components: 保留的约束轴数。
        scale: 是否按特征标准化。

    Returns:
        dict: coords、sites_constrained、sites_residual、explained、env_biplot、
        feature_biplot、constrained_inertia、total_inertia、transform、n_components。

    Raises:
        ValueError: 共有样本不足或 env 全为常数。
    """
    mat = clr_transform(as_numeric_matrix(table))
    env_arr, env_names, common = _prepare_env(env, list(mat.columns))
    x = mat[common].to_numpy(dtype=float).T
    core = _constrained_core(
        x, env_arr, n_components,
        scale=scale, weights=None, center=True,
        prefix="RDA",
        sample_names=common,
        feature_names=list(mat.index),
        env_names=env_names,
    )
    core["transform"] = "clr"
    return core


def dbrda(
    table: pd.DataFrame,
    env: pd.DataFrame,
    metric: str = "braycurtis",
    n_components: int = 4,
) -> dict:
    """基于距离的冗余分析（dbRDA，McArdle & Anderson 2001）。

    先算样本距离矩阵 → PCoA 取全部正特征轴得到 Gower 中心化坐标 → 对坐标做 RDA。

    Args:
        table: 丰度表（行 = 特征，列 = 样本）。
        env: 环境/分组变量表（样本 × 变量）。
        metric: BETA_METRICS 中的距离指标。
        n_components: 保留的约束轴数。

    Returns:
        dict: 与 rda 相同，另加 metric 与 pcoa_axes（正特征轴个数）。

    Raises:
        ValueError: metric 非法、样本不足或距离矩阵无正特征值。
    """
    if metric not in BETA_METRICS:
        raise ValueError(f"未知的距离指标：{metric}，可选 {_metric_choices()}。")
    mat = as_numeric_matrix(table)
    if mat.shape[1] < 4:
        raise ValueError(f"样本数不足（当前 {mat.shape[1]} 个），dbRDA 至少需要 4 个样本。")

    dist = distance_matrix(mat, metric=metric)
    coords_all, _ = pcoa(dist, n_axes=dist.shape[0])
    pcoa_axes = int(coords_all.shape[1])
    if pcoa_axes < 1:
        raise ValueError("距离矩阵没有正特征值（样本间距离全为 0），无法进行 dbRDA。")

    env_arr, env_names, common = _prepare_env(env, list(coords_all.index))
    x = coords_all.loc[common].to_numpy(dtype=float)
    core = _constrained_core(
        x, env_arr, n_components,
        scale=False, weights=None, center=True,
        prefix="dbRDA",
        sample_names=common,
        feature_names=list(coords_all.columns),
        env_names=env_names,
    )
    core["metric"] = metric
    core["pcoa_axes"] = pcoa_axes
    core["transform"] = f"pcoa({metric})"
    return core


def cca(table: pd.DataFrame, env: pd.DataFrame, n_components: int = 4) -> dict:
    """典范对应分析（CCA）。

    用卡方距离（对应分析）的加权形式：先构造卡方标准化残差矩阵
    Z = D_r^{-1/2} (P - r c^T) D_c^{-1/2}，再以样本质量 c 为权重、
    用 env 做加权约束回归并对拟合值做加权 PCA。总惯量即卡方统计量 / 总计数。

    Args:
        table: 丰度表（行 = 特征，列 = 样本，非负）。
        env: 环境/分组变量表（样本 × 变量）。
        n_components: 保留的约束轴数。

    Returns:
        dict: 结构同 rda（轴名前缀为 CCA），另加 chi_square（总卡方统计量）。

    Raises:
        ValueError: 表非正、样本不足或 env 全为常数。
    """
    mat = as_numeric_matrix(table)
    if (mat.to_numpy(dtype=float) < 0).any():
        raise ValueError("CCA 要求丰度表非负，请检查输入数据。")
    total_count = float(mat.to_numpy(dtype=float).sum())
    if total_count <= _EPS:
        raise ValueError("丰度表总计数为 0，无法进行 CCA。")
    if mat.shape[1] < 4:
        raise ValueError(f"样本数不足（当前 {mat.shape[1]} 个），CCA 至少需要 4 个样本。")

    env_arr, env_names, common = _prepare_env(env, list(mat.columns))
    sub = mat[common].to_numpy(dtype=float)
    p = sub / total_count
    row_mass = p.sum(axis=1)                     # 特征质量 r
    col_mass = p.sum(axis=0)                     # 样本质量 c
    expected = np.outer(row_mass, col_mass)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (p - expected) / np.sqrt(np.maximum(expected, _EPS))
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    chi_square = float(total_count * (z ** 2).sum())

    core = _constrained_core(
        z.T, env_arr, n_components,
        scale=False, weights=col_mass, center=False,
        prefix="CCA",
        sample_names=common,
        feature_names=list(mat.index),
        env_names=env_names,
    )
    core["chi_square"] = chi_square
    core["transform"] = "chi-square"
    return core


# --------------------------------------------------------------------------
# CAP（典范分析主坐标）
# --------------------------------------------------------------------------
def _lda_loo_accuracy(x: np.ndarray, y: np.ndarray) -> float:
    """用留一交叉验证评估 LDA 分类准确率。"""
    n = x.shape[0]
    if n < 3:
        return float("nan")
    correct = 0
    cv = LeaveOneOut()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train, test in cv.split(x):
            clf = LinearDiscriminantAnalysis(solver="svd")
            try:
                clf.fit(x[train], y[train])
                correct += int((clf.predict(x[test]) == y[test]).sum())
            except Exception:  # pragma: no cover - 退化子集，按错处理
                continue
    return correct / n


def cap(
    table: pd.DataFrame,
    groups: pd.Series | list | np.ndarray,
    metric: str = "braycurtis",
    n_components: int = 4,
) -> dict:
    """典范分析主坐标（CAP，Anderson 2003）。

    PCoA 取前 m 个轴 → 对 m 维坐标做线性判别分析 → 输出典范轴与留一交叉验证准确率。
    m 的选择：在 1..min(样本数-2, 正特征轴数, 20) 上做留一交叉验证，
    取准确率最高的最小 m。

    Args:
        table: 丰度表（行 = 特征，列 = 样本）。
        groups: 分组标签（Series，index 为样本名）。
        metric: BETA_METRICS 中的距离指标。
        n_components: 保留的典范轴数。

    Returns:
        dict: coords(CAP1..CAPk)、explained、accuracy、m、classes、metric。

    Raises:
        ValueError: 指标非法、样本/分组数不足、每组样本少于 2 个。
    """
    if metric not in BETA_METRICS:
        raise ValueError(f"未知的距离指标：{metric}，可选 {_metric_choices()}。")
    mat = as_numeric_matrix(table)
    labels = _prepare_groups(groups, list(mat.columns))
    common = [s for s in mat.columns if s in labels.index]
    if len(common) < 5:
        raise ValueError(f"样本数不足（当前 {len(common)} 个），CAP 至少需要 5 个样本。")
    lab = labels.loc[common]
    cats = sorted(pd.unique(lab))
    if len(cats) < 2:
        raise ValueError("CAP 至少需要 2 个分组。")
    counts = lab.value_counts()
    if int(counts.min()) < 2:
        raise ValueError("CAP 要求每个分组至少 2 个样本（否则无法交叉验证）。")

    dist = distance_matrix(mat[common], metric=metric)
    coords_all, explained_all = pcoa(dist, n_axes=len(common))
    k_all = int(coords_all.shape[1])
    m_max = int(min(len(common) - 2, k_all, 20))
    if m_max < 1:
        raise ValueError("可用于 CAP 的 PCoA 轴数为 0，无法进行判别分析。")

    y = lab.to_numpy()
    best_acc, best_m = -1.0, 1
    for m in range(1, m_max + 1):
        acc = _lda_loo_accuracy(coords_all.iloc[:, :m].to_numpy(dtype=float), y)
        if np.isfinite(acc) and acc > best_acc + 1e-12:
            best_acc, best_m = float(acc), int(m)

    sel = coords_all.iloc[:, :best_m].to_numpy(dtype=float)
    n_cap = int(min(n_components, best_m, len(cats) - 1, len(common) - 1))
    n_cap = max(n_cap, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lda = LinearDiscriminantAnalysis(solver="svd", n_components=n_cap)
        scores = lda.fit(sel, y).transform(sel)
    coords = pd.DataFrame(scores, index=common, columns=_axis_names("CAP", scores.shape[1]))
    expl = np.asarray(getattr(lda, "explained_variance_ratio_", np.zeros(scores.shape[1])), dtype=float)
    expl = np.nan_to_num(expl[: scores.shape[1]], nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "coords": coords,
        "explained": expl,
        "accuracy": float(best_acc),
        "m": int(best_m),
        "classes": [str(c) for c in cats],
        "metric": metric,
        "pcoa_explained": np.asarray(explained_all, dtype=float),
    }


# --------------------------------------------------------------------------
# 非线性降维
# --------------------------------------------------------------------------
def tsne_embedding(
    table: pd.DataFrame,
    transform: str = "clr",
    perplexity: float | None = None,
    seed: int = 42,
    n_components: int = 2,
) -> pd.DataFrame:
    """t-SNE 样本降维。

    Args:
        table: 丰度表（行 = 特征，列 = 样本）。
        transform: clr / hellinger / logcpm / relative / none。
        perplexity: 默认 min(30, max(5, (n_samples-1)//3))。
        seed: 随机种子。
        n_components: 嵌入维度（默认 2）。

    Returns:
        DataFrame（样本 × TSNE1..TSNEn）。

    Raises:
        ValueError: 样本数 < 5 或 n_components 不小于样本数。
    """
    mat, _ = _apply_transform(table, transform)
    x = _standardize(mat.to_numpy(dtype=float).T, scale=True)
    n = x.shape[0]
    if n < 5:
        raise ValueError(f"样本数不足（当前 {n} 个），t-SNE 至少需要 5 个样本。")
    if int(n_components) >= n:
        raise ValueError(f"n_components（{n_components}）必须小于样本数（{n}）。")

    if perplexity is None:
        perplexity = min(30.0, max(5.0, (n - 1) // 3))
    perp = float(min(max(float(perplexity), 1.0), n - 1))

    model = TSNE(
        n_components=int(n_components),
        perplexity=perp,
        random_state=seed,
        init="pca" if n > int(n_components) else "random",
        learning_rate="auto",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb = model.fit_transform(x)
    emb = np.nan_to_num(np.asarray(emb, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(emb, index=list(mat.columns), columns=_axis_names("TSNE", emb.shape[1]))


def umap_embedding(
    table: pd.DataFrame,
    transform: str = "clr",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
    n_components: int = 2,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """UMAP 样本降维（惰性导入 umap-learn）。

    Args:
        table: 丰度表（行 = 特征，列 = 样本）。
        transform: clr / hellinger / logcpm / relative / none。
        n_neighbors: 邻居数，自动截断为 min(n_neighbors, n_samples-1) 且至少为 2。
        min_dist: 嵌入点最小距离。
        seed: 随机种子。
        n_components: 嵌入维度（默认 2）。
        metric: 距离度量。

    Returns:
        DataFrame（样本 × UMAP1..UMAPn）。

    Raises:
        RuntimeError: 未安装 umap-learn。
        ValueError: 样本数 < 5 或 n_components 不小于样本数。
    """
    try:
        import umap  # 惰性导入：umap-learn 为可选依赖
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise RuntimeError(
            "未安装 umap-learn，无法执行 UMAP 降维。请先运行：pip install umap-learn"
        ) from exc

    mat, _ = _apply_transform(table, transform)
    x = _standardize(mat.to_numpy(dtype=float).T, scale=True)
    n = x.shape[0]
    if n < 5:
        raise ValueError(f"样本数不足（当前 {n} 个），UMAP 至少需要 5 个样本。")
    if int(n_components) >= n:
        raise ValueError(f"n_components（{n_components}）必须小于样本数（{n}）。")

    k = int(max(2, min(int(n_neighbors), n - 1)))
    reducer = umap.UMAP(
        n_neighbors=k,
        min_dist=float(min_dist),
        n_components=int(n_components),
        random_state=seed,
        metric=metric,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb = np.asarray(reducer.fit_transform(x), dtype=float)
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(emb, index=list(mat.columns), columns=_axis_names("UMAP", emb.shape[1]))


# --------------------------------------------------------------------------
# PLS-DA
# --------------------------------------------------------------------------
def vip_scores(x: np.ndarray, y: np.ndarray, model, n_components: int) -> np.ndarray:
    """计算 PLS-DA 的 VIP（Variable Importance in Projection）值。

    标准公式：VIP_j = sqrt( p * Σ_a SS_a (w_ja / ||w_a||)^2 / Σ_a SS_a )，
    其中 SS_a 为成分 a 对 Y 的解释平方和（t_a 与 y 载荷的乘积）。

    Args:
        x: 自变量矩阵（样本 × 特征）。
        y: 因变量矩阵（样本 × 类别数，one-hot 或指示变量）。
        model: 已拟合的 PLSRegression。
        n_components: 参与计算的成分数。

    Returns:
        ndarray，长度等于特征数 p。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = x.shape[1]
    t = np.asarray(getattr(model, "x_scores_", np.zeros((x.shape[0], 0))), dtype=float)
    w = np.asarray(getattr(model, "x_weights_", np.zeros((p, 0))), dtype=float)
    k = int(min(int(n_components), t.shape[1], w.shape[1]))
    if k < 1:
        return np.zeros(p)
    t, w = t[:, :k], w[:, :k]

    q = getattr(model, "y_loadings_", None)
    if q is not None:
        q = np.asarray(q, dtype=float)[:, :k]
        ss = (t ** 2).sum(axis=0) * (q ** 2).sum(axis=0)
    else:  # pragma: no cover - 兜底：由 t 与中心化后的 Y 现算载荷
        yc = y - y.mean(axis=0, keepdims=True)
        denom = np.where((t ** 2).sum(axis=0) > _EPS, (t ** 2).sum(axis=0), 1.0)
        q = (t.T @ yc) / denom[:, None]
        ss = (t ** 2).sum(axis=0) * (q ** 2).sum(axis=0)

    if not np.isfinite(ss).any() or float(np.nansum(ss)) <= _EPS:
        ss = (t ** 2).sum(axis=0)
    if float(np.nansum(ss)) <= _EPS:  # pragma: no cover - 完全退化
        return np.zeros(p)

    wnorm2 = (w ** 2).sum(axis=0)
    wnorm2 = np.where(wnorm2 > _EPS, wnorm2, 1.0)
    contrib = (w ** 2 / wnorm2) @ ss
    vip = np.sqrt(p * contrib / float(np.sum(ss)))
    return np.nan_to_num(vip, nan=0.0, posinf=0.0, neginf=0.0)


def _pls_fit(x: np.ndarray, y: np.ndarray, n_components: int) -> PLSRegression:
    """拟合 PLSRegression（成分数自动截断到合法范围）。"""
    k = int(max(1, min(int(n_components), x.shape[0] - 1, x.shape[1])))
    model = PLSRegression(n_components=k, scale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x, y)
    return model


def _pls_accuracy(x: np.ndarray, y_idx: np.ndarray, n_components: int, seed: int = 42) -> float:
    """PLS-DA 的分类准确率：留一交叉验证与分层 K 折中取较小者（更保守）。"""
    n_classes = int(y_idx.max()) + 1

    def _run(cv) -> float:
        correct = 0
        total = 0
        for train, test in cv.split(x, y_idx):
            if test.size == 0 or np.unique(y_idx[train]).size < 2:
                continue
            k = int(max(1, min(int(n_components), train.size - 1, x.shape[1])))
            model = PLSRegression(n_components=k, scale=False)
            ytr = np.eye(n_classes)[y_idx[train]]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x[train], ytr)
                    pred = np.asarray(model.predict(x[test]), dtype=float)
            except Exception:  # pragma: no cover - 退化折
                continue
            hit = np.argmax(pred, axis=1) == y_idx[test]
            correct += int(hit.sum())
            total += int(test.size)
        return correct / total if total > 0 else float("nan")

    accs = [_run(LeaveOneOut())]
    counts = np.bincount(y_idx, minlength=n_classes)
    n_splits = int(min(5, counts[counts > 0].min()))
    if n_splits >= 2 and x.shape[0] >= 2 * n_splits:
        accs.append(_run(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)))
    accs = [a for a in accs if np.isfinite(a)]
    return float(min(accs)) if accs else float("nan")


def plsda(
    table: pd.DataFrame,
    groups: pd.Series | list | np.ndarray,
    n_components: int = 2,
    transform: str = "clr",
    scale: bool = True,
) -> dict:
    """偏最小二乘判别分析（PLS-DA）。

    用 sklearn.cross_decomposition.PLSRegression，对分组做 one-hot 编码作为 Y。

    Args:
        table: 丰度表（行 = 特征，列 = 样本）。
        groups: 分组标签（Series，index 为样本名）。
        n_components: PLS 成分数。
        transform: clr / hellinger / logcpm / relative / none。
        scale: 是否按特征标准化。

    Returns:
        dict:
          coords    — 样本 × PLS1..PLSk 得分；
          explained — 各成分解释的 X 方差比例；
          vip       — DataFrame(columns=["feature", "VIP"]，按 VIP 降序)；
          accuracy  — 留一 / 分层 5 折中较小者的准确率；
          classes   — 类别列表；
          r2x / r2y — 累计解释度。

    Raises:
        ValueError: 样本或分组不足、每组样本少于 2 个。
    """
    mat, name = _apply_transform(table, transform)
    labels = _prepare_groups(groups, list(mat.columns))
    common = [s for s in mat.columns if s in labels.index]
    if len(common) < 5:
        raise ValueError(f"样本数不足（当前 {len(common)} 个），PLS-DA 至少需要 5 个样本。")
    lab = labels.loc[common]
    cats = sorted(pd.unique(lab))
    if len(cats) < 2:
        raise ValueError("PLS-DA 至少需要 2 个分组。")
    if int(lab.value_counts().min()) < 2:
        raise ValueError("PLS-DA 要求每个分组至少 2 个样本。")

    x = _standardize(mat[common].to_numpy(dtype=float).T, scale=scale)
    y_idx = np.array([cats.index(v) for v in lab.to_numpy()], dtype=int)
    y = np.eye(len(cats))[y_idx]

    k = int(max(1, min(int(n_components), x.shape[0] - 1, x.shape[1])))
    model = _pls_fit(x, y, k)
    t = np.asarray(model.x_scores_, dtype=float)
    p_load = np.asarray(model.x_loadings_, dtype=float)
    ss_x = float((x ** 2).sum())
    explained = np.array(
        [(t[:, i] ** 2).sum() * (p_load[:, i] ** 2).sum() / max(ss_x, _EPS) for i in range(t.shape[1])]
    )
    explained = np.clip(np.nan_to_num(explained, nan=0.0), 0.0, 1.0)
    if explained.sum() > 1.0:  # 数值上溢时归一
        explained = explained / explained.sum()

    y_pred = np.asarray(model.predict(x), dtype=float)
    ss_tot = float(((y - y.mean(axis=0, keepdims=True)) ** 2).sum())
    r2y = 1.0 - float(((y - y_pred) ** 2).sum()) / max(ss_tot, _EPS)

    vip = vip_scores(x, y, model, t.shape[1])
    vip_df = (
        pd.DataFrame({"feature": list(mat.index), "VIP": vip})
        .sort_values("VIP", ascending=False)
        .reset_index(drop=True)
    )

    coords = pd.DataFrame(t, index=common, columns=_axis_names("PLS", t.shape[1]))
    return {
        "coords": coords,
        "explained": explained,
        "vip": vip_df,
        "accuracy": _pls_accuracy(x, y_idx, k),
        "classes": [str(c) for c in cats],
        "r2x": float(explained.sum()),
        "r2y": float(r2y),
        "transform": name,
        "n_components": int(t.shape[1]),
    }


# --------------------------------------------------------------------------
# 碎石图数据
# --------------------------------------------------------------------------
def scree_table(
    coords_or_dist: pd.DataFrame | tuple | dict | np.ndarray,
    max_axes: int = 10,
) -> pd.DataFrame:
    """给碎石图用的数据表。

    Args:
        coords_or_dist: 以下任一
            * pcoa 的 (coords, explained) 元组；
            * 含 "explained" 键的 dict（如本模块 pca_ordination / rda 的返回值）；
            * 一维 explained 数组；
            * 距离矩阵 DataFrame（内部会做 PCoA）。
        max_axes: 最多输出的轴数。

    Returns:
        DataFrame(columns=["axis", "explained", "cumulative"])。
    """
    src = coords_or_dist
    if isinstance(src, tuple) and len(src) == 2:
        explained = src[1]
    elif isinstance(src, dict):
        explained = src.get("explained")
    elif isinstance(src, pd.DataFrame):
        _, explained = pcoa(src, n_axes=max(1, int(max_axes)))
    else:
        explained = src
    if explained is None:
        raise ValueError("无法从输入中解析出 explained（解释度）数组。")

    arr = np.asarray(explained, dtype=float).ravel()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr[: max(1, int(max_axes))]
    if arr.size == 0:
        arr = np.zeros(1)
    return pd.DataFrame(
        {
            "axis": np.arange(1, arr.size + 1),
            "explained": arr,
            "cumulative": np.cumsum(arr),
        }
    )


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------
def _make_grouped_table(seed: int = 42):
    """3 组 × 15 样本 × 200 特征的对数正态丰度表（含明显组效应）。"""
    rng = np.random.default_rng(seed)
    n_per, n_feat = 15, 200
    labels = np.array(["A"] * n_per + ["B"] * n_per + ["C"] * n_per)
    base = np.exp(rng.normal(0.0, 1.0, n_feat))
    effect = np.ones((n_feat, 3))
    for g, idx in enumerate(np.array_split(np.arange(n_feat), 3)):
        effect[idx, g] = np.exp(rng.normal(2.0, 0.25, idx.size))
    vals = np.zeros((n_feat, labels.size))
    for j, lab in enumerate(labels):
        g = {"A": 0, "B": 1, "C": 2}[lab]
        vals[:, j] = base * effect[:, g] * np.exp(rng.normal(0.0, 0.4, n_feat))
    table = pd.DataFrame(
        np.round(vals),
        index=[f"F{i:03d}" for i in range(n_feat)],
        columns=[f"S{j:02d}" for j in range(labels.size)],
    )
    return table, pd.Series(labels, index=table.columns)


def _make_gradient_table(seed: int = 7):
    """45 样本 × 200 特征的梯度数据：所有特征随连续梯度单调变化。"""
    rng = np.random.default_rng(seed)
    n, n_feat = 45, 200
    grad = np.linspace(-1.0, 1.0, n) + rng.normal(0.0, 0.05, n)
    base = np.exp(rng.normal(0.0, 0.8, n_feat))
    slope = rng.normal(0.0, 1.2, n_feat)
    vals = base[:, None] * np.exp(slope[:, None] * grad[None, :] + rng.normal(0.0, 0.3, (n_feat, n)))
    table = pd.DataFrame(
        vals,
        index=[f"G{i:03d}" for i in range(n_feat)],
        columns=[f"T{j:02d}" for j in range(n)],
    )
    env = pd.DataFrame({"Gradient": grad, "Noise": rng.normal(0.0, 1.0, n)}, index=table.columns)
    return table, env


def _kmeans_accuracy(coords: pd.DataFrame, labels: pd.Series) -> float:
    """KMeans 聚类后按多数投票映射标签的准确率。"""
    from sklearn.cluster import KMeans

    x = coords.to_numpy(dtype=float)
    y = labels.loc[coords.index].to_numpy()
    km = KMeans(n_clusters=len(pd.unique(y)), n_init=10, random_state=0)
    cl = km.fit_predict(x)
    correct = 0
    for c in np.unique(cl):
        true = y[cl == c]
        if true.size:
            correct += int((true == pd.Series(true).mode().iloc[0]).sum())
    return correct / y.size


def _self_check() -> None:
    """合成数据自检：全部断言通过才算成功。"""
    warnings.filterwarnings("ignore")

    grouped, groups = _make_grouped_table()
    grad_table, env = _make_gradient_table()
    results: dict = {}

    print("=" * 68)
    print("AmpliconDesk 排序模块自检（合成数据）")
    print("=" * 68)

    # ---------------- PCA ----------------
    pca = pca_ordination(grouped, n_components=4, transform="clr", scale=True)
    pc12 = float(pca["explained"][:2].sum())
    acc_pca = _kmeans_accuracy(pca["coords"].iloc[:, :2], groups)
    results["PCA"] = {"PC1-2 解释度": pc12, "KMeans 准确率": acc_pca,
                      "载荷形状": tuple(pca["loadings"].shape)}
    print(f"[PCA]    PC1={pca['explained'][0]:.3f} PC2={pca['explained'][1]:.3f} "
          f"PC1-2={pc12:.3f} | KMeans 准确率={acc_pca:.3f} | coords={pca['coords'].shape} "
          f"loadings={pca['loadings'].shape}")
    assert pc12 > 0.40, f"PCA 前两轴解释度不足：{pc12:.3f}"
    assert acc_pca > 0.90, f"PCA 前两轴分类准确率不足：{acc_pca:.3f}"

    # ---------------- RDA ----------------
    rda_res = rda(grad_table, env[["Gradient"]], n_components=4)
    r_rda = float(np.corrcoef(rda_res["coords"]["RDA1"].to_numpy(),
                              env.loc[rda_res["coords"].index, "Gradient"].to_numpy())[0, 1])
    ratio_rda = rda_res["constrained_inertia"] / max(rda_res["total_inertia"], 1e-12)
    results["RDA"] = {"|r(RDA1, Gradient)|": abs(r_rda), "约束/总惯量": ratio_rda}
    print(f"[RDA]    RDA1~Gradient r={r_rda:+.3f} (|r|={abs(r_rda):.3f}) | "
          f"约束惯量={rda_res['constrained_inertia']:.4f} 总惯量={rda_res['total_inertia']:.4f} "
          f"比值={ratio_rda:.3f} | 轴数={rda_res['n_components']}")
    assert abs(r_rda) > 0.80, f"RDA1 与环境变量相关性不足：{abs(r_rda):.3f}"
    assert ratio_rda > 0.30, f"RDA 约束惯量占比不足：{ratio_rda:.3f}"

    # ---------------- dbRDA ----------------
    dbr = dbrda(grad_table, env[["Gradient"]], metric="braycurtis", n_components=4)
    r_dbr = float(np.corrcoef(dbr["coords"]["dbRDA1"].to_numpy(),
                              env.loc[dbr["coords"].index, "Gradient"].to_numpy())[0, 1])
    results["dbRDA"] = {"|r(dbRDA1, Gradient)|": abs(r_dbr), "PCoA 正特征轴": dbr["pcoa_axes"]}
    print(f"[dbRDA]  dbRDA1~Gradient r={r_dbr:+.3f} (|r|={abs(r_dbr):.3f}) | "
          f"PCoA 正特征轴={dbr['pcoa_axes']} | 轴数={dbr['n_components']}")
    assert abs(r_dbr) > 0.80, f"dbRDA1 与环境变量相关性不足：{abs(r_dbr):.3f}"

    # ---------------- CCA ----------------
    cca_res = cca(grad_table, env[["Gradient"]], n_components=4)
    r_cca = float(np.corrcoef(cca_res["coords"]["CCA1"].to_numpy(),
                              env.loc[cca_res["coords"].index, "Gradient"].to_numpy())[0, 1])
    results["CCA"] = {"|r(CCA1, Gradient)|": abs(r_cca), "卡方": cca_res["chi_square"]}
    print(f"[CCA]    CCA1~Gradient r={r_cca:+.3f} (|r|={abs(r_cca):.3f}) | "
          f"总卡方={cca_res['chi_square']:.2f} | 约束/总惯量="
          f"{cca_res['constrained_inertia'] / max(cca_res['total_inertia'], 1e-12):.3f}")
    assert abs(r_cca) > 0.70, f"CCA1 与环境变量相关性不足：{abs(r_cca):.3f}"

    # ---------------- CAP ----------------
    cap_res = cap(grouped, groups, metric="braycurtis", n_components=4)
    results["CAP"] = {"留一准确率": cap_res["accuracy"], "m": cap_res["m"]}
    print(f"[CAP]    留一交叉验证准确率={cap_res['accuracy']:.3f} | m={cap_res['m']} | "
          f"CAP 轴数={cap_res['coords'].shape[1]} | 类别={cap_res['classes']}")
    assert cap_res["accuracy"] > 0.80, f"CAP 留一准确率不足：{cap_res['accuracy']:.3f}"

    # ---------------- t-SNE / UMAP ----------------
    ts = tsne_embedding(grouped, transform="clr", perplexity=None, seed=42)
    um = umap_embedding(grouped, transform="clr", seed=42)
    results["t-SNE"] = {"shape": tuple(ts.shape), "含 NaN": bool(ts.isna().any().any())}
    results["UMAP"] = {"shape": tuple(um.shape), "含 NaN": bool(um.isna().any().any())}
    print(f"[t-SNE]  shape={ts.shape} NaN={bool(ts.isna().any().any())} "
          f"列={list(ts.columns)}")
    print(f"[UMAP]   shape={um.shape} NaN={bool(um.isna().any().any())} "
          f"列={list(um.columns)}")
    assert ts.shape == (45, 2) and not ts.isna().any().any(), "t-SNE 输出异常"
    assert um.shape == (45, 2) and not um.isna().any().any(), "UMAP 输出异常"

    # ---------------- PLS-DA ----------------
    pls = plsda(grouped, groups, n_components=2, transform="clr", scale=True)
    vip_mean = float(pls["vip"]["VIP"].mean())
    results["PLS-DA"] = {"VIP 长度": int(pls["vip"].shape[0]), "VIP 均值": vip_mean,
                         "准确率": pls["accuracy"], "r2x": pls["r2x"], "r2y": pls["r2y"]}
    print(f"[PLS-DA] VIP 长度={pls['vip'].shape[0]} VIP 均值={vip_mean:.3f} | "
          f"准确率={pls['accuracy']:.3f} | R2X={pls['r2x']:.3f} R2Y={pls['r2y']:.3f} | "
          f"coords={pls['coords'].shape}")
    assert pls["vip"].shape[0] == grouped.shape[0], "VIP 长度与特征数不一致"
    assert abs(vip_mean - 1.0) <= 0.20, f"VIP 均值偏离 1：{vip_mean:.3f}"
    assert pls["accuracy"] > 0.85, f"PLS-DA 准确率不足：{pls['accuracy']:.3f}"

    # ---------------- scree_table ----------------
    st1 = scree_table((pca["coords"], pca["explained"]))
    st2 = scree_table(pca)
    st3 = scree_table(distance_matrix(grouped, "braycurtis"), max_axes=5)
    print(f"[scree]  元组输入={st1.shape} dict 输入={st2.shape} 距离矩阵输入={st3.shape} | "
          f"首行={st1.iloc[0].to_dict()}")
    assert list(st1.columns) == ["axis", "explained", "cumulative"]
    assert np.isclose(st2["cumulative"].iloc[-1], pca["explained"].sum())
    assert st3.shape[0] == 5

    # ---------------- 边界情况 ----------------
    try:
        tsne_embedding(grouped.iloc[:, :4])
        raise AssertionError("t-SNE 样本不足时应抛 ValueError")
    except ValueError as exc:
        print(f"[边界]   t-SNE 样本不足 → ValueError: {exc}")
    try:
        umap_embedding(grouped.iloc[:, :3])
        raise AssertionError("UMAP 样本不足时应抛 ValueError")
    except ValueError as exc:
        print(f"[边界]   UMAP 样本不足 → ValueError: {exc}")
    try:
        cca(grouped, env.iloc[:, :0])
        raise AssertionError("env 为空时应抛 ValueError")
    except ValueError as exc:
        print(f"[边界]   env 为空 → ValueError: {exc}")

    print("-" * 68)
    print("全部自检通过 ✅")


if __name__ == "__main__":
    _self_check()
