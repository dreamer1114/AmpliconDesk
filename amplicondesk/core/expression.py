"""表达谱分析：计数归一化、负二项差异表达、样本相关性与 PCA。

面向宏转录组 / 转录组的基因水平表达表（原始计数），提供：
  * TMM（edgeR 风格）与 DESeq2 中位数比值两种归一化
  * NB-GLM Wald 检验（共有离散度），并保留 Wilcoxon 作为稳健备选
  * log2-CPM 空间的样本相关性、PCA

全部用 numpy / scipy 实现，不依赖 statsmodels 或 R。
"""

# ============================================================================
# 【未启用】本模块不属于 16S 扩增子的分析流程
# ----------------------------------------------------------------------------
# 宏转录组专属：文库归一化（TMM / DESeq2 中位数比值 / CPM）与负二项 GLM Wald 差异表达检验。
#
# AmpliconDesk 定位为 16S 扩增子下游分析工具，**本模块不再被任何流程引用**，
# 保留是为了将来需要时可以直接接回。16S 的差异分析走
# ``amplicondesk.core.stats.differential_abundance``（CLR + Wilcoxon 秩和）。
#
# 如需重新启用：在 pipeline 的 ``run_differential`` 里按 analysis_type 分支调用，
# 并恢复 ``normalization`` / ``de_method`` 两个配置项。
# ============================================================================


from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .preprocess import as_numeric_matrix
from .stats import bh_fdr, cliffs_delta

EPS = 1e-12


# --------------------------------------------------------------------------
# 归一化
# --------------------------------------------------------------------------
def deseq_size_factors(counts: np.ndarray) -> np.ndarray:
    """DESeq2 的中位数比值法尺寸因子。counts 为 基因 × 样本。"""
    counts = np.asarray(counts, dtype=float)
    positive = (counts > 0).all(axis=1)
    if positive.sum() < 1:
        return np.ones(counts.shape[1])

    sub = counts[positive]
    log_geo = np.log(sub).mean(axis=1)
    ratios = sub / np.exp(log_geo)[:, None]
    factors = np.median(ratios, axis=0)
    factors = np.where(np.isfinite(factors) & (factors > 0), factors, 1.0)
    return factors / np.exp(np.log(factors).mean())


def tmm_factors(counts: np.ndarray, logratio_trim: float = 0.3, sum_trim: float = 0.05) -> np.ndarray:
    """edgeR 的 TMM 归一化因子。返回每个样本相对参考样本的缩放系数。"""
    counts = np.asarray(counts, dtype=float)
    lib = counts.sum(axis=0)
    lib = np.where(lib > 0, lib, 1.0)

    # 参考样本：上四分位数最接近均值的那个
    q75 = np.array([np.percentile(counts[:, j][counts[:, j] > 0], 75) if (counts[:, j] > 0).any() else 0.0
                    for j in range(counts.shape[1])])
    ref = int(np.argmin(np.abs(q75 - q75.mean()))) if q75.size else 0

    ref_counts = counts[:, ref]
    ref_lib = lib[ref]
    factors = np.ones(counts.shape[1])

    for j in range(counts.shape[1]):
        if j == ref:
            continue
        obs = counts[:, j]
        mask = (obs > 0) & (ref_counts > 0)
        if mask.sum() < 10:
            factors[j] = lib[j] / ref_lib
            continue
        obs_r = obs[mask] / lib[j]
        ref_r = ref_counts[mask] / ref_lib
        m = np.log2(obs_r / ref_r)
        a = 0.5 * (np.log2(obs_r) + np.log2(ref_r))

        # 按 M 与 A 做双侧截尾
        lo_m, hi_m = np.quantile(m, [logratio_trim, 1 - logratio_trim])
        keep = (m >= lo_m) & (m <= hi_m)
        if keep.sum() < 5:
            factors[j] = lib[j] / ref_lib
            continue
        a_kept = a[keep]
        lo_a, hi_a = np.quantile(a_kept, [sum_trim, 1 - sum_trim])
        keep2 = keep.copy()
        keep2[keep] = (a_kept >= lo_a) & (a_kept <= hi_a)
        if keep2.sum() < 5:
            factors[j] = lib[j] / ref_lib
            continue

        vw = 1.0 / (1.0 / (obs[mask][keep2] / lib[j]) + 1.0 / (ref_counts[mask][keep2] / ref_lib))
        m_kept = m[keep2]
        if vw.sum() <= 0 or not np.isfinite(vw).all():
            factors[j] = lib[j] / ref_lib
        else:
            factors[j] = float(2 ** (np.sum(m_kept * vw) / np.sum(vw)))

    factors = np.where(np.isfinite(factors) & (factors > 0), factors, 1.0)
    eff = factors * lib
    return eff / np.exp(np.log(eff).mean())


def normalize_counts(table: pd.DataFrame, method: str = "tmm") -> tuple[pd.DataFrame, np.ndarray]:
    """按指定方法归一化计数。

    method:
      tmm    — edgeR 风格 TMM 因子；deseq2 — 中位数比值；cpm — 每百万计数；
      none   — 仅除以库大小（等价 CPM 的计数版）
    返回 (归一化矩阵, 缩放因子)。除 cpm 外，返回的都是"等效计数"（列和≈一致）。
    """
    table = as_numeric_matrix(table)
    counts = table.to_numpy(dtype=float)

    method = (method or "tmm").lower()
    if method == "deseq2":
        factors = deseq_size_factors(counts)
        norm = counts / factors[None, :]
    elif method == "tmm":
        factors = tmm_factors(counts)
        norm = counts / factors[None, :]
    elif method in ("cpm", "none"):
        lib = counts.sum(axis=0)
        factors = np.where(lib > 0, lib, 1.0)
        norm = counts / factors[None, :] * 1e6
    else:
        raise ValueError(f"未知的归一化方法：{method}")

    return pd.DataFrame(norm, index=table.index, columns=table.columns), factors


def cpm(table: pd.DataFrame, log: bool = True, prior_count: float = 0.5,
        normalization: str = "tmm") -> pd.DataFrame:
    """CPM 转换（可选 log2）。

    先按 normalization 得到等效计数，再按有效库大小换算成每百万计数，
    因此不会出现重复除以库大小的问题。
    """
    norm, _ = normalize_counts(table, normalization)
    lib = norm.sum(axis=0).replace(0, np.nan)
    out = norm.div(lib, axis=1) * 1e6
    if log:
        out = np.log2(out + prior_count)
    return out.fillna(0.0)


# --------------------------------------------------------------------------
# 负二项 GLM Wald 检验
# --------------------------------------------------------------------------
def _poisson_closed_form(y: np.ndarray, sf: np.ndarray, in_b: np.ndarray) -> tuple[float, float]:
    """两组设计的 Poisson MLE（闭式解）：log mu = log sf + b0 + b1 * x。"""
    mu_a = y[~in_b].sum() / max(sf[~in_b].sum(), EPS)
    mu_b = y[in_b].sum() / max(sf[in_b].sum(), EPS)
    mu_a = max(mu_a, 1e-8)
    mu_b = max(mu_b, 1e-8)
    return float(np.log(mu_a)), float(np.log(mu_b / mu_a))


def estimate_common_dispersion(
    counts: np.ndarray, sf: np.ndarray, in_b: np.ndarray, trim: float = 0.1
) -> float:
    """用矩估计求共有离散度（edgeR estimateCommonDisp 的简化近似）。

    先按 Poisson 闭式解得到各基因的 mu，再用 (y-mu)^2 - mu 的矩估计求 phi，
    最后取截尾均值作为全局共有离散度。
    """
    n_genes, n_samples = counts.shape
    if n_samples <= 2:
        return 0.0

    # Poisson 闭式解可整体向量化：组均值 × 尺寸因子
    sf_safe = np.maximum(sf, EPS)
    in_b_f = in_b.astype(float)
    sum_sf_a = sf_safe[~in_b].sum()
    sum_sf_b = sf_safe[in_b].sum()
    mu_a = counts[:, ~in_b].sum(axis=1) / max(sum_sf_a, EPS)
    mu_b = counts[:, in_b].sum(axis=1) / max(sum_sf_b, EPS)
    mu_a = np.maximum(mu_a, 1e-8)
    mu_b = np.maximum(mu_b, 1e-8)

    eta = (np.log(mu_a)[:, None] + (in_b_f * np.log(mu_b / mu_a))[None, :]
           + np.log(sf_safe)[None, :])
    mu = np.exp(np.clip(eta, -50, 50))

    num = np.sum((counts - mu) ** 2 - mu, axis=1)
    den = np.sum(mu ** 2, axis=1)
    phi = np.where(den > 0, num / np.maximum(den, EPS), 0.0)
    phi = np.where(np.isfinite(phi), phi, 0.0)

    # 截尾，避免个别极端基因主导
    lo, hi = np.quantile(phi, [trim, 1 - trim]) if phi.size > 10 else (phi.min(), phi.max())
    kept = phi[(phi >= lo) & (phi <= hi)]
    return float(max(np.mean(kept) if kept.size else 0.0, 0.0))


def _nb_irls(y: np.ndarray, x: np.ndarray, offset: np.ndarray, phi: float,
             beta0: np.ndarray | None = None, max_iter: int = 60, tol: float = 1e-9):
    """负二项 GLM 的 IRLS 拟合，返回 (beta, cov)。"""
    n, p = x.shape
    beta = np.zeros(p) if beta0 is None else np.array(beta0, dtype=float)

    for _ in range(max_iter):
        eta = np.clip(x @ beta + offset, -40, 40)
        mu = np.exp(eta)
        w = mu / (1.0 + phi * mu)
        z = (y - mu) / np.maximum(mu, EPS)
        rhs = w * z
        xtwx = x.T @ (x * w[:, None])
        xtwz = x.T @ rhs
        try:
            new_beta = np.linalg.solve(xtwx, xtwz)
        except np.linalg.LinAlgError:
            new_beta = np.linalg.lstsq(xtwx, xtwz, rcond=None)[0]
        if not np.all(np.isfinite(new_beta)):
            return None, None
        if np.max(np.abs(new_beta - beta)) < tol:
            beta = new_beta
            break
        beta = new_beta

    eta = np.clip(x @ beta + offset, -40, 40)
    mu = np.exp(eta)
    w = mu / (1.0 + phi * mu)
    xtwx = x.T @ (x * w[:, None])
    cov = np.linalg.pinv(xtwx)
    return beta, cov


def nb_wald_test(
    table: pd.DataFrame, groups: pd.Series, group_a: str, group_b: str,
    normalization: str = "tmm", dispersion: float | None = None,
) -> pd.DataFrame:
    """负二项 GLM Wald 检验（两组比较）。

    模型：log(mu_ij) = log(sf_j) + b0 + b1 * I(样本属于 group_b)
    b1 即 group_b 相对 group_a 的自然对数倍数变化。
    """
    table = as_numeric_matrix(table)
    groups = groups.astype(str)

    in_a = [s for s in table.columns if groups.get(s) == group_a]
    in_b = [s for s in table.columns if groups.get(s) == group_b]
    if len(in_a) < 2 or len(in_b) < 2:
        raise ValueError(f"NB 检验要求每组至少 2 个样本（当前 {group_a}={len(in_a)}, {group_b}={len(in_b)}）。")

    cols = in_a + in_b
    counts = table[cols].to_numpy(dtype=float)
    x = np.zeros((len(cols), 2))
    x[:, 0] = 1.0
    x[len(in_a):, 1] = 1.0
    flag_b = np.array([False] * len(in_a) + [True] * len(in_b))

    norm, factors = normalize_counts(table[cols], normalization)
    sf = factors / np.exp(np.log(factors).mean())

    if dispersion is None:
        dispersion = estimate_common_dispersion(counts, sf, flag_b)
    phi = float(max(dispersion, 0.0))

    log2fc, se, stat, pvals, ok = [], [], [], [], []
    offset = np.log(np.maximum(sf, EPS))
    for g in range(counts.shape[0]):
        y = counts[g]
        b0, b1 = _poisson_closed_form(y, sf, flag_b)
        beta, cov = _nb_irls(y, x, offset, phi, beta0=np.array([b0, b1]))
        if beta is None or cov is None or cov[1, 1] <= 0 or not np.isfinite(cov[1, 1]):
            log2fc.append(np.nan); se.append(np.nan); stat.append(np.nan); pvals.append(np.nan); ok.append(False)
            continue
        s = float(np.sqrt(cov[1, 1]))
        z = float(beta[1] / s) if s > 0 else np.nan
        pv = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
        log2fc.append(float(beta[1] / np.log(2)))
        se.append(float(s / np.log(2)))
        stat.append(z)
        pvals.append(pv)
        ok.append(True)

    mean_a = norm[in_a].mean(axis=1).to_numpy()
    mean_b = norm[in_b].mean(axis=1).to_numpy()
    p_arr = np.asarray(pvals, dtype=float)

    res = pd.DataFrame(
        {
            "feature": table.index,
            "mean_norm_A": mean_a,
            "mean_norm_B": mean_b,
            "log2FC": log2fc,
            "lfcSE": se,
            "statistic": stat,
            "p": p_arr,
            "FDR": bh_fdr(p_arr),
            "effect_size": [
                cliffs_delta(norm.loc[f, in_a].to_numpy(dtype=float),
                             norm.loc[f, in_b].to_numpy(dtype=float))
                for f in table.index
            ],
            "method": f"NB-GLM Wald (dispersion={phi:.4f})",
        }
    )
    res["group_enriched"] = np.where(res["log2FC"] > 0, group_b, group_a)
    res["-log10FDR"] = -np.log10(np.clip(res["FDR"].to_numpy(dtype=float), 1e-300, 1.0))
    return res.sort_values("p", na_position="last").reset_index(drop=True)


def differential_expression(
    table: pd.DataFrame, groups: pd.Series, group_a: str, group_b: str,
    method: str = "nb", normalization: str = "tmm",
) -> pd.DataFrame:
    """差异表达总入口：method 为 nb（负二项）或 wilcoxon（非参数，稳健）。"""
    if (method or "nb").lower() in ("nb", "negbin", "glm", "deseq2", "edger"):
        try:
            return nb_wald_test(table, groups, group_a, group_b, normalization=normalization)
        except Exception as exc:
            from .stats import differential_abundance

            res = differential_abundance(table, groups, group_a, group_b, method="wilcoxon")
            res["method"] = f"Wilcoxon（NB 检验不可用：{exc}）"
            return res
    from .stats import differential_abundance

    res = differential_abundance(table, groups, group_a, group_b, method="wilcoxon")
    res["method"] = "Wilcoxon rank-sum + CLR"
    return res


# --------------------------------------------------------------------------
# 表达谱概览
# --------------------------------------------------------------------------
def expression_profile(table: pd.DataFrame, normalization: str = "tmm") -> dict:
    """归一化前后各样本的表达量分布，用于箱线图对比。"""
    table = as_numeric_matrix(table)
    norm, factors = normalize_counts(table, normalization)

    def stats_of(frame: pd.DataFrame) -> pd.DataFrame:
        lib = frame.sum(axis=0)
        detected = (frame > 0).sum(axis=0)
        return pd.DataFrame(
            {
                "LibrarySize": lib,
                "log10LibrarySize": np.log10(lib.replace(0, np.nan)),
                "DetectedGenes": detected,
                "MedianLog2": np.log2(frame.replace(0, np.nan)).median(axis=0),
            }
        )

    raw_stats = stats_of(table)
    norm_stats = stats_of(norm)
    # 归一化后按 CPM 呈现
    lib = table.sum(axis=0).replace(0, np.nan)
    norm_cpm = norm.div(lib, axis=1) * 1e6

    return {
        "raw": table,
        "normalized": norm,
        "normalized_cpm": norm_cpm.fillna(0.0),
        "factors": pd.Series(factors, index=table.columns, name="SizeFactor"),
        "raw_stats": raw_stats,
        "norm_stats": norm_stats,
        "normalization": normalization,
    }


def sample_correlation_matrix(table: pd.DataFrame, method: str = "pearson",
                              normalization: str = "tmm") -> pd.DataFrame:
    """样本间相关性矩阵（在 log2-CPM 空间上计算）。"""
    norm, _ = normalize_counts(table, "cpm")
    lib = as_numeric_matrix(table).sum(axis=0).replace(0, np.nan)
    log_cpm = np.log2(norm.div(lib, axis=1) * 1e6 + 0.5).fillna(0.0)
    return log_cpm.corr(method=method)


def pca_expression(table: pd.DataFrame, n_components: int = 4,
                   normalization: str = "tmm") -> tuple[pd.DataFrame, np.ndarray]:
    """在 log2-CPM 空间做样本 PCA，返回 (坐标, 各主成分解释度)。"""
    norm, _ = normalize_counts(table, "cpm")
    lib = as_numeric_matrix(table).sum(axis=0).replace(0, np.nan)
    log_cpm = np.log2(norm.div(lib, axis=1) * 1e6 + 0.5).fillna(0.0)

    # 只保留在足够多样本中表达的基因，降低噪声
    keep = (log_cpm > 0).sum(axis=1) >= max(2, int(0.2 * log_cpm.shape[1]))
    mat = log_cpm.loc[keep] if keep.sum() >= 10 else log_cpm
    if mat.shape[0] < 2:
        raise ValueError("有效基因数太少，无法做 PCA。")

    x = mat.to_numpy(dtype=float).T
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    x = x / sd

    u, s, vt = np.linalg.svd(x, full_matrices=False)
    n_comp = int(min(n_components, s.size, x.shape[0]))
    var = (s ** 2) / max((s ** 2).sum(), EPS)
    coords = u[:, :n_comp] * s[:n_comp]

    return (
        pd.DataFrame(coords, index=mat.columns, columns=[f"PC{i + 1}" for i in range(n_comp)]),
        np.asarray(var[:n_comp], dtype=float),
    )


def top_expressed(table: pd.DataFrame, groups: pd.Series, top_n: int = 20,
                  normalization: str = "tmm") -> pd.DataFrame:
    """丰度最高的基因及其在各分组的平均表达量。"""
    norm, _ = normalize_counts(table, normalization)
    lib = as_numeric_matrix(table).sum(axis=0).replace(0, np.nan)
    cpm = norm.div(lib, axis=1) * 1e6
    cpm = cpm.fillna(0.0)

    order = cpm.mean(axis=1).sort_values(ascending=False).head(top_n).index
    sub = cpm.loc[order]
    out = pd.DataFrame({"feature": order, "mean_CPM": cpm.loc[order].mean(axis=1).to_numpy()})
    for g in sorted(groups.astype(str).unique()):
        cols = [c for c in sub.columns if groups.get(c) == g]
        out[f"mean_{g}"] = sub[cols].mean(axis=1).to_numpy() if cols else np.nan
    return out
