"""组间统计：FDR 校正、差异丰度、LDA 效应量、随机森林与 ROC。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal, ttest_ind, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from .preprocess import as_numeric_matrix, clr_transform, to_relative


# --------------------------------------------------------------------------
# 多重检验校正
# --------------------------------------------------------------------------
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正。"""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    res = np.empty(n)
    res[order] = q
    out[ok] = res
    return out


def bonferroni(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = np.isfinite(p).sum()
    return np.clip(p * max(n, 1), 0, 1)


# --------------------------------------------------------------------------
# 效应量
# --------------------------------------------------------------------------
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta 非参数效应量，范围 [-1, 1]。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.nan
    diff = a[:, None] - b[None, :]
    gt = (diff > 0).sum()
    lt = (diff < 0).sum()
    return float((gt - lt) / (a.size * b.size))


# --------------------------------------------------------------------------
# 差异丰度分析
# --------------------------------------------------------------------------
def differential_abundance(
    table: pd.DataFrame,
    groups: pd.Series,
    group_a: str,
    group_b: str,
    method: str = "wilcoxon",
    use_clr: bool = True,
) -> pd.DataFrame:
    """两组差异丰度分析。

    在相对丰度上计算 log2 倍数变化，在 CLR（或相对丰度）上做检验，
    同时给出 Cliff's delta 效应量与 BH-FDR。
    """
    table = as_numeric_matrix(table)
    rel = to_relative(table)
    clr = clr_transform(table)

    groups = groups.astype(str)
    in_a = [s for s in table.columns if groups.get(s) == group_a]
    in_b = [s for s in table.columns if groups.get(s) == group_b]
    if len(in_a) == 0 or len(in_b) == 0:
        raise ValueError(f"分组 {group_a} / {group_b} 在本数据中没有对应样本。")

    eps = 1e-6
    mean_a_rel = rel[in_a].mean(axis=1)
    mean_b_rel = rel[in_b].mean(axis=1)
    mean_a_abs = table[in_a].mean(axis=1)
    mean_b_abs = table[in_b].mean(axis=1)

    log2fc = np.log2((mean_a_rel + eps) / (mean_b_rel + eps))
    effect_space = clr if use_clr else rel

    stats_p, stats_stat = [], []
    for feat in table.index:
        va = effect_space.loc[feat, in_a].to_numpy(dtype=float)
        vb = effect_space.loc[feat, in_b].to_numpy(dtype=float)
        try:
            if method == "wilcoxon":
                if len(va) >= 8 and len(vb) >= 8:
                    stat, p = mannwhitneyu(va, vb, alternative="two-sided")
                else:
                    stat, p = mannwhitneyu(va, vb, alternative="two-sided")
            elif method == "ttest":
                stat, p = ttest_ind(va, vb, equal_var=False)
            else:
                raise ValueError(f"未知检验方法：{method}")
        except ValueError:
            stat, p = np.nan, np.nan
        stats_stat.append(stat)
        stats_p.append(p)

    p_arr = np.asarray(stats_p, dtype=float)
    fdr = bh_fdr(p_arr)

    res = pd.DataFrame(
        {
            "feature": table.index,
            "mean_rel_A": mean_a_rel.to_numpy(),
            "mean_rel_B": mean_b_rel.to_numpy(),
            "mean_abund_A": mean_a_abs.to_numpy(),
            "mean_abund_B": mean_b_abs.to_numpy(),
            "log2FC": log2fc.to_numpy(),
            "statistic": stats_stat,
            "p": p_arr,
            "FDR": fdr,
            "effect_size": [
                cliffs_delta(
                    effect_space.loc[f, in_a].to_numpy(dtype=float),
                    effect_space.loc[f, in_b].to_numpy(dtype=float),
                )
                for f in table.index
            ],
        }
    )
    res["group_enriched"] = np.where(res["log2FC"] > 0, group_a, group_b)
    res["-log10FDR"] = -np.log10(np.clip(res["FDR"].to_numpy(dtype=float), 1e-300, 1.0))
    res = res.sort_values("p", na_position="last").reset_index(drop=True)
    return res


def multi_group_test(
    table: pd.DataFrame, groups: pd.Series, method: str = "kruskal", use_clr: bool = True
) -> pd.DataFrame:
    """多组整体差异检验（Kruskal-Wallis），返回每个特征的 p 与 FDR。"""
    table = as_numeric_matrix(table)
    mat = clr_transform(table) if use_clr else to_relative(table)
    groups = groups.astype(str)
    samples = [s for s in mat.columns if s in groups.index]
    cats = sorted(set(groups.loc[samples]))
    rows = []
    for feat in mat.index:
        vals = [mat.loc[feat, [s for s in samples if groups[s] == g]].to_numpy(dtype=float) for g in cats]
        vals = [v for v in vals if v.size > 0]
        try:
            stat, p = kruskal(*vals) if len(vals) >= 2 and all(v.size >= 1 for v in vals) else (np.nan, np.nan)
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append({"feature": feat, "statistic": stat, "p": p})
    res = pd.DataFrame(rows)
    res["FDR"] = bh_fdr(res["p"].to_numpy())
    res["-log10FDR"] = -np.log10(np.clip(res["FDR"].to_numpy(dtype=float), 1e-300, 1.0))
    return res.sort_values("p", na_position="last").reset_index(drop=True)


def lda_effect_size(
    table: pd.DataFrame, groups: pd.Series, group_a: str, group_b: str
) -> pd.DataFrame:
    """LEfSe 风格的线性判别分析效应量（LDA score）。

    在 CLR 变换后的数据上拟合二分类 LDA，取线性判别系数作为效应量，
    并按 LEfSe 惯例取 log10 标度。
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    table = as_numeric_matrix(table)
    mat = clr_transform(table)
    groups = groups.astype(str)
    samples = [s for s in mat.columns if groups.get(s) in (group_a, group_b)]
    if len(samples) < 4:
        return pd.DataFrame(columns=["feature", "LDA_score", "group_enriched"])

    x = mat[samples].T.to_numpy(dtype=float)
    y = np.array([0 if groups[s] == group_a else 1 for s in samples])

    n1 = int((y == 0).sum())
    n2 = int((y == 1).sum())
    if n1 < 2 or n2 < 2:
        return pd.DataFrame(columns=["feature", "LDA_score", "group_enriched"])

    lda = LinearDiscriminantAnalysis(solver="svd", shrinkage=None)
    try:
        lda.fit(x, y)
        coef = np.asarray(lda.coef_).ravel()
    except Exception:
        return pd.DataFrame(columns=["feature", "LDA_score", "group_enriched"])

    score = np.log10(np.abs(coef) * 1e6 + 1.0)
    mean_diff = mat[samples].T.groupby(y).mean()
    enriched = np.where(mean_diff.iloc[0].to_numpy() > mean_diff.iloc[1].to_numpy(), group_a, group_b)

    out = pd.DataFrame({"feature": mat.index, "LDA_score": score, "group_enriched": enriched})
    out["signed_score"] = np.where(out["group_enriched"] == group_a, out["LDA_score"], -out["LDA_score"])
    return out.sort_values("LDA_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 随机森林与 ROC
# --------------------------------------------------------------------------
def random_forest_analysis(
    table: pd.DataFrame,
    groups: pd.Series,
    n_estimators: int = 500,
    cv_folds: int = 5,
    seed: int = 42,
    top_features: int = 20,
) -> dict:
    """随机森林分类：交叉验证准确率、ROC、特征重要性。"""
    table = as_numeric_matrix(table)
    mat = clr_transform(table)
    groups = groups.astype(str)

    samples = [s for s in mat.columns if s in groups.index]
    labels = groups.loc[samples]
    cats = sorted(labels.unique())
    if len(cats) < 2 or len(samples) < 8:
        return {"ok": False, "reason": "样本数或类别数不足，无法建模（建议每类至少 5 个样本）。"}

    counts = labels.value_counts()
    if counts.min() < 2:
        return {"ok": False, "reason": "存在样本数少于 2 的组，无法做交叉验证。"}

    x = mat[samples].T.to_numpy(dtype=float)
    y = labels.to_numpy()

    n_splits = int(min(cv_folds, counts.min()))
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1, class_weight="balanced_subsample"
    )

    try:
        acc_scores = cross_val_score(clf, x, y, cv=cv, scoring="accuracy")
        proba = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")
        pred = cross_val_predict(clf, x, y, cv=cv)
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "reason": f"建模失败：{exc}"}

    from sklearn.metrics import confusion_matrix

    labels_sorted = cats
    cm = pd.DataFrame(
        confusion_matrix(y, pred, labels=labels_sorted),
        index=[f"真实: {c}" for c in labels_sorted],
        columns=[f"预测: {c}" for c in labels_sorted],
    )

    clf.fit(x, y)
    importance = (
        pd.DataFrame({"feature": mat.index, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    roc_data: dict = {}
    if len(cats) == 2:
        pos = list(clf.classes_).index(cats[1])
        y_bin = (y == cats[1]).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, proba[:, pos])
        roc_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(auc(fpr, tpr)),
            "positive_class": cats[1],
        }

    oob = float(np.mean(clf.oob_score_)) if getattr(clf, "oob_score_", None) else None

    return {
        "ok": True,
        "accuracy": float(np.mean(acc_scores)),
        "accuracy_std": float(np.std(acc_scores)),
        "accuracy_folds": acc_scores.tolist(),
        "n_splits": n_splits,
        "classes": cats,
        "importance": importance,
        "importance_top": importance.head(top_features),
        "roc": roc_data,
        "oob_score": oob,
        "n_samples": len(samples),
        "confusion": cm,
    }
