"""丰度表预处理：过滤、抽平、标准化、按分类级聚合。"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]


def as_numeric_matrix(table: pd.DataFrame) -> pd.DataFrame:
    """确保是纯数值矩阵，缺失值补 0。"""
    out = table.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    out = out.loc[out.sum(axis=1) > 0, out.sum(axis=0) > 0]
    return out


def filter_features(
    table: pd.DataFrame,
    min_count: float = 0.0,
    min_prevalence: float = 0.0,
    min_total_frac: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """按丰度/出现率/总占比过滤低丰度特征。

    min_count      : 至少在 min_prevalence 比例的样本中达到该丰度
    min_prevalence : 出现率阈值 (0~1)
    min_total_frac : 该特征总丰度占全表比例下限 (0~1)
    """
    table = as_numeric_matrix(table)
    n0 = table.shape[0]
    keep = np.ones(table.shape[0], dtype=bool)

    if min_prevalence > 0 and min_count >= 0:
        present = (table >= min_count).sum(axis=1) / max(1, table.shape[1])
        keep &= (present >= min_prevalence).to_numpy()

    if min_total_frac > 0:
        frac = table.sum(axis=1) / max(table.to_numpy().sum(), 1e-12)
        keep &= (frac >= min_total_frac).to_numpy()

    filtered = table.loc[keep]
    if filtered.shape[0] == 0:
        filtered = table
        keep = np.ones(table.shape[0], dtype=bool)

    stats = {
        "n_features_before": int(n0),
        "n_features_after": int(filtered.shape[0]),
        "n_features_removed": int(n0 - filtered.shape[0]),
    }
    return filtered, stats


def rarefy(table: pd.DataFrame, depth: int | None = None, seed: int = 42) -> tuple[pd.DataFrame, int]:
    """无放回抽平到统一深度。返回 (抽平后矩阵, 实际深度)。"""
    table = as_numeric_matrix(table)
    counts = table.to_numpy(dtype=np.int64).T  # 样本 × 特征
    lib = counts.sum(axis=1)
    if depth is None:
        depth = int(lib.min())
    depth = int(min(depth, lib.min()))
    if depth <= 0:
        raise ValueError("测序深度为 0，无法抽平。")

    rng = np.random.default_rng(seed)
    out = np.zeros_like(counts)
    for i, row in enumerate(counts):
        total = int(row.sum())
        if total <= depth:
            out[i] = row
            continue
        # 超几何抽样：逐个样本做无放回抽取
        pool = np.repeat(np.arange(row.size), row)
        picked = rng.choice(pool, size=depth, replace=False)
        out[i] = np.bincount(picked, minlength=row.size)
    return pd.DataFrame(out.T, index=table.index, columns=table.columns), depth


def to_relative(table: pd.DataFrame) -> pd.DataFrame:
    table = as_numeric_matrix(table)
    totals = table.sum(axis=0).replace(0, np.nan)
    return table.div(totals, axis=1).fillna(0.0)


def clr_transform(table: pd.DataFrame, pseudocount: float = 0.5) -> pd.DataFrame:
    """中心化对数比变换，适合宏基因组/宏转录组的成分数据。"""
    table = as_numeric_matrix(table)
    x = table + pseudocount
    rel = x.div(x.sum(axis=0), axis=1)
    logx = np.log(rel)
    return logx.sub(logx.mean(axis=0), axis=1)


def aggregate_by_taxonomy(
    table: pd.DataFrame, taxonomy: pd.DataFrame, rank: str
) -> pd.DataFrame:
    """把特征表按指定分类级聚合。"""
    if rank not in taxonomy.columns:
        raise KeyError(f"分类表中没有 {rank} 级。可选：{list(taxonomy.columns)}")
    common = [f for f in table.index if f in taxonomy.index]
    if not common:
        raise ValueError("丰度表特征名与分类表无法对应，请检查首列是否为特征 ID。")
    sub = table.loc[common]
    labels = taxonomy.loc[common, rank].fillna("Unclassified").replace("", "Unclassified")
    labels = labels.where(labels.notna(), "Unclassified")
    return sub.groupby(labels).sum()


def detect_abundance_type(table: pd.DataFrame) -> dict:
    """判断输入表是原始计数还是相对丰度。

    判定依据（按优先级）：
      1) 各列和都接近 1       → 相对丰度（比例）
      2) 各列和都接近 100     → 相对丰度（百分比）
      3) 所有值 ≤ 1.0001      → 视为比例
      4) 绝大多数是整数       → 原始计数
      5) 其余                 → 未知，按计数处理

    返回 dict: {"type", "is_relative", "scale", "reason", "column_sum_median", "integer_ratio"}
    """
    mat = table.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    if mat.empty or mat.to_numpy().sum() <= 0:
        return {"type": "empty", "is_relative": False, "scale": 1.0,
                "reason": "数据为空", "column_sum_median": 0.0, "integer_ratio": 0.0}

    col_sums = mat.sum(axis=0).to_numpy(dtype=float)
    med_sum = float(np.median(col_sums))
    vmax = float(mat.to_numpy().max())

    vals = mat.to_numpy(dtype=float).ravel()
    nz = vals[vals > 0]
    if nz.size:
        integer_ratio = float(np.mean(np.isclose(nz, np.round(nz), atol=1e-6)))
    else:
        integer_ratio = 1.0

    def near(a: float, b: float, tol: float = 0.02) -> bool:
        return abs(a - b) <= tol * max(abs(b), 1e-9)

    if med_sum > 0 and near(med_sum, 1.0, 0.05):
        return {"type": "relative_fraction", "is_relative": True, "scale": 1.0,
                "reason": f"各样本列和约为 1（中位数 {med_sum:.4f}），判定为相对丰度",
                "column_sum_median": med_sum, "integer_ratio": integer_ratio}
    if med_sum > 0 and near(med_sum, 100.0, 0.05):
        return {"type": "relative_percent", "is_relative": True, "scale": 100.0,
                "reason": f"各样本列和约为 100（中位数 {med_sum:.2f}），判定为百分比形式的相对丰度",
                "column_sum_median": med_sum, "integer_ratio": integer_ratio}
    if vmax <= 1.0001:
        return {"type": "relative_fraction", "is_relative": True, "scale": 1.0,
                "reason": f"所有取值均 ≤ 1（最大 {vmax:.4f}），判定为相对丰度",
                "column_sum_median": med_sum, "integer_ratio": integer_ratio}
    if integer_ratio >= 0.95:
        return {"type": "count", "is_relative": False, "scale": 1.0,
                "reason": f"取值中 {integer_ratio:.0%} 为整数、列和中位数 {med_sum:,.0f}，判定为原始计数",
                "column_sum_median": med_sum, "integer_ratio": integer_ratio}
    return {"type": "unknown", "is_relative": False, "scale": 1.0,
            "reason": f"取值中仅 {integer_ratio:.0%} 为整数，无法确定类型，按原始计数处理",
            "column_sum_median": med_sum, "integer_ratio": integer_ratio}


def ensure_relative(table: pd.DataFrame, info: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """确保拿到相对丰度表：不是相对丰度就转换。

    返回 (相对丰度表, 判定信息)。百分比表会被归一化到 0~1。
    """
    info = info or detect_abundance_type(table)
    mat = as_numeric_matrix(table)
    if info.get("is_relative"):
        if info.get("type") == "relative_percent":
            return to_relative(mat), info
        # 已是比例，但仍重新归一化以消除浮点误差
        return to_relative(mat), info
    info = {**info, "converted": True,
            "convert_note": "输入为原始计数，已按样本总和转换为相对丰度后进行组成类分析"}
    return to_relative(mat), info


def collapse_to_top(
    rel_table: pd.DataFrame, top_n: int = 10, other_label: str = "Others"
) -> pd.DataFrame:
    """按总体平均丰度保留 Top-N，其余归入 Others。"""
    if rel_table.shape[0] <= top_n:
        return rel_table.copy()
    order = rel_table.mean(axis=1).sort_values(ascending=False)
    keep = list(order.index[:top_n])
    rest = rel_table.drop(index=keep)
    out = pd.concat([rel_table.loc[keep], rest.sum(axis=0).to_frame(other_label).T])
    return out
