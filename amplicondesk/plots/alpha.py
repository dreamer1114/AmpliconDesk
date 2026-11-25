"""Alpha 多样性、稀释曲线、物种累积曲线、等级丰度曲线。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..style import Style
from .base import (
    finish_axes, new_figure, panel_grid, place_annotation, place_axis_labels,
    place_suptitle, place_title, resolve_legend, save, stat_box, style_axes,
)

__all__ = [
    "plot_alpha_panel", "plot_alpha_heatmap", "plot_rarefaction",
    "plot_alpha_rarefaction", "plot_accumulation", "plot_rank_abundance",
    "plot_sample_quality",
]


def _p_label(p: float) -> str:
    if not np.isfinite(p):
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return "ns"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# --------------------------------------------------------------------------
def plot_alpha_panel(
    alpha: pd.DataFrame, groups: pd.Series, metrics: list[str],
    out_path: str | Path, style: Style, test_results: dict | None = None,
) -> dict:
    metrics = [m for m in metrics if m in alpha.columns]
    if not metrics:
        raise ValueError("没有可绘制的 alpha 多样性指标。")

    fig, axes, ncol = panel_grid(style, len(metrics))
    merged = alpha.join(groups.rename("Group"), how="inner")
    merged["Group"] = merged["Group"].astype(str)
    colors = style.group_colors(merged["Group"])
    order = sorted(merged["Group"].unique())
    width = style.num("element.box_width", 0.6)
    jitter = style.num("element.jitter", 0.18)
    rotate = style.num("element.axis_rotate", 45)

    for k, metric in enumerate(metrics):
        ax = axes[k // ncol][k % ncol]
        sns.boxplot(data=merged, x="Group", y=metric, order=order, ax=ax, hue="Group",
                    palette=colors, width=width, linewidth=1.0, fliersize=0,
                    showfliers=False, legend=False)
        if style.flag("element.point_labels", True):
            sns.stripplot(data=merged, x="Group", y=metric, order=order, ax=ax,
                          color="#2F2F2F", size=3.0, alpha=0.65, jitter=jitter)
        style_axes(ax, style)
        ax.set_xlabel("")
        place_title(ax, style, metric, allow_override=False)

        res = (test_results or {}).get(metric)
        if res and np.isfinite(res.get("p", np.nan)):
            ymax, ymin = merged[metric].max(), merged[metric].min()
            span = (ymax - ymin) or 1.0
            ax.set_ylim(ymin - 0.06 * span, ymax + 0.24 * span)
            place_annotation(
                ax, f"{res['test']}  {_p_label(res['p'])} {_stars(res['p'])}", style
            )
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(style.font_size("tick_label"))
            lbl.set_color(style.font_color("tick_label"))
        if len(order) > 6 or rotate >= 45:
            ax.tick_params(axis="x", rotation=rotate if len(order) > 4 else 0)

    for k in range(len(metrics), axes.shape[0] * axes.shape[1]):
        axes[k // ncol][k % ncol].axis("off")

    place_suptitle(fig, style, "Alpha 多样性组间比较")
    fig.tight_layout()
    return save(fig, out_path, style,
                "箱线图展示各分组的 alpha 多样性指数分布，散点为单个样本；"
                "标注为组间检验结果（两组用 Mann-Whitney U，多组用 Kruskal-Wallis）。",
                "Alpha 多样性组间比较")


def plot_alpha_heatmap(alpha: pd.DataFrame, groups: pd.Series,
                       out_path: str | Path, style: Style) -> dict:
    metrics = [m for m in ["Observed", "Chao1", "Shannon", "Simpson", "Pielou"] if m in alpha.columns]
    if not metrics:
        metrics = list(alpha.columns[:4])
    mat = alpha[metrics].copy()
    meta = groups.reindex(mat.index).astype(str)
    order = sorted(mat.index, key=lambda s: (meta.get(s, ""), s))
    mat, meta = mat.loc[order], meta.loc[order]

    z = (mat - mat.mean()) / mat.std(ddof=0).replace(0, np.nan)
    z = z.fillna(0.0)

    fig, ax = new_figure(style, max(5.5, 0.30 * len(order) + 3.2), 3.6)

    # 分组色条
    colors = style.group_colors(meta)
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle

    for i, s in enumerate(order):
        ax.add_patch(Rectangle((i, len(metrics) + 0.06), 1, 0.22,
                               facecolor=colors.get(str(meta[s]), "#999999"),
                               edgecolor="none", clip_on=False))

    sns.heatmap(z.T, cmap=style.cmap("heatmap"), center=0, ax=ax,
                cbar_kws={"label": "Z-score"}, xticklabels=True, yticklabels=True,
                annot=style.flag("element.heatmap_values", False),
                annot_kws={"fontsize": max(5, style.font_size("annotation") * 0.7)})
    ax.set_xticklabels(order, rotation=90, fontsize=style.font_size("tick_label") * 0.78)
    ax.set_yticklabels(metrics, rotation=0, fontsize=style.font_size("tick_label"))
    if not style.flag("element.grid", True):
        ax.grid(False)
    place_title(ax, style, "Alpha 多样性指数一览（按组排序）")
    fig.tight_layout()
    return save(fig, out_path, style,
                "各样本 alpha 多样性指数的 Z-score 热图，顶部色条为分组归属。",
                "Alpha 多样性热图")


# --------------------------------------------------------------------------
def _sample_pool(counts_row: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(counts_row.size), counts_row.astype(np.int64))


def plot_rarefaction(table: pd.DataFrame, groups: pd.Series | None,
                     out_path: str | Path, style: Style, n_points: int = 15) -> dict:
    counts = table.to_numpy(dtype=float).T
    lib = counts.sum(axis=1)
    if lib.max() <= 0:
        raise ValueError("数据为空，无法绘制稀释曲线。")
    n_points = max(3, int(style.num("element.label_top_n", n_points)))
    depths = np.unique(np.linspace(max(1, lib.min() * 0.05), lib.min(), min(n_points, 20)).astype(int))

    fig, ax = new_figure(style, 7.4, 4.8)
    colors = style.group_colors(groups) if groups is not None else {}
    rng = np.random.default_rng(0)
    alpha = 0.55 * style.num("palette.alpha", 1.0)

    for i, row in enumerate(counts):
        if row.sum() <= 0:
            continue
        pool = _sample_pool(row)
        ys = []
        for d in depths:
            d = int(min(d, pool.size))
            sub = rng.choice(pool, size=d, replace=False) if d < pool.size else pool
            ys.append(np.unique(sub).size)
        color, label = "#7A7A7A", None
        if groups is not None:
            sname = table.columns[i]
            if sname in groups.index:
                label = str(groups[sname])
                color = colors.get(label, "#7A7A7A")
        ax.plot(depths, ys, color=color, alpha=alpha, lw=1.1, label=label)
        ax.scatter(depths[-1], ys[-1], color=color, s=10, alpha=min(1.0, alpha + 0.25))

    finish_axes(ax, style, legend=groups is not None,
                xlabel="抽样深度 (reads)", ylabel="观测到的特征数",
                title="稀释曲线 (Rarefaction curve)")
    resolve_legend(style, ax, title="Group")
    fig.tight_layout()
    return save(fig, out_path, style,
                "每条曲线代表一个样本，曲线趋于平缓说明测序深度已接近覆盖该样本的全部多样性。",
                "稀释曲线")


def plot_alpha_rarefaction(
    table: pd.DataFrame, groups: pd.Series, out_path: str | Path, style: Style,
    metrics: tuple[str, ...] = ("Observed", "Shannon"), n_points: int = 12,
    n_repeats: int = 5,
) -> dict:
    """Alpha 多样性随测序深度的变化曲线（按分组取均值 ± 标准差）。"""
    from ..core.diversity import alpha_diversity

    counts = table.to_numpy(dtype=float).T
    lib = counts.sum(axis=1)
    if lib.max() <= 0 or table.shape[1] < 2:
        raise ValueError("数据不足，无法绘制 alpha 稀释曲线。")

    max_depth = int(lib.min())
    depths = np.unique(np.linspace(max(10, int(max_depth * 0.05)), max_depth,
                                   min(max(4, n_points), 20)).astype(int))
    rng = np.random.default_rng(42)
    groups = groups.astype(str)
    cats = sorted(groups.unique())
    colors = style.group_colors(cats)

    fig, axes, ncol = panel_grid(style, len(metrics), ncol=len(metrics))
    for mi, metric in enumerate(metrics):
        ax = axes[0][mi]
        for g in cats:
            cols = [c for c in table.columns if groups.get(c) == g]
            if not cols:
                continue
            sub = table[cols]
            means, sds = [], []
            for d in depths:
                vals = []
                for rep in range(max(1, n_repeats)):
                    sampled = {}
                    for c in cols:
                        row = sub[c].to_numpy(dtype=float)
                        pool = _sample_pool(row)
                        dd = int(min(d, pool.size))
                        pick = rng.choice(pool, size=dd, replace=False) if dd < pool.size else pool
                        sampled[c] = np.bincount(pick, minlength=row.size).astype(float)
                    ad = alpha_diversity(pd.DataFrame(sampled))
                    if metric in ad.columns:
                        vals.append(float(ad[metric].mean()))
                means.append(np.mean(vals) if vals else np.nan)
                sds.append(np.std(vals) if vals else 0.0)
            means = np.asarray(means, dtype=float)
            sds = np.asarray(sds, dtype=float)
            ax.plot(depths, means, color=colors[g], lw=1.8, marker="o", ms=3.5, label=g)
            ax.fill_between(depths, means - sds, means + sds, color=colors[g], alpha=0.18)
        finish_axes(ax, style, legend=(mi == len(metrics) - 1),
                    xlabel="抽样深度 (reads)", ylabel=metric)
        place_title(ax, style, metric, allow_override=False)
        if mi == len(metrics) - 1:
            resolve_legend(style, ax)

    for k in range(len(metrics), axes.shape[0] * axes.shape[1]):
        axes[k // ncol][k % ncol].axis("off")
    place_suptitle(fig, style, "Alpha 多样性稀释曲线")
    fig.tight_layout()
    return save(fig, out_path, style,
                "曲线为各分组在不同抽样深度下的 alpha 多样性均值，阴影为标准差。"
                "曲线进入平台期说明测序深度已充分。",
                "Alpha 多样性稀释曲线")


def plot_accumulation(table: pd.DataFrame, groups: pd.Series | None,
                      out_path: str | Path, style: Style,
                      n_perm: int = 20) -> dict:
    """物种累积曲线（样本数 vs 累计发现特征数）。"""
    binary = (table.to_numpy(dtype=float) > 0).T  # 样本 × 特征
    n = binary.shape[0]
    if n < 2:
        raise ValueError("样本数太少，无法绘制累积曲线。")

    fig, ax = new_figure(style, 7.2, 4.6)
    rng = np.random.default_rng(7)

    def curve(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = np.arange(1, rows.shape[0] + 1)
        acc = np.zeros((n_perm, rows.shape[0]))
        for p in range(n_perm):
            order = rng.permutation(rows.shape[0])
            cum = np.maximum.accumulate(rows[order], axis=0)
            acc[p] = cum.sum(axis=1)
        return xs, acc

    groups_ = groups.astype(str) if groups is not None else None
    colors = style.group_colors(groups_) if groups_ is not None else {}

    xs, acc = curve(binary)
    ax.plot(xs, acc.mean(axis=0), color="#333333", lw=2.2, label="全部样本", zorder=4)
    ax.fill_between(xs, acc.min(axis=0), acc.max(axis=0), color="#333333", alpha=0.12)

    if groups_ is not None:
        for g in sorted(groups_.unique()):
            idx = [i for i, c in enumerate(table.columns) if groups_.get(c) == g]
            if len(idx) < 2:
                continue
            xs_g, acc_g = curve(binary[idx])
            ax.plot(xs_g, acc_g.mean(axis=0), color=colors[g], lw=1.7, label=g)
        ax.legend(loc="lower right", frameon=style.flag("legend.frame", False),
                  fontsize=style.font_size("legend_text"))

    finish_axes(ax, style, legend=False,
                xlabel="样本数", ylabel="累计观测特征数", title="物种累积曲线")
    fig.tight_layout()
    return save(fig, out_path, style,
                "曲线为随机抽取 n 个样本时累计发现的特征数（阴影为多次随机排序的极值范围）。"
                "曲线趋于平缓说明继续增加样本难以发现新特征。",
                "物种累积曲线")


def plot_rank_abundance(table: pd.DataFrame, groups: pd.Series | None,
                        out_path: str | Path, style: Style, max_rank: int = 200) -> dict:
    """等级丰度曲线（Rank-abundance）。"""
    rel = table.div(table.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    rel = rel.loc[rel.mean(axis=1).sort_values(ascending=False).index[:max_rank]]

    fig, ax = new_figure(style, 7.2, 4.6)
    groups_ = groups.astype(str) if groups is not None else None
    colors = style.group_colors(groups_) if groups_ is not None else {}
    alpha = 0.55 * style.num("palette.alpha", 1.0)

    for i, s in enumerate(rel.columns):
        vals = rel[s].sort_values(ascending=False).to_numpy()
        vals = vals[vals > 0]
        if vals.size == 0:
            continue
        color, label = "#7A7A7A", None
        if groups_ is not None and s in groups_.index:
            label = groups_[s]
            color = colors.get(label, "#7A7A7A")
        ax.plot(np.arange(1, vals.size + 1), vals, color=color, alpha=alpha, lw=1.0, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    finish_axes(ax, style, legend=groups_ is not None,
                xlabel="丰度排名（对数）", ylabel="相对丰度（对数）", title="等级丰度曲线")
    resolve_legend(style, ax, title="Group")
    fig.tight_layout()
    return save(fig, out_path, style,
                "每条折线为一个样本，曲线越陡峭表示群落均匀度越低（由少数优势类群主导）。",
                "等级丰度曲线")


def plot_sample_quality(table: pd.DataFrame, groups: pd.Series, out_path: str | Path,
                        style: Style, outlier_z: float = 2.0) -> dict:
    """样本质控概览：文库大小 vs 检出特征数，自动标出可能的离群样本。

    离群判定：文库大小或检出特征数偏离全体中位数超过 outlier_z 倍 MAD（稳健 z 分数）。
    """
    lib = table.sum(axis=0).astype(float)
    detected = (table > 0).sum(axis=0).astype(float)
    groups = groups.reindex(table.columns).astype(str)

    def robust_z(x: pd.Series) -> pd.Series:
        med = x.median()
        mad = (x - med).abs().median()
        scale = mad * 1.4826 if mad > 0 else (x.std(ddof=0) or 1.0)
        return (x - med) / (scale if scale else 1.0)

    z = pd.concat([robust_z(lib).rename("lib"), robust_z(detected).rename("det")], axis=1)
    flag = (z.abs() >= outlier_z).any(axis=1)

    fig, ax = new_figure(style, 8.2, 5.4)
    colors = style.group_colors(groups)
    size = style.num("element.point_size", 68)
    alpha = style.num("element.point_alpha", 0.9) * style.num("palette.alpha", 1.0)

    for g in sorted(groups.unique()):
        idx = [s for s in table.columns if groups[s] == g]
        ax.scatter(lib[idx], detected[idx], s=size, color=colors[g], label=g,
                   edgecolor="white" if style.flag("element.point_edge", True) else "none",
                   linewidth=0.9, alpha=alpha, zorder=3)

    out_idx = [s for s in table.columns if bool(flag.get(s, False))]
    if out_idx:
        ax.scatter(lib[out_idx], detected[out_idx], s=size * 2.2, facecolor="none",
                   edgecolor="#C44E52", linewidth=1.6, zorder=4)
        for s in out_idx:
            ax.annotate(str(s)[:22], (lib[s], detected[s]),
                        fontsize=style.font_size("annotation"),
                        color="#C44E52", xytext=(6, 5), textcoords="offset points", zorder=6)

    style_axes(ax, style)
    ax.set_xscale("log")
    place_axis_labels(ax, style, "文库大小（reads，对数）", "检出特征数")
    place_title(ax, style, "样本质控概览")
    place_annotation(
        ax,
        f"检出特征中位数 = {detected.median():.0f}   文库中位数 = {lib.median():,.0f}   离群样本 {len(out_idx)} 个",
        style, x=0.02, y=0.98, ha="left", va="top",
    )
    resolve_legend(style, ax, title="Group", preferred="lower right")
    fig.tight_layout()
    return save(fig, out_path, style,
                "每个点为一个样本。文库大小或检出特征数显著偏离整体的样本会被红圈标出——"
                "这类样本可能需要在下游分析（尤其是差异分析）中特别关注或剔除。",
                "样本质控概览")
