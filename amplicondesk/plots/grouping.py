"""分组视角的绘图：先看组间差异，再看组内异质性。

科研上正确的顺序是：
  1) 先确认"分组之间有没有差异"（组间）
  2) 再确认"每组内部是否一致"（组内）
  如果组内本身就散得很开，那么组间差异的解读就要打折扣。
本模块把这两步的图放在一起，并在图注里写清楚怎么读。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..style import Style
from .base import (
    finish_axes, new_figure, panel_grid, place_annotation, place_axis_labels,
    place_suptitle, place_title, resolve_legend, save, style_axes,
)

__all__ = [
    "plot_group_overview", "plot_within_group_distance", "plot_within_group_diversity",
    "plot_between_vs_within",
]


# --------------------------------------------------------------------------
# 单组内两两距离
# --------------------------------------------------------------------------
def _within_distances(distance: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """把距离矩阵拆成"每个组内部的两两距离"长表。"""
    common = [s for s in distance.index if s in groups.index]
    if len(common) < 3:
        return pd.DataFrame(columns=["Group", "distance", "pair"])
    d = distance.loc[common, common].to_numpy(dtype=float)
    labels = groups.loc[common].astype(str).to_numpy()
    rows = []
    n = len(common)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                rows.append({"Group": labels[i], "distance": float(d[i, j]),
                             "pair": f"{common[i]} | {common[j]}"})
    return pd.DataFrame(rows)


def _between_distances(distance: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """组间两两距离长表。"""
    common = [s for s in distance.index if s in groups.index]
    if len(common) < 3:
        return pd.DataFrame(columns=["Group", "distance", "pair"])
    d = distance.loc[common, common].to_numpy(dtype=float)
    labels = groups.loc[common].astype(str).to_numpy()
    rows = []
    n = len(common)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                pair = " vs ".join(sorted([labels[i], labels[j]]))
                rows.append({"Group": pair, "distance": float(d[i, j]),
                             "pair": f"{common[i]} | {common[j]}"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def plot_between_vs_within(distance: pd.DataFrame, groups: pd.Series,
                           out_path: str | Path, style: Style,
                           metric_label: str = "", permanova_p: float | None = None) -> dict:
    """组间 vs 组内距离分布（并排小提琴 + 箱线），一眼看出分组是否成立。"""
    within = _within_distances(distance, groups)
    between = _between_distances(distance, groups)
    if within.empty and between.empty:
        raise ValueError("可用样本不足，无法比较组内与组间距离。")

    within = within.assign(kind="组内")
    between = between.assign(kind="组间")
    df = pd.concat([between, within], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=style.figsize(11.0, 4.8),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    fig.patch.set_facecolor(str(style.get("figure.background", "#FFFFFF")))
    for a in axes:
        a.set_facecolor(str(style.get("figure.facecolor_axes", "#FFFFFF")))

    # 左：组间 vs 组内 汇总
    ax = axes[0]
    order = ["组间", "组内"]
    palette = [style.single_color(3), style.single_color(0)]
    sns.violinplot(data=df, x="kind", y="distance", order=order, ax=ax,
                   palette=palette, inner=None, cut=0, linewidth=0.8,
                   alpha=0.55 * style.num("palette.alpha", 1.0))
    sns.boxplot(data=df, x="kind", y="distance", order=order, ax=ax,
                width=0.16, showcaps=True, showfliers=False,
                boxprops=dict(facecolor="white", edgecolor="#444444", linewidth=1.0),
                medianprops=dict(color="#C44E52", linewidth=1.6),
                whiskerprops=dict(color="#444444"), capprops=dict(color="#444444"))
    style_axes(ax, style)
    ax.set_xlabel("")
    place_axis_labels(ax, style, "", f"{metric_label} 距离".strip())
    place_title(ax, style, "组间 vs 组内距离", allow_override=False)
    med_b = float(between["distance"].median()) if not between.empty else np.nan
    med_w = float(within["distance"].median()) if not within.empty else np.nan
    if np.isfinite(med_b) and np.isfinite(med_w) and med_w > 0:
        txt = f"组间中位数 {med_b:.3f} / 组内中位数 {med_w:.3f}，相差 {med_b / med_w:.2f} 倍"
        place_annotation(ax, txt, style, x=0.5, y=0.98)

    # 右：每个组的组内距离
    ax2 = axes[1]
    if not within.empty:
        cat_order = sorted(within["Group"].unique(),
                           key=lambda g: float(within.loc[within["Group"] == g, "distance"].median()))
        colors = style.group_colors(cat_order)
        sns.violinplot(data=within, x="Group", y="distance", order=cat_order, ax=ax2,
                       hue="Group", palette=colors, legend=False,
                       inner=None, cut=0, linewidth=0.8,
                       alpha=0.45 * style.num("palette.alpha", 1.0))
        sns.boxplot(data=within, x="Group", y="distance", order=cat_order, ax=ax2,
                    width=0.14, showcaps=True, showfliers=False,
                    boxprops=dict(facecolor="white", edgecolor="#333333", linewidth=0.9))
        sns.stripplot(data=within, x="Group", y="distance", order=cat_order, ax=ax2,
                      color="#333333", size=2.2, alpha=0.45,
                      jitter=style.num("element.jitter", 0.2))
        ax2.tick_params(axis="x", rotation=style.num("element.axis_rotate", 45))
        # 组内异质性指数 = 组内距离中位数 / 总体组内中位数
        if np.isfinite(med_w) and med_w > 0:
            for i, g in enumerate(cat_order):
                med_g = float(within.loc[within["Group"] == g, "distance"].median())
                ax2.text(i, med_g, f" {med_g / med_w:.2f}×", ha="center", va="bottom",
                         fontsize=style.font_size("annotation") * 0.85,
                         color=style.font_color("annotation"))
    style_axes(ax2, style)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    place_title(ax2, style, "各组内部的两两距离", allow_override=False)

    place_suptitle(fig, style, f"组间与组内差异比较（{metric_label}）" if metric_label
                   else "组间与组内差异比较")
    fig.tight_layout()
    caption = (
        "左图比较组间距离与组内距离：组间明显大于组内，说明分组能解释群落差异；"
        "若两者接近，说明分组结构不明显。右图展示每个组内部样本两两之间的距离，"
        "标注的倍数 = 该组组内距离中位数 / 全部组内距离中位数，越大表示该组内部越不一致。"
    )
    if permanova_p is not None and np.isfinite(permanova_p):
        caption += f" 组间差异的 PERMANOVA p = {permanova_p:.3f}。"
    return save(fig, out_path, style, caption, "组间 vs 组内差异")


def plot_within_group_distance(distance: pd.DataFrame, groups: pd.Series,
                               out_path: str | Path, style: Style,
                               metric_label: str = "") -> dict:
    """组内异质性分析：各组内部两两距离的分布 + 组内平均相似度。"""
    within = _within_distances(distance, groups)
    if within.empty:
        raise ValueError("每组样本数不足（需要每组至少 2 个样本）才能计算组内距离。")

    cats = sorted(within["Group"].unique())
    colors = style.group_colors(cats)

    fig, axes = plt.subplots(1, 2, figsize=style.figsize(11.0, 4.6),
                             gridspec_kw={"width_ratios": [1.3, 1.0]})
    fig.patch.set_facecolor(str(style.get("figure.background", "#FFFFFF")))
    for a in axes:
        a.set_facecolor(str(style.get("figure.facecolor_axes", "#FFFFFF")))

    ax = axes[0]
    sns.violinplot(data=within, x="Group", y="distance", order=cats, ax=ax,
                   hue="Group", palette=colors, legend=False, inner=None, cut=0,
                   linewidth=0.8, alpha=0.5 * style.num("palette.alpha", 1.0))
    sns.boxplot(data=within, x="Group", y="distance", order=cats, ax=ax,
                width=0.15, showfliers=False,
                boxprops=dict(facecolor="white", edgecolor="#333333", linewidth=0.9))
    style_axes(ax, style)
    ax.set_xlabel("")
    place_axis_labels(ax, style, "", f"组内{(' ' + metric_label) if metric_label else ''}距离".strip())
    place_title(ax, style, "组内距离分布", allow_override=False)
    ax.tick_params(axis="x", rotation=style.num("element.axis_rotate", 45))

    # 右：组内异质性指数（变异系数）
    ax2 = axes[1]
    summary = within.groupby("Group")["distance"].agg(["median", "mean", "std", "count"])
    summary["cv"] = summary["std"] / summary["mean"].replace(0, np.nan)
    summary = summary.reindex(cats)
    bars = ax2.bar(np.arange(len(summary)), summary["cv"].fillna(0),
                   color=[colors.get(g, "#999999") for g in summary.index],
                   edgecolor="white" if style.flag("element.bar_edge", True) else "none",
                   width=0.66)
    for b, (g, row) in zip(bars, summary.iterrows()):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{row['cv']:.3f}\nn={int(row['count'])}",
                 ha="center", va="bottom", fontsize=style.font_size("annotation") * 0.85)
    style_axes(ax2, style)
    ax2.set_xticks(np.arange(len(summary)))
    ax2.set_xticklabels(summary.index, rotation=style.num("element.axis_rotate", 45))
    place_axis_labels(ax2, style, "", "组内距离变异系数 (CV)")
    place_title(ax2, style, "组内异质性指数", allow_override=False)
    ax2.set_ylim(0, float(summary["cv"].max() or 1) * 1.25)

    place_suptitle(fig, style, "组内差异分析")
    fig.tight_layout()
    return save(fig, out_path, style,
                "左图为各组内部样本两两之间的距离分布；右图为组内距离的变异系数（CV），"
                "CV 越大表示该组样本彼此差异越大、组内一致性越低。"
                "若某组 CV 明显偏高，说明该组可能存在亚群或离群样本，解读组间差异时需谨慎。",
                "组内差异分析")


def plot_within_group_diversity(alpha: pd.DataFrame, groups: pd.Series,
                                out_path: str | Path, style: Style,
                                metrics: tuple[str, ...] = ("Shannon", "Observed", "Pielou")) -> dict:
    """组内多样性一致性：每个组内多样性指标的离散程度。"""
    metrics = tuple(m for m in metrics if m in alpha.columns)
    if not metrics:
        raise ValueError("没有可用的 alpha 多样性指标。")

    merged = alpha.join(groups.rename("Group"), how="inner")
    merged["Group"] = merged["Group"].astype(str)
    cats = sorted(merged["Group"].unique())
    colors = style.group_colors(cats)

    fig, axes, ncol = panel_grid(style, len(metrics), ncol=len(metrics),
                                 base_w=3.8, base_h=3.6)
    for k, metric in enumerate(metrics):
        ax = axes[0][k]
        # 每个样本相对本组均值的偏离（标准化到本组标准差），反映组内一致性
        sub = merged[[metric, "Group"]].dropna()
        stats = sub.groupby("Group")[metric].agg(["mean", "std", "count"])
        stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)

        sns.boxplot(data=sub, x="Group", y=metric, order=cats, ax=ax, hue="Group",
                    palette=colors, width=style.num("element.box_width", 0.6),
                    fliersize=0, showfliers=False, legend=False)
        sns.stripplot(data=sub, x="Group", y=metric, order=cats, ax=ax,
                      color="#2F2F2F", size=3, alpha=0.65,
                      jitter=style.num("element.jitter", 0.18))
        # 用虚线标出各组均值，虚线越密说明组内越一致
        for i, g in enumerate(cats):
            m = stats.loc[g, "mean"]
            if np.isfinite(m):
                ax.hlines(m, i - 0.3, i + 0.3, color=colors.get(g, "#333333"),
                          lw=style.num("element.line_width", 1.8), zorder=4)
        style_axes(ax, style)
        ax.set_xlabel("")
        cv_txt = "  ".join(f"{g} CV={stats.loc[g, 'cv']:.3f}"
                           for g in cats if pd.notna(stats.loc[g, "cv"]))
        place_annotation(ax, cv_txt or "CV 不可用", style, x=0.5, y=0.98)
        place_title(ax, style, metric, allow_override=False)
        ax.tick_params(axis="x", rotation=style.num("element.axis_rotate", 45))

    for k in range(len(metrics), axes.shape[0] * axes.shape[1]):
        axes[k // ncol][k % ncol].axis("off")
    place_suptitle(fig, style, "各组内部多样性一致性")
    fig.tight_layout()
    return save(fig, out_path, style,
                "粗横线为各组均值，标注为组内变异系数（CV = 标准差 / 均值）。"
                "CV 越小说明该组样本之间的多样性越一致；某组 CV 明显偏大时，"
                "该组可能包含异质亚群，会影响组间比较的可靠性。",
                "组内多样性一致性")


def plot_group_overview(groups: pd.Series, table: pd.DataFrame,
                        distance: pd.DataFrame | None, alpha: pd.DataFrame | None,
                        out_path: str | Path, style: Style) -> dict:
    """分组总览：一张图看清分组构成、组内距离、组内多样性与文库大小。"""
    groups = groups.astype(str)
    cats = sorted(groups.unique())
    colors = style.group_colors(cats)

    fig, axes = plt.subplots(2, 2, figsize=style.figsize(11.0, 8.0))
    fig.patch.set_facecolor(str(style.get("figure.background", "#FFFFFF")))
    for row in axes:
        for a in row:
            a.set_facecolor(str(style.get("figure.facecolor_axes", "#FFFFFF")))

    # 1) 样本数
    ax = axes[0][0]
    counts = groups.value_counts().reindex(cats).fillna(0)
    ax.bar(np.arange(len(counts)), counts.to_numpy(),
           color=[colors.get(g, "#999") for g in counts.index],
           edgecolor="white" if style.flag("element.bar_edge", True) else "none", width=0.66)
    for i, v in enumerate(counts.to_numpy()):
        ax.text(i, v, f" {int(v)}", ha="center", va="bottom",
                fontsize=style.font_size("annotation") * 0.9)
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(counts.index, rotation=style.num("element.axis_rotate", 45))
    style_axes(ax, style)
    place_title(ax, style, "各组样本数", allow_override=False)
    place_axis_labels(ax, style, "", "样本数")

    # 2) 文库大小
    ax = axes[0][1]
    lib = table.sum(axis=0)
    df = pd.DataFrame({"Group": [groups.get(s, "NA") for s in lib.index],
                       "lib": lib.to_numpy(dtype=float)})
    sns.boxplot(data=df, x="Group", y="lib", order=cats, ax=ax, hue="Group",
                palette=colors, width=0.6, fliersize=0, showfliers=False, legend=False)
    sns.stripplot(data=df, x="Group", y="lib", order=cats, ax=ax, color="#333",
                  size=3, alpha=0.6, jitter=style.num("element.jitter", 0.18))
    style_axes(ax, style)
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=style.num("element.axis_rotate", 45))
    place_title(ax, style, "各样本文库大小", allow_override=False)
    place_axis_labels(ax, style, "", "reads（对数）")

    # 3) 组内距离
    ax = axes[1][0]
    if distance is not None and not distance.empty:
        within = _within_distances(distance, groups)
        if not within.empty:
            sns.boxplot(data=within, x="Group", y="distance", order=cats, ax=ax,
                        hue="Group", palette=colors, width=0.6, fliersize=0,
                        showfliers=False, legend=False)
            sns.stripplot(data=within, x="Group", y="distance", order=cats, ax=ax,
                          color="#333", size=2.4, alpha=0.5,
                          jitter=style.num("element.jitter", 0.2))
        else:
            ax.text(0.5, 0.5, "每组样本不足", ha="center", va="center",
                    transform=ax.transAxes, color="#999")
    else:
        ax.text(0.5, 0.5, "无距离矩阵", ha="center", va="center",
                transform=ax.transAxes, color="#999")
    style_axes(ax, style)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=style.num("element.axis_rotate", 45))
    place_title(ax, style, "组内两两距离", allow_override=False)

    # 4) 组内 alpha 离散度
    ax = axes[1][1]
    if alpha is not None and "Shannon" in alpha.columns:
        merged = alpha[["Shannon"]].join(groups.rename("Group"), how="inner")
        merged["Group"] = merged["Group"].astype(str)
        sns.violinplot(data=merged, x="Group", y="Shannon", order=cats, ax=ax,
                       hue="Group", palette=colors, legend=False, inner=None, cut=0,
                       linewidth=0.8, alpha=0.5)
        sns.boxplot(data=merged, x="Group", y="Shannon", order=cats, ax=ax,
                    width=0.15, showfliers=False,
                    boxprops=dict(facecolor="white", edgecolor="#333", linewidth=0.9))
    else:
        ax.text(0.5, 0.5, "无 alpha 数据", ha="center", va="center",
                transform=ax.transAxes, color="#999")
    style_axes(ax, style)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=style.num("element.axis_rotate", 45))
    place_title(ax, style, "组内 Shannon 多样性", allow_override=False)

    place_suptitle(fig, style, "分组总览")
    fig.tight_layout()
    return save(fig, out_path, style,
                "四张图合起来回答「分组是否可靠」：样本量是否均衡、文库大小是否可比、"
                "组内样本是否彼此接近（组内距离越小越一致）、组内多样性是否集中。"
                "任何一格出现明显异常，都应在解读组间差异前先排查。",
                "分组总览")
