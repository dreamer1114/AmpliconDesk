"""群落组成：堆叠柱状图、面积图、热图、优势类群箱线图、Venn/UpSet。"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
from matplotlib_venn import venn2, venn3  # type: ignore

from ..style import Style
from .base import (
    finish_axes, new_figure, panel_grid, place_annotation, place_suptitle,
    place_title, resolve_legend, save, style_axes,
    _save_clustermap as _save_clustermap,
)

__all__ = [
    "plot_composition_bar", "plot_composition_area", "plot_heatmap",
    "plot_top_taxa_box", "plot_venn", "plot_core_microbiome", "plot_ternary",
]


def _to_relative(table: pd.DataFrame) -> pd.DataFrame:
    ab = table.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    totals = ab.sum(axis=0)
    if float(totals.max()) > 1.5:
        ab = ab.div(totals.replace(0, np.nan), axis=1).fillna(0.0)
    return ab


def _collapse_top(rel: pd.DataFrame, top_n: int, other_label: str = "Others") -> pd.DataFrame:
    if rel.shape[0] <= top_n:
        return rel.copy()
    order = rel.mean(axis=1).sort_values(ascending=False)
    keep = list(order.index[:top_n])
    rest = rel.drop(index=keep)
    if rest.empty:
        return rel.loc[keep].copy()
    return pd.concat([rel.loc[keep], rest.sum(axis=0).to_frame(other_label).T])


def _feature_colors(style: Style, labels: list[str]) -> dict:
    """类群配色：优先使用 element 级连续色图循环，保证同一类群颜色稳定。"""
    cmap = plt.get_cmap("tab20")
    extra = plt.get_cmap("tab20b")
    colors = {}
    for i, f in enumerate(labels):
        colors[f] = cmap(i % 20) if i < 20 else extra((i - 20) % 20)
    return colors


# --------------------------------------------------------------------------
def plot_composition_bar(rel_table: pd.DataFrame, groups: pd.Series,
                         out_path: str | Path, style: Style,
                         rank_label: str = "Feature", top_n: int = 10) -> dict:
    ab = _to_relative(rel_table)
    data = _collapse_top(ab, top_n)

    order = [s for s in data.columns if s in groups.index] or list(data.columns)
    meta = groups.loc[order].astype(str)
    sorted_cols = sorted(order, key=lambda s: (str(meta[s]), s))
    data = data[sorted_cols]
    plot_df = data.T

    n = len(sorted_cols)
    fig, ax = new_figure(style, max(7.5, 0.34 * n + 3.6), 5.0)
    bottom = np.zeros(len(plot_df))
    x = np.arange(len(plot_df))
    labels = list(plot_df.columns)
    color_map = _feature_colors(style, labels)

    edge = "white" if style.flag("element.bar_edge", True) else "none"
    alpha = style.num("palette.alpha", 1.0)
    for f in labels:
        vals = plot_df[f].to_numpy(dtype=float) * 100
        ax.bar(x, vals, bottom=bottom, color=color_map[f], width=0.82,
               edgecolor=edge, linewidth=0.35, alpha=alpha, label=str(f)[:48])
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_cols, rotation=90, fontsize=style.font_size("tick_label") * 0.78)
    ax.set_ylabel("相对丰度 (%)", fontsize=style.font_size("axis_label"),
                  color=style.font_color("axis_label"))
    ax.set_ylim(0, 100)

    if style.flag("layout.bar_group_gap", True):
        cats = [str(meta[s]) for s in sorted_cols]
        for i in range(1, len(cats)):
            if cats[i] != cats[i - 1]:
                ax.axvline(i - 0.5, color="#333333", lw=1.0, ls="--", alpha=0.7)
        positions: dict[str, list[int]] = {}
        for i, c in enumerate(cats):
            positions.setdefault(c, []).append(i)
        for c, pos in positions.items():
            ax.text(np.mean(pos), 101.5, c, ha="center", va="bottom",
                    fontsize=style.font_size("tick_label"), color=style.font_color("tick_label"))

    style_axes(ax, style)
    place_title(ax, style, f"群落组成（{rank_label}，Top {min(top_n, len(labels))} + Others）")
    resolve_legend(style, ax, title=rank_label, preferred="outside right")
    fig.tight_layout()
    return save(fig, out_path, style,
                "堆叠柱状图展示各样本在指定分类级上的相对丰度构成，虚线分隔不同分组。",
                f"群落组成（{rank_label}）")


def plot_composition_area(rel_table: pd.DataFrame, groups: pd.Series,
                          out_path: str | Path, style: Style,
                          rank_label: str = "Feature", top_n: int = 12) -> dict:
    """面积图：按分组排序后展示类群丰度的连续变化。"""
    ab = _to_relative(rel_table)
    data = _collapse_top(ab, top_n)

    order = [s for s in data.columns if s in groups.index] or list(data.columns)
    meta = groups.loc[order].astype(str)
    sorted_cols = sorted(order, key=lambda s: (str(meta[s]), s))
    data = data[sorted_cols]
    labels = list(data.index)
    color_map = _feature_colors(style, labels)

    fig, ax = new_figure(style, 8.4, 4.8)
    x = np.arange(len(sorted_cols))
    ax.stackplot(x, *[data.loc[f].to_numpy(dtype=float) * 100 for f in labels],
                 labels=[str(f)[:44] for f in labels],
                 colors=[color_map[f] for f in labels],
                 alpha=style.num("palette.alpha", 1.0), edgecolor="white", linewidth=0.2)
    ax.set_xlim(0, max(1, len(sorted_cols) - 1))
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_cols, rotation=90, fontsize=style.font_size("tick_label") * 0.78)

    if style.flag("layout.bar_group_gap", True):
        cats = [str(meta[s]) for s in sorted_cols]
        for i in range(1, len(cats)):
            if cats[i] != cats[i - 1]:
                ax.axvline(i - 0.5, color="#333333", lw=1.0, ls="--", alpha=0.7)

    style_axes(ax, style)
    place_title(ax, style, f"群落组成面积图（{rank_label}，Top {min(top_n, len(labels))} + Others）")
    ax.set_ylabel("相对丰度 (%)", fontsize=style.font_size("axis_label"))
    resolve_legend(style, ax, title=rank_label, preferred="outside right")
    fig.tight_layout()
    return save(fig, out_path, style,
                "面积图按分组顺序展示各分类单元的连续丰度变化，便于观察组内/组间的渐变趋势。",
                f"群落组成面积图（{rank_label}）")


def plot_heatmap(rel_table: pd.DataFrame, groups: pd.Series, out_path: str | Path,
                 style: Style, top_n: int = 30, rank_label: str = "Feature",
                 cluster: bool = True) -> dict:
    rel = _to_relative(rel_table)
    top = rel.mean(axis=1).sort_values(ascending=False).head(top_n).index
    mat = rel.loc[top]
    mat = mat[[s for s in mat.columns if s in groups.index]]
    if mat.shape[1] == 0:
        raise ValueError("没有可用的样本。")

    meta = groups.loc[mat.columns].astype(str)
    order = sorted(mat.columns, key=lambda s: (str(meta[s]), s))
    mat, meta = mat[order], meta[order]
    plot_mat = np.log10(mat + 1e-4)

    width = max(7.0, 0.32 * mat.shape[1] + 3.4)
    height = max(5.2, 0.24 * mat.shape[0] + 2.0)
    cmap = style.cmap("heatmap")
    show_values = style.flag("element.heatmap_values", False)
    annot = plot_mat.round(2) if show_values else False
    colors = style.group_colors(meta)

    if cluster and style.flag("element.dendrogram", True):
        g = sns.clustermap(
            plot_mat, cmap=cmap, col_cluster=False, row_cluster=True,
            col_colors=meta.map(colors), xticklabels=True, yticklabels=True,
            figsize=style.figsize(width, height), cbar_pos=(0.015, 0.84, 0.022, 0.12),
            dendrogram_ratio=(0.13, 0.02),
            annot=annot, annot_kws={"fontsize": max(4, style.font_size("annotation") * 0.55)},
        )
        reordered = g.dendrogram_row.reordered_ind
        g.ax_heatmap.set_yticklabels([str(plot_mat.index[i])[:52] for i in reordered],
                                     fontsize=style.font_size("tick_label") * 0.8)
        g.ax_heatmap.set_xticklabels(mat.columns, rotation=90,
                                     fontsize=style.font_size("tick_label") * 0.76)
        g.ax_heatmap.set_xlabel("")
        g.ax_heatmap.set_ylabel("")
        g.ax_heatmap.grid(False)
        g.fig.suptitle(f"Top {mat.shape[0]} {rank_label} 丰度热图", y=1.02,
                       fontsize=style.font_size("suptitle"),
                       fontweight=style.font_weight("suptitle"),
                       color=style.font_color("suptitle"))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        made = _save_clustermap(g.fig, out_path, style)
        plt.close(g.fig)
        main = made.get("png") or made.get("pdf") or str(out_path)
        return {"path": main, "name": Path(main).name, "files": made,
                "pdf": made.get("pdf"),
                "caption": "热图为 log10 转换后的相对丰度，颜色越红丰度越高；"
                           "行为特征（已按相似度层次聚类），列为样本并按分组排序。",
                "title": f"Top {mat.shape[0]} {rank_label} 丰度热图"}

    fig, ax = new_figure(style, width, height)
    sns.heatmap(plot_mat, cmap=cmap, ax=ax, annot=annot,
                annot_kws={"fontsize": max(4, style.font_size("annotation") * 0.55)},
                cbar_kws={"label": "log10(相对丰度)"})
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=style.font_size("tick_label") * 0.76)
    ax.set_yticklabels([str(t)[:52] for t in mat.index], fontsize=style.font_size("tick_label") * 0.8)
    ax.grid(False)
    place_title(ax, style, f"Top {mat.shape[0]} {rank_label} 丰度热图")
    fig.tight_layout()
    return save(fig, out_path, style,
                "热图为 log10 转换后的相对丰度。", f"Top {mat.shape[0]} {rank_label} 丰度热图")


def plot_top_taxa_box(rel_table: pd.DataFrame, groups: pd.Series, out_path: str | Path,
                      style: Style, rank_label: str = "Feature", top_n: int = 9) -> dict:
    """优势类群在分组间的丰度箱线图（每个类群一个面板）。"""
    rel = _to_relative(rel_table)
    top = rel.mean(axis=1).sort_values(ascending=False).head(top_n).index
    merged = rel.loc[top].T.join(groups.rename("Group"), how="inner")
    merged["Group"] = merged["Group"].astype(str)
    colors = style.group_colors(merged["Group"])
    order = sorted(merged["Group"].unique())

    fig, axes, ncol = panel_grid(style, len(top), base_w=3.4, base_h=3.0)
    for k, feat in enumerate(top):
        ax = axes[k // ncol][k % ncol]
        sns.boxplot(data=merged, x="Group", y=feat, order=order, ax=ax, hue="Group",
                    palette=colors, width=style.num("element.box_width", 0.6),
                    fliersize=0, showfliers=False, legend=False)
        if style.flag("element.point_labels", True):
            sns.stripplot(data=merged, x="Group", y=feat, order=order, ax=ax,
                          color="#2F2F2F", size=2.6, alpha=0.6,
                          jitter=style.num("element.jitter", 0.18))
        style_axes(ax, style)
        ax.set_xlabel("")
        ax.set_ylabel("相对丰度", fontsize=style.font_size("axis_label") * 0.9)
        place_title(ax, style, str(feat)[:34], allow_override=False, scale=0.9)
        if len(order) > 4:
            ax.tick_params(axis="x", rotation=style.num("element.axis_rotate", 45))

    for k in range(len(top), axes.shape[0] * axes.shape[1]):
        axes[k // ncol][k % ncol].axis("off")
    place_suptitle(fig, style, f"优势 {rank_label} 组间丰度比较")
    fig.tight_layout()
    return save(fig, out_path, style,
                "每个面板为一个优势类群，箱线图展示其相对丰度在分组间的分布差异。",
                f"优势 {rank_label} 组间箱线图")


# --------------------------------------------------------------------------
def _presence_sets(table: pd.DataFrame, groups: pd.Series, threshold: float) -> dict[str, set]:
    sets: dict[str, set] = {}
    for g in sorted(groups.astype(str).unique()):
        cols = [c for c in table.columns if str(groups.get(c)) == g]
        if not cols:
            continue
        sub = table[cols]
        present = (sub > 0).sum(axis=1) / len(cols) >= threshold
        sets[g] = set(sub.index[present])
    return sets


def plot_venn(table: pd.DataFrame, groups: pd.Series, out_path: str | Path,
              style: Style, threshold: float = 0.5, max_groups: int = 5,
              rank_label: str = "Feature") -> dict:
    """特征共享情况：2~3 组画 Venn 图，4 组及以上画 UpSet 图。"""
    sets = _presence_sets(table, groups, threshold)
    if len(sets) < 2:
        raise ValueError("至少需要 2 个分组才能绘制共享特征图。")

    names = list(sets.keys())
    colors = style.group_colors(names)
    bg = str(style.get("figure.background", "#FFFFFF"))
    caption = f"判定标准：某特征在该组至少有 {threshold:.0%} 的样本中检出，即视为该组存在。"

    # ---- 2 或 3 组：Venn ----
    if len(sets) <= 3:
        fig, ax = new_figure(style, 6.4, 5.8)
        ax.set_aspect("equal")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if len(sets) == 2:
                a, b = names
                v = venn2([sets[a], sets[b]], set_labels=(a, b), ax=ax)
                patch_colors = {"10": colors[a], "01": colors[b], "11": "#C9C9C9"}
            else:
                a, b, c = names
                v = venn3([sets[a], sets[b], sets[c]], set_labels=(a, b, c), ax=ax)
                patch_colors = {k: "#C9C9C9" for k in
                                ["100", "010", "001", "110", "101", "011", "111"]}
                patch_colors["100"], patch_colors["010"], patch_colors["001"] = (
                    colors[a], colors[b], colors[c])

        for key, col in patch_colors.items():
            patch = v.get_patch_by_id(key)
            if patch is not None:
                patch.set_color(col)
                patch.set_alpha(0.72)
                patch.set_edgecolor("white")

        for lbl in getattr(v, "set_labels", []) or []:
            if lbl:
                lbl.set_fontsize(style.font_size("legend_text") + 1.5)
                lbl.set_color(style.font_color("legend_text"))
        for lbl in getattr(v, "subset_labels", []) or []:
            if lbl:
                lbl.set_fontsize(style.font_size("annotation"))
                lbl.set_color(style.font_color("annotation"))

        place_title(ax, style, f"各分组共有 / 特有{rank_label}")
        fig.tight_layout()
        return save(fig, out_path, style,
                    caption + " 图中数字为该区域的特征数量。", f"Venn 图（{rank_label}）")

    # ---- 4 组及以上：UpSet ----
    used = names[:max_groups]
    all_features = sorted(set().union(*[sets[g] for g in used]))
    rows = []
    for feat in all_features:
        members = tuple(g for g in used if feat in sets[g])
        if members:
            rows.append(members)
    combo = pd.Series(rows).value_counts()

    top = combo.head(20)
    fig, (ax_bar, ax_dot) = plt.subplots(
        2, 1, figsize=style.figsize(9.0, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.3], "hspace": 0.06},
    )
    fig.patch.set_facecolor(bg)
    x = np.arange(len(top))
    bar_color = style.single_color(0)
    ax_bar.bar(x, top.to_numpy(), color=bar_color, width=0.62,
               edgecolor="white", linewidth=0.6)
    for xi, val in zip(x, top.to_numpy()):
        ax_bar.text(xi, val, str(val), ha="center", va="bottom",
                    fontsize=style.font_size("annotation") * 0.9)
    ax_bar.set_ylabel("特征数", fontsize=style.font_size("axis_label"))
    place_title(ax_bar, style, f"UpSet：各分组组合的共有 / 特有{rank_label}")
    style_axes(ax_bar, style)

    for i in range(len(used)):
        ax_dot.scatter(x, [i] * len(x), s=26, color="#DDDDDD", zorder=2)
    for xi, members in enumerate(top.index):
        idx = [used.index(g) for g in members]
        ax_dot.scatter([xi] * len(idx), idx, s=64, color=bar_color, zorder=3)
        if idx:
            ax_dot.plot([xi, xi], [min(idx), max(idx)], color=bar_color, lw=1.6, zorder=2)

    ax_dot.set_yticks(range(len(used)))
    ax_dot.set_yticklabels(used, fontsize=style.font_size("tick_label"))
    ax_dot.set_ylim(-0.7, len(used) - 0.3)
    ax_dot.set_xticks([])
    ax_dot.invert_yaxis()
    style_axes(ax_dot, style)
    fig.tight_layout()
    return save(fig, out_path, style,
                caption + " 展示特征数最多的前 20 种分组组合；点阵表示该组合包含哪些分组。",
                f"UpSet 图（{rank_label}）")


def plot_core_microbiome(table: pd.DataFrame, groups: pd.Series, out_path: str | Path,
                         style: Style, prevalence_steps: int = 40) -> dict:
    """核心微生物组分析：平均相对丰度 vs 出现率。

    按不同出现率阈值画出保留的特征数，并把高丰度 / 高出现率的"核心"特征高亮。
    """
    rel = _to_relative(table)
    if rel.shape[1] < 2:
        raise ValueError("样本数不足，无法做核心微生物组分析。")

    prevalence = (rel > 0).sum(axis=1) / rel.shape[1]
    mean_abund = rel.mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=style.figsize(11.0, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    fig.patch.set_facecolor(str(style.get("figure.background", "#FFFFFF")))
    for a in (ax1, ax2):
        a.set_facecolor(str(style.get("figure.facecolor_axes", "#FFFFFF")))

    # --- 左：丰度 vs 出现率 ---
    core_mask = (prevalence >= 0.8) & (mean_abund >= mean_abund.median())
    base = style.single_color(0)
    ax1.scatter(mean_abund[~core_mask], prevalence[~core_mask], s=22, alpha=0.55,
                color="#B9C2CC", edgecolor="none", label=f"其他 ({int((~core_mask).sum())})")
    ax1.scatter(mean_abund[core_mask], prevalence[core_mask], s=42, alpha=0.92,
                color=base, edgecolor="white", linewidth=0.4,
                label=f"核心特征 ({int(core_mask.sum())})")
    ax1.set_xscale("log")
    ax1.axhline(0.8, color="#888888", ls="--", lw=1.0)
    ax1.axvline(float(mean_abund.median()), color="#888888", ls="--", lw=1.0)
    style_axes(ax1, style)
    ax1.set_xlabel("平均相对丰度（对数）", fontsize=style.font_size("axis_label"))
    ax1.set_ylabel("出现率（样本比例）", fontsize=style.font_size("axis_label"))
    ax1.set_ylim(-0.03, 1.05)
    place_title(ax1, style, "丰度 - 出现率分布", allow_override=False)
    resolve_legend(style, ax1)

    # --- 右：不同出现率阈值下保留的特征数 ---
    thresholds = np.linspace(0, 1, max(5, int(prevalence_steps)))
    counts = [(prevalence >= t).sum() for t in thresholds]
    ax2.plot(thresholds * 100, counts, color=base, lw=2.2, marker="o", ms=3.5)
    ax2.axvline(80, color="#C44E52", ls="--", lw=1.2)
    ax2.annotate(f"80% 阈值保留 {int((prevalence >= 0.8).sum())} 个",
                 xy=(80, int((prevalence >= 0.8).sum())),
                 xytext=(10, 12), textcoords="offset points",
                 fontsize=style.font_size("annotation"),
                 color=style.font_color("annotation"))
    style_axes(ax2, style)
    ax2.set_xlabel("出现率阈值 (%)", fontsize=style.font_size("axis_label"))
    ax2.set_ylabel("保留特征数", fontsize=style.font_size("axis_label"))
    place_title(ax2, style, "核心特征数随阈值的变化")

    for a in (ax1, ax2):
        a.grid(style.flag("element.grid", True),
               alpha=style.num("element.grid_alpha", 0.25), linewidth=0.6)
    place_suptitle(fig, style, "核心微生物组分析")
    fig.tight_layout()
    return save(fig, out_path, style,
                "左图每个点为一个特征：横轴平均相对丰度，纵轴出现率。右上角高丰度且高出现率的特征"
                "构成核心群落。右图展示出现率阈值收紧时保留的特征数如何下降。",
                "核心微生物组分析")


def plot_ternary(rel_table: pd.DataFrame, groups: pd.Series, out_path: str | Path,
                 style: Style, rank_label: str = "Feature", top_n: int = 300) -> dict:
    """三元图：三组样本中各类群的相对丰度构成（需要恰好 3 个分组）。"""
    cats = sorted(groups.astype(str).unique())
    if len(cats) != 3:
        raise ValueError(f"三元图需要恰好 3 个分组，当前有 {len(cats)} 个。")

    rel = _to_relative(rel_table)
    cols = [s for s in rel.columns if s in groups.index]
    if len(cols) < 3:
        raise ValueError("可用样本不足。")
    rel = rel[cols]

    means = pd.DataFrame({
        c: rel[[s for s in cols if str(groups[s]) == c]].mean(axis=1) for c in cats
    }).fillna(0.0)
    total = means.sum(axis=1)
    means = means[total > 0]
    if means.empty:
        raise ValueError("没有可绘制的特征。")
    total = means.sum(axis=1)
    prop = means.div(total, axis=0)

    order = total.sort_values(ascending=False).head(top_n).index
    prop = prop.loc[order]
    sizes = total.loc[order]

    a = prop[cats[0]].to_numpy(dtype=float)
    b = prop[cats[1]].to_numpy(dtype=float)
    c = prop[cats[2]].to_numpy(dtype=float)
    x = 0.5 * (2 * b + c)
    y = (np.sqrt(3) / 2) * c

    dominant = np.argmax(prop.to_numpy(dtype=float), axis=1)
    colors = style.group_colors(cats)

    fig, ax = new_figure(style, 7.2, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # 三角形边框与网格
    for frac in (0.25, 0.5, 0.75):
        for i in range(3):
            pts = _ternary_gridline(frac, i)
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color="#DDDDDD", lw=0.8, zorder=1)
    tri = [(0, 0), (1, 0), (0.5, np.sqrt(3) / 2), (0, 0)]
    ax.plot([p[0] for p in tri], [p[1] for p in tri], color="#555555", lw=1.4, zorder=2)

    for i, cat in enumerate(cats):
        ax.scatter(x[dominant == i], y[dominant == i], s=18 + 260 * sizes.to_numpy()[dominant == i]
                   / max(sizes.max(), 1e-12), color=colors[cat], alpha=0.72,
                   edgecolor="white", linewidth=0.3, label=cat, zorder=3)

    off = 0.06
    fs = style.font_size("legend_text") + 1.5
    fc = style.font_color("legend_text")
    ax.text(-off, -off, f"{cats[0]}\n100%", ha="right", va="top", fontsize=fs, color=fc,
            linespacing=1.3)
    ax.text(1 + off, -off, f"{cats[1]}\n100%", ha="left", va="top", fontsize=fs, color=fc,
            linespacing=1.3)
    ax.text(0.5, np.sqrt(3) / 2 + off, f"{cats[2]} 100%", ha="center", va="bottom",
            fontsize=fs, color=fc)

    ax.set_xlim(-0.28, 1.30)
    ax.set_ylim(-0.22, np.sqrt(3) / 2 + 0.16)
    place_title(ax, style, f"三元图（{rank_label}，前 {len(prop)} 个类群）")
    resolve_legend(style, ax, title="优势分组", preferred="upper right")
    fig.tight_layout()
    return save(fig, out_path, style,
                "每个点为一个类群的丰度构成，越靠近某个角表示在该分组中占比越高；"
                "点的大小代表该类群的整体丰度。用于直观找出各组特异富集的类群。",
                f"三元图（{rank_label}）")


def _ternary_gridline(frac: float, side: int) -> list[tuple[float, float]]:
    """三元网格线。顶点 A=(0,0)、B=(1,0)、C=(0.5,h)，重心坐标 (a,b,c) 映射为 x=b+0.5c, y=h*c。

    side=0 → c 恒为 frac（平行底边）；side=1 → a 恒为 frac（平行右边）；
    side=2 → b 恒为 frac（平行左边）。
    """
    h = np.sqrt(3) / 2
    if side == 0:
        return [(0.5 * frac, frac * h), (1 - 0.5 * frac, frac * h)]
    if side == 1:
        return [(1 - frac, 0.0), (0.5 * (1 - frac), (1 - frac) * h)]
    return [(frac, 0.0), (0.5 + 0.5 * frac, (1 - frac) * h)]
