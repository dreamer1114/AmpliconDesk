"""宏转录组 / 表达谱绘图与功能富集绘图。"""

# ============================================================================
# 【未启用】本模块不属于 16S 扩增子的分析流程
# ----------------------------------------------------------------------------
# 宏转录组专属绘图：表达量分布、归一化因子、样本相关性热图、表达谱 PCA、高表达基因箱线图。
#
# AmpliconDesk 定位为 16S 扩增子下游分析工具，**本模块的图不再被任何流程产出**，
# 保留是为了将来需要时可以直接接回。
#
# 函数仍在 ``amplicondesk.plots`` 中导出，可以单独调用。
# ============================================================================


from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..style import Style
from .base import (
    finish_axes, new_figure, panel_grid, place_annotation, place_title,
    resolve_legend, save, style_axes,
    _save_clustermap as _save_clustermap,
)

__all__ = [
    "plot_expression_boxplot", "plot_correlation_heatmap", "plot_expression_pca",
    "plot_size_factors", "plot_enrichment_bubble", "plot_enrichment_bar",
]


def plot_expression_boxplot(profile: dict, groups: pd.Series, out_path: str | Path,
                            style: Style) -> dict:
    """归一化前后的表达量分布对比（并列面板）。"""
    raw_stats = profile["raw_stats"]
    norm_cpm = profile["normalized_cpm"]
    factors = profile["factors"]
    meta = groups.reindex(raw_stats.index).astype(str)
    colors = style.group_colors(meta)
    order = sorted(meta.unique())

    fig, axes, ncol = panel_grid(style, 4, ncol=4, base_w=4.0, base_h=3.6)

    def _box(ax, values, title, ylabel, log10=False):
        df = pd.DataFrame({
            "Group": [meta.get(s, "NA") for s in values.index],
            "value": values.to_numpy(dtype=float),
        })
        if log10:
            df = df[df["value"] > 0]
            df["value"] = np.log10(df["value"])
        sns.boxplot(data=df, x="Group", y="value", order=order, ax=ax, hue="Group",
                    palette=colors, width=style.num("element.box_width", 0.6),
                    fliersize=0, showfliers=False, legend=False)
        sns.stripplot(data=df, x="Group", y="value", order=order, ax=ax,
                      color="#2F2F2F", size=3, alpha=0.6,
                      jitter=style.num("element.jitter", 0.18))
        style_axes(ax, style)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=style.font_size("axis_label") * 0.92)
        place_title(ax, style, title, allow_override=False, scale=0.95)
        if len(order) > 4:
            ax.tick_params(axis="x", rotation=style.num("element.axis_rotate", 45))

    _box(axes[0][0], raw_stats["LibrarySize"], "归一化前：文库大小", "reads", log10=True)
    _box(axes[0][1], factors, "文库缩放因子", "缩放因子")
    _box(axes[0][2], norm_cpm.mean(axis=0), "归一化后：平均 log2 CPM",
         "log2(CPM+0.5)")
    axes[0][2].clear()
    log_cpm_means = np.log2(norm_cpm.mean(axis=0) + 0.5)
    _box(axes[0][2], log_cpm_means, "归一化后：平均表达水平", "log2(CPM+0.5)")
    _box(axes[0][3], (norm_cpm > 0).sum(axis=0).astype(float),
         "检出基因数", "detected genes")

    for k in range(4, axes.shape[0] * axes.shape[1]):
        axes[k // ncol][k % ncol].axis("off")

    from .base import place_suptitle

    method = str(profile.get("normalization", "")).upper()
    place_suptitle(fig, style, f"表达量分布（{method} 归一化）")
    fig.tight_layout()
    return save(fig, out_path, style,
                f"展示 {method} 归一化前后的样本表达量分布。归一化后各样本的分布应大致对齐，"
                "否则说明存在明显的文库组成偏差。",
                "表达量分布")


def plot_size_factors(factors: pd.Series, groups: pd.Series, out_path: str | Path,
                      style: Style) -> dict:
    meta = groups.reindex(factors.index).astype(str)
    colors = style.group_colors(meta)
    df = pd.DataFrame({"sample": factors.index, "factor": factors.to_numpy(dtype=float),
                       "Group": meta.to_numpy()})
    fig, ax = new_figure(style, max(6.4, 0.4 * len(df) + 3.0), 4.2)
    ax.bar(np.arange(len(df)), df["factor"],
           color=[colors.get(g, "#999999") for g in df["Group"]],
           edgecolor="white" if style.flag("element.bar_edge", True) else "none", width=0.8)
    ax.axhline(1.0, color="#333333", ls="--", lw=1.0)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["sample"], rotation=90, fontsize=style.font_size("tick_label") * 0.8)
    style_axes(ax, style)
    ax.set_ylabel("归一化因子")
    place_title(ax, style, "各样本归一化因子")
    fig.tight_layout()
    return save(fig, out_path, style,
                "归一化因子反映各样本相对测序深度/文库组成；虚线为基准 1.0，偏离越远说明该样本需要越大的校正。",
                "归一化因子")


def plot_correlation_heatmap(corr: pd.DataFrame, groups: pd.Series, out_path: str | Path,
                             style: Style, cluster: bool = True) -> dict:
    meta = groups.reindex(corr.index).astype(str)
    colors = style.group_colors(meta)
    width = max(6.4, 0.34 * corr.shape[1] + 3.6)
    height = max(5.6, 0.30 * corr.shape[0] + 3.0)

    if cluster and style.flag("element.dendrogram", True):
        g = sns.clustermap(corr, cmap=style.cmap("diverging"), vmin=float(np.nanmin(corr.to_numpy())),
                           vmax=1.0, row_colors=meta.map(colors), col_colors=meta.map(colors),
                           figsize=style.figsize(width, height),
                           cbar_pos=(0.02, 0.83, 0.022, 0.13),
                           dendrogram_ratio=(0.12, 0.12),
                           xticklabels=True, yticklabels=True)
        g.ax_heatmap.set_xticklabels(corr.columns, rotation=90,
                                     fontsize=style.font_size("tick_label") * 0.78)
        g.ax_heatmap.set_yticklabels(corr.index, fontsize=style.font_size("tick_label") * 0.78)
        g.ax_heatmap.grid(False)
        g.fig.suptitle("样本相关性热图", y=1.02, fontsize=style.font_size("suptitle"),
                       fontweight=style.font_weight("suptitle"),
                       color=style.font_color("suptitle"))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        made = _save_clustermap(g.fig, out_path, style)
        plt.close(g.fig)
        main = made.get("png") or made.get("pdf") or str(out_path)
        return {"path": main, "name": Path(main).name, "files": made,
                "pdf": made.get("pdf"),
                "caption": "样本间相关性（log2-CPM 空间，Pearson）。侧边色条为分组归属。"
                           "组内相关性应高于组间；个别样本与所有样本相关性都低时提示为离群样本。",
                "title": "样本相关性热图"}

    fig, ax = new_figure(style, width, height)
    sns.heatmap(corr, cmap=style.cmap("diverging"), ax=ax, vmin=float(np.nanmin(corr.to_numpy())),
                vmax=1.0, cbar_kws={"label": "相关系数"}, annot=False)
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=style.font_size("tick_label") * 0.78)
    ax.set_yticklabels(corr.index, fontsize=style.font_size("tick_label") * 0.78)
    ax.grid(False)
    place_title(ax, style, "样本相关性热图")
    fig.tight_layout()
    return save(fig, out_path, style,
                "样本间相关性（log2-CPM 空间），数值越接近 1 表示表达谱越相似。",
                "样本相关性热图")


def plot_expression_pca(coords: pd.DataFrame, explained, groups: pd.Series,
                        out_path: str | Path, style: Style) -> dict:
    from .ordination import _draw_ordination

    stat = f"PC1 {explained[0] * 100:.1f}%   PC2 {explained[1] * 100:.1f}%"
    return _draw_ordination(
        coords, groups, style, "表达谱 PCA", out_path,
        "基于 log2-CPM 的样本主成分分析，用于检查样本聚类与离群情况。",
        explained=explained, stat_text=stat)


# --------------------------------------------------------------------------
# 功能富集
# --------------------------------------------------------------------------
def plot_enrichment_bubble(enrich: pd.DataFrame, out_path: str | Path, style: Style,
                           top_n: int = 20, title: str = "功能富集分析") -> dict:
    if enrich is None or enrich.empty:
        raise ValueError("富集结果为空。")
    df = enrich.sort_values("p").head(top_n).copy()
    df = df.iloc[::-1]
    df["-log10P"] = -np.log10(np.clip(df["p"].to_numpy(dtype=float), 1e-300, 1.0))
    df["ratio"] = df["n_query"] / np.maximum(df["n_term"], 1)

    fig, ax = new_figure(style, 8.2, max(4.0, 0.40 * len(df) + 1.8))
    sizes = 40 + 320 * (df["ratio"] / max(df["ratio"].max(), 1e-9))
    sc = ax.scatter(df["-log10P"], np.arange(len(df)), s=sizes,
                    c=df["-log10P"], cmap=style.cmap("continuous"),
                    edgecolor="#555555", linewidth=0.5, alpha=style.num("palette.alpha", 1.0))
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels([str(t)[:44] for t in df["term"]],
                       fontsize=style.font_size("tick_label") * 0.9)
    ax.axvline(-np.log10(style.num("element.label_threshold", 0.05)),
               color="#888888", ls="--", lw=1.0)
    style_axes(ax, style)
    ax.set_xlabel("-log10 (p value)", fontsize=style.font_size("axis_label"))
    place_title(ax, style, title)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.036, pad=0.03)
    cbar.set_label("-log10(p)")
    fig.tight_layout()
    return save(fig, out_path, style,
                "气泡图展示显著富集的功能条目。横轴为显著性，点大小表示命中该条目的基因占该条目总基因数的比例。",
                title)


def plot_enrichment_bar(enrich: pd.DataFrame, out_path: str | Path, style: Style,
                        top_n: int = 20, title: str = "功能富集分析（条形图）") -> dict:
    if enrich is None or enrich.empty:
        raise ValueError("富集结果为空。")
    df = enrich.sort_values("p").head(top_n).iloc[::-1].copy()
    df["-log10P"] = -np.log10(np.clip(df["p"].to_numpy(dtype=float), 1e-300, 1.0))

    fig, ax = new_figure(style, 8.0, max(4.0, 0.36 * len(df) + 1.8))
    ax.barh([str(t)[:46] for t in df["term"]], df["-log10P"], color=style.single_color(2),
            edgecolor="white" if style.flag("element.bar_edge", True) else "none", linewidth=0.5)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["-log10P"], i, f"  {int(row['n_query'])}/{int(row['n_term'])}",
                va="center", fontsize=style.font_size("annotation") * 0.85)
    ax.axvline(-np.log10(style.num("element.label_threshold", 0.05)),
               color="#C44E52", ls="--", lw=1.1)
    style_axes(ax, style)
    ax.set_xlabel("-log10 (p value)", fontsize=style.font_size("axis_label"))
    place_title(ax, style, title)
    ax.tick_params(axis="y", labelsize=style.font_size("tick_label") * 0.9)
    fig.tight_layout()
    return save(fig, out_path, style,
                "条形图展示显著富集条目；标注为命中基因数 / 该条目总基因数。",
                title)
