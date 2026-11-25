"""排序分析与多元统计的绘图：PCoA/NMDS/PCA/RDA/dbRDA/CCA/CAP/t-SNE/UMAP/PLS-DA、
碎石图、距离分布、PERMDISP、envfit、Mantel。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..style import Style
from .base import (
    add_biplot_arrows, finish_axes, new_figure, place_annotation, place_axis_labels,
    place_title, resolve_legend, save, scatter_groups, stat_box, style_axes,
)

__all__ = [
    "plot_pcoa", "plot_nmds", "plot_pca", "plot_rda", "plot_dbrda", "plot_cca",
    "plot_cap", "plot_tsne", "plot_umap", "plot_plsda", "plot_plsda_vip",
    "plot_scree", "plot_distance_boxplot", "plot_permdisp", "plot_envfit", "plot_mantel",
]


def _p_label(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _stars(p: float) -> str:
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _axis_labels(explained, ncol: int, coords: pd.DataFrame, prefix: str) -> tuple[str, str]:
    if explained is not None and len(explained) >= 2:
        a = f"{coords.columns[0]} ({float(explained[0]) * 100:.1f}%)"
        b = f"{coords.columns[1]} ({float(explained[1]) * 100:.1f}%)"
        return a, b
    return coords.columns[0], coords.columns[1]


# --------------------------------------------------------------------------
# 通用排序绘图
# --------------------------------------------------------------------------
def _draw_ordination(
    coords: pd.DataFrame, groups: pd.Series, style: Style,
    title: str, out_path: str | Path, caption: str,
    explained=None, metric_label: str = "", stat_text: str = "",
    env_arrows: pd.DataFrame | None = None, feature_arrows: pd.DataFrame | None = None,
    base_w: float = 7.0, base_h: float = 5.6,
) -> dict:
    fig, ax = new_figure(style, base_w, base_h)
    colors = scatter_groups(ax, coords, groups, style)

    label_a, label_b = _axis_labels(explained, coords.shape[1], coords, title)
    if metric_label:
        place_title(ax, style, f"{title}（{metric_label}）")
    else:
        place_title(ax, style, title)

    # biplot 箭头需要按坐标范围缩放，否则会飞出画面
    extra = []
    if env_arrows is not None and len(env_arrows) > 0:
        item = _overlay_arrows(ax, coords, env_arrows, style, color="#7A4E9E",
                               label="环境/分组因子")
        if item:
            extra.append(item)
    if feature_arrows is not None and len(feature_arrows) > 0:
        item = _overlay_arrows(ax, coords, feature_arrows, style, color="#3C8D6E",
                               label="主要特征", top_n=8)
        if item:
            extra.append(item)

    style_axes(ax, style)
    _apply_axis_labels(ax, style, label_a, label_b)
    if stat_text:
        place_annotation(ax, stat_text, style)
    resolve_legend(style, ax, extra_handles=extra, title="Group")
    fig.tight_layout()
    return save(fig, out_path, style, caption, title)


def _apply_axis_labels(ax, style: Style, xlabel: str, ylabel: str):
    from .base import place_axis_labels

    place_axis_labels(ax, style, xlabel, ylabel)


def _overlay_arrows(ax, coords: pd.DataFrame, arrows: pd.DataFrame, style: Style,
                    color: str, label: str, top_n: int = 12, scale_to: float = 0.72,
                    legend: bool = True):
    """把箭头按当前散点的坐标范围缩放到合适长度。

    箭头数据通常是相关系数（量纲与排序坐标不同），因此先在"轴范围归一化空间"里
    计算各箭头的长度，再把最长的那支缩放到画幅的 scale_to 倍，最后换算回数据单位。

    返回 (handle, label) 供调用方合并进统一图例；不需要时返回 None。
    """
    if arrows is None or len(arrows) == 0 or arrows.shape[1] < 2:
        return None

    cols = list(arrows.columns[:2])
    x = coords[coords.columns[0]].to_numpy(dtype=float)
    y = coords[coords.columns[1]].to_numpy(dtype=float)
    xr = float(np.nanmax(x) - np.nanmin(x)) or 1.0
    yr = float(np.nanmax(y) - np.nanmin(y)) or 1.0

    sub = arrows[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.empty:
        return None

    # 归一化空间下的向量长度，用于挑最重要的箭头并按比例缩放
    norm = np.sqrt((sub.iloc[:, 0] / xr) ** 2 + (sub.iloc[:, 1] / yr) ** 2)
    if not np.isfinite(norm).any() or float(norm.max()) <= 0:
        return None
    order = norm.sort_values(ascending=False).index
    sub = sub.loc[order].head(top_n)

    scale = (scale_to / 2.0) / float(norm.max())
    scaled = sub * scale

    for k, (name, row) in enumerate(scaled.iterrows()):
        dx, dy = float(row.iloc[0]), float(row.iloc[1])
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.4, alpha=0.9),
                    zorder=5, annotation_clip=False)
        # 箭头标签上下交替偏移，减少多个箭头方向相近时的文字重叠
        ox, oy = (5, 4) if k % 2 == 0 else (5, -9)
        ax.annotate(str(name)[:22], (dx, dy),
                    fontsize=style.font_size("annotation") * 0.9, color=color, alpha=0.95,
                    xytext=(ox, oy), textcoords="offset points", zorder=6)

    if not legend:
        return None
    from matplotlib.lines import Line2D

    return (Line2D([0], [0], color=color, lw=1.4, marker=">", markersize=5), label)


# --------------------------------------------------------------------------
# 各排序方法
# --------------------------------------------------------------------------
def plot_pcoa(coords: pd.DataFrame, groups: pd.Series, explained, metric_label: str,
              out_path: str | Path, style: Style, permanova_res: dict | None = None) -> dict:
    stat = ""
    if permanova_res and np.isfinite(permanova_res.get("p", np.nan)):
        stat = (f"PERMANOVA  R² = {permanova_res['R2']:.3f}   F = {permanova_res['F']:.2f}   "
                f"{_p_label(permanova_res['p'])} {_stars(permanova_res['p'])}")
    caption = "主坐标分析（PCoA）散点图，椭圆为各分组 95% 置信区间。"
    if stat:
        caption += f" {stat.replace('  ', '，')}。"
    return _draw_ordination(coords, groups, style, "PCoA", out_path, caption,
                            explained=explained, metric_label=metric_label, stat_text=stat)


def plot_nmds(coords: pd.DataFrame, groups: pd.Series, stress: float, metric_label: str,
              out_path: str | Path, style: Style) -> dict:
    stat = f"stress = {stress:.3f}" if np.isfinite(stress) else ""
    return _draw_ordination(
        coords, groups, style, "NMDS", out_path,
        f"非度量多维标度（NMDS）排序图，stress = {stress:.3f}（< 0.2 表示排序可信）。"
        if np.isfinite(stress) else "非度量多维标度（NMDS）排序图。",
        metric_label=metric_label, stat_text=stat)


def plot_pca(table: pd.DataFrame, groups: pd.Series, out_path: str | Path,
             style: Style, transform: str = "clr", n_features_arrows: int = 10) -> dict:
    from ..core.ordination import pca_ordination

    res = pca_ordination(table, n_components=4, transform=transform)
    coords, explained = res["coords"], res["explained"]
    loadings = res.get("loadings")
    arrows = None
    if loadings is not None and not loadings.empty:
        arrows = loadings.iloc[:, :2].copy()

    stat = f"PC1 解释度 {explained[0] * 100:.1f}%   PC2 解释度 {explained[1] * 100:.1f}%"
    caption = (f"主成分分析（PCA）散点图，数据经 {transform.upper()} 变换。"
               "绿色箭头为主要特征（变量载荷）的投影方向，箭头越长表示该特征对该轴贡献越大。")
    return _draw_ordination(coords, groups, style, "PCA", out_path, caption,
                            explained=explained, stat_text=stat,
                            feature_arrows=arrows)


def plot_rda(table: pd.DataFrame, env: pd.DataFrame, groups: pd.Series,
             out_path: str | Path, style: Style) -> dict:
    from ..core.ordination import rda

    res = rda(table, env, n_components=4)
    pct = (res["constrained_inertia"] / max(res["total_inertia"], 1e-12)) * 100
    stat = f"约束惯量占比 {pct:.1f}%"
    caption = ("冗余分析（RDA）三序图。紫色箭头为约束变量（环境/分组因子），"
               "绿色箭头为主要特征；箭头方向反映其与排序轴的相关性。")
    return _draw_ordination(res["coords"], groups, style, "RDA", out_path, caption,
                            explained=res["explained"], stat_text=stat,
                            env_arrows=res.get("env_biplot"),
                            feature_arrows=res.get("feature_biplot"))


def plot_dbrda(table: pd.DataFrame, env: pd.DataFrame, groups: pd.Series,
               out_path: str | Path, style: Style, metric: str = "braycurtis") -> dict:
    from ..core.ordination import dbrda

    res = dbrda(table, env, metric=metric, n_components=4)
    pct = (res["constrained_inertia"] / max(res["total_inertia"], 1e-12)) * 100
    stat = f"约束惯量占比 {pct:.1f}%"
    return _draw_ordination(
        res["coords"], groups, style, "db-RDA", out_path,
        f"基于距离的冗余分析（db-RDA），距离指标为 {metric}。紫色箭头为约束变量。",
        explained=res["explained"], stat_text=stat, env_arrows=res.get("env_biplot"))


def plot_cca(table: pd.DataFrame, env: pd.DataFrame, groups: pd.Series,
             out_path: str | Path, style: Style) -> dict:
    from ..core.ordination import cca

    res = cca(table, env, n_components=4)
    pct = (res["constrained_inertia"] / max(res["total_inertia"], 1e-12)) * 100
    stat = f"约束惯量占比 {pct:.1f}%"
    caption = ("典范对应分析（CCA）三序图，基于卡方距离。紫色箭头为约束变量，"
               "绿色箭头为主要特征。")
    return _draw_ordination(res["coords"], groups, style, "CCA", out_path, caption,
                            explained=res["explained"], stat_text=stat,
                            env_arrows=res.get("env_biplot"),
                            feature_arrows=res.get("feature_biplot"))


def plot_cap(table: pd.DataFrame, groups: pd.Series, out_path: str | Path,
             style: Style, metric: str = "braycurtis") -> dict:
    from ..core.ordination import cap

    res = cap(table, groups, metric=metric, n_components=3)
    coords = res["coords"]
    if coords.shape[1] < 2:
        # 只有一个判别轴（例如恰好两组且 m=1）时，补一个零轴以便二维展示
        coords = coords.copy()
        coords["CAP2"] = 0.0
    stat = f"留一交叉验证准确率 = {res['accuracy']:.1%}   (使用前 {res['m']} 个 PCoA 轴)"
    return _draw_ordination(
        coords, groups, style, "CAP", out_path,
        f"典范分析主坐标（CAP），距离指标 {metric}。"
        f"类别可分性由留一交叉验证准确率评估：{res['accuracy']:.1%}。",
        explained=res.get("explained"), stat_text=stat)


def plot_tsne(table: pd.DataFrame, groups: pd.Series, out_path: str | Path,
              style: Style, transform: str = "clr", perplexity: float | None = None) -> dict:
    from ..core.ordination import tsne_embedding

    coords = tsne_embedding(table, transform=transform, perplexity=perplexity)
    return _draw_ordination(
        coords, groups, style, "t-SNE", out_path,
        f"t-SNE 降维散点图（基于 {transform.upper()} 变换，{coords.shape[1]} 维嵌入）。"
        "t-SNE 保留局部结构，簇间距离不宜直接解读为差异大小。",
        explained=None, stat_text=f"perplexity = {perplexity if perplexity else 'auto'}")


def plot_umap(table: pd.DataFrame, groups: pd.Series, out_path: str | Path,
              style: Style, transform: str = "clr", n_neighbors: int = 15,
              min_dist: float = 0.1) -> dict:
    from ..core.ordination import umap_embedding

    coords = umap_embedding(table, transform=transform, n_neighbors=n_neighbors, min_dist=min_dist)
    return _draw_ordination(
        coords, groups, style, "UMAP", out_path,
        f"UMAP 降维散点图（基于 {transform.upper()} 变换）。"
        "UMAP 同时保留局部与部分全局结构，可用于观察样本聚类。",
        stat_text=f"n_neighbors = {n_neighbors}   min_dist = {min_dist}")


def plot_plsda(table: pd.DataFrame, groups: pd.Series, out_path: str | Path,
               style: Style, transform: str = "clr", n_components: int = 2) -> dict:
    from ..core.ordination import plsda

    res = plsda(table, groups, n_components=max(2, n_components), transform=transform)
    coords, explained = res["coords"], res["explained"]
    stat = f"交叉验证准确率 = {res['accuracy']:.1%}   R²Y = {res.get('r2y', float('nan')):.3f}"
    return _draw_ordination(
        coords, groups, style, "PLS-DA", out_path,
        "偏最小二乘判别分析（PLS-DA）得分图。"
        f"交叉验证准确率 {res['accuracy']:.1%}，可用于评估分组是否可判别。",
        explained=explained, stat_text=stat)


def plot_plsda_vip(vip: pd.DataFrame, out_path: str | Path, style: Style,
                   top_n: int = 20, threshold: float = 1.0) -> dict:
    df = vip.reindex(vip["VIP"].sort_values(ascending=False).index).head(top_n).iloc[::-1]
    fig, ax = new_figure(style, 7.4, max(3.6, 0.34 * len(df) + 1.8))
    colors = [style.single_color(0) if v >= threshold else "#BBBBBB" for v in df["VIP"]]
    ax.barh([str(f)[:46] for f in df["feature"]], df["VIP"], color=colors,
            edgecolor="white" if style.flag("element.bar_edge", True) else "none", linewidth=0.5)
    ax.axvline(threshold, color="#C44E52", ls="--", lw=1.2)
    style_axes(ax, style)
    ax.set_xlabel("VIP 值")
    place_title(ax, style, f"PLS-DA 变量投影重要性 Top {len(df)}")
    ax.tick_params(axis="y", labelsize=style.font_size("tick_label") * 0.9)
    fig.tight_layout()
    return save(fig, out_path, style,
                f"VIP > {threshold} 的特征（着色）通常被认为对分组判别有实质性贡献。",
                "PLS-DA VIP 图")


# --------------------------------------------------------------------------
# 辅助图
# --------------------------------------------------------------------------
def plot_scree(explained, out_path: str | Path, style: Style, label: str = "PCoA") -> dict:
    exp = np.asarray(explained, dtype=float)
    fig, ax = new_figure(style, 6.4, 4.4)
    x = np.arange(1, exp.size + 1)
    ax.bar(x, exp * 100, color=style.single_color(0), width=0.62,
           edgecolor="white" if style.flag("element.bar_edge", True) else "none")
    ax2 = ax.twinx()
    ax2.plot(x, np.cumsum(exp) * 100, color="#C44E52", marker="o", ms=4.5, lw=1.8, label="累计")
    ax2.set_ylabel("累计解释度 (%)", color="#C44E52")
    ax2.tick_params(axis="y", colors="#C44E52")
    ax2.set_ylim(0, 105)
    ax2.grid(False)

    style_axes(ax, style)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}{i}" for i in x], rotation=0)
    ax.set_xlabel("排序轴")
    ax.set_ylabel("解释度 (%)")
    place_title(ax, style, f"{label} 各轴解释度（碎石图）")
    fig.tight_layout()
    return save(fig, out_path, style,
                "柱状图为各排序轴的解释度，红色折线为累计解释度，用于判断保留多少个轴合适。",
                f"{label} 碎石图")


def plot_distance_boxplot(long_df: pd.DataFrame, metric_label: str, out_path: str | Path,
                          style: Style, test_res: dict | None = None) -> dict:
    order = sorted(long_df["pair"].unique(), key=lambda s: (not s.startswith("Within"), s))
    fig, ax = new_figure(style, max(6.0, 0.62 * long_df["pair"].nunique()), 4.6)
    sns.boxplot(data=long_df, x="pair", y="distance", order=order, ax=ax, hue="pair",
                palette=style.palette_colors(), width=style.num("element.box_width", 0.62),
                fliersize=0, showfliers=False, legend=False)
    sns.stripplot(data=long_df, x="pair", y="distance", order=order, ax=ax,
                  color="#333333", size=2.6, alpha=0.5,
                  jitter=style.num("element.jitter", 0.2))
    style_axes(ax, style)
    ax.set_xlabel("")
    ax.set_ylabel(f"{metric_label} 距离")
    place_title(ax, style, "组内 / 组间距离分布")
    ax.tick_params(axis="x", rotation=style.num("element.axis_rotate", 45))

    caption = "比较组内与组间的样本距离分布，组间距离显著大于组内距离说明分组结构明显。"
    if test_res and np.isfinite(test_res.get("p", np.nan)):
        rtxt = f"ANOSIM R = {test_res['R']:.3f}" if "R" in test_res else ""
        place_annotation(ax, f"{rtxt}  {_p_label(test_res['p'])} {_stars(test_res['p'])}", style)
        caption += f" 统计检验：{test_res.get('test', '')}，{_p_label(test_res['p'])}。"
    fig.tight_layout()
    return save(fig, out_path, style, caption, "组内/组间距离分布")


def plot_permdisp(res: dict, out_path: str | Path, style: Style, metric_label: str = "") -> dict:
    """多元离散度齐性检验：各组样本到组质心的距离分布。"""
    dist = res.get("distances")
    if dist is None or dist.empty:
        raise ValueError("PERMDISP 结果缺少距离数据。")
    fig, ax = new_figure(style, 6.6, 4.6)
    order = sorted(dist["Group"].astype(str).unique())
    colors = style.group_colors(order)
    sns.boxplot(data=dist, x="Group", y="distance", order=order, ax=ax, hue="Group",
                palette=colors, width=style.num("element.box_width", 0.6),
                fliersize=0, showfliers=False, legend=False)
    sns.stripplot(data=dist, x="Group", y="distance", order=order, ax=ax,
                  color="#2F2F2F", size=3, alpha=0.6, jitter=style.num("element.jitter", 0.18))
    style_axes(ax, style)
    ax.set_xlabel("")
    ax.set_ylabel(f"到组质心的{(' ' + metric_label) if metric_label else ''}距离")
    place_title(ax, style, "多元离散度齐性检验（PERMDISP）")
    place_annotation(ax, f"F = {res['F']:.3f}   {_p_label(res['p'])} {_stars(res['p'])}", style)
    fig.tight_layout()
    return save(fig, out_path, style,
                "检验各组内部离散程度是否相同。若显著，说明组间差异可能来自离散度不同"
                "（而非质心不同），解读 PERMANOVA 结果时需谨慎。",
                "PERMDISP 离散度检验")


def plot_envfit(envfit_df: pd.DataFrame, out_path: str | Path, style: Style) -> dict:
    if envfit_df is None or envfit_df.empty:
        raise ValueError("envfit 结果为空。")
    df = envfit_df.sort_values("r2", ascending=True)
    fig, ax = new_figure(style, 7.4, max(3.4, 0.36 * len(df) + 1.6))
    colors = [style.single_color(0) if p < 0.05 else "#BBBBBB" for p in df["p"]]
    ax.barh([str(v)[:40] for v in df["variable"]], df["r2"], color=colors,
            edgecolor="white" if style.flag("element.bar_edge", True) else "none", linewidth=0.5)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["r2"], i, f"  {_p_label(row['p'])}{_stars(row['p'])}",
                va="center", fontsize=style.font_size("annotation") * 0.9)
    style_axes(ax, style)
    ax.set_xlabel("R²（对排序结构的解释度）")
    place_title(ax, style, "环境 / 分组因子与排序结构的拟合（envfit）")
    fig.tight_layout()
    return save(fig, out_path, style,
                "每个变量对排序结构的解释度 R² 及置换检验 p 值；显著（着色）者才是排序格局的有效驱动因子。",
                "envfit 因子拟合")


def plot_mantel(mantel_df: pd.DataFrame, out_path: str | Path, style: Style,
                distance_label: str = "群落距离") -> dict:
    if mantel_df is None or mantel_df.empty:
        raise ValueError("Mantel 结果为空。")
    df = mantel_df.copy()
    df["absr"] = df["r"].abs()
    df = df.sort_values("absr", ascending=True)
    fig, ax = new_figure(style, 7.6, max(3.4, 0.36 * len(df) + 1.6))
    colors = [style.single_color(0) if p < 0.05 else "#BBBBBB" for p in df["p"]]
    ax.barh([str(v)[:40] for v in df["variable"]], df["r"], color=colors,
            edgecolor="white" if style.flag("element.bar_edge", True) else "none", linewidth=0.5)
    ax.axvline(0, color="#333333", lw=0.9)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["r"], i, f"  {_p_label(row['p'])}{_stars(row['p'])}",
                va="center", fontsize=style.font_size("annotation") * 0.9)
    style_axes(ax, style)
    ax.set_xlabel(f"与{distance_label}的 Mantel 相关系数 r")
    place_title(ax, style, "Mantel 检验")
    fig.tight_layout()
    return save(fig, out_path, style,
                f"比较{distance_label}矩阵与各环境/分组变量距离矩阵的相关性，"
                "显著（着色）表示该变量与群落差异显著相关。",
                "Mantel 检验")


def plot_procrustes(res: dict, groups: pd.Series, out_path: str | Path, style: Style,
                    label_a: str = "排序 1", label_b: str = "排序 2") -> dict:
    """Procrustes 一致性分析：两套排序结果经最优相似变换后的吻合程度。

    实心点为第一套排序坐标，空心点为对齐后的第二套坐标，连线长度反映单个样本的不一致程度。
    """
    c1 = res.get("coords1_fitted")
    c2 = res.get("coords2_aligned")
    if c1 is None or c2 is None or c1.empty or c2.empty:
        raise ValueError("Procrustes 结果缺少对齐后的坐标。")

    common = [s for s in c1.index if s in c2.index and s in groups.index]
    if len(common) < 3:
        raise ValueError("可用于 Procrustes 绘图的样本不足。")
    c1, c2 = c1.loc[common], c2.loc[common]
    groups = groups.loc[common].astype(str)
    colors = style.group_colors(groups)

    xa, ya = c1.columns[0], c1.columns[1]
    xb, yb = c2.columns[0], c2.columns[1]

    fig, ax = new_figure(style, 7.2, 6.0)

    # 同一样本在两套排序中的位置连线
    for s in common:
        ax.plot([c1.loc[s, xa], c2.loc[s, xb]], [c1.loc[s, ya], c2.loc[s, yb]],
                color="#AAAAAA", lw=0.9, alpha=0.75, zorder=1)

    size = style.num("element.point_size", 68)
    alpha = style.num("element.point_alpha", 0.92) * style.num("palette.alpha", 1.0)
    for g in sorted(groups.unique()):
        idx = [s for s in common if groups[s] == g]
        col = colors[g]
        ax.scatter(c1.loc[idx, xa], c1.loc[idx, ya], s=size, color=col, label=g,
                   edgecolor="white", linewidth=0.9, alpha=alpha, zorder=3)
        ax.scatter(c2.loc[idx, xb], c2.loc[idx, yb], s=size * 0.7, facecolor="none",
                   edgecolor=col, linewidth=1.3, alpha=0.95, zorder=3)

    if style.flag("element.zero_lines", True):
        ax.axhline(0, color="#BBBBBB", lw=0.8, zorder=0)
        ax.axvline(0, color="#BBBBBB", lw=0.8, zorder=0)

    style_axes(ax, style)
    place_axis_labels(ax, style, f"{label_a} 轴 1", f"{label_a} 轴 2")
    place_title(ax, style, "Procrustes 一致性分析")

    m2 = res.get("m2")
    r = res.get("r")
    p = res.get("p")
    if m2 is not None and np.isfinite(m2):
        place_annotation(ax, f"m² = {m2:.4f}   r = {r:.3f}   {_p_label(p)} {_stars(p)}", style)

    from matplotlib.lines import Line2D

    extra = [(
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="#555555", markersize=7, linestyle="none"),
        f"对齐后的 {label_b}",
    )]
    resolve_legend(style, ax, extra_handles=extra, title="Group")
    fig.tight_layout()
    return save(fig, out_path, style,
                "两套排序结果经平移 + 缩放 + 旋转最优对齐后的吻合程度。"
                "m² 越小、r 越接近 1 表示两套排序越一致；连线越长说明该样本在两套排序中位置差异越大。",
                "Procrustes 一致性分析")
