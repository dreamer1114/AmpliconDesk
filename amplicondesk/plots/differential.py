"""差异分析绘图：火山图、MA 图、差异条形图、森林图、单特征箱线图、LDA 效应量。"""

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
)

__all__ = [
    "plot_volcano", "plot_ma", "plot_diff_bar", "plot_forest",
    "plot_feature_box", "plot_lda", "plot_diff_heatmap",
]


def _prep_diff(diff: pd.DataFrame) -> pd.DataFrame:
    df = diff.copy()
    df["log2FC"] = pd.to_numeric(df["log2FC"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    p = pd.to_numeric(df["p"], errors="coerce").to_numpy(dtype=float)
    df["-log10P"] = -np.log10(np.clip(p, 1e-300, 1.0))
    df["-log10P"] = df["-log10P"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "FDR" not in df.columns:
        df["FDR"] = df["p"]
    return df


def _mean_columns(diff: pd.DataFrame) -> tuple[str, str] | None:
    for a, b in [("mean_rel_A", "mean_rel_B"), ("mean_norm_A", "mean_norm_B"),
                 ("mean_abund_A", "mean_abund_B")]:
        if a in diff.columns and b in diff.columns:
            return a, b
    return None


# --------------------------------------------------------------------------
def plot_volcano(diff: pd.DataFrame, out_path: str | Path, style: Style,
                 group_a: str, group_b: str, label_top: int | None = None) -> dict:
    df = _prep_diff(diff)
    fdr_cut = style.num("element.label_threshold", 0.05)
    fc_cut = style.num("element.volcano_fc_cut", 1.0)
    label_top = int(label_top if label_top is not None else style.num("element.label_top_n", 12))

    def classify(row):
        if not (row["FDR"] < fdr_cut):
            return "Not significant"
        if row["log2FC"] >= fc_cut:
            return f"Enriched in {group_a}"
        if row["log2FC"] <= -fc_cut:
            return f"Enriched in {group_b}"
        return "Not significant"

    df["class"] = df.apply(classify, axis=1)
    colors = {
        "Not significant": "#BFBFBF",
        f"Enriched in {group_a}": style.single_color(3),
        f"Enriched in {group_b}": style.single_color(0),
    }

    fig, ax = new_figure(style, 7.6, 5.8)
    alpha = 0.78 * style.num("palette.alpha", 1.0)
    for cls, sub in df.groupby("class"):
        ax.scatter(sub["log2FC"], sub["-log10P"], s=26, alpha=alpha,
                   color=colors.get(cls, "#999999"), label=f"{cls} (n={len(sub)})",
                   edgecolor="white" if style.flag("element.point_edge", True) else "none",
                   linewidth=0.4)
    if style.flag("element.zero_lines", True):
        ax.axhline(-np.log10(fdr_cut), color="#666666", ls="--", lw=1.0)
        ax.axvline(fc_cut, color="#666666", ls="--", lw=1.0)
        ax.axvline(-fc_cut, color="#666666", ls="--", lw=1.0)

    cand = df[df["class"] != "Not significant"].sort_values("p")
    if label_top > 0:
        x_span = float(df["log2FC"].max() - df["log2FC"].min()) or 1.0
        y_span = float(df["-log10P"].max() - df["-log10P"].min()) or 1.0
        min_dx, min_dy = 0.11 * x_span, 0.05 * y_span
        placed: list[tuple[float, float]] = []
        for _, row in cand.iterrows():
            if len(placed) >= label_top:
                break
            px, py = float(row["log2FC"]), float(row["-log10P"])
            if all(abs(px - qx) > min_dx or abs(py - qy) > min_dy for qx, qy in placed):
                placed.append((px, py))
                ax.annotate(str(row["feature"])[:28], (px, py),
                            fontsize=style.font_size("annotation") * 0.85,
                            color=style.font_color("annotation"), alpha=0.95,
                            xytext=(5, 4), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.55))

    y_top = float(df["-log10P"].max()) or 1.0
    ax.set_ylim(top=y_top * 1.10)
    style_axes(ax, style)
    ax.set_xlabel(f"log2 Fold Change  ({group_a} / {group_b})",
                  fontsize=style.font_size("axis_label"), color=style.font_color("axis_label"))
    ax.set_ylabel("-log10 (p value)", fontsize=style.font_size("axis_label"),
                  color=style.font_color("axis_label"))
    place_title(ax, style, f"差异丰度火山图：{group_a} vs {group_b}")
    resolve_legend(style, ax, preferred="outside bottom")
    fig.tight_layout()
    return save(fig, out_path, style,
                f"火山图展示各特征的差异倍数与显著性。虚线为 log2FC = ±{fc_cut} 与 "
                f"p = {fdr_cut} 阈值；{group_a} 富集在一侧，{group_b} 富集在另一侧。",
                f"火山图 {group_a} vs {group_b}")


def plot_ma(diff: pd.DataFrame, out_path: str | Path, style: Style,
            group_a: str, group_b: str) -> dict:
    """MA 图：平均丰度 vs 对数倍数变化。"""
    df = _prep_diff(diff)
    means = _mean_columns(diff)
    if means:
        ma, mb = means
        a = pd.to_numeric(df[ma], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        b = pd.to_numeric(df[mb], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mean_ab = (a + b) / 2.0
    else:
        mean_ab = np.ones(len(df))
    fdr_cut = style.num("element.label_threshold", 0.05)

    sig = df["FDR"] < fdr_cut
    fig, ax = new_figure(style, 7.4, 5.4)
    ax.scatter(mean_ab[~sig], df["log2FC"].to_numpy()[~sig], s=22, color="#BFBFBF",
               alpha=0.75, edgecolor="none", label=f"Not significant (n={int((~sig).sum())})")
    ax.scatter(mean_ab[sig], df["log2FC"].to_numpy()[sig], s=30, color=style.single_color(3),
               alpha=0.9, edgecolor="white",
               linewidth=0.35 if style.flag("element.point_edge", True) else 0,
               label=f"Significant (n={int(sig.sum())})")
    if style.flag("element.zero_lines", True):
        ax.axhline(0, color="#333333", lw=1.0)
        for cut in (style.num("element.volcano_fc_cut", 1.0),):
            ax.axhline(cut, color="#888888", ls="--", lw=0.9)
            ax.axhline(-cut, color="#888888", ls="--", lw=0.9)

    try:
        ax.set_xscale("log")
    except Exception:
        pass
    style_axes(ax, style)
    ax.set_xlabel("平均丰度", fontsize=style.font_size("axis_label"))
    ax.set_ylabel(f"log2 Fold Change  ({group_a} / {group_b})",
                  fontsize=style.font_size("axis_label"))
    place_title(ax, style, f"MA 图：{group_a} vs {group_b}")
    resolve_legend(style, ax)
    fig.tight_layout()
    return save(fig, out_path, style,
                "MA 图用平均丰度作横轴，可同时判断差异倍数是否与丰度水平相关"
                "（低丰度特征往往倍数变化更大但更不稳定）。",
                f"MA 图 {group_a} vs {group_b}")


def plot_diff_bar(diff: pd.DataFrame, out_path: str | Path, style: Style,
                  group_a: str, group_b: str, top_n: int = 20) -> dict:
    df = _prep_diff(diff)
    fdr_cut = style.num("element.label_threshold", 0.05)
    sig = df[df["FDR"] < fdr_cut]
    if sig.empty:
        sig = df.nsmallest(min(top_n, len(df)), "p")
    sub = sig.reindex(sig["log2FC"].abs().sort_values(ascending=False).index).head(top_n).iloc[::-1]
    if sub.empty:
        raise ValueError("没有可展示的差异特征。")

    colors = [style.single_color(3) if v > 0 else style.single_color(0) for v in sub["log2FC"]]
    fig, ax = new_figure(style, 7.8, max(3.6, 0.34 * len(sub) + 1.8))
    ax.barh([str(f)[:46] for f in sub["feature"]], sub["log2FC"], color=colors,
            edgecolor="white" if style.flag("element.bar_edge", True) else "none", linewidth=0.5)
    ax.axvline(0, color="#333333", lw=0.9)
    style_axes(ax, style)
    ax.set_xlabel(f"log2 Fold Change  ({group_a} / {group_b})",
                  fontsize=style.font_size("axis_label"))
    place_title(ax, style, f"差异丰度 Top {len(sub)} 特征")
    ax.tick_params(axis="y", labelsize=style.font_size("tick_label") * 0.9)
    fig.tight_layout()
    return save(fig, out_path, style,
                f"横向条形图展示差异最显著特征的 log2 倍数变化；正值偏向 {group_a}，负值偏向 {group_b}。",
                f"差异特征 log2FC {group_a} vs {group_b}")


def plot_forest(diff: pd.DataFrame, out_path: str | Path, style: Style,
                group_a: str, group_b: str, top_n: int = 20) -> dict:
    """森林图：log2FC 点估计 + 置信区间（有 lfcSE 时绘制误差棒）。"""
    df = _prep_diff(diff)
    fdr_cut = style.num("element.label_threshold", 0.05)
    sig = df[df["FDR"] < fdr_cut]
    if sig.empty:
        sig = df.nsmallest(min(top_n, len(df)), "p")
    sub = sig.reindex(sig["log2FC"].abs().sort_values(ascending=False).index).head(top_n).iloc[::-1]
    if sub.empty:
        raise ValueError("没有可展示的差异特征。")

    has_se = "lfcSE" in sub.columns and pd.to_numeric(sub["lfcSE"], errors="coerce").notna().any()
    y = np.arange(len(sub))
    fig, ax = new_figure(style, 7.8, max(3.8, 0.36 * len(sub) + 1.9))

    if has_se:
        se = pd.to_numeric(sub["lfcSE"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        lower = sub["log2FC"].to_numpy(dtype=float) - 1.96 * se
        upper = sub["log2FC"].to_numpy(dtype=float) + 1.96 * se
        ax.hlines(y, lower, upper, color="#888888", lw=1.4, zorder=2)
        ax.scatter(sub["log2FC"], y, s=46, color=style.single_color(0), zorder=3,
                   edgecolor="white", linewidth=0.6)
    else:
        colors = [style.single_color(3) if v > 0 else style.single_color(0)
                  for v in sub["log2FC"]]
        ax.barh(y, sub["log2FC"], color=colors,
                edgecolor="white" if style.flag("element.bar_edge", True) else "none", height=0.62)

    ax.axvline(0, color="#333333", lw=1.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([str(f)[:44] for f in sub["feature"]],
                       fontsize=style.font_size("tick_label") * 0.9)
    style_axes(ax, style)
    ax.set_xlabel(f"log2 Fold Change  ({group_a} / {group_b})",
                  fontsize=style.font_size("axis_label"))
    place_title(ax, style, f"差异特征森林图：{group_a} vs {group_b}")
    fig.tight_layout()
    caption = ("森林图展示各差异特征的效应量点估计"
               + ("与 95% 置信区间（误差棒）" if has_se else "")
               + f"；中线为 0（无差异），点位于右侧表示在 {group_a} 中更高。")
    return save(fig, out_path, style, caption, f"森林图 {group_a} vs {group_b}")


def plot_feature_box(table: pd.DataFrame, groups: pd.Series, features: list[str],
                     out_path: str | Path, style: Style,
                     label_a: str = "", label_b: str = "", top_n: int = 9) -> dict:
    """指定特征在两组间的丰度箱线图（差异分析最常用的附图）。"""
    feats = [f for f in features if f in table.index][:top_n]
    if not feats:
        raise ValueError("没有可绘制的特征。")

    rel = table.div(table.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    merged = rel.loc[feats].T.join(groups.rename("Group"), how="inner")
    merged["Group"] = merged["Group"].astype(str)
    colors = style.group_colors(merged["Group"])
    order = sorted(merged["Group"].unique())

    fig, axes, ncol = panel_grid(style, len(feats), base_w=3.5, base_h=3.1)
    for k, feat in enumerate(feats):
        ax = axes[k // ncol][k % ncol]
        sns.boxplot(data=merged, x="Group", y=feat, order=order, ax=ax, hue="Group",
                    palette=colors, width=style.num("element.box_width", 0.6),
                    fliersize=0, showfliers=False, legend=False)
        sns.stripplot(data=merged, x="Group", y=feat, order=order, ax=ax,
                      color="#2F2F2F", size=3, alpha=0.65,
                      jitter=style.num("element.jitter", 0.18))
        style_axes(ax, style)
        ax.set_xlabel("")
        ax.set_ylabel("相对丰度", fontsize=style.font_size("axis_label") * 0.9)
        place_title(ax, style, str(feat)[:32], allow_override=False, scale=0.92)
        if len(order) > 4:
            ax.tick_params(axis="x", rotation=style.num("element.axis_rotate", 45))

    for k in range(len(feats), axes.shape[0] * axes.shape[1]):
        axes[k // ncol][k % ncol].axis("off")
    if style.show_text("suptitle"):
        from .base import place_suptitle

        place_suptitle(fig, style, f"差异特征丰度分布（{label_a} vs {label_b}）"
                       if label_a else "差异特征丰度分布")
    fig.tight_layout()
    return save(fig, out_path, style,
                "每个面板为一个差异显著特征，箱线图展示其相对丰度在两（多）组间的分布。",
                "差异特征丰度箱线图")


def plot_lda(lda: pd.DataFrame, out_path: str | Path, style: Style,
             group_a: str, group_b: str, top_n: int = 20) -> dict:
    if lda is None or lda.empty:
        raise ValueError("LDA 结果为空。")
    df = lda.reindex(lda["LDA_score"].sort_values(ascending=False).index).head(top_n).iloc[::-1]
    colors = [style.single_color(3) if g == group_a else style.single_color(0)
              for g in df["group_enriched"]]

    fig, ax = new_figure(style, 7.8, max(3.6, 0.36 * len(df) + 1.8))
    ax.barh([str(f)[:46] for f in df["feature"]], df["LDA_score"], color=colors,
            edgecolor="white" if style.flag("element.bar_edge", True) else "none", linewidth=0.5)
    style_axes(ax, style)
    ax.set_xlabel("LDA score (log10)")
    place_title(ax, style, "LEfSe 风格 LDA 效应量")

    from matplotlib.patches import Patch

    handles = [Patch(facecolor=style.single_color(3)), Patch(facecolor=style.single_color(0))]
    ax.legend(handles, [group_a, group_b], title="Enriched in", loc="lower right",
              frameon=style.flag("legend.frame", False),
              fontsize=style.font_size("legend_text"))
    ax.tick_params(axis="y", labelsize=style.font_size("tick_label") * 0.9)
    fig.tight_layout()
    return save(fig, out_path, style,
                "基于 CLR 转换数据的线性判别分析效应量，数值越大表示该特征对区分两组贡献越大。",
                "LDA 效应量")


def plot_diff_heatmap(table: pd.DataFrame, groups: pd.Series, diff: pd.DataFrame,
                      out_path: str | Path, style: Style,
                      group_a: str, group_b: str, top_n: int = 30) -> dict:
    """差异显著特征在样本间的丰度热图（按分组排序）。"""
    from .composition import _to_relative

    fdr_cut = style.num("element.label_threshold", 0.05)
    sig = diff[diff["FDR"] < fdr_cut]
    if sig.empty:
        sig = diff.nsmallest(min(top_n, len(diff)), "p")
    feats = [f for f in sig.head(top_n)["feature"].tolist() if f in table.index]
    if not feats:
        raise ValueError("没有可用于热图的差异特征。")

    sub = table.loc[feats]
    sub = sub[[s for s in sub.columns if groups.get(s) in (group_a, group_b)]]
    if sub.shape[1] < 2:
        raise ValueError("可用于热图的样本不足。")

    meta = groups.loc[sub.columns].astype(str)
    order = sorted(sub.columns, key=lambda s: (str(meta[s]), s))
    sub, meta = sub[order], meta[order]

    rel = _to_relative(sub)
    z = rel.sub(rel.mean(axis=1), axis=0).div(rel.std(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    width = max(7.0, 0.32 * z.shape[1] + 3.4)
    height = max(5.0, 0.24 * z.shape[0] + 2.0)
    colors = style.group_colors(meta)

    from matplotlib.patches import Rectangle
    import matplotlib.colors as mcolors

    fig, ax = new_figure(style, width, height)
    sns.heatmap(z, cmap=style.cmap("diverging"), center=0, ax=ax,
                cbar_kws={"label": "Z-score"}, yticklabels=True, xticklabels=True,
                annot=style.flag("element.heatmap_values", False))
    for i, s in enumerate(z.columns):
        ax.add_patch(Rectangle((i, 0), 1, -0.35, facecolor=colors.get(str(meta[s]), "#999999"),
                               edgecolor="none", clip_on=False))
    ax.set_xticklabels(z.columns, rotation=90, fontsize=style.font_size("tick_label") * 0.76)
    ax.set_yticklabels([str(t)[:50] for t in z.index], fontsize=style.font_size("tick_label") * 0.8)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    place_title(ax, style, f"差异特征丰度热图（{group_a} vs {group_b}）")
    fig.tight_layout()
    return save(fig, out_path, style,
                "差异最显著特征的 Z-score 热图，列为样本（按分组排序，顶部色条指示分组），"
                "用于观察这些特征是否能把两组样本分开。",
                f"差异特征热图 {group_a} vs {group_b}")
