"""高级图形模块：小提琴 / 云雨 / 密度 / 蜂群 / 棒棒糖 / 气泡 / 桑基 / 维诺树状图 / 流向图。

设计要点：
  * 全部使用纯 matplotlib + numpy + scipy 实现，不引入 plotly、matplotlib-sankey 等新依赖；
    贝塞尔连边用 ``matplotlib.path.Path`` + ``PathPatch``，多边形用 ``Polygon``。
  * 所有文字统一走 ``place_title`` / ``place_axis_labels`` / ``place_annotation``，
    颜色统一走 ``style.group_colors`` / ``style.single_color`` / ``style.cmap``，
    尺寸统一走 ``style.num("element.*")``，保证界面上的字体 / 配色 / 元素设置对每张图都生效。
  * 退化输入一律抛中文 ``ValueError``，便于 Web 端直接把消息展示给用户。
  * 文件末尾 ``__main__`` 自检会真的把 9 张 PNG 写到 ``preview/`` 目录，
    并验证维诺面积比例误差、桑基流量守恒、蜂群点间距等硬指标。

输入表约定：**行 = 特征（类群 / 基因），列 = 样本**；``groups`` 为 index=样本名、值为分组的 Series。
"""

from __future__ import annotations

import math
import os as _os
import sys as _sys
import string as _string
from pathlib import Path
from typing import Any, Sequence

# 允许直接以 `python amplicondesk/plots/advanced.py` 运行本文件做自检：
# 该写法下模块的 __package__ 为空，相对导入（from ..style import ...）会失败，
# 因此这里先把项目根目录放入 sys.path，再把自身按包内路径重新导入一次。
if __package__ in (None, ""):  # pragma: no cover - 仅直接执行脚本时触发
    _root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from amplicondesk.plots import advanced as _self_module

    _self_module._selftest()  # type: ignore[attr-defined]
    raise SystemExit(0)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, PathPatch, Polygon
from matplotlib.path import Path as MplPath
from scipy.spatial import Voronoi

from ..style import Style, text_style_kwargs
from .base import (
    new_figure, panel_grid, place_annotation, place_axis_labels, place_suptitle,
    place_title, resolve_legend, save, style_axes,
)

__all__ = [
    "plot_violin", "plot_raincloud", "plot_density", "plot_beeswarm",
    "plot_lollipop", "plot_bubble", "plot_sankey",
    "plot_voronoi_treemap", "plot_stacked_flow",
]


# ==========================================================================
# 基础工具
# ==========================================================================
def _to_relative(table: pd.DataFrame) -> pd.DataFrame:
    """把丰度表转为相对丰度（列 = 样本，每列求和为 1）。"""
    ab = table.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    totals = ab.sum(axis=0)
    if float(totals.max()) > 1.5:
        ab = ab.div(totals.replace(0, np.nan), axis=1).fillna(0.0)
    return ab


def _paired(rel_table: pd.DataFrame, groups: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """取出既有丰度又有分组的样本，返回 (样本×特征 的相对丰度矩阵, 分组)。

    Raises:
        ValueError: 没有任何样本同时出现在丰度表列名与 groups 索引中。
    """
    if not isinstance(groups, pd.Series):
        groups = pd.Series(groups)
    cols = [c for c in rel_table.columns if c in set(groups.index)]
    if not cols:
        raise ValueError("丰度表的列名与分组表的索引没有任何交集，无法配对样本。")
    rel = rel_table[cols]
    meta = groups.loc[cols].astype(str)
    return rel.T, meta


def _top_features(rel: pd.DataFrame, groups: pd.Series, top_n: int) -> tuple[pd.DataFrame, list[str]]:
    """按组均值挑选优势特征，返回 (样本×特征 矩阵, 特征名列表)。

    Raises:
        ValueError: 样本数不足 2、分组数不足 2、或特征数为 0。
    """
    mat, meta = _paired(rel, groups)
    if mat.shape[1] == 0:
        raise ValueError("所选分类级下没有可用的特征。")
    if len(meta) < 2:
        raise ValueError("可用样本不足 2 个，无法比较分布。")
    if meta.nunique() < 2:
        raise ValueError("至少需要 2 个分组才能比较特征分布，当前只有 1 组。")

    mean_ab = mat.mean(axis=0).sort_values(ascending=False)
    feats = [str(f) for f in mean_ab.index[:max(1, int(top_n))]]
    return mat, feats


def _group_order(meta: pd.Series) -> list[str]:
    """分组展示顺序（按名称排序，保证多次运行一致）。"""
    return sorted(meta.astype(str).unique().tolist())


def _finite_or_raise(fig, kind: str) -> None:
    """检查图上所有数据元素的坐标都是有限值，否则抛中文 ValueError。"""
    bad: list[str] = []
    for ax in fig.axes:
        for line in [*ax.lines, *getattr(ax, "collections", [])]:
            try:
                xs = np.asarray(line.get_xdata() if hasattr(line, "get_xdata")
                                else line.get_offsets(), dtype=float)
            except Exception:
                continue
            if xs.size and not np.all(np.isfinite(xs)):
                bad.append(type(line).__name__)
    if bad:
        raise ValueError(f"{kind}：图中出现 NaN / inf 坐标（涉及 {sorted(set(bad))}），已中止导出。")


def _axis_text(style: Style, default: str) -> str:
    """坐标轴标签的默认文字；若用户在图层面板里填了自定义内容，则用用户的。"""
    return style.text("font.axis_label.text", default)


def _point_radius(style: Style, ax) -> float:
    """把 ``element.point_size``（平方磅）换算成轴归一化坐标下的点半径。

    Args:
        style: 样式对象，读取 ``element.point_size``。
        ax: 目标轴，用来取得绘图区的像素宽高。

    Returns:
        点半径，单位是"轴宽/轴高归一化"后的相对长度。
    """
    size = max(1.0, style.num("element.point_size", 68))
    try:
        bbox = ax.get_window_extent()
        w = max(1.0, float(bbox.width))
        h = max(1.0, float(bbox.height))
    except Exception:
        w, h = 320.0, 300.0
    return 0.5 * math.sqrt(size) / w, 0.5 * math.sqrt(size) / h


# ==========================================================================
# 核密度估计
# ==========================================================================
def _kde(values: np.ndarray, grid: np.ndarray, bw: float | None = None) -> np.ndarray:
    """一维高斯核密度估计（Silverman 经验带宽，纯 numpy 实现）。

    Args:
        values: 一维样本值。
        grid: 求值网格点。
        bw: 手动指定带宽；为 None 时按 Silverman 规则估计。

    Returns:
        与 ``grid`` 等长的密度数组；样本退化（全同值）时返回以该值为中心的窄峰。
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return np.zeros_like(grid, dtype=float)
    if n == 1 or np.allclose(v, v[0]):
        z = np.full_like(grid, -0.5 * ((grid - v[0]) / (0.02 * (np.ptp(grid) or 1.0))) ** 2)
        dens = np.exp(z)
        return dens / max(dens.sum(), 1e-12)

    sd = float(np.std(v, ddof=1))
    iqr = float(np.subtract(*np.percentile(v, [75, 25])))
    scale = min(sd, iqr / 1.349) if iqr > 0 else sd
    if not np.isfinite(scale) or scale <= 0:
        scale = sd if sd > 0 else 1.0
    h = float(bw) if bw else 0.9 * scale * n ** (-1.0 / 5.0)
    h = max(h, 1e-9 * (float(np.ptp(v)) or 1.0))

    z = (grid[:, None] - v[None, :]) / h
    dens = np.exp(-0.5 * z ** 2).sum(axis=1)
    total = dens.sum()
    return dens / total if total > 0 else dens


def _kde_curve(values: np.ndarray, n_grid: int = 220,
               pad: float = 0.16) -> tuple[np.ndarray, np.ndarray]:
    """在样本范围外扩 ``pad`` 后求核密度曲线。

    Returns:
        (grid, density)，密度已归一化为最大值为 1（便于画小提琴/云雨图的半宽）。
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    lo, hi = float(np.min(v)), float(np.max(v))
    span = hi - lo
    if span <= 0:
        span = abs(lo) * 0.1 or 1.0
    grid = np.linspace(lo - pad * span, hi + pad * span, int(n_grid))
    dens = _kde(v, grid)
    peak = float(np.max(dens)) if dens.size else 0.0
    return grid, (dens / peak if peak > 0 else dens)


# ==========================================================================
# 1. 小提琴图
# ==========================================================================
def _draw_violin(ax, values: np.ndarray, center: float, color: str, half_width: float,
                 style: Style, side: str = "both") -> None:
    """在 ``center`` 处画一个（半）小提琴。

    Args:
        ax: 目标轴。
        values: 该组的样本值。
        center: 小提琴在横轴上的中心位置。
        color: 填充色。
        half_width: 单侧最大宽度（数据坐标）。
        style: 样式对象。
        side: ``both`` / ``left`` / ``right``，控制画整只还是半边。
    """
    grid, dens = _kde_curve(values)
    alpha = style.num("palette.alpha", 1.0) * 0.55
    lw = style.num("element.line_width", 1.8) * 0.5
    if side in ("both", "right"):
        ax.fill_betweenx(grid, center, center + dens * half_width, facecolor=color,
                         edgecolor=color, linewidth=lw, alpha=alpha, zorder=2)
    if side in ("both", "left"):
        ax.fill_betweenx(grid, center, center - dens * half_width, facecolor=color,
                         edgecolor=color, linewidth=lw, alpha=alpha, zorder=2)


def _draw_narrow_box(ax, values: np.ndarray, center: float, color: str,
                     width: float, style: Style) -> None:
    """在 ``center`` 处叠加窄箱线（含须）与中位数点。"""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    q1, q2, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    iqr = q3 - q1
    lo = float(np.min(v[v >= q1 - 1.5 * iqr])) if np.any(v >= q1 - 1.5 * iqr) else float(v.min())
    hi = float(np.max(v[v <= q3 + 1.5 * iqr])) if np.any(v <= q3 + 1.5 * iqr) else float(v.max())
    lw = max(0.7, style.num("element.line_width", 1.8) * 0.55)
    dark = tuple(min(1.0, c * 0.65) for c in mcolors.to_rgb(color))

    ax.plot([center, center], [lo, q1], color=dark, lw=lw, zorder=4)
    ax.plot([center, center], [q3, hi], color=dark, lw=lw, zorder=4)
    ax.plot([center - width * 0.45, center + width * 0.45], [lo, lo],
            color=dark, lw=lw, zorder=4)
    ax.plot([center - width * 0.45, center + width * 0.45], [hi, hi],
            color=dark, lw=lw, zorder=4)
    ax.add_patch(plt.Rectangle((center - width / 2, q1), width, max(q3 - q1, 1e-9),
                               facecolor="white", edgecolor=dark, linewidth=lw, zorder=5))
    ax.plot([center], [q2], marker="o", ms=3.4, color=dark, zorder=6)


def plot_violin(rel_table: pd.DataFrame, groups: pd.Series, out_path: str | Path,
                style: Style, rank_label: str = "Feature", top_n: int = 9) -> dict:
    """小提琴图（多面板，每个面板一个优势特征，横轴分组）。

    每个面板内：整只小提琴（按键密度塑形）+ 窄箱线 + 中位数点。

    Args:
        rel_table: 行=特征、列=样本的丰度表（自动转相对丰度）。
        groups: index=样本名、值为分组的 Series。
        out_path: 输出图片路径。
        style: 样式对象。
        rank_label: 分类级中文名称，用于标题/图注。
        top_n: 展示的优势特征个数（每个特征一个面板）。

    Returns:
        ``save()`` 的结果字典（path / name / caption / title）。

    Raises:
        ValueError: 样本、分组或特征数量不足以绘图。
    """
    rel = _to_relative(rel_table)
    mat, feats = _top_features(rel, groups, top_n)
    meta = mat.index.to_series().map(groups.astype(str))
    order = _group_order(meta)
    colors = style.group_colors(order)
    width = style.num("element.box_width", 0.6)
    alpha = style.num("element.point_alpha", 0.92) * style.num("palette.alpha", 1.0)
    edge = "white" if style.flag("element.point_edge", True) else "none"
    jitter = style.num("element.jitter", 0.18)

    fig, axes, ncol = panel_grid(style, len(feats), base_w=3.3, base_h=3.0)
    x = np.arange(len(order))
    for k, feat in enumerate(feats):
        ax = axes[k // ncol][k % ncol]
        rng = np.random.default_rng(20240517 + k)
        for i, g in enumerate(order):
            vals = mat.loc[meta == g, feat].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            _draw_violin(ax, vals, float(i), colors[g], 0.36, style)
            _draw_narrow_box(ax, vals, float(i), colors[g], width * 0.30, style)
            ax.scatter(i + rng.uniform(-jitter, jitter, vals.size), vals, s=9,
                       color="#3A3A3A", alpha=alpha * 0.65, edgecolor=edge,
                       linewidth=0.25, zorder=6)

        ax.set_xticks(x)
        ax.set_xticklabels(order, fontsize=style.font_size("tick_label") * 0.9)
        if len(order) > 4:
            ax.tick_params(axis="x", rotation=style.num("element.axis_rotate", 45))
        style_axes(ax, style)
        place_axis_labels(ax, style, "", _axis_text(style, "相对丰度"))
        place_title(ax, style, f"{str(feat)[:30]}", allow_override=False, scale=0.92)

    for k in range(len(feats), axes.shape[0] * axes.shape[1]):
        axes[k // ncol][k % ncol].axis("off")

    handles = [Patch(facecolor=colors[g], edgecolor="none", label=g) for g in order]
    axes[0][0].legend(handles=handles, loc="upper right", frameon=style.flag("legend.frame", False),
                      fontsize=style.font_size("legend_text"), title=None)
    place_suptitle(fig, style, f"优势 {rank_label} 的小提琴图")
    fig.tight_layout()
    result = save(
        fig, out_path, style,
        f"多面板小提琴图：每个面板是一个优势 {rank_label}，横轴为分组。"
        "琴身宽度表示该组样本丰度的核密度（越宽说明该区间样本越密集），"
        "内部白色窄箱线给出四分位距，黑点为中位数，散点为单个样本。"
        "读图时重点比较同组内分布的宽窄与各组中位线的相对高低。",
        f"优势 {rank_label} 小提琴图",
    )
    result["features"] = feats
    return result


# ==========================================================================
# 2. 云雨图
# ==========================================================================
def plot_raincloud(rel_table: pd.DataFrame, groups: pd.Series, out_path: str | Path,
                   style: Style, rank_label: str = "Feature", top_n: int = 6) -> dict:
    """云雨图：半边小提琴 + 窄箱线 + 散点云（多面板）。

    每个面板的左半边是半小提琴（"云"），中间是窄箱线（"伞柄"），
    右半边是按局部密度横向错开的散点（"雨"）。

    Args:
        rel_table: 行=特征、列=样本的丰度表。
        groups: index=样本名、值为分组的 Series。
        out_path: 输出图片路径。
        style: 样式对象。
        rank_label: 分类级中文名称。
        top_n: 展示的优势特征个数。

    Returns:
        ``save()`` 的结果字典。

    Raises:
        ValueError: 样本、分组或特征数量不足以绘图。
    """
    rel = _to_relative(rel_table)
    mat, feats = _top_features(rel, groups, top_n)
    meta = mat.index.to_series().map(groups.astype(str))
    order = _group_order(meta)
    colors = style.group_colors(order)
    alpha = style.num("element.point_alpha", 0.92) * style.num("palette.alpha", 1.0)
    size = style.num("element.point_size", 68) * 0.42

    fig, axes, ncol = panel_grid(style, len(feats), base_w=3.5, base_h=3.0)
    x = np.arange(len(order))
    for k, feat in enumerate(feats):
        ax = axes[k // ncol][k % ncol]
        ax.set_yticks(x)
        ax.set_yticklabels(order, fontsize=style.font_size("tick_label") * 0.9)
        for i, g in enumerate(order):
            vals = mat.loc[meta == g, feat].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            # 云：半边小提琴（朝下）
            grid, dens = _kde_curve(vals)
            ax.fill_betweenx(grid, float(i) - 0.02, float(i) - 0.02 - dens * 0.40,
                             facecolor=colors[g], edgecolor=colors[g],
                             linewidth=style.num("element.line_width", 1.8) * 0.5,
                             alpha=0.55 * style.num("palette.alpha", 1.0), zorder=2)
            # 伞柄：窄箱线（旋转成横向）
            q1, q2, q3 = (float(v) for v in np.percentile(vals, [25, 50, 75]))
            iqr = q3 - q1
            lo = float(vals[vals >= q1 - 1.5 * iqr].min()) if np.any(vals >= q1 - 1.5 * iqr) else q1
            hi = float(vals[vals <= q3 + 1.5 * iqr].max()) if np.any(vals <= q3 + 1.5 * iqr) else q3
            dark = tuple(min(1.0, c * 0.65) for c in mcolors.to_rgb(colors[g]))
            lw = max(0.7, style.num("element.line_width", 1.8) * 0.55)
            ax.plot([lo, q1], [i, i], color=dark, lw=lw, zorder=4)
            ax.plot([q3, hi], [i, i], color=dark, lw=lw, zorder=4)
            ax.add_patch(plt.Rectangle((q1, i - 0.085), max(q3 - q1, 1e-9), 0.17,
                                       facecolor="white", edgecolor=dark,
                                       linewidth=lw, zorder=5))
            ax.plot([q2], [i], marker="o", ms=3.4, color=dark, zorder=6)
            # 雨：按密度错开的散点
            dens_norm = _kde(vals, vals)
            dens_norm = dens_norm / max(dens_norm.max(), 1e-12)
            span = float(np.ptp(vals)) or 1.0
            offs = dens_norm * 0.34 - 0.17
            rng = np.random.default_rng(9001 + k * 31 + i)
            offs = offs + rng.uniform(-0.025, 0.025, offs.size)
            ax.scatter((float(i) + 0.045) + offs, vals, s=size, color=colors[g],
                       alpha=alpha * 0.8, edgecolor="white", linewidth=0.25, zorder=3)

        ax.axhline(0, color="#DDDDDD", lw=0.6, zorder=0)
        ax.set_xlim(-0.85, 0.55)
        style_axes(ax, style)
        place_axis_labels(ax, style, "", _axis_text(style, "相对丰度"))
        place_title(ax, style, f"{str(feat)[:30]}", allow_override=False, scale=0.92)

    for k in range(len(feats), axes.shape[0] * axes.shape[1]):
        axes[k // ncol][k % ncol].axis("off")

    handles = [Patch(facecolor=colors[g], edgecolor="none", label=g) for g in order]
    axes[0][0].legend(handles=handles, loc="lower right", frameon=style.flag("legend.frame", False),
                      fontsize=style.font_size("legend_text"))
    place_suptitle(fig, style, f"优势 {rank_label} 云雨图（半边小提琴 + 箱线 + 散点云）")
    fig.tight_layout()
    result = save(
        fig, out_path, style,
        f"多面板云雨图：每个面板是一个优势 {rank_label}，纵轴为分组。"
        "每行左侧的“云”是该组丰度的半核密度曲线，中间的窄箱线给出四分位距与中位数，"
        "右侧的“雨”是全部样本点（按局部密度横向错开，因此越密集处越宽）。"
        "相比箱线图，它同时保留了分布形状、离散程度和每一个原始样本。",
        f"优势 {rank_label} 云雨图",
    )
    result["features"] = feats
    return result


# ==========================================================================
# 3. 密度图
# ==========================================================================
def _rug(ax, values: np.ndarray, y0: float, color: str, style: Style,
         side: str = "bottom", height: float = 0.055) -> None:
    """在轴的边缘画 rug（短竖线），标出每个样本的实际取值。"""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    lw = max(0.5, style.num("element.line_width", 1.8) * 0.35)
    if side == "bottom":
        ax.vlines(v, y0, y0 + height, color=color, lw=lw, alpha=0.75, zorder=7)
    else:
        ax.vlines(v, y0 - height, y0, color=color, lw=lw, alpha=0.75, zorder=7)


def plot_density(rel_table: pd.DataFrame, groups: pd.Series, out_path: str | Path,
                 style: Style, rank_label: str = "Feature", top_n: int = 6,
                 *, rug: bool = True) -> dict:
    """密度图：每个面板一个特征的核密度曲线，按分组着色并填充半透明，底部可加 rug。

    Args:
        rel_table: 行=特征、列=样本的丰度表。
        groups: index=样本名、值为分组的 Series。
        out_path: 输出图片路径。
        style: 样式对象。
        rank_label: 分类级中文名称。
        top_n: 展示的优势特征个数。
        rug: 是否在面板底部画 rug（每个样本一根短竖线）。

    Returns:
        ``save()`` 的结果字典。

    Raises:
        ValueError: 样本、分组或特征数量不足以绘图。
    """
    rel = _to_relative(rel_table)
    mat, feats = _top_features(rel, groups, top_n)
    meta = mat.index.to_series().map(groups.astype(str))
    order = _group_order(meta)
    colors = style.group_colors(order)
    lw = style.num("element.line_width", 1.8)
    alpha = 0.28 * style.num("palette.alpha", 1.0)

    fig, axes, ncol = panel_grid(style, len(feats), base_w=3.4, base_h=2.9)
    for k, feat in enumerate(feats):
        ax = axes[k // ncol][k % ncol]
        peak = 0.0
        for g in order:
            vals = mat.loc[meta == g, feat].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size < 2:
                continue
            grid, dens = _kde_curve(vals)
            peak = max(peak, 1.0)
            ax.plot(grid, dens, color=colors[g], lw=lw, label=g, zorder=4)
            ax.fill_between(grid, 0, dens, color=colors[g], alpha=alpha, zorder=2)
            if rug:
                _rug(ax, vals, 0.0, colors[g], style)

        ax.set_ylim(-0.06, max(1.22, peak + 0.22))
        ax.set_xlabel(_axis_text(style, "相对丰度"),
                      fontsize=style.font_size("axis_label"),
                      color=style.font_color("axis_label"))
        style_axes(ax, style)
        place_title(ax, style, f"{str(feat)[:30]}", allow_override=False, scale=0.92)

    for k in range(len(feats), axes.shape[0] * axes.shape[1]):
        axes[k // ncol][k % ncol].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[0][0].legend(loc="upper right", frameon=style.flag("legend.frame", False),
                          fontsize=style.font_size("legend_text"), ncol=int(style.num("legend.ncol", 1)))
    place_suptitle(fig, style, f"优势 {rank_label} 丰度密度分布")
    fig.tight_layout()
    result = save(
        fig, out_path, style,
        f"多面板密度图：每个面板是一个优势 {rank_label}，横轴为相对丰度、纵轴为核密度（已按峰值归一）。"
        "每条彩色曲线对应一个分组，曲线错开说明该特征在组间分布不同；"
        "曲线峰值位置差得越远、重叠越少，说明组间区分度越好。"
        + (" 底部短竖线（rug）标出每个样本的实际取值。" if rug else ""),
        f"优势 {rank_label} 密度分布",
    )
    result["features"] = feats
    return result


# ==========================================================================
# 4. 蜂群图
# ==========================================================================
def _swarm_offsets(y: np.ndarray | None = None, *, radius: float = 0.02,
                   gap: float = 0.02,
                   scale: tuple[float, float] | None = None,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """蜂群错开算法：为每个点计算横向偏移，保证圆点互不重叠且左右对称。

    核心是"蟹行"式贪心搜索：先在 (Δx/scale_x, Δy/scale_y) 的等比空间里做
    圆-圆不重叠判定，候选偏移按 0, +r, -r, +2r, -2r ... 交替试放，
    取第一个与所有已放置点距离 ≥ ``2*radius`` 的位置，因此结果天然左右对称。

    Args:
        y: 点的纵坐标（数据坐标）。为 None 时返回空数组。
        radius: 点的显示半径，单位是轴高归一化坐标（= 轴宽的归一化半径）。
        gap: 额外留出的间隙，防止边缘刚好相切。
        scale: ``(x_pad, y_pad)``，把数据坐标同比缩放到归一化空间的换算系数；
            省略时按 y 的极差自适应。
        rng: 随机数发生器（仅用于打破完全对称时的并列选择）。

    Returns:
        与 ``y`` 等长的横向偏移量（数据坐标，正数向右）。

    Raises:
        ValueError: ``y`` 中含有非有限值。
    """
    if y is None:
        return np.zeros(0, dtype=float)
    yy = np.asarray(y, dtype=float)
    if yy.size == 0:
        return np.zeros(0, dtype=float)
    if not np.all(np.isfinite(yy)):
        raise ValueError("蜂群图：样本值中出现 NaN / inf，请先清洗数据。")

    rng = rng or np.random.default_rng(20240517)
    if scale is None:
        span_y = float(np.ptp(yy)) or 1.0
        y_pad = span_y * 1.06
        x_pad = y_pad * 0.45
    else:
        x_pad, y_pad = float(scale[0]), float(scale[1])
    x_pad = x_pad if x_pad > 0 else 1.0
    y_pad = y_pad if y_pad > 0 else 1.0

    min_dist = 2.0 * (radius + gap)  # 归一化空间里的最小圆心距
    order = np.argsort(yy, kind="stable")
    placed: list[tuple[float, float]] = []   # 归一化坐标
    norm_off = np.zeros(yy.size, dtype=float)

    # 归一化空间中的"单位步长"：换算回数据坐标后恰好等于点直径
    step_data = min_dist * x_pad
    for idx in order:
        nyv = float(yy[idx]) / y_pad
        # 只有纵向差异小于点直径的已放点才可能碰撞
        near = [(nx, ny) for nx, ny in placed if abs(ny - nyv) < min_dist]
        pick = 0.0
        if near:
            k = 0
            while k < 5000:
                if k == 0:
                    cand = 0.0
                else:
                    mag = ((k + 1) // 2) * step_data
                    cand = mag if k % 2 == 1 else -mag
                    if k > 1:
                        cand += float(rng.uniform(-0.06, 0.06)) * step_data
                nxc = cand / x_pad
                blocked = False
                for nx, ny in near:
                    if (nxc - nx) ** 2 + (nyv - ny) ** 2 < min_dist ** 2 - 1e-12:
                        blocked = True
                        break
                if not blocked:
                    pick = cand
                    break
                k += 1
        norm_off[idx] = pick / x_pad
        placed.append((pick / x_pad, nyv))

    return norm_off * x_pad


def plot_beeswarm(rel_table: pd.DataFrame, groups: pd.Series, out_path: str | Path,
                  style: Style, rank_label: str = "Feature", top_n: int = 9) -> dict:
    """蜂群图：点按局部密度横向错开，避免重叠（自行实现的错开算法）。

    所有面板共用一个纵横联合坐标系，因此点间距在全局都满足"圆心距 ≥ 点直径"。

    Args:
        rel_table: 行=特征、列=样本的丰度表。
        groups: index=样本名、值为分组的 Series。
        out_path: 输出图片路径。
        style: 样式对象。
        rank_label: 分类级中文名称。
        top_n: 展示的优势特征个数。

    Returns:
        ``save()`` 的结果字典，额外含 ``swarm_min_ratio``（实际最小间距 / 点直径）。

    Raises:
        ValueError: 样本、分组或特征数量不足以绘图。
    """
    rel = _to_relative(rel_table)
    mat, feats = _top_features(rel, groups, top_n)
    meta = mat.index.to_series().map(groups.astype(str))
    order = _group_order(meta)
    colors = style.group_colors(order)
    alpha = style.num("element.point_alpha", 0.92) * style.num("palette.alpha", 1.0)
    size = style.num("element.point_size", 68)
    edge = "white" if style.flag("element.point_edge", True) else "none"

    fig, axes, ncol = panel_grid(style, len(feats), base_w=3.3, base_h=3.0)
    panel_h = 3.0 * style.num("figure.scale", 1.0)
    x_span = 0.5 * 2.0 + 4.0  # 每个面板横向可用宽度（数据坐标）的保守估计
    y_span = 1.06 * float(np.ptp(mat[feats].to_numpy(dtype=float)) or 1.0)

    # 点半径：换算到"数据坐标"里，横向每单位 = 1/x_span 个面板宽
    rad_pt = 0.5 * math.sqrt(max(1.0, size))
    radius_norm = rad_pt / 300.0            # 近似：一格 ≈ 300pt 宽
    scale = (x_span * radius_norm * 2.0 / 2.0, y_span / (panel_h * 72.0 / 300.0))
    scale = (max(scale[0], 1e-6), max(scale[1], 1e-6))

    min_ratio = float("inf")
    for k, feat in enumerate(feats):
        ax = axes[k // ncol][k % ncol]
        rng = np.random.default_rng(20240518 + k)
        for i, g in enumerate(order):
            vals = mat.loc[meta == g, feat].to_numpy(dtype=float)
            vals = np.asarray(vals[np.isfinite(vals)], dtype=float)
            if vals.size == 0:
                continue
            offs = _swarm_offsets(vals, radius=radius_norm, gap=radius_norm * 0.25,
                                  scale=scale, rng=rng)
            xs = float(i) + offs * 2.0
            ax.scatter(xs, vals, s=size, color=colors[g], alpha=alpha,
                       edgecolor=edge, linewidth=0.4, zorder=4, label=str(g))
            # 记录实际最小间距 / 点直径
            for a in range(vals.size):
                for b in range(a + 1, vals.size):
                    dx = (xs[a] - xs[b]) / scale[0]
                    dy = (vals[a] - vals[b]) / scale[1]
                    dist = math.hypot(dx, dy)
                    min_ratio = min(min_ratio, dist / (2.0 * radius_norm))

        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, fontsize=style.font_size("tick_label") * 0.9)
        if len(order) > 4:
            ax.tick_params(axis="x", rotation=style.num("element.axis_rotate", 45))
        style_axes(ax, style)
        place_axis_labels(ax, style, "", _axis_text(style, "相对丰度"))
        place_title(ax, style, f"{str(feat)[:30]}", allow_override=False, scale=0.92)

    for k in range(len(feats), axes.shape[0] * axes.shape[1]):
        axes[k // ncol][k % ncol].axis("off")

    handles = [Patch(facecolor=colors[g], edgecolor="none", label=g) for g in order]
    axes[0][0].legend(handles=handles, loc="upper right", frameon=style.flag("legend.frame", False),
                      fontsize=style.font_size("legend_text"))
    place_suptitle(fig, style, f"优势 {rank_label} 蜂群图")
    fig.tight_layout()
    result = save(
        fig, out_path, style,
        f"多面板蜂群图：每个面板是一个优势 {rank_label}，横轴为分组、纵轴为相对丰度。"
        "每个点是一个样本；为避免重叠，同一高度上过密的点会被左右对称地推开，"
        "点的横向铺开宽度因此直观反映了该丰度区间的样本密度。"
        "适合小样本量（每组 3~30 个）时判断分布形状、离群点与组间重叠程度。",
        f"优势 {rank_label} 蜂群图",
    )
    result["features"] = feats
    result["swarm_min_ratio"] = float(min_ratio)
    return result


# ==========================================================================
# 5. 棒棒糖图
# ==========================================================================
def plot_lollipop(df: pd.DataFrame, out_path: str | Path, style: Style,
                  value_col: str = "log2FC", label_col: str = "feature",
                  group_a: str = "", group_b: str = "", top_n: int = 20) -> dict:
    """棒棒糖图（横向）：每条目一个点 + 从 0 引出的细线，按数值排序，正负两色。

    Args:
        df: 差异分析结果表，至少包含 ``label_col`` 与 ``value_col``。
        out_path: 输出图片路径。
        style: 样式对象。
        value_col: 数值列名（如 log2FC）。
        label_col: 条目名列名。
        group_a: 正向富集组名（用于图注与轴标题）。
        group_b: 负向富集组名。
        top_n: 展示条目数（按数值绝对值排序）。

    Returns:
        ``save()`` 的结果字典。

    Raises:
        ValueError: 缺少必需列、数值列全为空、或没有可展示的条目。
    """
    if value_col not in df.columns:
        raise ValueError(f"数据缺少数值列 “{value_col}”。")
    if label_col not in df.columns:
        raise ValueError(f"数据缺少标签列 “{label_col}”。")

    work = df[[label_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])
    if work.empty:
        raise ValueError(f"列 “{value_col}” 全为空值，无法绘制棒棒糖图。")

    sub = work.reindex(work[value_col].abs().sort_values(ascending=False).index)
    sub = sub.head(max(1, int(top_n))).iloc[::-1]
    if sub.empty:
        raise ValueError("没有可展示的条目。")

    pos_c = style.single_color(3)
    neg_c = style.single_color(0)
    labels = [str(v)[:46] for v in sub[label_col]]
    values = sub[value_col].to_numpy(dtype=float)
    y = np.arange(len(sub))

    fig, ax = new_figure(style, 7.6, max(3.4, 0.34 * len(sub) + 1.9))
    lw = max(0.6, style.num("element.line_width", 1.8) * 0.6)
    for yi, v in zip(y, values):
        color = pos_c if v >= 0 else neg_c
        ax.plot([0, v], [yi, yi], color=color, lw=lw, alpha=0.85, zorder=2)
        ax.scatter([v], [yi], s=style.num("element.point_size", 68) * 0.62, color=color,
                   alpha=style.num("element.point_alpha", 0.92), zorder=3,
                   edgecolor="white" if style.flag("element.point_edge", True) else "none",
                   linewidth=0.6)

    if style.flag("element.zero_lines", True):
        ax.axvline(0, color="#333333", lw=1.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=style.font_size("tick_label") * 0.9)
    ax.tick_params(axis="y", length=0)
    style_axes(ax, style)

    xlabel = f"log2 Fold Change  ({group_a} / {group_b})" if (group_a or group_b) else "数值"
    place_axis_labels(ax, style, xlabel, "")
    place_title(ax, style, f"特征效应量棒棒糖图（Top {len(sub)}）")
    handles = [Patch(facecolor=pos_c, edgecolor="none", label=f"上调（{group_a or '正值'}）"),
               Patch(facecolor=neg_c, edgecolor="none", label=f"下调（{group_b or '负值'}）")]
    resolve_legend(style, ax, extra_handles=handles, preferred="lower right")
    fig.tight_layout()
    result = save(
        fig, out_path, style,
        "横向棒棒糖图：每个条目一根从 0 出发的细线，末端圆点表示其效应量大小。"
        f"线段朝右（暖色）表示在 {group_a or '组 A'} 中更高，朝左（冷色）表示在 "
        f"{group_b or '组 B'} 中更高；线条越长差异倍数越大。"
        "相比条形图，它更强调数值落点，适合在有限版面里展示 Top 差异特征。",
        "特征效应量棒棒糖图",
    )
    result["n_items"] = int(len(sub))
    return result


# ==========================================================================
# 6. 气泡图
# ==========================================================================
def plot_bubble(df: pd.DataFrame, out_path: str | Path, style: Style,
                x_col: str, y_col: str, size_col: str, color_col: str | None = None,
                label_col: str | None = None, top_n: int = 40,
                title: str = "气泡图") -> dict:
    """通用气泡图（也可当包珠图用）：x/y 定位，size 决定点大小，color 决定颜色。

    颜色既支持连续色图（``color_col`` 为数值、且唯一值多于 8 个），
    也支持分组色板（``color_col`` 为分类变量）。

    Args:
        df: 数据表。
        out_path: 输出图片路径。
        style: 样式对象。
        x_col: 横坐标列名。
        y_col: 纵坐标列名。
        size_col: 决定点大小的列名。
        color_col: 决定颜色的列名；None 时用调色板主色。
        label_col: 需要标注文字的列名。
        top_n: 参与绘图的最大条目数（按 size 降序取）。
        title: 图标题（中文）。

    Returns:
        ``save()`` 的结果字典。

    Raises:
        ValueError: 缺少必需列、数值列全为空、或没有可用条目。
    """
    for col in (x_col, y_col, size_col):
        if col not in df.columns:
            raise ValueError(f"数据缺少必需列 “{col}”。")
    if color_col is not None and color_col not in df.columns:
        raise ValueError(f"数据缺少颜色列 “{color_col}”。")
    if label_col is not None and label_col not in df.columns:
        raise ValueError(f"数据缺少标签列 “{label_col}”。")

    cols = [x_col, y_col, size_col] + [c for c in (color_col, label_col) if c]
    work = df[list(dict.fromkeys(cols))].copy()
    for c in (x_col, y_col, size_col):
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col, size_col])
    if work.empty:
        raise ValueError("x / y / size 列去掉空值后没有可用条目。")

    work = work.sort_values(size_col, ascending=False).head(max(1, int(top_n)))
    if work.empty:
        raise ValueError("没有可展示的条目。")

    base = style.num("element.point_size", 68)
    raw = work[size_col].to_numpy(dtype=float)
    lo, hi = float(np.min(raw)), float(np.max(raw))
    frac = np.ones_like(raw) if hi <= lo else (raw - lo) / (hi - lo)
    sizes = base * (0.35 + 2.6 * frac)
    alpha = style.num("element.point_alpha", 0.92) * style.num("palette.alpha", 1.0)
    edge = "white" if style.flag("element.point_edge", True) else "none"

    fig, ax = new_figure(style, 7.4, 5.8)
    legend_extra: list[tuple[Any, str]] = []
    if color_col is None:
        ax.scatter(work[x_col], work[y_col], s=sizes, color=style.single_color(0),
                   alpha=alpha, edgecolor=edge, linewidth=0.6, zorder=3)
    else:
        cvals = work[color_col]
        numeric = pd.to_numeric(cvals, errors="coerce")
        use_continuous = bool(numeric.notna().mean() > 0.95 and numeric.nunique() > 8)
        if use_continuous:
            cres = pd.to_numeric(cvals, errors="coerce").fillna(float(numeric.min()))
            sc = ax.scatter(work[x_col], work[y_col], s=sizes, c=cres.to_numpy(dtype=float),
                            cmap=style.cmap("continuous"), alpha=alpha,
                            edgecolor=edge, linewidth=0.6, zorder=3)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label(str(color_col), fontsize=style.font_size("legend_text"),
                           color=style.font_color("legend_text"))
            cbar.ax.tick_params(labelsize=style.font_size("tick_label") * 0.85)
        else:
            cats = cvals.astype(str)
            cmap_colors = style.group_colors(sorted(cats.unique().tolist()))
            for cat in sorted(cats.unique().tolist()):
                m = (cats == cat).to_numpy()
                ax.scatter(work[x_col].to_numpy()[m], work[y_col].to_numpy()[m],
                           s=sizes[m], color=cmap_colors[cat], alpha=alpha, label=str(cat),
                           edgecolor=edge, linewidth=0.6, zorder=3)
            legend_extra = [(Patch(facecolor=cmap_colors[c], edgecolor="none", label=str(c)),
                             str(c)) for c in sorted(cats.unique().tolist())]

    if label_col is not None:
        xspan = float(np.ptp(work[x_col].to_numpy(dtype=float))) or 1.0
        yspan = float(np.ptp(work[y_col].to_numpy(dtype=float))) or 1.0
        limit = int(style.num("element.label_top_n", 12)) or 12
        placed: list[tuple[float, float]] = []
        for _, row in work.iterrows():
            if len(placed) >= limit:
                break
            px, py = float(row[x_col]), float(row[y_col])
            if all(abs(px - qx) > 0.06 * xspan or abs(py - qy) > 0.05 * yspan
                   for qx, qy in placed):
                placed.append((px, py))
                ax.annotate(str(row[label_col])[:28], (px, py),
                            fontsize=style.font_size("annotation") * 0.85,
                            color=style.font_color("annotation"),
                            xytext=(6, 5), textcoords="offset points", zorder=6)

    if style.flag("element.zero_lines", True) and float(np.min(work[y_col])) < 0 < float(np.max(work[y_col])):
        ax.axhline(0, color="#BBBBBB", lw=0.8, ls="--", zorder=1)

    style_axes(ax, style)
    place_axis_labels(ax, style, f"{x_col}", f"{y_col}")
    place_title(ax, style, str(title))

    # 气泡大小图例
    handles, labels = ax.get_legend_handles_labels()
    for frac_ref in (0.0, 0.5, 1.0):
        val = lo + frac_ref * (hi - lo)
        handles.append(plt.Line2D([], [], marker="o", linestyle="none",
                                  markersize=math.sqrt(base * (0.35 + 2.6 * frac_ref)) * 0.52,
                                  markerfacecolor="#B0B0B0", markeredgecolor="none"))
        labels.append(f"{size_col} = {val:.3g}")
    for h, l in legend_extra:
        handles.append(h)
        labels.append(l)
    resolve_legend(style, ax, extra_handles=list(zip(handles, labels)), preferred="upper right")
    fig.tight_layout()
    result = save(
        fig, out_path, style,
        f"气泡图：横轴为 {x_col}，纵轴为 {y_col}，气泡面积正比于 {size_col}"
        + (f"，颜色代表 {color_col}（连续变量用色带、分类变量用分组色）" if color_col else "")
        + "。右上角气泡越大越靠上表示该条目在两个维度上都更突出，"
        "常用于富集分析（通路）或差异特征的多指标汇总展示。",
        str(title),
    )
    result["n_items"] = int(len(work))
    return result


# ==========================================================================
# 7. 桑基图（纯 matplotlib）
# ==========================================================================
def _clean_flow(flow: pd.DataFrame) -> pd.DataFrame:
    """把两列 / 三列的流向表统一整理成 ``source`` / ``target`` / ``weight`` 三列。

    Raises:
        ValueError: 少于两列、列名缺失或权重含非有限值。
    """
    if flow is None or len(flow) == 0:
        raise ValueError("流向表为空，无法绘制桑基图。")
    df = flow.copy()
    if isinstance(df.index, pd.MultiIndex) and df.shape[1] == 0:
        df = df.reset_index()

    cols = list(df.columns)
    lowered = [str(c).strip().lower() for c in cols]

    def _find(names: Sequence[str]) -> Any:
        for want in names:
            if want in lowered:
                return cols[lowered.index(want)]
        return None

    src = _find(["source", "src", "from", "来源", "组"])
    tgt = _find(["target", "dst", "to", "去向", "目标"])
    wgt = _find(["weight", "value", "count", "flow", "权重", "丰度"])

    if src is None or tgt is None:
        if len(cols) < 2:
            raise ValueError("桑基图需要至少两列（source, target）或三列（source, target, weight）。")
        src, tgt = (src or cols[0]), (tgt or cols[1])
        if wgt is None and len(cols) >= 3 and pd.api.types.is_numeric_dtype(df[cols[2]]):
            wgt = cols[2]

    out = pd.DataFrame({
        "source": df[src].astype(str).str.strip(),
        "target": df[tgt].astype(str).str.strip(),
    })
    if wgt is not None and wgt not in (src, tgt):
        w = pd.to_numeric(df[wgt], errors="coerce")
    else:
        w = pd.Series(np.ones(len(df)), index=df.index)
    out["weight"] = w.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["weight"])
    out = out[(out["source"] != "") & (out["target"] != "")]
    if out.empty:
        raise ValueError("流向表清洗后没有有效记录（空标签或非法权重）。")
    if not np.all(np.isfinite(out["weight"].to_numpy(dtype=float))):
        raise ValueError("流向表的权重列存在非有限值。")
    out = out.groupby(["source", "target"], as_index=False, sort=False)["weight"].sum()
    if float(out["weight"].sum()) <= 0:
        raise ValueError("流向表的权重总和为 0，无法绘制桑基图。")
    return out


def _sankey_geometry(flow: pd.DataFrame, min_flow: float = 0.0, max_links: int = 60,
                     gap_frac: float = 0.045, scale_frac: float = 0.92,
                     thickness_frac: float = 0.055,
                     x_left: float = 0.02, x_right: float = 0.98) -> dict:
    """计算桑基图的节点矩形与连边扇形带几何。

    节点按该层的总流量排序，节点高度严格正比于其流量；连边厚度正比于流量，
    因此"左层总高 = 右层总高 = 所有连边厚度之和"，流量守恒在几何上是结构性的。

    Args:
        flow: 已清洗的 ``source`` / ``target`` / ``weight`` 三列表。
        min_flow: 低于该流量的连边并入 ``(其他)`` 汇总边。
        max_links: 最多绘制的连边数（按流量降序）。
        gap_frac: 节点之间的间隙占可用高度的比例。
        scale_frac: 节点总高占可用高度的比例。
        thickness_frac: 节点矩形宽度（以 x 轴单位计）。
        x_left: 左层节点的 x 中心。
        x_right: 右层节点的 x 中心。

    Returns:
        dict，含 ``nodes``（含 layer/name/total/y0/y1/x0/x1）、``links``
        （含 source/target/weight/vertices）、``source_totals``、``target_totals``。

    Raises:
        ValueError: 清洗后仍无有效连边。
    """
    work = flow.copy()
    other = work["weight"] < float(min_flow)
    if bool(other.any()):
        work = pd.concat([work[~other],
                          pd.DataFrame({"source": ["(其他)"], "target": ["(其他)"],
                                        "weight": [float(work.loc[other, "weight"].sum())]})],
                         ignore_index=True)
    work = work.sort_values("weight", ascending=False).head(max(1, int(max_links)))
    work = work[work["weight"] > 0]
    if work.empty:
        raise ValueError("筛选后的流向表没有有效连边，请降低 min_flow 或放大 max_links。")

    sources = list(dict.fromkeys(work["source"].tolist()))
    targets = list(dict.fromkeys(work["target"].tolist()))
    s_tot = work.groupby("source")["weight"].sum().to_dict()
    t_tot = work.groupby("target")["weight"].sum().to_dict()
    sources.sort(key=lambda s: (-float(s_tot.get(s, 0.0)), str(s)))
    targets.sort(key=lambda t: (-float(t_tot.get(t, 0.0)), str(t)))
    max_links_n = max(len(sources), len(targets))
    if max_links_n <= 0:
        raise ValueError("流量图没有任何节点。")

    scale = (1.0 - gap_frac * (max_links_n - 1)) * scale_frac / max(
        max(sum(s_tot.values()), sum(t_tot.values())), 1e-12)

    def _place(names: list[str], totals: dict, x_center: float) -> dict[str, dict]:
        blocks: list[dict] = []
        cursor = 0.0
        for i, nm in enumerate(names):
            h = float(totals.get(nm, 0.0)) * scale
            blocks.append({"name": nm, "height": h, "y0": cursor})
            cursor += h
            if i != len(names) - 1:
                cursor += gap_frac
        offset = max((1.0 - cursor) / 2.0, 0.0)
        out: dict[str, dict] = {}
        for b in blocks:
            out[b["name"]] = {
                "name": b["name"], "height": b["height"],
                "y0": b["y0"] + offset, "y1": b["y0"] + b["height"] + offset,
                "x0": x_center - thickness_frac / 2.0,
                "x1": x_center + thickness_frac / 2.0,
            }
        return out

    s_nodes = _place(sources, s_tot, x_left if targets else 0.5)
    t_nodes = _place(targets, t_tot, x_right if sources else 0.5)
    for nm, nd in s_nodes.items():
        nd["layer"] = "source"
    for nm, nd in t_nodes.items():
        nd["layer"] = "target"

    # 连边：源节点内按目标 y 排序、目标节点内按源 y 排序，减少交叉
    s_cursor = {nm: float(s_nodes[nm]["y0"]) for nm in sources}
    t_cursor = {nm: float(t_nodes[nm]["y0"]) for nm in targets}
    order = work.copy()
    order["_sy"] = order["source"].map(lambda s: float(s_nodes[s]["y0"]))
    order["_ty"] = order["target"].map(lambda t: float(t_nodes[t]["y0"]))
    order = order.sort_values(["source", "_ty", "_sy", "target"], kind="stable")

    links: list[dict] = []
    for _, row in order.iterrows():
        s, t, w = str(row["source"]), str(row["target"]), float(row["weight"])
        th = w * scale
        sy0, sy1 = s_cursor[s], s_cursor[s] + th
        ty0, ty1 = t_cursor[t], t_cursor[t] + th
        s_cursor[s] = sy1
        t_cursor[t] = ty1
        top = _bezier_ridge(s_nodes[s]["x1"], sy1, t_nodes[t]["x0"], ty1, th, +0.5)
        bot = _bezier_ridge(s_nodes[s]["x1"], sy0, t_nodes[t]["x0"], ty0, th, -0.5)
        verts = top + bot[::-1]
        links.append({"source": s, "target": t, "weight": w,
                      "thickness": th, "vertices": verts})

    return {
        "nodes": {**s_nodes, **t_nodes},
        "sources": sources, "targets": targets,
        "links": links, "scale": scale,
        "source_totals": {k: float(v) for k, v in s_tot.items()},
        "target_totals": {k: float(v) for k, v in t_tot.items()},
        "height": {"source": float(sum(x["height"] for x in s_nodes.values())),
                   "target": float(sum(x["height"] for x in t_nodes.values())),
                   "thickness": float(sum(l["thickness"] for l in links))},
    }


def _bezier_ridge(x0: float, y0: float, x1: float, y1: float, thickness: float,
                  side: float, samples: int = 64) -> list[tuple[float, float]]:
    """采样一条三次贝塞尔中心线的上下边缘，返回多边形折线顶点。

    控制点取 ``x0 + 0.55*(x1-x0)`` 与 ``x1 - 0.55*(x1-x0)``，
    使连边呈现桑基图典型的"横 - 斜 - 横"S 形。

    Args:
        x0: 起点 x；y0: 起点中心 y。
        x1: 终点 x；y1: 终点中心 y。
        thickness: 连边厚度。
        side: ``+0.5`` 取上边缘，``-0.5`` 取下边缘。
        samples: 采样点数。

    Returns:
        顶点列表 [(x, y), ...]，从起点到终点。
    """
    dx = x1 - x0
    cx1, cx2 = x0 + 0.55 * dx, x1 - 0.55 * dx
    t = np.linspace(0.0, 1.0, max(8, int(samples)))[:, None]
    omt = 1.0 - t
    px = (omt ** 3) * x0 + 3 * (omt ** 2) * t * cx1 + 3 * omt * (t ** 2) * cx2 + (t ** 3) * x1
    py = (omt ** 3) * y0 + 3 * (omt ** 2) * t * y0 + 3 * omt * (t ** 2) * y1 + (t ** 3) * y1
    dpx = 3 * (omt ** 2) * (cx1 - x0) + 6 * omt * t * (cx2 - cx1) + 3 * (t ** 2) * (x1 - cx2)
    dpy = 3 * (omt ** 2) * (y0 - y0) + 6 * omt * t * (y1 - y0) + 3 * (t ** 2) * (y1 - y1)
    norm = np.hypot(dpx, dpy)
    norm[norm < 1e-12] = 1e-12
    nx, ny = -dpy / norm, dpx / norm
    off = side * thickness
    return list(zip((px + off * nx).ravel().tolist(), (py + off * ny).ravel().tolist()))


def _link_patch(vertices: list[tuple[float, float]], color: str, alpha: float,
                zorder: int = 2) -> PathPatch:
    """把（可能带间隙的）顶点列表拆成若干闭合多边形，返回一个拼接后的 PathPatch。"""
    verts: list[tuple[float, float]] = []
    codes: list[int] = []
    chunk: list[tuple[float, float]] = []
    for xy in vertices:
        if xy is None:
            if len(chunk) >= 3:
                verts.append(chunk[0])
                codes.append(MplPath.MOVETO)
                for p in chunk[1:]:
                    verts.append(p)
                    codes.append(MplPath.LINETO)
                verts.append(chunk[0])
                codes.append(MplPath.CLOSEPOLY)
            chunk = []
            continue
        chunk.append(xy)
    if len(chunk) >= 3:
        verts.append(chunk[0])
        codes.append(MplPath.MOVETO)
        for p in chunk[1:]:
            verts.append(p)
            codes.append(MplPath.LINETO)
        verts.append(chunk[0])
        codes.append(MplPath.CLOSEPOLY)
    path = MplPath(verts, codes)
    return PathPatch(path, facecolor=color, edgecolor="none", alpha=alpha,
                     linewidth=0.0, zorder=zorder)


def _darken(color: str, factor: float = 0.72) -> tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def plot_sankey(flow: pd.DataFrame, out_path: str | Path, style: Style,
                title: str = "桑基图", min_flow: float = 0.0, max_links: int = 60) -> dict:
    """桑基图：两列（source, target）或三列（source, target, weight）的流向表。

    节点按该层总流量排序、高度严格正比于流量；连边用 ``Path`` + ``PathPatch``
    绘制的扇形带（三次贝塞尔曲线填充）。纯 matplotlib 实现，不依赖 plotly。

    Args:
        flow: 流向 DataFrame。
        out_path: 输出图片路径。
        style: 样式对象。
        title: 中文标题。
        min_flow: 低于该流量的连边汇总为 "(其他)"。
        max_links: 最多绘制的连边数。

    Returns:
        ``save()`` 的结果字典，额外含 ``sankey``（几何与流量守恒信息）。

    Raises:
        ValueError: 流向表格式不合法或清洗后无有效连边。
    """
    clean = _clean_flow(flow)
    geo = _sankey_geometry(clean, min_flow=min_flow, max_links=max_links)
    nodes = geo["nodes"]
    with_flow = [n for n in nodes.values() if n["height"] > 1e-9]
    all_colors = list(nodes.keys())
    colors = style.group_colors(all_colors)
    alpha = 0.42 * style.num("palette.alpha", 1.0)

    fig, ax = new_figure(style, 8.6, 5.6)
    y_lo = min((n["y0"] for n in with_flow), default=0.0)
    y_hi = max((n["y1"] for n in with_flow), default=1.0)
    node_h = np.mean([n["height"] for n in with_flow]) if with_flow else 0.05

    for link in geo["links"]:
        patch = _link_patch(link["vertices"], colors[link["source"]], alpha, zorder=2)
        ax.add_patch(patch)

    edge = "white" if style.flag("element.bar_edge", True) else "none"
    for layer, names in (("source", geo["sources"]), ("target", geo["targets"])):
        for dist, nm in enumerate(names):
            nd = nodes[nm]
            if nd["height"] <= 1e-9:
                continue
            ax.add_patch(plt.Rectangle((nd["x0"], nd["y0"]),
                                       max(nd["x1"] - nd["x0"], 1e-4), nd["height"],
                                       facecolor=colors[nm], edgecolor=edge,
                                       linewidth=0.6, zorder=4))
            if nd["height"] >= 0.07 * node_h and style.show_text("annotation"):
                ax.annotate(f"{nm}  {geo['source_totals'].get(nm, geo['target_totals'].get(nm, 0.0)):.1%}",
                            xy=(nd["x0"] if layer == "source" else nd["x1"],
                                (nd["y0"] + nd["y1"]) / 2.0),
                            xytext=(-6 if layer == "source" else 6, 0),
                            textcoords="offset points",
                            ha="right" if layer == "source" else "left", va="center",
                            **text_style_kwargs(style, "annotation"), zorder=6)

    pad = 0.06 * max(y_hi - y_lo, 1e-6)
    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.axis("off")
    place_title(ax, style, str(title))
    fig.tight_layout()
    result = save(
        fig, out_path, style,
        "桑基图展示流量如何从左侧（来源）分配到右侧（去向）。"
        "每个矩形的高度正比于该节点承载的总流量，连边（扇形带）的宽度正比于具体流量，"
        "颜色跟随来源节点。读图时看某个来源主要流向哪些去向、以及各去向由谁构成。",
        str(title),
    )
    result["sankey"] = {
        "n_nodes": len(nodes), "n_links": len(geo["links"]),
        "source_total": float(sum(geo["source_totals"].values())),
        "target_total": float(sum(geo["target_totals"].values())),
        "link_total": float(sum(l["weight"] for l in geo["links"])),
        "height": geo["height"],
        "min_flow": float(min_flow), "max_links": int(max_links),
    }
    return result


# ==========================================================================
# 8. 维诺树状图
# ==========================================================================
def _polygon_centroid(px: np.ndarray, py: np.ndarray) -> tuple[float, float]:
    """多边形面积与质心（鞋带公式）；退化时回退到顶点均值。"""
    n = px.size
    if n == 0:
        return 0.0, 0.0
    if n < 3:
        return float(px.mean()), float(py.mean())
    x2, y2 = np.roll(px, -1), np.roll(py, -1)
    cross = px * y2 - x2 * py
    a = 0.5 * float(cross.sum())
    if abs(a) < 1e-15:
        return float(px.mean()), float(py.mean())
    cx = float(((px + x2) * cross).sum()) / (6.0 * a)
    cy = float(((py + y2) * cross).sum()) / (6.0 * a)
    if not (math.isfinite(cx) and math.isfinite(cy)):
        return float(px.mean()), float(py.mean())
    return cx, cy


def _polygon_area(pts: np.ndarray) -> float:
    """鞋带公式求有向面积（逆时针为正）。"""
    arr = np.asarray(pts, dtype=float).reshape(-1, 2)
    if arr.shape[0] < 3:
        return 0.0
    x, y = arr[:, 0], arr[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    return float((x * y2 - x2 * y).sum() / 2.0)


def _order_ccw(pts: np.ndarray) -> np.ndarray:
    """把多边形顶点统一成逆时针顺序。

    **不要**按质心角度重排顶点：半平面裁剪（Sutherland–Hodgman）本身就保持
    顶点顺序，输出仍是凸多边形且首尾相接；按角度重排会在质心靠近某条边时
    打乱顺序、把多边形变成自交图形，使面积算错。这里只按有向面积决定是否
    整体反转顶点顺序，保证返回值恒为逆时针。
    """
    arr = np.asarray(pts, dtype=float).reshape(-1, 2)
    if arr.shape[0] < 3:
        return arr
    return arr[::-1] if _polygon_area(arr) < 0 else arr


def _clip_halfplane(pts: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """用半平面 ``a*x + b*y + c >= 0`` 裁剪凸多边形（Sutherland–Hodgman，向量化）。

    逐边判断两个端点是否在保留侧，跨越时插入交点；用 numpy 布尔掩码一次性
    拼出新顶点数组，避免 Python 逐点 append 造成的性能瓶颈
    （维诺剖分每轮要调用上万次）。
    """
    p = np.asarray(pts, dtype=float)
    n = p.shape[0]
    if n == 0:
        return p
    p2 = np.roll(p, -1, axis=0)
    f1 = a * p[:, 0] + b * p[:, 1] + c
    f2 = a * p2[:, 0] + b * p2[:, 1] + c
    keep1 = f1 >= 0.0
    cross = keep1 != (f2 >= 0.0)
    if not np.any(cross):
        return p[keep1]
    denom = f1 - f2
    safe = cross & (np.abs(denom) > 1e-15)
    if not np.any(safe):
        return p[keep1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(safe, f1 / np.where(safe, denom, 1.0), 0.0)
    inter = p + t[:, None] * (p2 - p)
    # 保持 Sutherland–Hodgman 的顶点顺序：每条边先放"保留的起点"、再放"出射交点"
    i_keep = np.nonzero(keep1)[0]
    i_cross = np.nonzero(safe)[0]
    order = np.concatenate([2 * i_keep, 2 * i_cross + 1])
    verts = p[i_keep] if i_keep.size else np.zeros((0, 2), dtype=float)
    pts_new = inter[i_cross] if i_cross.size else np.zeros((0, 2), dtype=float)
    stacked = np.vstack([verts, pts_new])
    return stacked[np.argsort(order, kind="stable")]


def _circle_chords(radius: float, precision: int) -> list[tuple[float, float, float]]:
    """预计算用正多边形逼近圆盘时每条弦对应的半平面 ``(a, b, c)``。

    多边形边数越多，圆盘边界越精确；160~240 边时面积误差 < 0.1%。
    """
    precision = max(24, int(precision))
    ang = np.linspace(0.0, 2.0 * np.pi, precision, endpoint=False)
    circ = np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])
    out: list[tuple[float, float, float]] = []
    for i in range(precision):
        x1, y1 = circ[i]
        x2, y2 = circ[(i + 1) % precision]
        a, b = float(y2 - y1), float(-(x2 - x1))
        c = -(a * x1 + b * y1)
        if c < 0:  # 保证圆心 (0,0) 落在内侧
            a, b, c = -a, -b, -c
        out.append((a, b, c))
    return out


def _clip_circle(pts: np.ndarray, precision: int = 240,
                 chords: list[tuple[float, float, float]] | None = None) -> np.ndarray:
    """用圆盘裁剪凸多边形（等价于与内接正多边形求交，再统一成逆时针）。"""
    if pts.shape[0] == 0:
        return pts
    if chords is None:
        chords = _circle_chords(1.0, precision)
    out = _order_ccw(pts)
    for a, b, c in chords:
        out = _clip_halfplane(out, a, b, c)
        if out.shape[0] == 0:
            break
    return _order_ccw(out)


def _sector_sites(targets: np.ndarray, radius: float,
                  rng: np.random.Generator) -> np.ndarray:
    """按目标面积划分扇形扇区，在每个扇区中线上放置初始站点。

    每个单元分到一个圆心角正比于其目标面积的扇区（这样初始剖分已经"面积∝权重"的
    角向近似），站点放在扇区中心线上的同一半径处；随后幂图迭代只需在半径方向
    上微调即可收敛。相比随机撒点，这种确定性初始化不会出现站点挤在圆心附近的退化。
    """
    n = targets.size
    frac = targets / targets.sum()
    # 扇区边界：按面积比例切分 2π（用累积和避免累积误差）
    edges = np.concatenate([[0.0], np.cumsum(frac)]) * 2.0 * math.pi
    mid = 0.5 * (edges[:-1] + edges[1:])
    # 站点半径略有差别 + 轻微抖动，避免所有站点共圆导致的对称退化
    rr = radius * (0.42 + 0.10 * np.sqrt(frac / frac.max()))
    rr = rr + rng.uniform(-0.01, 0.01, n) * radius
    return np.column_stack([rr * np.cos(mid), rr * np.sin(mid)])


def voronoi_treemap_geometry(weights: Sequence[float], n_iter: int = 200,
                             alpha: float = 1.0, radius: float = 1.0,
                             seed: int = 20240517,
                             clip_precision: int = 240,
                             tol: float = 0.005) -> dict:
    """加权 Voronoi 剖分：让每个单元的面积正比于给定权重。

    算法（幂图 + 迭代重加权 / Lloyd 松弛）：
      1. **扇形初始化**：按目标面积把圆盘切成 ``n`` 个圆心角正比于面积的扇区，
         站点放在各扇区中线上，因此初始剖分已具备正确的角向比例；
      2. 每轮用**幂图（power diagram）**剖分圆盘：站点 ``s_i`` 带权重 ``w_i``，
         单元 = 所有半平面
         ``2(s_j-s_i)·x + (|s_i|²-w_i) - (|s_j|²-w_j) >= 0`` 与圆盘的交集。
         这里对**所有其它站点**取半平面（而不是只取无权重 Delaunay 邻居）：
         权重一变幂图的邻接关系就会变，只取无权重邻居会漏掉约束，
         留下超大的错误单元、或把某些单元错误地挤空；
      3. 用鞋带公式算单元面积与质心，按面积误差更新权重
         ``w_i += alpha * (1 - target_i/area_i) * R²/(2|s_i|)``
         （面积偏小 → 权重抬高 → 单元扩张），并把站点移向质心；
      4. 重复直到最大相对面积误差 < ``tol`` 或达到 ``n_iter``。

    Returns:
        dict，含 ``cells``（单元→逆时针多边形顶点）、``weights``（最终幂图权重）、
        ``areas``、``targets``、``rel_error``、``iterations``、``converged``、
        ``error_history``、``radius``。

    Raises:
        ValueError: 权重数量不足 2、出现非正/非有限值、或权重差异过大无法收敛。
    """
    w = np.asarray([float(x) for x in weights], dtype=float)
    if w.size < 2:
        raise ValueError("维诺树状图至少需要 2 个分类单元，当前不足。")
    if not np.all(np.isfinite(w)):
        raise ValueError("维诺树状图的权重中出现 NaN / inf，请先清洗数据。")
    if np.any(w <= 0):
        raise ValueError("维诺树状图的权重必须全部为正数（面积为正）。")

    ratio = float(w.max() / w.min())
    if ratio > 5e4:
        top = [int(i) + 1 for i in np.argsort(-w)[:3]]
        small = [int(i) + 1 for i in np.argsort(w)[:3]]
        raise ValueError(
            f"维诺树状图的权重大小差异过大（最大/最小 = {ratio:.3g}）："
            f"最大单元序号 {top}、最小单元序号 {small}。"
            "面积正比于权重时极小权重会被几何精度吞掉。建议：对丰度先做 sqrt 或 "
            "log1p 变换、把极小丰度的单元合并为“其他”，或减少 top_n 让权重更接近同一量级。"
        )

    targets = w / w.sum()
    n = w.size
    rng = np.random.default_rng(seed)
    sites = _sector_sites(targets, radius, rng)

    # 权重初值：角向已按面积比例划分，半径方向先给一致权重起算
    wp = (sites ** 2).sum(axis=1) - 1.15 * (radius ** 2) * targets

    circle_area = math.pi * radius ** 2
    bbox = radius * 2.0
    boundary = np.asarray([[-bbox, -bbox], [bbox, -bbox], [bbox, bbox], [-bbox, bbox]],
                          dtype=float)
    chords = _circle_chords(radius, clip_precision)

    areas = np.zeros(n)
    polys: list[np.ndarray] = [np.zeros((0, 2))] * n
    history: list[float] = []
    it = 0

    for it in range(max(1, int(n_iter))):
        max_err = 0.0
        for i in range(n):
            si = sites[i]
            order = [int(j) for j in np.argsort(np.hypot(*(sites - si).T)) if j != i]
            poly = boundary
            for j in order:
                sj = sites[j]
                a = 2.0 * (sj[0] - si[0])
                b = 2.0 * (sj[1] - si[1])
                c = float((si @ si) - wp[i] - (sj @ sj) + wp[j])
                # 快速排除 1：整个圆盘都满足该半平面 → 约束不可能切到图形
                if c >= 0 and math.hypot(a, b) * radius <= c:
                    continue
                # 快速排除 2：当前多边形的顶点已全部满足该半平面
                if poly.shape[0] and float(max(a * poly[:, 0].max() + b * poly[:, 1].max(),
                                               a * poly[:, 0].min() + b * poly[:, 1].min())) + c <= 1e-12:
                    continue
                poly = _clip_halfplane(poly, a, b, c)
                if poly.shape[0] == 0:
                    break
            if poly.shape[0] >= 3:
                poly = _clip_circle(poly, clip_precision, chords)
            if poly.shape[0] < 3:
                polys[i] = np.zeros((0, 2))
                areas[i] = 0.0
                max_err = 1.0
                radius_i = float(np.hypot(si[0], si[1]))
                wp[i] += float(alpha) * (radius ** 2) / (2.0 * max(radius_i, 0.05 * radius))
                continue

            polys[i] = poly
            area = max(_polygon_area(poly), 0.0)
            areas[i] = area
            frac = area / circle_area
            max_err = max(max_err, float(abs(frac - targets[i]) / targets[i]))

            # 权重增量 δ 会让单元沿 |s_i| 方向扩张约 δ/(2|s_i|)，
            # 因此把面积相对误差换算成权重修正量。
            radius_i = float(np.hypot(si[0], si[1]))
            step_i = (radius ** 2) / (2.0 * max(radius_i, 0.15 * radius))
            wp[i] += float(alpha) * (1.0 - targets[i] / max(frac, 1e-9)) * step_i

        history.append(float(max_err))
        if max_err < tol:
            break

        # 所有单元都有效时才更新站点位置（有单元为空时先只调权重）
        if all(p.shape[0] >= 3 for p in polys):
            cents = np.array([_polygon_centroid(p[:, 0], p[:, 1]) for p in polys])
            bad = ~np.all(np.isfinite(cents), axis=1)
            if np.any(bad):
                cents[bad] = sites[bad]
            sites = sites + 0.7 * (cents - sites)
        wp = wp - float(np.mean(wp))  # 整体平移不改变剖分，仅用于数值稳定

    if not all(p.shape[0] >= 3 for p in polys):
        empty = [i + 1 for i, p in enumerate(polys) if p.shape[0] < 3]
        raise ValueError(
            f"维诺剖分未能为全部 {n} 个分类单元分配到面积（空单元序号 {empty[:8]}，"
            f"权重比 {ratio:.3g}）。建议：把极小丰度的单元合并为“其他”、"
            "对丰度做 sqrt / log1p 变换，或减少 top_n。"
        )

    total = float(np.sum(areas))
    rel_err = np.abs(areas / total - targets) / targets
    max_rel = float(np.max(rel_err))
    return {
        "cells": polys, "weights": wp, "areas": areas, "targets": targets,
        "rel_error": rel_err, "iterations": it + 1, "radius": float(radius),
        "total_area": total, "circle_area": circle_area,
        "max_rel_error": max_rel,
        "mean_rel_error": float(np.mean(rel_err)),
        "error_history": history,
        "converged": bool(max_rel < max(tol, 0.15)),
    }


def _voronoi_area_error(weights: Sequence[float], geometry: dict | None = None,
                        **kwargs) -> float:
    """自检辅助：返回维诺剖分的**面积比例最大相对误差**（越小越好）。

    Args:
        weights: 输入权重（丰度）。
        geometry: 已算好的 ``voronoi_treemap_geometry`` 结果；为 None 时现算。
        **kwargs: 传给 ``voronoi_treemap_geometry`` 的参数。

    Returns:
        最大相对误差（0.12 表示 12%）。
    """
    geo = geometry or voronoi_treemap_geometry(weights, **kwargs)
    return float(geo["max_rel_error"])


def plot_voronoi_treemap(rel_table: pd.DataFrame, out_path: str | Path, style: Style,
                         rank_label: str = "Feature", top_n: int = 22,
                         *, n_iter: int = 150) -> dict:
    """维诺树状图：把各分类单元的丰度映射为面积，用加权 Voronoi 剖分填满一个圆。

    Args:
        rel_table: 行=特征、列=样本的丰度表。
        out_path: 输出图片路径。
        style: 样式对象。
        rank_label: 分类级中文名称。
        top_n: 展示的分类单元数量。
        n_iter: 迭代重加权的最大轮数。

    Returns:
        ``save()`` 的结果字典，额外含 ``voronoi``（面积、误差、迭代信息）与 ``names``。

    Raises:
        ValueError: 分类单元不足 2 个、权重非正、或权重差异过大导致不收敛。
    """
    rel = _to_relative(rel_table)
    if rel.shape[0] == 0:
        raise ValueError("所选分类级下没有可用的特征。")
    mean_ab = rel.mean(axis=1).sort_values(ascending=False)
    mean_ab = mean_ab[mean_ab > 0]
    if mean_ab.size < 2:
        raise ValueError("维诺树状图至少需要 2 个非零丰度的分类单元。")

    taken = mean_ab.head(max(2, int(top_n)))
    names = [str(x) for x in taken.index]
    weights = taken.to_numpy(dtype=float)
    if weights.max() / max(weights.min(), 1e-300) > 5e4 and taken.size > 2:
        # 极端偏斜时按平方根压缩量纲，避免迭代不收敛；面积仍严格正比于压缩后的权重
        weights = np.sqrt(weights)

    try:
        geo = voronoi_treemap_geometry(weights, n_iter=n_iter)
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"维诺剖分失败：{exc}") from exc

    if geo["max_rel_error"] >= 0.15:
        raise ValueError(
            f"维诺剖分不收敛：面积比例最大误差 {geo['max_rel_error']:.1%} ≥ 15%，"
            "请减少分类单元数量或先压缩丰度量纲。"
        )

    cmap = plt.get_cmap(style.cmap("continuous"))
    n = len(names)
    frac = geo["targets"] / max(geo["targets"].max(), 1e-12)
    colors = [cmap(0.22 + 0.72 * float(f)) for f in frac]

    fig, ax = new_figure(style, 7.0, 6.6)
    ax.set_aspect("equal")
    circle = plt.Circle((0, 0), geo["radius"], facecolor="#F7F7F7",
                        edgecolor="#9A9A9A", linewidth=0.9, zorder=1)
    ax.add_patch(circle)

    shown = 0
    for i, nm in enumerate(names):
        poly = geo["cells"][i]
        if poly is None or poly.shape[0] < 3:
            continue
        ax.add_patch(Polygon(poly, closed=True, facecolor=colors[i], edgecolor="white",
                             linewidth=0.9, alpha=0.95, zorder=2))
        shown += 1
        area_frac = float(geo["areas"][i]) / max(geo["total_area"], 1e-12)
        if area_frac >= 0.035 and style.show_text("annotation"):
            cx, cy = _polygon_centroid(poly[:, 0], poly[:, 1])
            ax.annotate(f"{nm[:16]}\n{area_frac:.1%}", (cx, cy), ha="center", va="center",
                        **text_style_kwargs(style, "annotation"), zorder=6)

    lim = geo["radius"] * 1.12
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis("off")
    place_title(ax, style, f"{rank_label}丰度维诺树状图（前 {shown} 个）")
    place_annotation(ax, f"面积误差 ≤ {geo['max_rel_error']:.1%}", style,
                     x=0.02, y=0.02, ha="left", va="bottom")

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=mcolors.Normalize(vmin=float(weights.min()),
                                                      vmax=float(weights.max())))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.02, shrink=0.7)
    cbar.set_label(f"{rank_label}平均相对丰度", fontsize=style.font_size("legend_text"),
                   color=style.font_color("legend_text"))
    cbar.ax.tick_params(labelsize=style.font_size("tick_label") * 0.85)
    fig.tight_layout()
    result = save(
        fig, out_path, style,
        f"维诺树状图用不规则多边形铺满圆盘，每个多边形的**面积正比于该 {rank_label} 的平均相对丰度**"
        "（多边形越大多表示该单元越占优势）。颜色深浅同样对应丰度，"
        "标注文字给出名称与占比。它比饼图更容易比较大量分类单元的相对份额，"
        f"并且不依赖矩形切分。本图剖分迭代 {geo['iterations']} 轮，"
        f"面积比例与权重比例的平均误差 {geo['mean_rel_error']:.1%}、最大误差 {geo['max_rel_error']:.1%}。",
        f"{rank_label}丰度维诺树状图",
    )
    result["names"] = names
    result["weights"] = [float(x) for x in weights]
    result["voronoi"] = {
        "area_rel_error_mean": float(geo["mean_rel_error"]),
        "area_rel_error_max": float(geo["max_rel_error"]),
        "iterations": int(geo["iterations"]),
        "areas": [float(a) for a in geo["areas"]],
        "n_cells": shown,
    }
    return result


# ==========================================================================
# 9. 分组 → 分类单元的流向图（组均值版）
# ==========================================================================
def plot_stacked_flow(groups: pd.Series, composition: pd.DataFrame, out_path: str | Path,
                      style: Style, rank_label: str = "Feature", top_n: int = 10,
                      *, max_links: int = 60) -> dict:
    """分组 → 分类单元的桑基/流向图（组均值版），比通用桑基图更贴近生信场景。

    左层为分组，右层为分类单元（行=分类单元、列=样本的丰度表），
    连边宽度 = 该分类单元在该组内的平均相对丰度，因此每个分组的总流出量恒等于 1。

    Args:
        groups: index=样本名、值为分组的 Series。
        composition: 行=分类单元、列=样本的丰度表。
        out_path: 输出图片路径。
        style: 样式对象。
        rank_label: 分类级中文名称。
        top_n: 右层保留的分类单元数（其余合并为 "其他"）。
        max_links: 最多绘制的连边数。

    Returns:
        ``save()`` 的结果字典，额外含 ``flow``（流向表）与 ``sankey``（守恒信息）。

    Raises:
        ValueError: 分组数不足 2、样本无法配对或没有非零丰度的分类单元。
    """
    rel = _to_relative(composition)
    mat, meta = _paired(rel, groups)
    if meta.nunique() < 2:
        raise ValueError("至少需要 2 个分组才能绘制流向图，当前只有 1 组。")
    order = _group_order(meta)

    mean_by_group = pd.DataFrame({
        g: mat.loc[meta == g].mean(axis=0) for g in order
    }).fillna(0.0)
    means = mean_by_group.mean(axis=1).sort_values(ascending=False)

    top = means.head(max(1, int(top_n)))
    rows: list[dict] = []
    for g in order:
        for feat in top.index:
            rows.append({"source": g, "target": str(feat), "weight": float(mean_by_group.loc[feat, g])})
        rest = float(mean_by_group.loc[~mean_by_group.index.isin(top.index), g].sum())
        if rest > 1e-12:
            rows.append({"source": g, "target": "其他", "weight": rest})
    flow = pd.DataFrame(rows)
    flow = flow[flow["weight"] > 0]
    if flow.empty:
        raise ValueError("各组平均丰度全为 0，无法绘制流向图。")

    clean = _clean_flow(flow)
    geo = _sankey_geometry(clean, min_flow=0.0, max_links=max_links,
                           gap_frac=0.03, scale_frac=0.94, thickness_frac=0.022)
    nodes = geo["nodes"]
    target_names = geo["targets"]

    # 分组用分组色板，分类单元用连续色图（与组成分析保持一致的语义）
    group_colors = style.group_colors(geo["sources"])
    cmap = plt.get_cmap(style.cmap("continuous"))
    feat_colors = {nm: cmap(0.18 + 0.70 * (i / max(len(target_names) - 1, 1)))
                   for i, nm in enumerate(target_names)}
    colors = {**group_colors, **feat_colors}

    fig, ax = new_figure(style, 8.8, 6.0)
    alpha = 0.40 * style.num("palette.alpha", 1.0)
    for link in geo["links"]:
        ax.add_patch(_link_patch(link["vertices"], colors.get(link["target"], "#999999"),
                                 alpha, zorder=2))

    edge = "white" if style.flag("element.bar_edge", True) else "none"
    for layer, names in (("source", geo["sources"]), ("target", geo["targets"])):
        for nm in names:
            nd = nodes[nm]
            if nd["height"] <= 1e-9:
                continue
            ax.add_patch(plt.Rectangle((nd["x0"], nd["y0"]),
                                       max(nd["x1"] - nd["x0"], 1e-4), nd["height"],
                                       facecolor=colors.get(nm, "#999999"),
                                       edgecolor=edge, linewidth=0.6, zorder=4))
            share = (geo["target_totals"].get(nm, 0.0) / max(len(geo["sources"]), 1)
                     if layer == "target" else 1.0)
            label = f"{nm[:30]}  {share:.1%}" if layer == "target" else str(nm)
            if nd["height"] >= 0.045 and style.show_text("annotation"):
                ax.annotate(label,
                            xy=(nd["x0"] if layer == "source" else nd["x1"],
                                (nd["y0"] + nd["y1"]) / 2.0),
                            xytext=(-6 if layer == "source" else 6, 0),
                            textcoords="offset points",
                            ha="right" if layer == "source" else "left", va="center",
                            **text_style_kwargs(style, "annotation"), zorder=6)

    y_lo = min((n["y0"] for n in nodes.values() if n["height"] > 1e-9), default=0.0)
    y_hi = max((n["y1"] for n in nodes.values() if n["height"] > 1e-9), default=1.0)
    pad = 0.06 * max(y_hi - y_lo, 1e-6)
    ax.set_xlim(-0.34, 1.34)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.axis("off")
    place_title(ax, style, f"分组 → {rank_label} 组成流向图")
    fig.tight_layout()
    result = save(
        fig, out_path, style,
        f"流向图左侧是分组、右侧是优势 {rank_label}，连边宽度等于该分类单元在该组内的平均相对丰度。"
        "因为组内各分类单元占比之和为 1，每个分组流出的总宽度都相同，"
        "所以右侧某个节点越粗，说明它在所有分组中整体越占优势；"
        "观察同一个右侧节点从左层不同分组流入的带宽比例，即可看出该分类单元的组间偏好。",
        f"分组 {rank_label} 流向图",
    )
    result["flow"] = clean
    result["sankey"] = {
        "n_links": len(geo["links"]),
        "source_total": float(sum(geo["source_totals"].values())),
        "target_total": float(sum(geo["target_totals"].values())),
        "link_total": float(sum(l["weight"] for l in geo["links"])),
        "height": geo["height"],
    }
    return result


# ==========================================================================
# 自检
# ==========================================================================
def _make_demo_data(n_samples: int = 30, n_features: int = 60, n_groups: int = 3,
                    seed: int = 20240517) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """构造合成丰度表 + 分组 + 差异分析结果表。

    丰度用对数正态分布生成（模拟微生物组/转录组的偏态丰度），
    并给前 3 个特征叠加组效应，使小提琴/密度/蜂群图有肉眼可见的组间差异。

    Args:
        n_samples: 样本数。
        n_features: 特征数。
        n_groups: 分组数。
        seed: 随机种子。

    Returns:
        (丰度表, 分组 Series, 差异分析结果表)，丰度表行=特征、列=样本。
    """
    rng = np.random.default_rng(seed)
    names = [f"F{i + 1:03d}" for i in range(n_features)]
    samples = [f"S{i + 1:02d}" for i in range(n_samples)]
    group_names = [f"Group{_string.ascii_uppercase[i]}" for i in range(n_groups)]
    groups = pd.Series(
        [group_names[i % n_groups] for i in range(n_samples)], index=samples, name="Group"
    )

    base = rng.lognormal(mean=0.4, sigma=1.15, size=(n_features, n_samples))
    for f in range(3):
        for gi, g in enumerate(group_names):
            cols = [j for j, s in enumerate(samples) if groups[s] == g]
            base[f, cols] *= 1.0 + 0.9 * gi
    for f in range(3, 12):
        g = group_names[f % n_groups]
        cols = [j for j, s in enumerate(samples) if groups[s] == g]
        base[f, cols] *= 1.7
    table = pd.DataFrame(base, index=names, columns=samples)

    # ---- 差异分析结果表 ----
    a_cols = [s for s in samples if groups[s] == group_names[0]]
    b_cols = [s for s in samples if groups[s] == group_names[1]]
    mean_a = table[a_cols].mean(axis=1)
    mean_b = table[b_cols].mean(axis=1)
    log2fc = np.log2((mean_a + 1e-6) / (mean_b + 1e-6))
    effect = np.linspace(1.8, 0.02, n_features)
    pvals = np.clip(np.abs(log2fc) * 0.0 + effect * rng.uniform(0.4, 1.0, n_features), 1e-12, 1.0)
    diff = pd.DataFrame({
        "feature": names,
        "log2FC": log2fc.to_numpy(dtype=float),
        "p": pvals,
        "FDR": np.clip(pvals * n_features / max(n_features, 1), 1e-12, 1.0),
        "lfcSE": rng.uniform(0.05, 0.4, n_features),
        "mean_rel_A": mean_a.to_numpy(dtype=float),
        "mean_rel_B": mean_b.to_numpy(dtype=float),
        "baseMean": ((mean_a + mean_b) / 2).to_numpy(dtype=float),
        "enriched_in": [group_names[0] if v > 0 else group_names[1] for v in log2fc],
    })
    return table, groups, diff


def _make_demo_sankey() -> pd.DataFrame:
    """构造一张三层的合成流向表（属 → 属 → 功能），用于自检桑基图。"""
    rng = np.random.default_rng(7)
    phyla = ["Firmicutes", "Bacteroidetes", "Proteobacteria", "Actinobacteria"]
    genera = ["Bacteroides", "Prevotella", "Faecalibacterium", "Escherichia",
              "Clostridium", "Bifidobacterium"]
    funcs = ["SCFA 合成", "碳代谢", "氮代谢", "耐药", "其他功能"]
    rows: list[dict] = []
    for p in phyla:
        for g in genera:
            w = float(rng.uniform(0.2, 1.0))
            if rng.random() < 0.55:
                rows.append({"source": p, "target": g, "weight": w})
    for g in genera:
        for fn in funcs:
            if rng.random() < 0.5:
                rows.append({"source": g, "target": fn, "weight": float(rng.uniform(0.2, 1.0))})
    return pd.DataFrame(rows)


def _image_info(path: Path) -> tuple[tuple[int, int], int]:
    """读取 PNG 的像素尺寸与字节数（不依赖 Pillow）。"""
    data = path.read_bytes()
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path.name} 不是有效的 PNG 文件。")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return (width, height), len(data)


def _swarm_min_spacing(rel_table: pd.DataFrame, groups: pd.Series, style: Style,
                       top_n: int = 9) -> float:
    """独立复算蜂群图的横坐标最小间距 / 点直径，用于自检（值越接近 1 越紧凑）。"""
    rel = _to_relative(rel_table)
    mat, feats = _top_features(rel, groups, top_n)
    meta = mat.index.to_series().map(groups.astype(str))
    order = _group_order(meta)
    size = style.num("element.point_size", 68)
    panel_h = 3.0 * style.num("figure.scale", 1.0)
    y_span = 1.06 * float(np.ptp(mat[feats].to_numpy(dtype=float)) or 1.0)
    x_span = 0.5 * 2.0 + 4.0
    rad_pt = 0.5 * math.sqrt(max(1.0, size))
    radius_norm = rad_pt / 300.0
    scale = (x_span * radius_norm * 2.0 / 2.0, y_span / (panel_h * 72.0 / 300.0))
    worst = float("inf")
    for k, feat in enumerate(feats):
        rng = np.random.default_rng(20240518 + k)
        for g in order:
            vals = np.asarray(mat.loc[meta == g, feat].to_numpy(dtype=float), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size < 2:
                continue
            offs = _swarm_offsets(vals, radius=radius_norm, gap=radius_norm * 0.25,
                                  scale=scale, rng=rng)
            xs = offs * 2.0
            for a in range(vals.size):
                for b in range(a + 1, vals.size):
                    dx = (xs[a] - xs[b]) / scale[0]
                    dy = (vals[a] - vals[b]) / scale[1]
                    worst = min(worst, math.hypot(dx, dy) / (2.0 * radius_norm))
    return float(worst)


def _selftest() -> int:
    """自检主流程：生成合成数据 → 调用全部 8 类图 → 校验硬指标 → 打印报告。"""
    style = Style()
    style.update({"figure.dpi": 150, "palette.preset": "科研配色1"})
    out_dir = Path(r"D:\wjw\dsh\preview")
    out_dir.mkdir(parents=True, exist_ok=True)

    table, groups, diff = _make_demo_data()
    print("=" * 74)
    print("高级图形模块自检")
    print("=" * 74)
    print(f"合成丰度表：{table.shape[0]} 特征 × {table.shape[1]} 样本；"
          f"分组：{dict(groups.value_counts())}")
    data_csv = out_dir / "advanced_demo_abundance.csv"
    table.round(6).to_csv(data_csv, encoding="utf-8-sig")
    diff_csv = out_dir / "advanced_demo_diff.csv"
    diff.to_csv(diff_csv, index=False, encoding="utf-8-sig")

    # ---------- 维诺树状图的面积比例硬指标 ----------
    rel = _to_relative(table)
    weights = rel.mean(axis=1).sort_values(ascending=False).head(22)
    geo = voronoi_treemap_geometry(weights.to_numpy(dtype=float), n_iter=150)
    err = _voronoi_area_error(weights.to_numpy(dtype=float), geo)
    print(f"\n[硬指标] 维诺剖分：{weights.size} 个单元，迭代 {geo['iterations']} 轮；"
          f"平均面积误差 {geo['mean_rel_error']:.3%}，最大误差 {err:.3%}")
    print("        逐单元 目标占比 vs 实际占比：")
    for nm, tg, ar in zip(weights.index, geo["targets"], geo["areas"]):
        print(f"          {str(nm)[:12]:<12} 目标 {tg:7.3%}  实际 {ar / geo['total_area']:7.3%}"
              f"  相对误差 {abs(ar / geo['total_area'] - tg) / tg:7.2%}")
    if err >= 0.15:
        raise ValueError(f"维诺面积比例误差 {err:.3%} 超过 15% 的硬指标。")

    # ---------- 桑基图流量守恒 ----------
    flow3 = _make_demo_sankey()
    print(f"\n[硬指标] 桑基图输入：{len(flow3)} 条连边，"
          f"{flow3['source'].nunique()} 个源节点 / {flow3['target'].nunique()} 个目标节点")
    clean = _clean_flow(flow3)
    s_total = float(clean.groupby("source")["weight"].sum().sum())
    t_total = float(clean.groupby("target")["weight"].sum().sum())
    print(f"        源侧总流量 = {s_total:.10f}；目标侧总流量 = {t_total:.10f}；"
          f"差值 = {abs(s_total - t_total):.3e}")

    # ---------- 蜂群图最小间距 ----------
    ratio = _swarm_min_spacing(table, groups, style, top_n=9)
    print(f"\n[硬指标] 蜂群图最小圆心距 / 点直径 = {ratio:.4f}（要求 ≥ 1.0）")
    if ratio < 1.0:
        raise ValueError(f"蜂群图最小间距 {ratio:.4f} 小于点半径要求。")

    # ---------- 逐图调用 ----------
    outputs: list[tuple[str, Path]] = []

    def _run(label: str, func, *args, **kwargs) -> dict:
        res = func(*args, **kwargs)
        p = Path(res["path"])
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"{label} 未产出有效文件：{p}")
        outputs.append((label, p))
        size, nbytes = _image_info(p)
        print(f"  ✓ {label:<22} {p.name:<34} {size[0]}×{size[1]} px  {nbytes:,} B")
        return res

    print("\n[1/9] 小提琴图 …")
    v = _run("plot_violin", plot_violin, table, groups, out_dir / "advanced_violin.png",
             style, rank_label="Genus", top_n=9)
    print(f"        面板特征：{v['features']}")

    print("[2/9] 云雨图 …")
    _run("plot_raincloud", plot_raincloud, table, groups, out_dir / "advanced_raincloud.png",
         style, rank_label="Genus", top_n=6)

    print("[3/9] 密度图 …")
    _run("plot_density", plot_density, table, groups, out_dir / "advanced_density.png",
         style, rank_label="Genus", top_n=6)

    print("[4/9] 蜂群图 …")
    b = _run("plot_beeswarm", plot_beeswarm, table, groups, out_dir / "advanced_beeswarm.png",
             style, rank_label="Genus", top_n=9)
    print(f"        图中实测最小间距 / 点直径 = {b['swarm_min_ratio']:.4f}")

    print("[5/9] 棒棒糖图 …")
    _run("plot_lollipop", plot_lollipop, diff, out_dir / "advanced_lollipop.png", style,
         value_col="log2FC", label_col="feature", group_a="GroupA", group_b="GroupB", top_n=20)

    print("[6/9] 气泡图 …")
    _run("plot_bubble", plot_bubble, diff, out_dir / "advanced_bubble.png", style,
         x_col="log2FC", y_col="baseMean", size_col="FDR",
         color_col="enriched_in", label_col="feature", top_n=40, title="差异特征气泡图")
    _run("plot_bubble(连续色)", plot_bubble, diff, out_dir / "advanced_bubble_cont.png", style,
         x_col="log2FC", y_col="baseMean", size_col="baseMean",
         color_col="lfcSE", label_col="feature", top_n=30, title="差异特征气泡图（连续配色）")

    print("[7/9] 桑基图 …")
    sk = _run("plot_sankey", plot_sankey, flow3, out_dir / "advanced_sankey.png", style,
              title="分类层级流向桑基图", min_flow=0.0, max_links=60)
    info = sk["sankey"]
    print(f"        节点 {info['n_nodes']} 个 / 连边 {info['n_links']} 条；"
          f"源总量 {info['source_total']:.10f} == 目标总量 {info['target_total']:.10f}"
          f"（差 {abs(info['source_total'] - info['target_total']):.3e}）；"
          f"节点高度 源 {info['height']['source']:.6f} / 目标 {info['height']['target']:.6f}"
          f" / 连边厚度和 {info['height']['thickness']:.6f}")
    if abs(info["source_total"] - info["target_total"]) > 1e-9 * max(info["source_total"], 1.0):
        raise ValueError("桑基图流量不守恒：源总量与目标总量不相等。")
    if abs(info["height"]["source"] - info["height"]["thickness"]) > 1e-9:
        raise ValueError("桑基图节点高度与连边总厚度不一致。")

    print("[8/9] 维诺树状图 …")
    vt = _run("plot_voronoi_treemap", plot_voronoi_treemap, table,
              out_dir / "advanced_voronoi.png", style, rank_label="Genus", top_n=22)
    print(f"        单元 {vt['voronoi']['n_cells']} 个，迭代 {vt['voronoi']['iterations']} 轮，"
          f"面积最大误差 {vt['voronoi']['area_rel_error_max']:.3%}")
    if vt["voronoi"]["area_rel_error_max"] >= 0.15:
        raise ValueError("维诺树状图面积比例误差超出 15% 的硬指标。")

    print("[9/9] 分组 → 分类单元流向图 …")
    sf = _run("plot_stacked_flow", plot_stacked_flow, groups, table,
              out_dir / "advanced_stacked_flow.png", style, rank_label="Genus", top_n=10)
    sinfo = sf["sankey"]
    print(f"        连边 {sinfo['n_links']} 条；源总量 {sinfo['source_total']:.10f} == "
          f"目标总量 {sinfo['target_total']:.10f}（差 "
          f"{abs(sinfo['source_total'] - sinfo['target_total']):.3e}）")
    if abs(sinfo["source_total"] - sinfo["target_total"]) > 1e-9:
        raise ValueError("分组流向图流量不守恒。")

    # ---------- 退化输入必须抛中文 ValueError ----------
    print("\n[退化输入检查]")
    cases = [
        ("只有 1 组", lambda: plot_violin(table, pd.Series("G", index=table.columns),
                                          out_dir / "_x.png", style)),
        ("特征不足（维诺）", lambda: plot_voronoi_treemap(
            table.head(1), out_dir / "_x.png", style, top_n=22)),
        ("缺列（棒棒糖）", lambda: plot_lollipop(diff.drop(columns=["log2FC"]),
                                                 out_dir / "_x.png", style)),
        ("空流向表", lambda: plot_sankey(pd.DataFrame({"source": [], "target": []}),
                                         out_dir / "_x.png", style)),
        ("权重差异过大（维诺）", lambda: plot_voronoi_treemap(
            pd.DataFrame(np.array([[1e-9], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]) * 1e-8,
                         index=[f"K{i}" for i in range(8)], columns=["S1"]),
            out_dir / "_x.png", style, top_n=8)),
    ]
    for name, fn in cases:
        try:
            fn()
        except ValueError as exc:
            msg = str(exc)
            if not any("\u4e00" <= ch <= "\u9fff" for ch in msg):
                raise AssertionError(f"退化输入 “{name}” 的报错不是中文：{msg}") from exc
            print(f"  ✓ {name:<20} → ValueError：{msg}")
        else:
            raise AssertionError(f"退化输入 “{name}” 未抛出 ValueError。")

    # ---------- 汇总 ----------
    print("\n" + "=" * 74)
    print(f"自检通过：{len(outputs)} 张 PNG 全部生成，无 NaN 坐标（导出前逐图校验）。")
    print(f"输出目录：{out_dir}")
    for label, p in outputs:
        size, nbytes = _image_info(p)
        print(f"  {label:<26} {p.name:<36} {size[0]}×{size[1]} px {nbytes:>10,} B")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    import sys
    import warnings

    warnings.filterwarnings("error", category=UserWarning, module="matplotlib")
    sys.exit(_selftest())
