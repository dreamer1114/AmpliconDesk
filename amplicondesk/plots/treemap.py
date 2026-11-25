"""面积正比于权重的维诺树状图（Voronoi treemap）。

采用**乘方图（power diagram / Laguerre 剖分）**而不是普通 Voronoi：
乘方图的胞腔是若干半平面的交，可以做多边形裁剪快速精确求解；
而且理论保证——站点位置固定时，通过调节各站点的权重，
胞腔面积向量可以覆盖整个单纯形，也就是说一定能达到任意指定的面积比例。

算法：
  1. 在一个正多边形（近似圆）边界内撒点，先做几轮无权 Lloyd 松弛把站点摊开；
  2. 交替执行
       a. 用当前权重建乘方图，精确计算各胞腔面积；
       b. 按相对面积误差调整权重（带回溯线搜索，误差不降就缩步长）；
       c. 让站点朝各自胞腔质心移动一小步（改善形状，避免出现细长胞腔）。
  3. 达到目标误差或迭代上限即停。

纯 numpy 实现，不依赖 scipy.spatial，速度在毫秒级。
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..style import Style
from .base import new_figure, place_annotation, place_title, resolve_legend, save, style_axes

__all__ = [
    "power_diagram", "voronoi_treemap_geometry", "plot_voronoi_treemap",
    "plot_treemap_fallback", "plot_treemap_auto",
]


def plot_treemap_auto(rel_table: pd.DataFrame, out_path: str | Path, style: Style,
                      rank_label: str = "Feature", top_n: int = 18, **kw) -> dict:
    """优先画维诺树状图；不收敛时自动改用矩形树图。

    两种图的**面积都严格正比于平均相对丰度**，区别只在外观：
    维诺是自然的有机多边形，矩形是规整的方块。单元数很多或丰度极其悬殊时，
    维诺剖分在几何上难以收敛，此时矩形树图是更稳妥的表达。
    """
    try:
        return plot_voronoi_treemap(rel_table, out_path, style,
                                    rank_label=rank_label, top_n=top_n, **kw)
    except Exception as exc:
        info = plot_treemap_fallback(rel_table, out_path, style,
                                     rank_label=rank_label, top_n=top_n)
        info["fallback_from"] = "voronoi"
        info["fallback_reason"] = str(exc)
        info["caption"] = (
            f"（维诺剖分未能收敛，已自动改用矩形树图——面积同样严格正比于平均相对丰度。"
            f"原因：{exc}）" + info.get("caption", "")
        )
        info["title"] = f"矩形树图（{rank_label}）"
        return info


# --------------------------------------------------------------------------
# 几何基础
# --------------------------------------------------------------------------
def _regular_polygon(n: int, radius: float = 1.0) -> np.ndarray:
    """正 n 边形（近似圆形边界），顶点逆时针。"""
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False) + math.pi / n
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])


def _clip_halfplane(poly: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Sutherland–Hodgman：保留 a*x + b*y <= c 的一侧。"""
    if len(poly) == 0:
        return poly
    out: list[np.ndarray] = []
    n = len(poly)
    vals = a * poly[:, 0] + b * poly[:, 1] - c
    for i in range(n):
        j = (i + 1) % n
        vi, vj = vals[i], vals[j]
        pi, pj = poly[i], poly[j]
        if vi <= 0:
            out.append(pi)
            if vj > 0:
                t = vi / (vi - vj)
                out.append(pi + t * (pj - pi))
        elif vj <= 0:
            t = vi / (vi - vj)
            out.append(pi + t * (pj - pi))
    return np.array(out) if out else np.empty((0, 2))


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _polygon_centroid(poly: np.ndarray) -> np.ndarray:
    if len(poly) < 3:
        return poly.mean(axis=0) if len(poly) else np.zeros(2)
    x, y = poly[:, 0], poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-14:
        return poly.mean(axis=0)
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * a)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6 * a)
    return np.array([cx, cy])


def power_diagram(sites: np.ndarray, weights: np.ndarray,
                  boundary: np.ndarray) -> list[np.ndarray]:
    """乘方图剖分。返回每个站点的胞腔多边形（可能为空）。

    站点 i 的胞腔 = {p : |p-s_i|² - w_i ≤ |p-s_j|² - w_j, ∀j}，
    等价于一族半平面 2(s_j-s_i)·p ≤ |s_j|² - w_j - |s_i|² + w_i 与边界的交。
    """
    n = len(sites)
    cells: list[np.ndarray] = []
    sq = (sites ** 2).sum(axis=1)
    for i in range(n):
        poly = boundary
        si, wi, sqi = sites[i], weights[i], sq[i]
        for j in range(n):
            if j == i:
                continue
            a = 2.0 * (sites[j, 0] - si[0])
            b = 2.0 * (sites[j, 1] - si[1])
            c = sq[j] - weights[j] - sqi + wi
            poly = _clip_halfplane(poly, a, b, c)
            if len(poly) == 0:
                break
        cells.append(poly)
    return cells


# --------------------------------------------------------------------------
# 求解面积正比于权重
# --------------------------------------------------------------------------
def voronoi_treemap_geometry(weights, n_iter: int = 120, tol: float = 0.01,
                             boundary_sides: int = 220, seed: int = 42,
                             time_budget: float | None = None) -> dict:
    """求出面积正比于 weights 的乘方图剖分。

    返回 dict: sites / weights / cells / areas / targets / max_rel_error / iterations / radius
    """
    import time

    w = np.asarray([float(x) for x in weights], dtype=float)
    if w.size < 2:
        raise ValueError("维诺树状图至少需要 2 个分类单元。")
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("权重必须都是正数（请先去掉丰度为 0 的分类单元）。")

    target = w / w.sum()
    radius = 1.0
    boundary = _regular_polygon(boundary_sides, radius)
    total_area = _polygon_area(boundary)

    rng = np.random.default_rng(seed)
    n = w.size

    # ---- 初始站点：同心环 + 抖动，然后做几轮无权 Lloyd 松弛 ----
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rad = radius * 0.55 * np.sqrt(target / target.max())
    sites = np.column_stack([rad * np.cos(angles), rad * np.sin(angles)])
    sites += rng.normal(0, 0.01 * radius, sites.shape)

    zero_w = np.zeros(n)
    for _ in range(12):
        cells = power_diagram(sites, zero_w, boundary)
        new_sites = sites.copy()
        for i, cell in enumerate(cells):
            if len(cell) >= 3:
                new_sites[i] = _polygon_centroid(cell)
        sites = 0.5 * sites + 0.5 * new_sites

    # 把差别大的权重转成初始权重的量级
    scale = total_area / n
    weights_v = np.zeros(n)
    best_err = float("inf")
    best = None
    step = 0.6 * scale
    started = time.time()

    for it in range(max(1, int(n_iter))):
        if time_budget is not None and (time.time() - started) > time_budget:
            break
        cells = power_diagram(sites, weights_v, boundary)
        areas = np.array([_polygon_area(c) for c in cells], dtype=float)
        tot = areas.sum()
        if tot <= 0:
            raise ValueError("乘方图剖分失败：所有胞腔面积为零，请检查权重。")

        rel = areas / tot
        err = np.abs(rel - target) / target
        max_err = float(err.max())
        if max_err < best_err:
            best_err = max_err
            best = (sites.copy(), weights_v.copy(), [c.copy() for c in cells],
                    areas.copy())

        if max_err < tol:
            break

        # 有站点被完全挤掉时温和补权（不打断正常迭代，否则会破坏收敛）
        base_w = weights_v
        dead = areas <= 1e-12
        if dead.any():
            base_w = weights_v.copy()
            base_w[dead] += 0.35 * scale
            base_w -= base_w.mean()

        # ---- 权重更新：面积偏小就加大权重（带回溯）----
        trial_w = base_w + step * (target - rel) / np.maximum(target, 1e-9)
        trial_w -= trial_w.mean()
        trial_cells = power_diagram(sites, trial_w, boundary)
        trial_areas = np.array([_polygon_area(c) for c in trial_cells], dtype=float)
        tt = trial_areas.sum()
        if tt <= 0:
            step *= 0.5
            continue
        trial_rel = trial_areas / tt
        trial_err = float((np.abs(trial_rel - target) / target).max())
        if trial_err < max_err:
            weights_v = trial_w
            step *= 1.05
        else:
            step *= 0.5
            if step < 1e-9:
                break

        # ---- 站点朝胞腔质心微调，改善形状 ----
        if it % 2 == 1:
            for i, cell in enumerate(cells):
                if len(cell) >= 3:
                    sites[i] = 0.85 * sites[i] + 0.15 * _polygon_centroid(cell)

    if best is None:  # pragma: no cover
        raise ValueError("维诺剖分未能得到有效结果。")

    sites_b, weights_b, cells_b, areas_b = best
    tot = areas_b.sum()
    rel = areas_b / tot if tot > 0 else np.full(n, 1.0 / n)
    return {
        "sites": sites_b,
        "weights": weights_b,
        "cells": cells_b,
        "areas": areas_b / max(tot, 1e-300),
        "targets": target,
        "max_rel_error": float((np.abs(rel - target) / target).max()),
        "iterations": it + 1,
        "radius": radius,
        "total_area": total_area,
    }


# --------------------------------------------------------------------------
# 绘图
# --------------------------------------------------------------------------
def plot_voronoi_treemap(rel_table: pd.DataFrame, out_path: str | Path, style: Style,
                         rank_label: str = "Feature", top_n: int = 18,
                         *, n_iter: int = 120, max_error: float = 0.12) -> dict:
    """维诺树状图：各分类单元的丰度映射为面积占比。"""
    from .composition import _to_relative

    rel = _to_relative(rel_table)
    if rel.shape[0] == 0:
        raise ValueError("所选分类级下没有可用的特征。")

    mean_ab = rel.mean(axis=1).sort_values(ascending=False)
    mean_ab = mean_ab[mean_ab > 0]
    if mean_ab.size < 2:
        raise ValueError("维诺树状图至少需要 2 个非零丰度的分类单元。")

    taken = mean_ab.head(max(2, int(top_n)))
    names = [str(x) for x in taken.index]
    w = taken.to_numpy(dtype=float)

    # 丰度极度悬殊时按平方根压缩，否则会出现宽度接近 0 的细长胞腔
    compressed = False
    if w.max() / max(w.min(), 1e-300) > 300 and w.size > 2:
        w = np.sqrt(w)
        compressed = True

    geo = voronoi_treemap_geometry(w, n_iter=n_iter)
    if geo["max_rel_error"] > max_error:
        raise ValueError(
            f"维诺剖分未能收敛：最大面积误差 {geo['max_rel_error']:.1%}（阈值 {max_error:.0%}）。"
            f"分类单元数 {w.size}，权重最大比 {w.max() / w.min():.1f}。"
            "建议减少分类单元数量，或选择丰度更均匀的分类级。"
        )

    targets = geo["targets"]
    frac = targets / targets.max()
    cmap = plt.get_cmap(style.cmap("continuous"))
    colors = [cmap(0.18 + 0.74 * float(f)) for f in frac]

    fig, ax = new_figure(style, 7.2, 6.8)
    ax.set_aspect("equal")

    boundary = _regular_polygon(220, geo["radius"])
    ax.fill(boundary[:, 0], boundary[:, 1], color="#FAFAFA", zorder=0)
    line_w = style.num("element.line_width", 1.8) * 0.6

    for i, cell in enumerate(geo["cells"]):
        if len(cell) < 3:
            continue
        ax.fill(cell[:, 0], cell[:, 1], facecolor=colors[i],
                edgecolor="white", linewidth=line_w, zorder=2)
        cx, cy = _polygon_centroid(cell)
        area = geo["areas"][i]
        # 只在胞腔够大时写标签，避免文字糊成一团
        if area >= 0.022:
            label = names[i]
            if len(label) > 16:
                label = label[:15] + "…"
            size = style.font_size("annotation") * (0.72 + 0.55 * math.sqrt(area / max(geo["areas"].max(), 1e-9)))
            ax.text(cx, cy, f"{label}\n{area * 100:.1f}%", ha="center", va="center",
                    fontsize=max(5.0, size), color="#1B2430", zorder=4,
                    fontfamily=style.font_stack("annotation"))

    ax.set_xlim(-1.22, 1.22)
    ax.set_ylim(-1.22, 1.22)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)

    place_title(ax, style, f"维诺树状图（{rank_label}，前 {len(names)} 个）")
    note = (f"面积误差 {geo['max_rel_error']:.1%}    "
            f"迭代 {geo['iterations']} 次")
    if compressed:
        note += "    权重已开方压缩"
    place_annotation(ax, note, style, x=0.5, y=0.02, va="bottom")

    # 颜色条说明
    sm = plt.cm.ScalarMappable(
        cmap=style.cmap("continuous"),
        norm=plt.Normalize(vmin=0, vmax=float(mean_ab.max() * 100)))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.82)
    cbar.set_label("平均相对丰度 (%)")
    fig.tight_layout()

    info = save(fig, out_path, style,
                "每个多边形为一个分类单元，**面积严格正比于其平均相对丰度**，"
                "相邻单元由维诺剖分自然分隔。标注的数字是该单元占总丰度的百分比；"
                "面积过小的单元不标注文字以免重叠。",
                f"维诺树状图（{rank_label}）")
    info["voronoi"] = {
        "max_rel_error": geo["max_rel_error"],
        "iterations": geo["iterations"],
        "n_units": len(names),
        "compressed": compressed,
        "areas": geo["areas"].tolist(),
        "targets": geo["targets"].tolist(),
    }
    info["names"] = names
    return info


def plot_treemap_fallback(rel_table: pd.DataFrame, out_path: str | Path, style: Style,
                          rank_label: str = "Feature", top_n: int = 18) -> dict:
    """矩形树图（squarified treemap）：维诺剖分不可用时的稳健替代。

    同样满足面积正比于丰度，且永不会不收敛。
    """
    from .composition import _to_relative

    rel = _to_relative(rel_table)
    mean_ab = rel.mean(axis=1).sort_values(ascending=False)
    mean_ab = mean_ab[mean_ab > 0]
    if mean_ab.size < 2:
        raise ValueError("至少需要 2 个非零丰度的分类单元。")
    taken = mean_ab.head(max(2, int(top_n)))
    names = [str(x) for x in taken.index]
    frac = (taken / taken.sum()).to_numpy(dtype=float)

    rects = _squarify(frac, 0.0, 0.0, 1.0, 1.0)

    cmap = plt.get_cmap(style.cmap("continuous"))
    fig, ax = new_figure(style, 7.4, 6.4)
    for i, (x, y, dx, dy) in enumerate(rects):
        ax.add_patch(plt.Rectangle((x, y), dx, dy,
                                   facecolor=cmap(0.18 + 0.74 * (frac[i] / frac.max())),
                                   edgecolor="white",
                                   linewidth=style.num("element.line_width", 1.8) * 0.6))
        if frac[i] >= 0.02:
            label = names[i] if len(names[i]) <= 14 else names[i][:13] + "…"
            ax.text(x + dx / 2, y + dy / 2, f"{label}\n{frac[i] * 100:.1f}%",
                    ha="center", va="center",
                    fontsize=max(5.0, style.font_size("annotation") * (0.7 + 0.6 * math.sqrt(frac[i] / frac.max()))),
                    color="#1B2430", fontfamily=style.font_stack("annotation"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    place_title(ax, style, f"矩形树图（{rank_label}，前 {len(names)} 个）")
    fig.tight_layout()
    return save(fig, out_path, style,
                "每个矩形的**面积严格正比于该分类单元的平均相对丰度**。"
                "维诺剖分在单元较多或丰度极不均匀时可能不收敛，此时自动改用矩形树图。",
                f"矩形树图（{rank_label}）")


def _squarify(frac: np.ndarray, x: float, y: float, w: float, h: float) -> list[tuple]:
    """squarified treemap 布局，返回 [(x, y, dx, dy), ...]。"""
    order = np.argsort(-frac)
    values = frac[order]
    rects_sorted: list[tuple] = []

    def worst(row: list[float], length: float) -> float:
        s = sum(row)
        if s <= 0 or length <= 0:
            return float("inf")
        side = s / length
        return max(max(side / (v / side) if v > 0 else float("inf"),
                       (v / side) / side if v > 0 else float("inf")) for v in row)

    def layout_row(row: list[float], x0: float, y0: float, w0: float, h0: float):
        s = sum(row)
        out = []
        if w0 >= h0:      # 竖着排一行
            rw = s / h0
            yy = y0
            for v in row:
                rh = (v / s) * h0 if s > 0 else 0
                out.append((x0, yy, rw, rh))
                yy += rh
            return out, (x0 + rw, y0, max(w0 - rw, 0.0), h0)
        rh = s / w0
        xx = x0
        for v in row:
            rwd = (v / s) * w0 if s > 0 else 0
            out.append((xx, y0, rwd, rh))
            xx += rwd
        return out, (x0, y0 + rh, w0, max(h0 - rh, 0.0))

    cx, cy, cw, ch = x, y, w, h
    row: list[float] = []
    idx_map: list[int] = []
    for i, v in enumerate(values):
        if cw <= 0 or ch <= 0:
            break
        length = ch if cw >= ch else cw
        if not row or worst(row + [v], length) <= worst(row, length):
            row.append(v)
            idx_map.append(i)
        else:
            placed, (cx, cy, cw, ch) = layout_row(row, cx, cy, cw, ch)
            rects_sorted.extend(placed)
            row, idx_map = [v], [i]
    if row and cw > 0 and ch > 0:
        placed, _ = layout_row(row, cx, cy, cw, ch)
        rects_sorted.extend(placed)

    # 还原成输入顺序
    out = [None] * len(frac)
    for pos, orig in enumerate(order):
        if pos < len(rects_sorted):
            out[orig] = rects_sorted[pos]
    return [r if r is not None else (0.0, 0.0, 0.0, 0.0) for r in out]
