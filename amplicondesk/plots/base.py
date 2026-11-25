"""绘图基础设施：风格应用、保存、轴美化、图例、置信椭圆、面板布局。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

from ..style import Style, apply_style, export_formats, resolve_font, text_style_kwargs  # noqa: F401

__all__ = [
    "save", "new_figure", "panel_grid", "style_axes", "finish_axes",
    "place_title", "place_suptitle", "place_axis_labels", "place_annotation",
    "conf_ellipse", "shift_artist", "figure_style", "stat_box", "tick_kwargs",
]


# --------------------------------------------------------------------------
# 保存
# --------------------------------------------------------------------------
def save(fig, path: str | Path, style: Style, caption: str = "", title: str = "") -> dict:
    """保存图片。

    * 先强制渲染一次（``canvas.draw``），让所有文字的实际尺寸确定下来，再按 tight bbox 输出。
      否则在刚切换字体/字号后，matplotlib 偶尔会用未更新的字体度量计算包围盒，
      导致边缘的坐标轴标签被裁掉（尤其是图例放到轴外侧时）。
    * 按 ``export.formats`` 同时或分别输出 PNG / PDF，PDF 为矢量图便于后续编辑。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bg = str(style.get("figure.background", "#FFFFFF"))
    dpi = style.savefig_dpi()
    transparent = style.flag("export.transparent", False)
    facecolor = "none" if transparent else bg

    try:
        apply_font_stack(fig, style)
        fig.canvas.draw()
    except Exception:
        pass

    formats = export_formats(style)
    made: dict[str, str] = {}
    for fmt in formats:
        target = path if path.suffix.lower() == f".{fmt}" else path.with_suffix(f".{fmt}")
        try:
            fig.savefig(target, bbox_inches="tight", pad_inches=0.08,
                        facecolor=facecolor, dpi=dpi, format=fmt)
            made[fmt] = str(target)
        except Exception:
            try:
                fig.savefig(target, bbox_inches="tight", facecolor=facecolor, dpi=dpi)
                made[fmt] = str(target)
            except Exception:
                pass

    plt.close(fig)

    # 以 PNG 为主产物；只输出 PDF 时用 PDF 充当主产物
    main = made.get("png") or made.get("pdf") or str(path)
    main_path = Path(main)
    return {
        "path": main,
        "name": main_path.name,
        "caption": caption,
        "title": title or main_path.stem,
        "files": made,
        "pdf": made.get("pdf"),
    }


def save_clustermap(fig, path: str | Path, style: Style) -> dict:
    """保存 seaborn clustermap 的图（它自带 figure，不能走 save()）。

    按 export.formats 输出 PNG / PDF，返回 {格式: 路径}。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bg = str(style.get("figure.background", "#FFFFFF"))
    dpi = style.savefig_dpi()
    try:
        apply_font_stack(fig, style)
        fig.canvas.draw()
    except Exception:
        pass
    made: dict[str, str] = {}
    for fmt in export_formats(style):
        tgt = path if path.suffix.lower() == f".{fmt}" else path.with_suffix(f".{fmt}")
        try:
            fig.savefig(tgt, bbox_inches="tight", facecolor=bg, dpi=dpi, format=fmt)
            made[fmt] = str(tgt)
        except Exception:
            pass
    return made


# 各绘图模块用这个名字导入
_save_clustermap = save_clustermap


def new_figure(style: Style, base_w: float, base_h: float):
    apply_style(style)
    fig, ax = plt.subplots(figsize=style.figsize(base_w, base_h))
    fig.patch.set_facecolor(str(style.get("figure.background", "#FFFFFF")))
    ax.set_facecolor(str(style.get("figure.facecolor_axes", "#FFFFFF")))
    return fig, ax


def panel_grid(style: Style, n_panels: int, ncol: int | None = None,
               base_w: float = 4.2, base_h: float = 3.6):
    """创建多面板网格，返回 (fig, axes二维数组, 面板数)。"""
    apply_style(style)
    if n_panels <= 0:
        raise ValueError("没有可绘制的面板。")
    ncol = int(ncol or style.num("layout.panel_columns", 3))
    ncol = max(1, min(ncol, n_panels))
    nrow = int(np.ceil(n_panels / ncol))

    fig, axes = plt.subplots(
        nrow, ncol, figsize=style.figsize(base_w * ncol, base_h * nrow), squeeze=False
    )
    fig.patch.set_facecolor(str(style.get("figure.background", "#FFFFFF")))
    for row in axes:
        for ax in row:
            ax.set_facecolor(str(style.get("figure.facecolor_axes", "#FFFFFF")))
    return fig, axes, ncol


# --------------------------------------------------------------------------
# 轴与文本
# --------------------------------------------------------------------------
def style_axes(ax, style: Style) -> None:
    show_grid = style.flag("element.grid", True)
    ax.grid(show_grid, alpha=style.num("element.grid_alpha", 0.25),
            linewidth=style.num("element.grid_width", 0.6))
    ax.set_axisbelow(True)

    show_spines = style.flag("element.spines", True)
    full_frame = style.flag("element.frame", True)
    color = str(style.get("element.frame_color", "#4A4A4A"))
    lw = style.num("element.axes_linewidth", 0.9)

    if not show_spines:
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0)
        return

    for name, spine in ax.spines.items():
        if not full_frame and name in ("top", "right"):
            spine.set_visible(False)
            continue
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(lw)
    ax.tick_params(width=style.num("element.tick_width", 0.9),
                   length=style.num("element.tick_length", 3.5),
                   direction=str(style.get("element.tick_direction", "out")),
                   color=color)


def shift_artist(artist, style: Style, kind: str):
    """按 style 里的 dx/dy 平移文本对象（单位：磅）。"""
    import matplotlib.transforms as mtransforms

    dx, dy = style.offset(kind)
    if not dx and not dy:
        return artist
    trans = mtransforms.ScaledTranslation(dx / 72.0, dy / 72.0, artist.figure.dpi_scale_trans)
    artist.set_transform(artist.get_transform() + trans)
    return artist


def tick_kwargs(style: Style, prefix: str = "tick_label") -> dict:
    """刻度标签的字体参数字典。"""
    kw = {
        "size": style.font_size(prefix),
        "color": style.font_color(prefix),
    }
    fam = style.font_family(prefix)
    if fam:
        kw["family"] = fam
    return kw


def apply_font_stack(fig, style: Style) -> int:
    """给图中**所有**文字对象挂上字体回退链，返回处理的文字数量。

    matplotlib 只有在 fontfamily 收到列表时才逐字符回退。项目里绝大多数文字都经过
    ``text_style_kwargs`` 拿到了列表，但仍有少量直接 ``ax.text`` / ``set_xlabel`` 的调用。
    这里在保存前统一兜底：保留每个元素**原本优先的字体**，把中文字体追加在后面，
    这样即使用户选了 Times New Roman，中文也不会变成方块。
    """
    import matplotlib.text as mtext

    stack = style.font_stack()
    if not stack:
        return 0
    n = 0
    for txt in fig.findobj(mtext.Text):
        try:
            cur = txt.get_fontfamily()
            fams = list(cur) if isinstance(cur, (list, tuple)) else [cur]
            ordered: list[str] = []
            for f in list(fams) + list(stack):
                if f and f not in ordered and f not in ("sans-serif", "serif", "monospace"):
                    ordered.append(f)
            if ordered:
                txt.set_fontfamily(ordered)
                n += 1
        except Exception:
            continue
    return n


def _apply_to_ticks(ax, style: Style, prefix: str = "tick_label") -> None:
    """把字体族/颜色/字号/斜体应用到已有刻度标签上。"""
    stack = style.font_stack(prefix)
    italic = style.font_style(prefix)
    size = style.font_size(prefix)
    color = style.font_color(prefix)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(size)
        lbl.set_color(color)
        lbl.set_fontfamily(stack)
        lbl.set_fontstyle(italic)


def place_title(ax, style: Style, default: str = "", prefix: str = "title",
                allow_override: bool = True, scale: float = 1.0):
    """设置子图标题。

    allow_override=False 时忽略 ``font.title.text`` 覆盖与 dx/dy 位移，
    用于多面板图中各面板的独立小标题（此时总标题由 suptitle 承担）。
    """
    if not style.show_text(prefix):
        return None
    text = style.text(f"font.{prefix}.text", default) if allow_override else default
    kw = text_style_kwargs(style, prefix, pad=8)
    if scale != 1.0:
        kw["fontsize"] = kw["fontsize"] * scale
    ax.set_title(text, **kw)
    if allow_override:
        shift_artist(ax.title, style, prefix)
    return ax.title


def place_suptitle(fig, style: Style, default: str = ""):
    if not style.show_text("suptitle"):
        return None
    text = style.text("font.suptitle.text", default)
    fig.suptitle(text, y=1.005, **text_style_kwargs(style, "suptitle"))
    if fig._suptitle is not None:
        shift_artist(fig._suptitle, style, "suptitle")
    return fig._suptitle


def place_axis_labels(ax, style: Style, xlabel: str = "", ylabel: str = ""):
    if style.show_text("axis_label"):
        if xlabel:
            ax.set_xlabel(style.text("font.axis_label.text", xlabel),
                          **text_style_kwargs(style, "axis_label"))
            shift_artist(ax.xaxis.label, style, "axis_label")
        if ylabel:
            ax.set_ylabel(ylabel, **text_style_kwargs(style, "axis_label"))
            shift_artist(ax.yaxis.label, style, "axis_label")
    _apply_to_ticks(ax, style)


def place_annotation(ax, text: str, style: Style, x: float = 0.5, y: float = 0.97,
                     ha: str = "center", va: str = "top"):
    """在轴内放置统计标注框。"""
    if not style.show_text("annotation") or not text:
        return None
    dx, dy = style.offset("annotation")
    ann = ax.annotate(
        style.text("font.annotation.text", text),
        xy=(x, y), xycoords="axes fraction", ha=ha, va=va,
        xytext=(dx, dy), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.32", fc="#F5F5F5", ec="#CCCCCC", lw=0.6, alpha=0.92),
        zorder=8,
        **text_style_kwargs(style, "annotation"),
    )
    return ann


def stat_box(ax, text: str, style: Style, x=0.5, y=0.97):
    return place_annotation(ax, text, style, x=x, y=y)


def finish_axes(ax, style: Style, legend: bool = True, title: str = "",
                xlabel: str = "", ylabel: str = "", **legend_kw):
    style_axes(ax, style)
    place_axis_labels(ax, style, xlabel, ylabel)
    if title:
        place_title(ax, style, title)
    if legend:
        resolve_legend(style, ax, **legend_kw)


def resolve_legend(style: Style, ax, extra_handles: list | None = None,
                   preferred: str | None = None, **kwargs):
    """统一图例摆放。

    preferred     : 该图类型的偏好位置，**仅当** style.legend.position 为默认的 "best"
                    时生效；用户在界面里显式选择的位置始终优先。
    extra_handles : [(handle, label), ...]，用于合并箭头等非散点句柄
                    （它们不在 ax.get_legend_handles_labels() 里）。
    """
    pos = str(style.get("legend.position", "best"))
    if pos == "best" and preferred:
        pos = preferred

    if pos == "none":
        leg = ax.get_legend()
        if leg:
            leg.remove()
        return None

    handles, labels = ax.get_legend_handles_labels()
    for item in extra_handles or []:
        try:
            h, l = item
        except (TypeError, ValueError):
            continue
        handles.append(h)
        labels.append(l)
    if not handles:
        return None

    ncol = int(style.num("legend.ncol", 1))
    frame = style.flag("legend.frame", False)
    scale = style.num("legend.marker_scale", 1.0)
    fs = style.font_size("legend_text")
    title_override = style.text("font.legend_title.text", "")

    seen, hh, ll = set(), [], []
    for h, l in zip(handles, labels):
        if l in seen or str(l).startswith("_"):
            continue
        seen.add(l)
        hh.append(h)
        ll.append(l)

    common = dict(ncol=ncol, frameon=frame, fontsize=fs, markerscale=scale, **kwargs)
    # 调用方万一仍传了 loc / bbox_to_anchor，这里摘掉，避免与下面的摆放参数冲突
    explicit_loc = common.pop("loc", None)
    explicit_anchor = common.pop("bbox_to_anchor", None)
    if title_override:
        common["title"] = title_override

    if pos.startswith("outside"):
        if pos.endswith("right"):
            loc, anchor = "upper left", explicit_anchor or (1.02, 1.0)
        else:
            loc, anchor = "upper center", explicit_anchor or (0.5, -0.30)
            common["ncol"] = max(int(common.get("ncol", 1)), 2)
        return _scaled(ax.legend(hh, ll, loc=explicit_loc or loc,
                                 bbox_to_anchor=anchor, **common), style)
    leg = ax.legend(hh, ll, loc=explicit_loc or pos, **common)
    return _scaled(leg, style)


def _scaled(leg, style: Style):
    _scale_legend_text(leg, style)
    return leg


def _scale_legend_text(leg, style: Style):
    """图例文字统一应用字号 / 字重 / 斜体 / 颜色 / 字体族。"""
    if leg is None:
        return
    stack = style.font_stack("legend_text")
    tstack = style.font_stack("legend_title")
    try:
        for t in leg.get_texts():
            t.set_fontsize(style.font_size("legend_text"))
            t.set_color(style.font_color("legend_text"))
            t.set_fontweight(style.font_weight("legend_text"))
            t.set_fontstyle(style.font_style("legend_text"))
            t.set_fontfamily(stack)
        title = leg.get_title()
        title.set_fontsize(style.font_size("legend_title"))
        title.set_color(style.font_color("legend_title"))
        title.set_fontweight(style.font_weight("legend_title"))
        title.set_fontstyle(style.font_style("legend_title"))
        title.set_fontfamily(tstack)
    except Exception:
        pass


# --------------------------------------------------------------------------
# 置信椭圆
# --------------------------------------------------------------------------
def conf_ellipse(x, y, ax, style: Style, color: str, n_std: float | None = None, **kwargs):
    if not style.flag("element.ellipse", True):
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return None
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)) or cov[0, 0] <= 0 or cov[1, 1] <= 0:
        return None

    n_std = float(n_std if n_std is not None else style.num("element.ellipse_std", 2.0))
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    rx = np.sqrt(1 + pearson)
    ry = np.sqrt(1 - pearson)

    ell = Ellipse((0, 0), width=rx * 2, height=ry * 2, facecolor="none",
                  edgecolor=color, lw=1.6,
                  alpha=style.num("element.ellipse_alpha", 0.85), zorder=2, **kwargs)
    tr = (matplotlib.transforms.Affine2D()
          .rotate_deg(45)
          .scale(np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std)
          .translate(np.mean(x), np.mean(y)))
    ell.set_transform(tr + ax.transData)
    ax.add_patch(ell)
    return ell


# --------------------------------------------------------------------------
# 分组散点
# --------------------------------------------------------------------------
def scatter_groups(ax, coords: pd.DataFrame, groups: pd.Series, style: Style,
                   xcol: str = None, ycol: str = None, ellipse: bool = True,
                   label_points: bool | None = None):
    """在排序图上按分组绘制散点 + 椭圆，返回 组名→颜色 映射。"""
    if coords.shape[1] < 2:
        raise ValueError("排序坐标至少需要 2 个轴。")
    xcol = xcol or coords.columns[0]
    ycol = ycol or coords.columns[1]

    merged = coords.join(groups.rename("Group"), how="inner")
    merged["Group"] = merged["Group"].astype(str)
    colors = style.group_colors(merged["Group"])

    size = style.num("element.point_size", 68)
    alpha = style.num("element.point_alpha", 0.92) * style.num("palette.alpha", 1.0)
    edge = "white" if style.flag("element.point_edge", True) else "none"
    show_labels = style.flag("element.point_labels", False) if label_points is None else label_points

    for g, sub in merged.groupby("Group", sort=True):
        ax.scatter(sub[xcol], sub[ycol], s=size, color=colors[str(g)], label=str(g),
                   edgecolor=edge, linewidth=0.9, alpha=alpha, zorder=3)
        if ellipse:
            conf_ellipse(sub[xcol], sub[ycol], ax, style, colors[str(g)])
        if show_labels:
            for name, row in sub.iterrows():
                if style.show_text("annotation"):
                    ax.annotate(str(name)[:24], (row[xcol], row[ycol]),
                                fontsize=style.font_size("annotation") * 0.85,
                                color=style.font_color("annotation"), alpha=0.9,
                                xytext=(4, 3), textcoords="offset points", zorder=6)

    if style.flag("element.zero_lines", True):
        ax.axhline(0, color="#BBBBBB", lw=0.8, zorder=1)
        ax.axvline(0, color="#BBBBBB", lw=0.8, zorder=1)
    return colors


def figure_style(style: Style | None, figure_id: str | None = None) -> Style:
    """取到某张图的最终样式（含逐图覆盖）。"""
    st = style or Style()
    return st.for_figure(figure_id) if figure_id else st


def add_biplot_arrows(ax, arrows: pd.DataFrame, style: Style, scale: float = 1.0,
                      color: str = "#7A4E9E", label: str = "环境因子", top_n: int = 10,
                      text_prefix: str = "annotation") -> None:
    """在排序图上加 biplot 箭头。"""
    if arrows is None or arrows.empty:
        return
    cols = list(arrows.columns[:2])
    sub = arrows.reindex(arrows[cols].pow(2).sum(axis=1).pow(0.5).sort_values(ascending=False).index)
    sub = sub.head(top_n)
    if sub.empty:
        return

    xmax = float(np.nanmax(np.abs(arrows[cols[0]].to_numpy(dtype=float)))) or 1.0
    ymax = float(np.nanmax(np.abs(arrows[cols[1]].to_numpy(dtype=float)))) or 1.0

    for i, (name, row) in enumerate(sub.iterrows()):
        dx = float(row[cols[0]]) * scale
        dy = float(row[cols[1]]) * scale
        ax.annotate(
            "", xy=(dx, dy), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.3, alpha=0.85),
            zorder=5,
        )
        ax.annotate(
            str(name)[:22], (dx, dy),
            fontsize=style.font_size(text_prefix) * 0.9,
            color=color, alpha=0.95,
            xytext=(4, 3), textcoords="offset points", zorder=6,
        )
    if label:
        from matplotlib.lines import Line2D

        ax.legend(
            handles=[Line2D([0], [0], color=color, lw=1.3, marker=">", markersize=5)],
            labels=[label], loc="lower right", frameon=False,
            fontsize=style.font_size("legend_text"),
        )
