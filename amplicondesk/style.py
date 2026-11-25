"""可视化风格系统。

设计要点：
  * 全部样式收敛为一组**扁平的 dotted-key**（如 ``font.title.size``），
    便于 JSON 存取、逐图覆盖，也便于 Web 界面按 schema 自动生成表单。
  * ``STYLE_SCHEMA`` 同时描述默认值、中文标签、控件类型与取值范围，
    是"界面自动生成风格编辑器"的唯一数据源。
  * 逐图覆盖：键 ``fig.<图ID>.<原键>`` 会覆盖该图上的 ``<原键>``。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# 字体
# --------------------------------------------------------------------------
CJK_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
    "WenQuanYi Zen Hei", "PingFang SC", "DengXian", "SimSun", "Arial Unicode MS",
]


def available_cjk_fonts() -> list[str]:
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    return [n for n in CJK_CANDIDATES if n in names]


def all_fonts() -> list[str]:
    from matplotlib import font_manager

    return sorted({f.name for f in font_manager.fontManager.ttflist})


# --------------------------------------------------------------------------
# 调色板
# --------------------------------------------------------------------------
PALETTES: dict[str, list[str]] = {
    "default": ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"],
    "okabe_ito": ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
                  "#56B4E9", "#F0E442", "#000000"],
    "nature": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
               "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"],
    "lancet": ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F",
               "#FDAF91", "#AD002A", "#ADB6B6", "#1B1919"],
    "jco": ["#0073C2", "#EFC000", "#868686", "#CD534C", "#7AA6DC",
            "#003C67", "#8F7700", "#3B3B3B", "#A73030", "#4A6990"],
    "nejm": ["#BC3C29", "#0072B5", "#E18727", "#20854E", "#7876B1",
             "#6F99AD", "#FFDC91", "#EE4C97"],
    "set2": ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854",
             "#FFD92F", "#E5C494", "#B3B3B3"],
    "paired": ["#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99",
               "#E31A1C", "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A"],
    "tableau10": ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
                  "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"],
    "grayscale": ["#2B2B2B", "#5A5A5A", "#808080", "#A6A6A6", "#C7C7C7",
                  "#DEDEDE", "#8C6D31", "#B58900"],
}

DIVERGING_CMAPS = ["RdBu_r", "RdYlBu_r", "coolwarm", "vlag", "icefire", "PiYG", "BrBG", "Spectral_r"]
SEQUENTIAL_CMAPS = ["viridis", "magma", "cividis", "YlGnBu", "YlOrRd", "Blues", "Greens", "Reds", "rocket", "mako"]

# --------------------------------------------------------------------------
# 科研配色方案（由用户提供的 Excel 导入，见 import_palettes.py）
# --------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
RESEARCH_PALETTE_FILE = DATA_DIR / "research_palettes.json"


def load_research_palettes(path: Path | None = None) -> dict[str, list[str]]:
    """读取内置的科研配色方案。找不到文件时返回空字典（不影响其它功能）。"""
    p = path or RESEARCH_PALETTE_FILE
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, list[str]] = {}
    for name, info in (payload.get("schemes") or {}).items():
        colors = [str(c) for c in (info.get("colors") or []) if str(c).strip()]
        if len(colors) >= 2:
            out[str(name)] = colors
    return out


RESEARCH_PALETTES: dict[str, list[str]] = load_research_palettes()

# 科研配色以 "科研配色N" 的名字并入统一调色板注册表
for _name, _colors in RESEARCH_PALETTES.items():
    PALETTES[_name] = _colors


def palette_names() -> list[str]:
    """全部可用调色板：内置 + 科研配色。"""
    return list(PALETTES.keys())


def research_palette_names() -> list[str]:
    return list(RESEARCH_PALETTES.keys())


def palette_colors_of(name: str) -> list[str]:
    return list(PALETTES.get(name, PALETTES["default"]))


# --------------------------------------------------------------------------
# Schema：默认值 + 界面元信息
# --------------------------------------------------------------------------
def _text_fields(prefix: str, label: str, size: float, weight: str = "normal",
                 color: str = "#1b2430") -> list[dict]:
    return [
        {"key": f"font.{prefix}.size", "label": f"{label}字号", "type": "number",
         "default": size, "min": 4, "max": 48, "step": 0.5, "group": "文字"},
        {"key": f"font.{prefix}.family", "label": f"{label}字体", "type": "font",
         "default": "", "group": "文字", "help": "留空则沿用全局字体"},
        {"key": f"font.{prefix}.weight", "label": f"{label}字重", "type": "select",
         "default": weight, "options": ["normal", "medium", "semibold", "bold"], "group": "文字"},
        {"key": f"font.{prefix}.style", "label": f"{label}字形", "type": "select",
         "default": "normal", "options": ["normal", "italic"], "group": "文字",
         "help": "italic = 斜体"},
        {"key": f"font.{prefix}.color", "label": f"{label}颜色", "type": "color",
         "default": color, "group": "文字"},
        {"key": f"font.{prefix}.dx", "label": f"{label}水平偏移", "type": "number",
         "default": 0.0, "min": -200, "max": 200, "step": 1, "group": "位置",
         "help": "单位：磅(pt)，正值向右"},
        {"key": f"font.{prefix}.dy", "label": f"{label}垂直偏移", "type": "number",
         "default": 0.0, "min": -200, "max": 200, "step": 1, "group": "位置",
         "help": "单位：磅(pt)，正值向上"},
        {"key": f"font.{prefix}.text", "label": f"{label}自定义内容", "type": "text",
         "default": "", "group": "文字", "help": "留空则使用自动生成的文字"},
        {"key": f"font.{prefix}.visible", "label": f"显示{label}", "type": "bool",
         "default": True, "group": "文字"},
    ]


STYLE_SCHEMA: list[dict] = [
    # ---------- 全局 ----------
    {"key": "font.family", "label": "全局字体", "type": "font",
     "default": "Times New Roman", "group": "全局",
     "help": "默认 Times New Roman；若系统没有该字体会自动回退到可用的中文字体"},
    {"key": "font.scale", "label": "字号整体缩放", "type": "number",
     "default": 1.0, "min": 0.5, "max": 2.5, "step": 0.05, "group": "全局"},
    {"key": "figure.scale", "label": "图幅整体缩放", "type": "number",
     "default": 1.0, "min": 0.5, "max": 2.5, "step": 0.05, "group": "全局"},
    {"key": "figure.dpi", "label": "输出分辨率 (DPI)", "type": "select",
     "default": 300, "options": [150, 200, 300, 400, 600], "group": "全局"},
    {"key": "figure.background", "label": "背景色", "type": "color",
     "default": "#FFFFFF", "group": "全局"},
    {"key": "figure.facecolor_axes", "label": "绘图区背景", "type": "color",
     "default": "#FFFFFF", "group": "全局"},

    # ---------- 输出格式 ----------
    {"key": "export.formats", "label": "导出格式", "type": "select",
     "default": "png+pdf", "options": ["png+pdf", "png", "pdf"], "group": "输出",
     "help": "PDF 为矢量图（文字可编辑，可在 Illustrator 中继续加工）；PNG 按下面的 DPI 输出。"
             "默认两种都出，只想要一种可在此切换"},
    {"key": "export.transparent", "label": "透明背景", "type": "bool",
     "default": False, "group": "输出"},

    # ---------- 配色 ----------
    {"key": "palette.preset", "label": "调色板预设", "type": "select",
     "default": "方案16", "options": palette_names(), "group": "配色",
     "help": "含 22 套科研配色与内置色板；分组较少时自动从色板中均匀取色"},
    {"key": "palette.custom", "label": "自定义颜色序列", "type": "colorlist",
     "default": [], "group": "配色",
     "help": "填写后覆盖预设；可按顺序循环用于各分组。可在界面上从配色方案里点选若干色生成"},
    {"key": "palette.group_map", "label": "分组指定颜色", "type": "colormap",
     "default": {}, "group": "配色",
     "help": "形如 {\"Control\": \"#4C72B0\", \"Treatment\": \"#C44E52\"}"},
    {"key": "palette.continuous", "label": "连续型色图", "type": "select",
     "default": "viridis", "options": SEQUENTIAL_CMAPS, "group": "配色"},
    {"key": "palette.diverging", "label": "双向色图", "type": "select",
     "default": "RdYlBu_r", "options": DIVERGING_CMAPS, "group": "配色"},
    {"key": "palette.heatmap", "label": "热图色图", "type": "select",
     "default": "RdYlBu_r", "options": DIVERGING_CMAPS + SEQUENTIAL_CMAPS, "group": "配色"},
    {"key": "palette.alpha", "label": "整体透明度", "type": "number",
     "default": 1.0, "min": 0.2, "max": 1.0, "step": 0.05, "group": "配色"},

    # ---------- 图例 ----------
    {"key": "legend.position", "label": "图例位置", "type": "select",
     "default": "best",
     "options": ["best", "upper right", "upper left", "lower right", "lower left",
                 "right", "center left", "center right", "upper center", "lower center",
                 "outside right", "outside bottom", "none"],
     "group": "图例"},
    {"key": "legend.ncol", "label": "图例列数", "type": "number",
     "default": 1, "min": 1, "max": 6, "step": 1, "group": "图例"},
    {"key": "legend.frame", "label": "显示图例边框", "type": "bool",
     "default": False, "group": "图例"},
    {"key": "legend.marker_scale", "label": "图例标记大小", "type": "number",
     "default": 1.0, "min": 0.3, "max": 3.0, "step": 0.1, "group": "图例"},

    # ---------- 元素 ----------
    {"key": "element.grid", "label": "显示网格", "type": "bool", "default": True, "group": "元素"},
    {"key": "element.grid_alpha", "label": "网格透明度", "type": "number",
     "default": 0.25, "min": 0, "max": 1, "step": 0.05, "group": "元素"},
    {"key": "element.grid_width", "label": "网格线宽", "type": "number",
     "default": 0.6, "min": 0, "max": 3, "step": 0.1, "group": "线条"},
    {"key": "element.spines", "label": "显示边框", "type": "bool", "default": True, "group": "线条",
     "help": "关闭后图中不画坐标轴边框线"},
    {"key": "element.frame", "label": "四周边框完整闭合", "type": "bool", "default": True,
     "group": "线条", "help": "关闭则只保留左边和下边两条轴线"},
    {"key": "element.frame_color", "label": "边框颜色", "type": "color",
     "default": "#4A4A4A", "group": "线条"},
    {"key": "element.axes_linewidth", "label": "边框线宽", "type": "number",
     "default": 0.9, "min": 0, "max": 4, "step": 0.1, "group": "线条"},
    {"key": "element.line_width", "label": "数据线宽", "type": "number",
     "default": 1.8, "min": 0.2, "max": 6, "step": 0.1, "group": "线条",
     "help": "曲线、折线、误差棒等数据元素的默认线宽"},
    {"key": "element.tick_width", "label": "刻度线宽", "type": "number",
     "default": 0.9, "min": 0, "max": 4, "step": 0.1, "group": "线条"},
    {"key": "element.tick_length", "label": "刻度线长度", "type": "number",
     "default": 3.5, "min": 0, "max": 12, "step": 0.5, "group": "线条"},
    {"key": "element.tick_direction", "label": "刻度方向", "type": "select",
     "default": "out", "options": ["out", "in", "inout"], "group": "线条"},
    {"key": "element.zero_lines", "label": "显示零轴参考线", "type": "bool",
     "default": True, "group": "元素"},
    {"key": "element.ellipse", "label": "置信椭圆", "type": "bool", "default": True, "group": "元素"},
    {"key": "element.ellipse_alpha", "label": "椭圆透明度", "type": "number",
     "default": 0.85, "min": 0, "max": 1, "step": 0.05, "group": "元素"},
    {"key": "element.ellipse_std", "label": "椭圆标准差倍数", "type": "number",
     "default": 2.0, "min": 1, "max": 4, "step": 0.25, "group": "元素"},
    {"key": "element.point_size", "label": "散点大小", "type": "number",
     "default": 68, "min": 4, "max": 400, "step": 2, "group": "元素"},
    {"key": "element.point_alpha", "label": "散点透明度", "type": "number",
     "default": 0.92, "min": 0.05, "max": 1, "step": 0.05, "group": "元素"},
    {"key": "element.point_edge", "label": "散点描边", "type": "bool", "default": True, "group": "元素"},
    {"key": "element.point_labels", "label": "显示样本名", "type": "bool",
     "default": False, "group": "元素", "help": "在排序图上标出每个样本的名称"},
    {"key": "element.jitter", "label": "散点抖动幅度", "type": "number",
     "default": 0.18, "min": 0, "max": 0.6, "step": 0.02, "group": "元素"},
    {"key": "element.box_width", "label": "箱体宽度", "type": "number",
     "default": 0.6, "min": 0.2, "max": 1.0, "step": 0.05, "group": "元素"},
    {"key": "element.bar_edge", "label": "柱体描边", "type": "bool", "default": True, "group": "元素"},
    {"key": "element.dendrogram", "label": "热图显示聚类树", "type": "bool",
     "default": True, "group": "元素"},
    {"key": "element.heatmap_values", "label": "热图显示数值", "type": "bool",
     "default": False, "group": "元素"},
    {"key": "element.axis_rotate", "label": "X 轴标签旋转", "type": "number",
     "default": 45, "min": 0, "max": 90, "step": 5, "group": "元素"},
    {"key": "element.label_threshold", "label": "标注显著性阈值", "type": "number",
     "default": 0.05, "min": 0.001, "max": 0.5, "step": 0.005, "group": "元素"},
    {"key": "element.label_top_n", "label": "标注条目数上限", "type": "number",
     "default": 12, "min": 0, "max": 60, "step": 1, "group": "元素"},
    {"key": "element.volcano_fc_cut", "label": "火山图 log2FC 阈值", "type": "number",
     "default": 1.0, "min": 0, "max": 6, "step": 0.1, "group": "元素"},

    # ---------- 布局 ----------
    {"key": "layout.subplot_spacing", "label": "子图间距", "type": "number",
     "default": 1.08, "min": 0.8, "max": 3.0, "step": 0.05, "group": "布局"},
    {"key": "layout.bar_group_gap", "label": "分组分隔线", "type": "bool",
     "default": True, "group": "布局"},
    {"key": "layout.panel_columns", "label": "多面板列数", "type": "number",
     "default": 3, "min": 1, "max": 6, "step": 1, "group": "布局"},
]

# 各文字元素
for _prefix, _label, _size, _weight in [
    ("suptitle", "总标题", 15, "bold"),
    ("title", "子图标题", 12, "semibold"),
    ("axis_label", "坐标轴标签", 11, "normal"),
    ("tick_label", "刻度标签", 10, "normal"),
    ("legend_text", "图例文字", 9.5, "normal"),
    ("legend_title", "图例标题", 10.5, "semibold"),
    ("annotation", "标注文字", 9, "normal"),
]:
    STYLE_SCHEMA.extend(_text_fields(_prefix, _label, _size, _weight))

STYLE_DEFAULTS: dict[str, Any] = {item["key"]: item["default"] for item in STYLE_SCHEMA}
SCHEMA_BY_KEY: dict[str, dict] = {item["key"]: item for item in STYLE_SCHEMA}

GROUPS = ["全局", "文字", "位置", "配色", "图例", "元素", "线条", "布局", "输出"]


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
@dataclass
class Style:
    """一组扁平样式键值，支持逐图覆盖与 JSON 往返。"""

    values: dict[str, Any] = field(default_factory=dict)
    figure_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ---- 基础读写 ----
    def get(self, key: str, default: Any = None) -> Any:
        if key in self.values:
            return self.values[key]
        if key in STYLE_DEFAULTS:
            return STYLE_DEFAULTS[key]
        return default

    def set(self, key: str, value: Any) -> "Style":
        self.values[key] = value
        return self

    def update(self, mapping: dict[str, Any]) -> "Style":
        self.values.update(mapping or {})
        return self

    def has_override(self, key: str) -> bool:
        return key in self.values

    # ---- 逐图覆盖 ----
    def for_figure(self, figure_id: str) -> "Style":
        """返回应用了 ``fig.<figure_id>.*`` 覆盖后的新样式。"""
        merged = Style(values=dict(self.values), figure_overrides=self.figure_overrides)
        prefix = f"fig.{figure_id}."
        for k, v in self.values.items():
            if k.startswith(prefix):
                merged.values[k[len(prefix):]] = v
        for k, v in (self.figure_overrides.get(figure_id) or {}).items():
            merged.values[k] = v
        return merged

    def set_figure(self, figure_id: str, key: str, value: Any) -> "Style":
        self.figure_overrides.setdefault(figure_id, {})[key] = value
        return self

    # ---- 便捷取值 ----
    def num(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def flag(self, key: str, default: bool = True) -> bool:
        v = self.get(key, default)
        if isinstance(v, str):
            return v.strip().lower() not in ("", "0", "false", "no", "off")
        return bool(v)

    def text(self, key: str, default: str = "") -> str:
        v = self.get(key, "")
        return str(v) if v not in (None, "") else default

    def font_size(self, prefix: str) -> float:
        return self.num(f"font.{prefix}.size", 10) * self.num("font.scale", 1.0)

    def font_weight(self, prefix: str) -> str:
        return str(self.get(f"font.{prefix}.weight", "normal"))

    def font_style(self, prefix: str) -> str:
        """normal 或 italic。"""
        v = str(self.get(f"font.{prefix}.style", "normal")).lower()
        return "italic" if v.startswith("ital") or v in ("oblique", "斜体") else "normal"

    def font_family(self, prefix: str = "") -> str:
        """某个文字元素的主字体；未单独指定时回退到全局字体。"""
        if prefix:
            own = str(self.get(f"font.{prefix}.family", "") or "").strip()
            if own:
                return own
        return str(self.get("font.family", "") or "").strip()

    def font_stack(self, prefix: str = "") -> list[str]:
        """matplotlib 的多字体回退链。

        matplotlib 只有在 ``fontfamily`` 收到**列表**时才做逐字符回退，
        所以主字体必须和中文字体一起作为一个列表传进去，
        否则用 Times New Roman 时中文会变成方块。
        """
        primary = self.font_family(prefix)
        chain: list[str] = []
        if primary and primary not in ("sans-serif", "serif", "monospace"):
            chain.append(primary)
        for f in available_cjk_fonts():
            if f not in chain:
                chain.append(f)
        if not chain:
            chain = ["DejaVu Sans"]
        return chain

    def font_color(self, prefix: str) -> str:
        return str(self.get(f"font.{prefix}.color", "#1b2430"))

    def offset(self, prefix: str) -> tuple[float, float]:
        return (self.num(f"font.{prefix}.dx", 0.0), self.num(f"font.{prefix}.dy", 0.0))

    def show_text(self, prefix: str) -> bool:
        return self.flag(f"font.{prefix}.visible", True)

    # ---- 颜色 ----
    def palette_colors(self) -> list[str]:
        custom = self.get("palette.custom") or []
        custom = [str(c) for c in custom if str(c).strip()]
        if custom:
            return custom
        preset = str(self.get("palette.preset", "default"))
        return list(PALETTES.get(preset, PALETTES["default"]))

    def group_colors(self, groups) -> dict[str, str]:
        """为分组列表生成 组名 → 颜色 的映射。

        当分组数少于调色板颜色数时，**均匀取色**而不是取前 N 个，
        否则两组时会拿到调色板开头两个相近的颜色（很多科研配色是渐变排布的）。
        """
        cats = list(dict.fromkeys([str(g) for g in groups]))
        colors = self.palette_colors()
        mapping = dict(self.get("palette.group_map") or {})
        n, m = len(cats), len(colors)

        if m <= 0:
            colors = PALETTES["default"]
            m = len(colors)

        if n > 1 and n < m:
            picks = [int(round(i * (m - 1) / (n - 1))) for i in range(n)]
        else:
            picks = [i % m for i in range(n)]

        out: dict[str, str] = {}
        used: set[str] = set()
        for i, c in enumerate(cats):
            if c in mapping and str(mapping[c]).strip():
                out[c] = str(mapping[c])
            else:
                pick = None
                for k in range(m):
                    cand = colors[(picks[i] + k) % m]
                    if cand not in used:
                        pick = cand
                        break
                out[c] = pick or colors[picks[i] % m]
            used.add(out[c])
        return out

    def single_color(self, index: int = 0) -> str:
        colors = self.palette_colors()
        return colors[index % len(colors)]

    def cmap(self, kind: str = "heatmap"):
        key = {"heatmap": "palette.heatmap", "continuous": "palette.continuous",
               "diverging": "palette.diverging"}.get(kind, "palette.heatmap")
        return str(self.get(key, "RdYlBu_r"))

    # ---- 尺寸 ----
    def figsize(self, base_w: float, base_h: float) -> tuple[float, float]:
        s = self.num("figure.scale", 1.0)
        return (round(base_w * s, 3), round(base_h * s, 3))

    def savefig_dpi(self) -> int:
        try:
            return int(float(self.get("figure.dpi", 300)))
        except (TypeError, ValueError):
            return 300

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        return {"values": copy.deepcopy(self.values),
                "figure_overrides": copy.deepcopy(self.figure_overrides)}

    @classmethod
    def from_dict(cls, data: dict | None) -> "Style":
        data = data or {}
        values = dict(data.get("values") or {})
        # 兼容直接传扁平字典
        if not values and data and "figure_overrides" not in data:
            values = {k: v for k, v in data.items() if k in STYLE_DEFAULTS or k.startswith("fig.")}
        return cls(values=values, figure_overrides=dict(data.get("figure_overrides") or {}))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Style":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def diff_from_default(self) -> dict:
        """只返回与默认值不同的键，便于保存"精简主题"。"""
        return {k: v for k, v in self.values.items()
                if k not in STYLE_DEFAULTS or STYLE_DEFAULTS[k] != v}


# --------------------------------------------------------------------------
# 预设主题
# --------------------------------------------------------------------------
PRESETS: dict[str, dict[str, Any]] = {
    "默认": {},
    "论文投稿": {
        "font.family": "Times New Roman",
        "font.scale": 0.95, "figure.dpi": 600,
        "font.title.weight": "bold", "font.title.size": 11,
        "font.axis_label.size": 10, "font.tick_label.size": 8.5,
        "font.legend_text.size": 8.5, "font.annotation.size": 8,
        "element.grid": False, "element.point_alpha": 1.0,
        "element.axes_linewidth": 1.1, "element.line_width": 1.6,
        "palette.preset": "方案14", "legend.frame": False,
        "element.axis_rotate": 45, "export.formats": "png+pdf",
    },
    "演示大屏": {
        "font.scale": 1.45, "figure.scale": 1.15, "figure.dpi": 200,
        "font.title.weight": "bold",
        "element.point_size": 130, "element.grid_alpha": 0.3,
        "palette.preset": "方案11", "legend.ncol": 2,
        "element.line_width": 2.6, "element.axes_linewidth": 1.2,
    },
    "极简黑白": {
        "palette.preset": "grayscale",
        "element.grid": False, "element.ellipse": False,
        "element.point_edge": True, "font.title.weight": "bold",
        "element.frame": False,
    },
    "答辩配色": {
        "palette.preset": "方案16", "font.scale": 1.2, "figure.scale": 1.1,
        "element.point_size": 100, "legend.ncol": 2, "element.line_width": 2.4,
    },
    "科研配色·冷色": {
        "palette.preset": "方案1", "font.family": "Times New Roman",
        "export.formats": "png+pdf",
    },
    "科研配色·暖色": {
        "palette.preset": "方案3", "font.family": "Times New Roman",
        "export.formats": "png+pdf",
    },
}


def preset_style(name: str, base: Style | None = None) -> Style:
    s = Style.from_dict(base.to_dict()) if base else Style()
    s.update(PRESETS.get(name, {}))
    return s


# --------------------------------------------------------------------------
# 应用到 matplotlib
# --------------------------------------------------------------------------
_CACHED_FALLBACK: str | None = None


def resolve_font(style: Style, prefix: str = "") -> str | None:
    """决定实际使用的字体族。

    用户指定了字体但系统没有时，回退到可用的中文字体（并保持可读性），
    而不是让 matplotlib 静默换成 DejaVu Sans 导致中文变方块。
    """
    global _CACHED_FALLBACK
    wanted = style.font_family(prefix)
    if _CACHED_FALLBACK is None:
        fonts = available_cjk_fonts()
        _CACHED_FALLBACK = fonts[0] if fonts else None

    if not wanted:
        return _CACHED_FALLBACK

    try:
        from matplotlib import font_manager

        installed = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        return wanted
    if wanted in installed:
        return wanted
    return _CACHED_FALLBACK or wanted


def apply_style(style: Style) -> None:
    """把全局部分写入 matplotlib rcParams。"""
    import matplotlib
    import matplotlib.pyplot as plt

    background = str(style.get("figure.background", "#FFFFFF"))
    axes_bg = str(style.get("figure.facecolor_axes", background))
    show_grid = style.flag("element.grid", True)
    show_spines = style.flag("element.spines", True)
    full_frame = style.flag("element.frame", True)
    frame_color = str(style.get("element.frame_color", "#4A4A4A"))
    axes_lw = style.num("element.axes_linewidth", 0.9)
    line_lw = style.num("element.line_width", 1.8)
    tick_w = style.num("element.tick_width", 0.9)
    tick_len = style.num("element.tick_length", 3.5)
    tick_dir = str(style.get("element.tick_direction", "out"))

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": style.savefig_dpi(),
            "savefig.bbox": "tight",
            "savefig.facecolor": background,
            "figure.facecolor": background,
            "axes.facecolor": axes_bg,
            "axes.edgecolor": frame_color if show_spines else "none",
            "axes.linewidth": axes_lw if show_spines else 0.0,
            "axes.grid": show_grid,
            "grid.alpha": style.num("element.grid_alpha", 0.25),
            "grid.linewidth": style.num("element.grid_width", 0.6),
            "grid.color": frame_color if show_grid else "none",
            "lines.linewidth": line_lw,
            "lines.markersize": 5.0,
            "patch.linewidth": 0.8,
            "errorbar.capsize": 3.0,
            "axes.titlesize": style.font_size("title"),
            "axes.titleweight": style.font_weight("title"),
            "axes.titlecolor": style.font_color("title"),
            "axes.labelsize": style.font_size("axis_label"),
            "axes.labelcolor": style.font_color("axis_label"),
            "xtick.labelsize": style.font_size("tick_label"),
            "ytick.labelsize": style.font_size("tick_label"),
            "xtick.color": style.font_color("tick_label"),
            "ytick.color": style.font_color("tick_label"),
            "xtick.major.width": tick_w,
            "ytick.major.width": tick_w,
            "xtick.major.size": tick_len,
            "ytick.major.size": tick_len,
            "xtick.direction": tick_dir,
            "ytick.direction": tick_dir,
            "legend.fontsize": style.font_size("legend_text"),
            "legend.title_fontsize": style.font_size("legend_title"),
            "legend.frameon": style.flag("legend.frame", False),
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,      # 让 PDF 里的文字保持可编辑
            "ps.fonttype": 42,
            "figure.autolayout": False,
        }
    )

    # 字体：把指定字体排到最前，后面跟上中文字体做兜底
    wanted = resolve_font(style)
    fallbacks = [f for f in available_cjk_fonts()]
    sans = []
    for f in ([wanted] if wanted else []) + fallbacks + list(plt.rcParams["font.sans-serif"]):
        if f and f not in sans:
            sans.append(f)
    plt.rcParams["font.sans-serif"] = sans
    plt.rcParams["font.family"] = "sans-serif"
    if not show_grid:
        plt.rcParams["axes.grid"] = False


# --------------------------------------------------------------------------
# 文本样式辅助
# --------------------------------------------------------------------------
def figure_id_list() -> list[str]:
    """可供单独覆盖的图 ID（与 pipeline 中的文件前缀一致）。"""
    return [
        "alpha_boxplot", "alpha_heatmap", "alpha_rarefaction", "sample_quality",
        "pcoa", "nmds", "pca", "rda", "dbrda", "cca", "cap", "tsne", "umap", "plsda",
        "scree", "distance_box", "betadisper", "envfit", "mantel", "procrustes",
        "rarefaction", "accumulation", "rank_abundance",
        "composition_bar", "composition_area", "heatmap", "top_taxa", "venn",
        "core_microbiome", "ternary", "violin", "raincloud", "density",
        "sankey", "lollipop", "bubble", "beeswarm", "voronoi",
        "volcano", "ma_plot", "diff_bar", "forest", "feature_box", "diff_heatmap", "lda",
        "rf_importance", "roc", "confusion",
        "expr_boxplot", "expr_correlation", "expr_pca", "plsda_vip",
        "enrichment_bubble", "enrichment_bar",
    ]


def text_style_kwargs(style: Style, prefix: str, default_text: str = "", **extra) -> dict:
    """构造 matplotlib 文本参数，统一处理字号 / 字重 / 斜体 / 颜色 / 字体族。

    这是全项目设置文字的唯一入口：任何图里的文字都应经过它，
    这样"加粗、斜体、换字体"才能对每一处文字生效。
    fontfamily 传的是**列表**，以便 matplotlib 逐字符回退到中文字体。
    """
    kw = {
        "fontsize": style.font_size(prefix),
        "fontweight": style.font_weight(prefix),
        "fontstyle": style.font_style(prefix),
        "color": style.font_color(prefix),
        "fontfamily": style.font_stack(prefix),
    }
    kw.update(extra)
    return kw


def export_formats(style: Style) -> list[str]:
    """按 export.formats 返回要写出的格式列表。"""
    v = str(style.get("export.formats", "png+pdf")).lower()
    if v in ("png+pdf", "both", "all"):
        return ["png", "pdf"]
    if v in ("pdf", "png"):
        return [v]
    return ["png", "pdf"]
