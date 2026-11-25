"""样式实时预览：用当前数据 + 给定样式快速渲染单张图。

界面里拖动字号/改颜色时，通过这个模块即时出图，避免每次都要跑完整流程。
结果按 (数据指纹 + 样式指纹) 缓存，重复请求直接复用。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import plots as pl
from ..core import diversity as dv
from ..core import stats as st
from ..core.io import orient_feature_table, read_group_table, read_table, read_taxonomy
from ..core.pipeline import ANALYSIS_TYPES, load_inputs, prepare_env
from ..core.preprocess import aggregate_by_taxonomy, filter_features, to_relative
from ..style import PRESETS, Style, preset_style

PREVIEW_DIR = Path(__file__).resolve().parents[2] / "preview"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# 可预览的图（键 = 前端标识，值 = 中文名）
PREVIEWABLE = [
    {"id": "alpha_boxplot", "label": "Alpha 多样性箱线图", "section": "Alpha 多样性"},
    {"id": "rarefaction", "label": "稀释曲线", "section": "测序深度"},
    {"id": "rank_abundance", "label": "等级丰度曲线", "section": "测序深度"},
    {"id": "pcoa", "label": "PCoA 主坐标分析", "section": "排序分析"},
    {"id": "nmds", "label": "NMDS 排序图", "section": "排序分析"},
    {"id": "pca", "label": "PCA 主成分分析", "section": "排序分析"},
    {"id": "umap", "label": "UMAP 降维", "section": "排序分析"},
    {"id": "tsne", "label": "t-SNE 降维", "section": "排序分析"},
    {"id": "plsda", "label": "PLS-DA 判别分析", "section": "排序分析"},
    {"id": "scree", "label": "碎石图", "section": "排序分析"},
    {"id": "distance_box", "label": "组内/组间距离分布", "section": "Beta 多样性"},
    {"id": "composition_bar", "label": "群落组成堆叠柱状图", "section": "群落组成"},
    {"id": "composition_area", "label": "群落组成面积图", "section": "群落组成"},
    {"id": "heatmap", "label": "丰度热图", "section": "群落组成"},
    {"id": "top_taxa", "label": "优势类群箱线图", "section": "群落组成"},
    {"id": "venn", "label": "Venn 共享特征图", "section": "群落组成"},
    {"id": "volcano", "label": "火山图", "section": "差异分析"},
    {"id": "ma_plot", "label": "MA 图", "section": "差异分析"},
    {"id": "diff_bar", "label": "差异特征条形图", "section": "差异分析"},
    {"id": "forest", "label": "森林图", "section": "差异分析"},
    {"id": "feature_box", "label": "差异特征箱线图", "section": "差异分析"},
    {"id": "lda", "label": "LDA 效应量", "section": "差异分析"},
    {"id": "rf_importance", "label": "随机森林特征重要性", "section": "机器学习"},
    {"id": "roc", "label": "ROC 曲线", "section": "机器学习"},
    {"id": "confusion", "label": "混淆矩阵", "section": "机器学习"},
]

_lock = threading.Lock()
_data_cache: dict[str, "_Prepared"] = {}
_file_cache: dict[str, Path] = {}
_CACHE_LIMIT = 24


@dataclass
class _Prepared:
    table: pd.DataFrame
    rel: pd.DataFrame
    groups: pd.Series
    meta: pd.DataFrame
    rank_table: pd.DataFrame | None
    rank_label: str
    distances: dict
    diff: pd.DataFrame | None
    alpha: pd.DataFrame
    alpha_tests: dict
    rf: dict | None
    label_a: str
    label_b: str


def _fingerprint(cfg: dict) -> str:
    keys = ["abundance_path", "metadata_path", "taxonomy_path", "group_column", "rank",
            "analysis_type", "rarefy"]
    payload = {k: cfg.get(k) for k in keys}
    payload["filter"] = cfg.get("filter")
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _build(cfg: dict) -> _Prepared:
    """加载并预处理数据（与正式流程保持一致）。"""
    load_cfg = {
        "abundance_path": cfg["abundance_path"],
        "metadata_path": cfg["metadata_path"],
        "taxonomy_path": cfg.get("taxonomy_path") or "",
    }
    table, meta, tax, env_raw, _info = load_inputs(load_cfg, _SilentProgress())

    group_col = cfg.get("group_column") or str(meta.columns[0])
    if group_col not in meta.columns:
        group_col = str(meta.columns[0])
    groups = meta[group_col].astype(str)

    flt = cfg.get("filter") or {}
    table, _ = filter_features(
        table,
        min_count=float(flt.get("min_count", 0) or 0),
        min_prevalence=float(flt.get("min_prevalence", 0) or 0),
        min_total_frac=float(flt.get("min_total_frac", 0) or 0),
    )
    rel = to_relative(table)

    rank_table, rank_label = None, "Feature"
    if tax is not None:
        rank = cfg.get("rank") or ANALYSIS_TYPES.get(
            cfg.get("analysis_type", "amplicon16s"), {}).get("default_rank", "Genus")
        try:
            rank_table = aggregate_by_taxonomy(table, tax, rank)
            rank_label = rank
        except Exception:
            rank_table, rank_label = None, "Feature"

    # 距离矩阵与差异分析：按需惰性计算太麻烦，这里直接算最常用的两个
    distances = {}
    for m in ("braycurtis", "jaccard"):
        try:
            distances[m] = dv.distance_matrix(table, m)
        except Exception:
            pass

    cats = sorted(groups.unique())
    label_a = cats[0] if cats else "A"
    label_b = cats[1] if len(cats) > 1 else (cats[0] if cats else "B")
    diff = None
    if len(cats) >= 2:
        try:
            diff = st.differential_abundance(table, groups, label_a, label_b, method="wilcoxon")
            diff["method"] = "Wilcoxon rank-sum + CLR"
        except Exception:
            diff = None

    alpha = dv.alpha_diversity(table)
    alpha_tests = {m: dv.group_test(alpha[m], groups)
                   for m in dv.ALPHA_METRICS if m in alpha.columns}

    rf = None
    try:
        rf = st.random_forest_analysis(table, groups, seed=42)
    except Exception:
        rf = None

    return _Prepared(
        table=table, rel=rel, groups=groups, meta=meta, rank_table=rank_table,
        rank_label=rank_label, distances=distances, diff=diff, alpha=alpha,
        alpha_tests=alpha_tests, rf=rf, label_a=label_a, label_b=label_b,
    )


class _SilentProgress:
    def update(self, *a, **k):
        pass

    logs: list = []


def get_prepared(cfg: dict) -> _Prepared:
    key = _fingerprint(cfg)
    with _lock:
        if key in _data_cache:
            return _data_cache[key]
    data = _build(cfg)
    with _lock:
        if len(_data_cache) >= _CACHE_LIMIT:
            _data_cache.pop(next(iter(_data_cache)))
        _data_cache[key] = data
    return data


def render_preview(cfg: dict, style_dict: dict, figure_id: str,
                   force: bool = False) -> tuple[Path, str]:
    """渲染一张预览图，返回 (文件路径, 图片说明)。"""
    if figure_id not in {p["id"] for p in PREVIEWABLE}:
        raise ValueError(f"不支持预览该图：{figure_id}")

    style = Style.from_dict(style_dict)
    data_key = _fingerprint(cfg)
    style_key = hashlib.md5(json.dumps(style.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:12]
    cache_key = f"{data_key}_{figure_id}_{style_key}"

    with _lock:
        cached = _file_cache.get(cache_key)
    if cached and cached.exists() and not force:
        return cached, "缓存"

    d = get_prepared(cfg)
    out = PREVIEW_DIR / f"{cache_key}.png"
    style_f = style.for_figure(figure_id)
    info = _dispatch(figure_id, d, style_f, out)

    with _lock:
        if len(_file_cache) >= _CACHE_LIMIT * 3:
            _file_cache.pop(next(iter(_file_cache)))
        _file_cache[cache_key] = out
    return Path(info["path"]), info.get("caption", "")


def _dist(table: pd.DataFrame, data: _Prepared) -> pd.DataFrame:
    """安全取 Bray-Curtis 距离矩阵。

    注意不能写 ``data.distances.get(...) or dv.distance_matrix(...)`` ——
    DataFrame 的真值判断会抛 "truth value is ambiguous"。
    """
    dist = data.distances.get("braycurtis")
    if dist is None:
        dist = dv.distance_matrix(table, "braycurtis")
        data.distances["braycurtis"] = dist
    return dist


def _dispatch(fid: str, d: _Prepared, style: Style, out: Path) -> dict:
    """按图 ID 调用对应绘图函数。"""
    T, G = d.table, d.groups

    if fid == "alpha_boxplot":
        return pl.plot_alpha_panel(d.alpha, G, dv.ALPHA_METRICS, out, style,
                                   test_results=d.alpha_tests)
    if fid == "rarefaction":
        return pl.plot_rarefaction(T, G, out, style)
    if fid == "rank_abundance":
        return pl.plot_rank_abundance(T, G, out, style)
    if fid == "pcoa":
        dist = _dist(T, d)
        coords, explained = dv.pcoa(dist)
        perm = dv.permanova(dist, G, permutations=199)
        return pl.plot_pcoa(coords, G, explained, "Bray-Curtis", out, style, permanova_res=perm)
    if fid == "nmds":
        dist = _dist(T, d)
        coords, stress = dv.nmds(dist, seed=42)
        return pl.plot_nmds(coords, G, stress, "Bray-Curtis", out, style)
    if fid == "pca":
        return pl.plot_pca(T, G, out, style)
    if fid == "umap":
        return pl.plot_umap(T, G, out, style)
    if fid == "tsne":
        return pl.plot_tsne(T, G, out, style)
    if fid == "plsda":
        return pl.plot_plsda(T, G, out, style)
    if fid == "scree":
        dist = _dist(T, d)
        _, explained = dv.pcoa(dist)
        return pl.plot_scree(explained, out, style, label="PCo")
    if fid == "distance_box":
        dist = _dist(T, d)
        long_df = dv.distance_boxplot_data(dist, G)
        anos = dv.anosim(dist, G, permutations=199)
        return pl.plot_distance_boxplot(long_df, "Bray-Curtis", out, style, test_res=anos)

    comp = d.rank_table if d.rank_table is not None else d.rel
    if fid == "composition_bar":
        return pl.plot_composition_bar(comp, G, out, style, rank_label=d.rank_label)
    if fid == "composition_area":
        return pl.plot_composition_area(comp, G, out, style, rank_label=d.rank_label)
    if fid == "heatmap":
        return pl.plot_heatmap(comp, G, out, style, rank_label=d.rank_label)
    if fid == "top_taxa":
        return pl.plot_top_taxa_box(comp, G, out, style, rank_label=d.rank_label)
    if fid == "venn":
        return pl.plot_venn(T, G, out, style, rank_label=d.rank_label)

    if d.diff is None:
        raise ValueError("当前数据无法进行差异分析（需要至少两个分组）。")
    diff = d.diff
    if fid == "volcano":
        return pl.plot_volcano(diff, out, style, group_a=d.label_a, group_b=d.label_b)
    if fid == "ma_plot":
        return pl.plot_ma(diff, out, style, group_a=d.label_a, group_b=d.label_b)
    if fid == "diff_bar":
        return pl.plot_diff_bar(diff, out, style, group_a=d.label_a, group_b=d.label_b)
    if fid == "forest":
        return pl.plot_forest(diff, out, style, group_a=d.label_a, group_b=d.label_b)
    if fid == "feature_box":
        feats = diff.head(6)["feature"].tolist()
        sub = T[[s for s in T.columns if G.get(s) in (d.label_a, d.label_b)]]
        return pl.plot_feature_box(sub, G.loc[sub.columns], feats, out, style,
                                   label_a=d.label_a, label_b=d.label_b, top_n=6)
    if fid == "lda":
        lda = st.lda_effect_size(T, G, d.label_a, d.label_b)
        if lda.empty:
            raise ValueError("数据不足以计算 LDA 效应量。")
        return pl.plot_lda(lda, out, style, group_a=d.label_a, group_b=d.label_b)

    if not (d.rf and d.rf.get("ok")):
        raise ValueError(f"随机森林未执行：{(d.rf or {}).get('reason', '原因未知')}")
    rf = d.rf
    if fid == "rf_importance":
        return pl.plot_rf_importance(rf["importance_top"], out, style)
    if fid == "roc":
        if not rf.get("roc"):
            raise ValueError("ROC 曲线仅适用于二分类分组。")
        return pl.plot_roc(rf["roc"], out, style, accuracy=rf["accuracy"])
    if fid == "confusion":
        return pl.plot_confusion(rf["confusion"], out, style, accuracy=rf["accuracy"])

    raise ValueError(f"未实现的预览类型：{fid}")


def list_presets() -> dict:
    return {"presets": list(PRESETS.keys()), "previewable": PREVIEWABLE}


def clear_cache() -> int:
    with _lock:
        n = len(_file_cache)
        _file_cache.clear()
        _data_cache.clear()
    for f in PREVIEW_DIR.glob("*.png"):
        try:
            f.unlink()
        except OSError:
            pass
    return n
