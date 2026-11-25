"""分析编排：从「丰度表 + 分组表」到「图集 + 表格 + HTML 报告」的全流程。

设计：每个分析模块由一个 ``run_*`` 私有函数负责，通过 ``_Ctx`` 上下文对象共享数据与产物，
任何单张图失败都只记入 warnings，不中断整体流程。
"""

from __future__ import annotations

import json
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import plots as pl
from ..style import Style
from . import diversity as dv
from . import multivariate as mv
from . import stats as st
from .io import meta_name_for, orient_feature_table, read_group_table, read_table, read_taxonomy
from .preprocess import (
    RANKS, aggregate_by_taxonomy, as_numeric_matrix, detect_abundance_type,
    ensure_relative, filter_features, rarefy, to_relative,
)

ANALYSIS_TYPES = {
    "amplicon16s": {
        "label": "16S 扩增子",
        "feature_label": "OTU/ASV",
        "rank_label": "属",
        "default_rank": "Genus",
        "default_beta": ["braycurtis", "jaccard"],
        "default_ord": ["pcoa", "nmds", "pca", "tsne", "umap"],
        "note": "基于 OTU/ASV 丰度表，默认按属水平展示群落组成。",
    },
}

# 本工具专注 16S 扩增子下游分析，固定使用这一个分析类型。
DEFAULT_ANALYSIS_TYPE = "amplicon16s"

ORDINATION_CHOICES = ["pcoa", "nmds", "pca", "tsne", "umap", "plsda", "rda", "dbrda", "cca", "cap"]
STAT_TEST_CHOICES = ["permanova", "anosim", "mrpp", "permdisp", "mantel", "envfit"]

ORD_LABELS = {
    "pcoa": "PCoA", "nmds": "NMDS", "pca": "PCA", "tsne": "t-SNE", "umap": "UMAP",
    "plsda": "PLS-DA", "rda": "RDA", "dbrda": "db-RDA", "cca": "CCA", "cap": "CAP",
}

STAT_LABELS = {
    "permanova": "PERMANOVA", "anosim": "ANOSIM", "mrpp": "MRPP",
    "permdisp": "PERMDISP", "mantel": "Mantel", "envfit": "envfit",
}


def default_config() -> dict:
    return {
        "analysis_type": DEFAULT_ANALYSIS_TYPE,
        "abundance_path": "",
        "metadata_path": "",
        "taxonomy_path": "",
        "env_path": "",
        "group_column": "",
        "compare_pairs": [],
        "rank": None,
        "style": {},

        # 预处理
        "filter": {"min_count": 0.0, "min_prevalence": 0.0, "min_total_frac": 0.0},
        "rarefy": False,
        "rarefy_depth": None,

        # 模块开关
        "run_alpha": True,
        "run_beta": True,
        "run_ordination": True,
        "run_multivariate": True,
        "run_within_group": True,
        "run_hypothesis": True,
        "run_advanced": True,
        "run_composition": True,
        "run_diff": True,
        "run_lda": True,
        "run_rf": True,

        # 详细设置
        "beta_metrics": None,
        "ordinations": None,
        "ordination_transform": "clr",
        "ordination_metric": "braycurtis",
        "stats_tests": None,
        "permutations": 999,
        "top_n_composition": 10,
        "top_n_heatmap": 30,
        "top_n_diff_features": 9,
        "venn_threshold": 0.5,
        "umap_neighbors": 15,
        "umap_min_dist": 0.1,
        "hypothesis_method": "auto",
        "p_adjust": "bh",
        "posthoc": True,
    }


# --------------------------------------------------------------------------
# 进度
# --------------------------------------------------------------------------
class Progress:
    def __init__(self, callback=None):
        self.callback = callback
        self.percent = 0.0
        self.stage = "初始化"
        self.logs: list[str] = []

    def update(self, percent: float, stage: str, message: str = ""):
        self.percent = float(max(0.0, min(100.0, percent)))
        self.stage = stage
        self.logs.append(f"[{self.percent:5.1f}%] {stage}" + (f" — {message}" if message else ""))
        if self.callback:
            try:
                self.callback(self.percent, stage, message)
            except Exception:
                pass


# --------------------------------------------------------------------------
# 上下文
# --------------------------------------------------------------------------
@dataclass
class _Ctx:
    cfg: dict
    style: Style
    out_dir: Path
    fig_dir: Path
    tab_dir: Path
    progress: Progress
    result: dict
    table: pd.DataFrame = None          # 过滤后的丰度表（特征 × 样本）
    table_div: pd.DataFrame = None      # 用于多样性的表（可能已抽平）
    rel: pd.DataFrame = None
    meta: pd.DataFrame = None
    groups: pd.Series = None
    tax: pd.DataFrame = None
    rank_table: pd.DataFrame = None
    rank_label: str = "Feature"
    env: pd.DataFrame = None            # 数值化环境因子（样本 × 变量）
    env_raw: pd.DataFrame = None        # 原始环境因子
    distances: dict = field(default_factory=dict)
    diff_tables: dict = field(default_factory=dict)
    counter: int = 0

    # ---- 产物登记 ----
    def add_fig(self, fig_dict: dict, section: str):
        self.result["figures"].append({**fig_dict, "section": section})

    def add_table(self, name: str, caption: str):
        self.result["tables"].append({"name": name, "caption": caption})

    def warn(self, msg: str):
        self.result["warnings"].append(msg)

    def next_index(self) -> str:
        self.counter += 1
        return f"{self.counter:02d}"

    def plot(self, figure_id: str, filename: str, section: str, fn, *args, **kwargs):
        """统一的绘图调用：解析逐图样式、执行、登记、错误隔离。"""
        style = self.style.for_figure(figure_id)
        try:
            info = fn(*args, out_path=self.fig_dir / filename, style=style, **kwargs)
            self.add_fig(info, section)
            return info
        except Exception as exc:
            self.warn(f"{filename} 绘制失败：{exc}")
            self.result.setdefault("figure_errors", []).append(
                {"figure": filename, "error": str(exc)}
            )
            return None


# --------------------------------------------------------------------------
# 输入
# --------------------------------------------------------------------------
def _match(a, b):
    import re

    def key(x):
        return re.sub(r"[\s_\-]+", "", str(x)).lower()

    bl = {key(x): x for x in b}
    common = [x for x in a if key(x) in bl]
    only_a = [x for x in a if key(x) not in bl]
    ka = {key(x) for x in common}
    only_b = [x for x in b if key(x) not in ka]
    return common, only_a, only_b


def load_inputs(cfg: dict, progress: Progress):
    p = progress
    p.update(2, "读取输入", "加载丰度表")
    table_raw = read_table(cfg["abundance_path"], index_col=0)
    meta = read_group_table(cfg["metadata_path"])

    tax = None
    if cfg.get("taxonomy_path"):
        try:
            tax = read_taxonomy(cfg["taxonomy_path"])
        except Exception as exc:
            p.logs.append(f"分类表读取失败，已忽略：{exc}")

    p.update(6, "读取输入", "识别表格方向并匹配样本")
    oriented, info = orient_feature_table(table_raw, list(meta.index))
    table = as_numeric_matrix(oriented)

    rename = {}
    for col in table.columns:
        m = meta_name_for(col, list(meta.index))
        if m is not None:
            rename[col] = m
    table = table.rename(columns=rename)
    table = table.loc[:, ~table.columns.duplicated()]

    common, only_table, only_meta = _match(table.columns, meta.index)
    if len(common) < 2:
        raise ValueError(
            "丰度表与分组表没有匹配上足够的样本。请检查样本名是否一致。\n"
            f"丰度表样本示例：{list(table.columns)[:5]}\n"
            f"分组表样本示例：{list(meta.index)[:5]}"
        )
    table, meta = table[common], meta.loc[common]

    if tax is not None:
        tax = tax.loc[[f for f in tax.index if f in table.index]]
        if tax.empty:
            tax = None

    info_out = {
        "transposed": info.get("transposed", False),
        "orient_reason": info.get("reason", ""),
        "n_samples": int(table.shape[1]),
        "n_features": int(table.shape[0]),
        "samples_only_in_table": [str(s) for s in only_table[:50]],
        "samples_only_in_metadata": [str(s) for s in only_meta[:50]],
        "n_only_in_table": len(only_table),
        "n_only_in_metadata": len(only_meta),
        "has_taxonomy": tax is not None,
        "metadata_columns": [str(c) for c in meta.columns],
        "library_sizes": table.sum(axis=0).astype(float).to_dict(),
    }

    # 环境因子表（可选）
    env_raw = None
    if cfg.get("env_path"):
        try:
            env_raw = read_table(cfg["env_path"], index_col=0)
            common_env, only_env, _ = _match(env_raw.index, meta.index)
            if len(common_env) >= 2:
                env_raw = env_raw.loc[[meta_name_for(s, list(meta.index)) or s for s in common_env]]
                info_out["env_columns"] = [str(c) for c in env_raw.columns]
                info_out["n_env_samples"] = int(env_raw.shape[0])
            else:
                p.logs.append("环境因子表与分组表样本无法匹配，已忽略。")
                env_raw = None
        except Exception as exc:
            p.logs.append(f"环境因子表读取失败，已忽略：{exc}")
            env_raw = None

    p.update(12, "读取输入", f"{table.shape[1]} 个样本 × {table.shape[0]} 个特征")
    return table, meta, tax, env_raw, info_out


def prepare_env(env_table: pd.DataFrame | None, groups: pd.Series) -> pd.DataFrame | None:
    """把环境因子表数值化：数值列保留，分类列做 one-hot。"""
    frames = []
    if env_table is not None and not env_table.empty:
        for col in env_table.columns:
            s = env_table[col]
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().sum() >= max(2, int(0.6 * len(s))):
                frames.append(num.astype(float).rename(str(col)))
            else:
                dummies = pd.get_dummies(s.astype(str), prefix=str(col), drop_first=True, dtype=float)
                frames.extend([dummies[c] for c in dummies.columns])

    # 分组本身也作为约束变量，保证没有环境表时 RDA/CCA 仍可用
    if groups is not None and groups.nunique() >= 2:
        dummies = pd.get_dummies(groups.astype(str), prefix="Group", drop_first=True, dtype=float)
        # 去掉 Group_ 前缀，图上直接显示组名，避免与图例重复且过长
        dummies.columns = [str(c).replace("Group_", "", 1) for c in dummies.columns]
        frames.extend([dummies[c] for c in dummies.columns])

    if not frames:
        return None
    env = pd.concat(frames, axis=1)
    env = env.loc[:, ~env.columns.duplicated()]
    env = env.dropna(axis=0, how="any")
    if env.shape[1] == 0 or env.shape[0] < 3:
        return None
    # 去掉零方差列
    env = env.loc[:, env.std(ddof=0) > 0]
    return env if env.shape[1] > 0 else None


# --------------------------------------------------------------------------
# 各分析模块
# --------------------------------------------------------------------------
def run_alpha(ctx: _Ctx):
    cfg, p = ctx.cfg, ctx.progress
    p.update(25, "Alpha 多样性", "计算多样性指数")
    alpha = dv.alpha_diversity(ctx.table_div)

    out = alpha.copy()
    out.insert(0, "Group", ctx.groups.reindex(out.index).astype(str))
    out.to_csv(ctx.tab_dir / "alpha_diversity.csv", encoding="utf-8-sig")
    ctx.add_table("alpha_diversity.csv", "各样本 alpha 多样性指数。")

    tests = {}
    for metric in dv.ALPHA_METRICS:
        if metric in alpha.columns:
            tests[metric] = dv.group_test(alpha[metric], ctx.groups)
    ctx.result["alpha_tests"] = tests

    pd.DataFrame(tests).T.reset_index().rename(columns={"index": "metric"}).to_csv(
        ctx.tab_dir / "alpha_tests.csv", index=False, encoding="utf-8-sig")
    ctx.add_table("alpha_tests.csv", "alpha 多样性组间检验结果。")

    ctx.plot("alpha_boxplot", "01_alpha_boxplot.png", "Alpha 多样性",
             pl.plot_alpha_panel, alpha, ctx.groups, dv.ALPHA_METRICS, test_results=tests)
    ctx.plot("alpha_heatmap", "02_alpha_heatmap.png", "Alpha 多样性",
             pl.plot_alpha_heatmap, alpha, ctx.groups)
    ctx.plot("alpha_rarefaction", "03_alpha_rarefaction.png", "Alpha 多样性",
             pl.plot_alpha_rarefaction, ctx.table_div, ctx.groups)
    ctx.plot("sample_quality", "03b_sample_quality.png", "Alpha 多样性",
             pl.plot_sample_quality, ctx.table, ctx.groups)

    ctx.result["alpha_table"] = alpha


def run_depth(ctx: _Ctx):
    p = ctx.progress
    p.update(30, "测序深度", "稀释曲线与累积曲线")
    ctx.plot("rarefaction", "04_rarefaction.png", "测序深度",
             pl.plot_rarefaction, ctx.table_div, ctx.groups)
    ctx.plot("accumulation", "05_accumulation.png", "测序深度",
             pl.plot_accumulation, ctx.table_div, ctx.groups)
    ctx.plot("rank_abundance", "06_rank_abundance.png", "测序深度",
             pl.plot_rank_abundance, ctx.table_div, ctx.groups)


def run_beta(ctx: _Ctx):
    cfg, p = ctx.cfg, ctx.progress
    info = ANALYSIS_TYPES.get(cfg["analysis_type"], ANALYSIS_TYPES["amplicon16s"])
    metrics = cfg.get("beta_metrics") or info["default_beta"]
    metrics = [m for m in metrics if m in dv.BETA_METRICS] or ["braycurtis"]
    perms = int(cfg.get("permutations", 999))
    tests = cfg.get("stats_tests") or STAT_TEST_CHOICES

    beta_summary = {}
    for i, metric in enumerate(metrics):
        label = dv.BETA_LABELS[metric]
        p.update(34 + i * 2, "Beta 多样性", f"{label} 距离矩阵")
        dist = dv.distance_matrix(ctx.table_div, metric)
        ctx.distances[metric] = dist
        dist.to_csv(ctx.tab_dir / f"distance_{metric}.csv", encoding="utf-8-sig")

        entry: dict = {"label": label, "permanova": {}, "anosim": {}, "mrpp": {}, "permdisp": {}}
        if "permanova" in tests:
            entry["permanova"] = dv.permanova(dist, ctx.groups, permutations=perms)
        if "anosim" in tests:
            entry["anosim"] = dv.anosim(dist, ctx.groups, permutations=perms)
        if "mrpp" in tests:
            try:
                entry["mrpp"] = mv.mrpp(dist, ctx.groups, permutations=perms)
            except Exception as exc:
                ctx.warn(f"MRPP（{label}）失败：{exc}")
        if "permdisp" in tests:
            try:
                entry["permdisp"] = mv.permdisp(dist, ctx.groups, permutations=perms)
            except Exception as exc:
                ctx.warn(f"PERMDISP（{label}）失败：{exc}")
        beta_summary[metric] = entry

        # 两两检验
        try:
            pair = mv.pairwise_test(dist, ctx.groups, method="permanova",
                                    permutations=min(perms, 999))
            if not pair.empty:
                pair.to_csv(ctx.tab_dir / f"pairwise_permanova_{metric}.csv",
                            index=False, encoding="utf-8-sig")
                pair_mrpp = mv.pairwise_test(dist, ctx.groups, method="mrpp",
                                             permutations=min(perms, 999))
                if not pair_mrpp.empty:
                    pair_mrpp.to_csv(ctx.tab_dir / f"pairwise_mrpp_{metric}.csv",
                                     index=False, encoding="utf-8-sig")
        except Exception as exc:
            ctx.warn(f"两两检验（{label}）失败：{exc}")

        # 距离分布
        try:
            long_df = dv.distance_boxplot_data(dist, ctx.groups)
            ctx.plot(f"distance_box", f"08_distance_{metric}.png", "组间差异",
                     pl.plot_distance_boxplot, long_df, label, test_res=entry["anosim"] or None)
        except Exception as exc:
            ctx.warn(f"距离箱线图（{label}）失败：{exc}")

        # 组间 vs 组内 对比（先把「分组是否成立」讲清楚）
        try:
            ctx.plot("between_vs_within", f"08b_between_within_{metric}.png", "组间差异",
                     pl.plot_between_vs_within, dist, ctx.groups, metric_label=label,
                     permanova_p=(entry.get("permanova") or {}).get("p"))
        except Exception as exc:
            ctx.warn(f"组间/组内对比图（{label}）失败：{exc}")

        # PERMDISP 可视化
        if entry.get("permdisp"):
            try:
                ctx.plot("betadisper", f"09_permdisp_{metric}.png", "多元统计检验",
                         pl.plot_permdisp, entry["permdisp"], metric_label=label)
            except Exception as exc:
                ctx.warn(f"PERMDISP 绘图（{label}）失败：{exc}")
        # Mantel
        if "mantel" in tests and ctx.env_raw is not None:
            try:
                mres = mv.mantel_heatmap(dist, ctx.env_raw, permutations=min(perms, 999))
                if not mres.empty:
                    mres.to_csv(ctx.tab_dir / f"mantel_{metric}.csv", index=False,
                                encoding="utf-8-sig")
                    beta_summary[metric]["mantel"] = mres.to_dict("records")
                    ctx.plot("mantel", f"10_mantel_{metric}.png", "多元统计检验",
                             pl.plot_mantel, mres, distance_label=label)
            except Exception as exc:
                ctx.warn(f"Mantel 检验（{label}）失败：{exc}")

    # 同时展示多种距离的 PCoA 对比面板
    ctx.result["beta"] = {k: _slim_beta(v) for k, v in beta_summary.items()}
    ctx.add_table("distance_*.csv", "各样本间的距离矩阵。")
    ctx.add_table("pairwise_permanova_*.csv", "两两分组 PERMANOVA（含 BH 校正）。")
    return metrics


def _slim_beta(entry: dict) -> dict:
    """剔除结果中不可 JSON 序列化的大对象（如 PERMDISP 的距离 DataFrame）。"""
    out = {}
    for key, val in entry.items():
        if key == "mantel":
            out[key] = val
            continue
        if isinstance(val, dict):
            slim = {k: v for k, v in val.items() if k not in ("distances",)}
            out[key] = slim
        else:
            out[key] = val
    return out


def _dist_for(ctx: _Ctx, metric: str) -> pd.DataFrame:
    """取距离矩阵：已算过就复用，否则现算。

    注意不能写 ``ctx.distances.get(m) or ...`` —— DataFrame 的真值判断会抛异常。
    """
    dist = ctx.distances.get(metric)
    if dist is None:
        dist = dv.distance_matrix(ctx.table_div, metric)
        ctx.distances[metric] = dist
    return dist


def run_ordination(ctx: _Ctx, beta_metrics: list[str]):
    cfg, p = ctx.cfg, ctx.progress
    choices = cfg.get("ordinations")
    if choices is None:
        choices = ANALYSIS_TYPES.get(cfg["analysis_type"], {}).get(
            "default_ord", ["pcoa", "nmds", "pca", "tsne", "umap"])
    choices = [c for c in choices if c in ORDINATION_CHOICES]
    if not choices:
        return

    transform = cfg.get("ordination_transform", "clr")
    metric = cfg.get("ordination_metric", "braycurtis")
    if metric not in dv.BETA_METRICS:
        metric = "braycurtis"
    perms = int(cfg.get("permutations", 999))
    tests = cfg.get("stats_tests") or STAT_TEST_CHOICES
    n = len(choices)
    summary: dict = {}

    for i, method in enumerate(choices):
        p.update(46 + i, "排序分析", ORD_LABELS.get(method, method))
        fid = method

        try:
            if method == "pcoa":
                for m2 in (beta_metrics or ["braycurtis"]):
                    dist = _dist_for(ctx, m2)
                    coords, explained = dv.pcoa(dist)
                    entry = (ctx.result.get("beta") or {}).get(m2, {})
                    ctx.plot(fid, f"07_pcoa_{m2}.png", "排序分析", pl.plot_pcoa,
                             coords, ctx.groups, explained, dv.BETA_LABELS[m2],
                             permanova_res=entry.get("permanova"))
                    ctx.plot("scree", f"07b_scree_{m2}.png", "排序分析", pl.plot_scree,
                             explained, label=f"PCo{m2[:2].upper()}")
                    summary.setdefault("pcoa", {})[m2] = explained.tolist()

            elif method == "nmds":
                for m2 in (beta_metrics or ["braycurtis"]):
                    dist = _dist_for(ctx, m2)
                    coords, stress = dv.nmds(dist, seed=42)
                    ctx.plot(fid, f"11_nmds_{m2}.png", "排序分析", pl.plot_nmds,
                             coords, ctx.groups, stress, dv.BETA_LABELS[m2])
                    summary.setdefault("nmds", {})[m2] = float(stress)

            elif method == "pca":
                ctx.plot(fid, "12_pca.png", "排序分析", pl.plot_pca,
                         ctx.table, ctx.groups, transform=transform)

            elif method == "tsne":
                ctx.plot(fid, "13_tsne.png", "排序分析", pl.plot_tsne,
                         ctx.table, ctx.groups, transform=transform)

            elif method == "umap":
                ctx.plot(fid, "14_umap.png", "排序分析", pl.plot_umap,
                         ctx.table, ctx.groups, transform=transform,
                         n_neighbors=int(cfg.get("umap_neighbors", 15)),
                         min_dist=float(cfg.get("umap_min_dist", 0.1)))

            elif method == "plsda":
                res_plsda = None
                try:
                    from .ordination import plsda as _plsda

                    res_plsda = _plsda(ctx.table, ctx.groups, n_components=2)
                except Exception as exc:
                    ctx.warn(f"PLS-DA 计算失败：{exc}")
                ctx.plot(fid, "15_plsda.png", "排序分析", pl.plot_plsda,
                         ctx.table, ctx.groups, transform=transform)
                if res_plsda is not None and not res_plsda["vip"].empty:
                    ctx.plot("plsda_vip", "15b_plsda_vip.png", "排序分析",
                             pl.plot_plsda_vip, res_plsda["vip"])
                    res_plsda["vip"].to_csv(ctx.tab_dir / "plsda_vip.csv",
                                            index=False, encoding="utf-8-sig")
                    ctx.add_table("plsda_vip.csv", "PLS-DA 变量投影重要性（VIP）。")
                    summary.setdefault("plsda", {})["accuracy"] = res_plsda["accuracy"]

            elif method in ("rda", "dbrda", "cca"):
                if ctx.env is None:
                    ctx.warn(f"{ORD_LABELS[method]} 需要环境因子或至少两个分组，已跳过。")
                    continue
                common = [s for s in ctx.table.columns if s in ctx.env.index]
                if len(common) < 4:
                    ctx.warn(f"{ORD_LABELS[method]} 可用样本不足（{len(common)}），已跳过。")
                    continue
                sub_table = ctx.table[common]
                sub_env = ctx.env.loc[common]
                sub_groups = ctx.groups.loc[common]
                if method == "rda":
                    ctx.plot(fid, "16_rda.png", "排序分析", pl.plot_rda,
                             sub_table, sub_env, sub_groups)
                elif method == "dbrda":
                    ctx.plot(fid, "17_dbrda.png", "排序分析", pl.plot_dbrda,
                             sub_table, sub_env, sub_groups, metric=metric)
                else:
                    ctx.plot(fid, "18_cca.png", "排序分析", pl.plot_cca,
                             sub_table, sub_env, sub_groups)

            elif method == "cap":
                ctx.plot(fid, "19_cap.png", "排序分析", pl.plot_cap,
                         ctx.table, ctx.groups, metric=metric)

        except Exception as exc:
            ctx.warn(f"{ORD_LABELS.get(method, method)} 分析失败：{exc}")

    ctx.result["ordination"] = summary


def run_grouping(ctx: _Ctx):
    """分组总览 + 组内差异分析。

    这是"先组间、后组内"里的后半段：在组间差异分析之后，
    再看每个组内部是否一致，避免把组内异质性误读成组间差异。
    """
    cfg, p = ctx.cfg, ctx.progress
    p.update(58, "组内差异", "计算各组内部异质性")

    # 组内距离
    dist = ctx.distances.get("braycurtis")
    if dist is None and ctx.distances:
        dist = next(iter(ctx.distances.values()))
    if dist is not None:
        label = dv.BETA_LABELS.get("braycurtis", "Bray-Curtis")
        try:
            ctx.plot("within_group", "20c_within_group.png", "组内差异",
                     pl.plot_within_group_distance, dist, ctx.groups, metric_label=label)
        except Exception as exc:
            ctx.warn(f"组内距离分析失败：{exc}")

        # 组内异质性写入结果与结论
        try:
            from ..plots.grouping import _within_distances

            within = _within_distances(dist, ctx.groups)
            if not within.empty:
                summ = within.groupby("Group")["distance"].agg(["median", "mean", "std", "count"])
                summ["CV"] = summ["std"] / summ["mean"].replace(0, np.nan)
                summ.round(4).to_csv(ctx.tab_dir / "within_group_distance.csv",
                                     encoding="utf-8-sig")
                ctx.add_table("within_group_distance.csv",
                              "各组内部两两距离的中位数/均值/标准差/变异系数。")
                ctx.result["within_group"] = {
                    "metric": label,
                    "by_group": {str(g): {
                        "median": float(r["median"]), "mean": float(r["mean"]),
                        "std": float(r["std"]) if pd.notna(r["std"]) else None,
                        "cv": float(r["CV"]) if pd.notna(r["CV"]) else None,
                        "n_pairs": int(r["count"]),
                    } for g, r in summ.iterrows()},
                }
        except Exception as exc:
            ctx.warn(f"组内异质性统计失败：{exc}")

    # 组内 alpha 一致性
    alpha = ctx.result.get("alpha_table")
    if alpha is not None:
        try:
            ctx.plot("within_alpha", "20d_within_alpha.png", "组内差异",
                     pl.plot_within_group_diversity, alpha, ctx.groups)
        except Exception as exc:
            ctx.warn(f"组内多样性一致性图失败：{exc}")

    # 分组总览
    try:
        ctx.plot("group_overview", "20e_group_overview.png", "组内差异",
                 pl.plot_group_overview, ctx.groups, ctx.table, dist, alpha)
    except Exception as exc:
        ctx.warn(f"分组总览图失败：{exc}")


def _forced_test(hyp, values, groups, method: str) -> dict:
    """按用户指定的方法做两组比较（多组时仍走多组流程）。"""
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame({"v": values, "g": groups.astype(str)}).dropna()
    cats = sorted(frame["g"].unique())
    if len(cats) < 2:
        return hyp.auto_test(values, groups)

    if len(cats) > 2:
        res = hyp.compare_multiple_groups(
            {c: frame.loc[frame["g"] == c, "v"].to_numpy(dtype=float) for c in cats},
            auto=(method == "auto"))
    else:
        a = frame.loc[frame["g"] == cats[0], "v"].to_numpy(dtype=float)
        b = frame.loc[frame["g"] == cats[1], "v"].to_numpy(dtype=float)
        res = hyp.compare_two_groups(a, b, auto=False, equal_var=(method in ("t", "anova")))
        if method == "wilcoxon_signed":
            res = hyp.compare_paired(a, b)
    res = dict(res)
    res["method_auto"] = False
    res.setdefault("reason", f"用户手动指定：{method}")
    return res


def run_hypothesis(ctx: _Ctx):
    """完整的假设检验：先查正态性与方差齐性，再自动选择合适的方法。

    覆盖：Shapiro-Wilk 正态性、Levene/Bartlett 方差齐性、t/Welch、Wilcoxon 秩和与符号秩、
    单因素 ANOVA + Tukey HSD、Kruskal-Wallis + Dunn、卡方 / Fisher 精确检验。
    """
    from . import hypothesis as hyp

    cfg, p = ctx.cfg, ctx.progress
    p.update(61, "假设检验", "正态性 / 方差齐性 / 组间比较")
    rows: list[dict] = []
    forced = str(cfg.get("hypothesis_method") or "auto").lower()

    # ---- 1. Alpha 多样性指标 ----
    alpha = ctx.result.get("alpha_table")
    if alpha is not None:
        for metric in dv.ALPHA_METRICS:
            if metric not in alpha.columns:
                continue
            try:
                res = hyp.auto_test(alpha[metric], ctx.groups) if forced == "auto" \
                    else _forced_test(hyp, alpha[metric], ctx.groups, forced)
                rows.append({"scope": "Alpha 多样性", "variable": metric, **res})
            except Exception as exc:
                ctx.warn(f"假设检验（{metric}）失败：{exc}")

    # ---- 2. 分组列之外的分类协变量 ----
    for col in ctx.meta.columns:
        name = str(col)
        if name == ctx.result.get("group_column"):
            continue
        vals = ctx.meta[col].astype(str)
        if vals.nunique() < 2 or vals.nunique() > max(10, len(vals) // 2):
            continue
        try:
            tab = pd.crosstab(ctx.groups, vals)
            res = hyp.compare_categorical(tab, str(ctx.result.get("group_column")), name)
            rows.append({
                "scope": "分类协变量关联", "variable": name,
                "test": res.get("test"), "p": res.get("p"),
                "statistic": res.get("statistic"),
                "effect_size": res.get("effect_size"),
                "effect_name": "Cramér's V",
                "reason": res.get("reason"),
                "n_groups": int(vals.nunique()),
            })
        except Exception as exc:
            ctx.warn(f"分类检验（{name}）失败：{exc}")

    # ---- 3. 差异最显著的特征（作为示例，展示具体用了哪种检验） ----
    for tag, diff in list(ctx.diff_tables.items())[:2]:
        if diff.empty:
            continue
        parts = tag.split("_vs_")
        a = parts[0] if parts else ""
        b = parts[1] if len(parts) > 1 else ""
        for feat in diff.head(3)["feature"].tolist():
            if feat not in ctx.table.index:
                continue
            try:
                fa = ctx.table.loc[feat, [s for s in ctx.table.columns if ctx.groups.get(s) == a]]
                fb = ctx.table.loc[feat, [s for s in ctx.table.columns if ctx.groups.get(s) == b]]
                res = hyp.compare_two_groups(fa.to_numpy(dtype=float), fb.to_numpy(dtype=float))
                rows.append({"scope": f"特征检验（{tag}）", "variable": str(feat), **res})
            except Exception as exc:
                ctx.warn(f"特征检验（{feat}）失败：{exc}")

    if not rows:
        return

    # 精简成可落表的格式
    flat: list[dict] = []
    for r in rows:
        flat.append({
            "分析范围": r.get("scope", ""),
            "变量": r.get("variable", ""),
            "组数": r.get("n_groups"),
            "检验方法": r.get("test", ""),
            "统计量": r.get("statistic"),
            "p 值": r.get("p"),
            "显著性": hyp.significance_stars(r.get("p")) if r.get("p") is not None else "",
            "效应量": r.get("effect_size"),
            "效应量类型": r.get("effect_name", ""),
            "是否自动选择": r.get("method_auto"),
            "选择理由": r.get("reason", ""),
        })
    df = pd.DataFrame(flat)
    df.to_csv(ctx.tab_dir / "hypothesis_tests.csv", index=False, encoding="utf-8-sig")
    ctx.add_table("hypothesis_tests.csv",
                  "完整的假设检验结果：含正态性/方差齐性判断、方法选择理由、统计量与效应量。")
    ctx.result["hypothesis"] = {
        "rows": flat,
        "n_tests": len(flat),
        "methods_used": sorted({str(r["检验方法"]) for r in flat if r["检验方法"]}),
    }
    p.update(63, "假设检验", f"完成 {len(flat)} 项检验")


def run_advanced_plots(ctx: _Ctx):
    """高级图形：小提琴图、云雨图、密度图、蜂群图、棒棒糖图、气泡图、桑基图、维诺树状图。"""
    cfg, p = ctx.cfg, ctx.progress
    p.update(87, "高级图形", "小提琴 / 云雨 / 密度 / 蜂群")
    comp = ctx.rank_table if ctx.rank_table is not None else ctx.rel
    label = ctx.rank_label
    top_n = int(cfg.get("top_n_composition", 10))

    for fid, fname, fn, kw in [
        ("violin", "42_violin.png", pl.plot_violin,
         {"rank_label": label, "top_n": min(9, max(3, top_n))}),
        ("raincloud", "43_raincloud.png", pl.plot_raincloud,
         {"rank_label": label, "top_n": 6}),
        ("density", "44_density.png", pl.plot_density,
         {"rank_label": label, "top_n": 6}),
        ("beeswarm", "45_beeswarm.png", pl.plot_beeswarm,
         {"rank_label": label, "top_n": min(9, max(3, top_n))}),
    ]:
        ctx.plot(fid, fname, "高级图形", fn, comp, ctx.groups, **kw)

    # 棒棒糖图：取第一个比较组的差异结果
    if ctx.diff_tables:
        tag, diff = next(iter(ctx.diff_tables.items()))
        parts = tag.split("_vs_")
        try:
            ctx.plot("lollipop", f"46_lollipop_{tag}.png", f"差异分析：{tag.replace('_vs_', ' vs ')}",
                     pl.plot_lollipop, diff, group_a=parts[0] if parts else "",
                     group_b=parts[1] if len(parts) > 1 else "")
        except Exception as exc:
            ctx.warn(f"棒棒糖图失败：{exc}")

    # 气泡图：丰度 vs 差异倍数 vs 显著性
    if ctx.diff_tables and ctx.rank_table is not None:
        try:
            tag, diff = next(iter(ctx.diff_tables.items()))
            d = diff.copy()
            abund = ctx.rel.mean(axis=1)
            d["abundance"] = d["feature"].map(abund).fillna(0.0)
            d = d[d["abundance"] > 0]
            if not d.empty:
                ctx.plot("bubble", f"47_bubble_{tag}.png", "高级图形",
                         pl.plot_bubble, d, x_col="abundance", y_col="log2FC",
                         size_col="abundance", color_col="-log10FDR",
                         label_col="feature", top_n=40,
                         title="丰度 - 倍数变化 - 显著性气泡图")
        except Exception as exc:
            ctx.warn(f"气泡图失败：{exc}")

    # 桑基图：分类级之间的流向（有分类表时用 Phylum → Class → Order，否则 分组 → Top 类群）
    try:
        flow = _build_sankey_flow(ctx)
        if flow is not None and not flow.empty:
            ctx.plot("sankey", "48_sankey.png", "高级图形", pl.plot_sankey, flow)
    except Exception as exc:
        ctx.warn(f"桑基图失败：{exc}")

    # 分组 → 类群的流向图
    try:
        ctx.plot("stacked_flow", "49_group_flow.png", "高级图形",
                 pl.plot_stacked_flow, ctx.groups, comp, rank_label=label, top_n=top_n)
    except Exception as exc:
        ctx.warn(f"分组流向图失败：{exc}")

    # 维诺树状图 / 矩形树图（自动选择，面积严格正比于丰度）
    try:
        from ..plots.treemap import plot_treemap_auto

        ctx.plot("voronoi", "50_voronoi.png", "高级图形",
                 plot_treemap_auto, comp, rank_label=label,
                 top_n=min(18, int(cfg.get("top_n_heatmap", 30))))
    except Exception as exc:
        ctx.warn(f"树状图失败：{exc}")


def _build_sankey_flow(ctx: _Ctx) -> pd.DataFrame | None:
    """构造分类级之间的流向表：优先用分类表的高→低两级，否则用 分组 → Top 类群。"""
    tax = ctx.tax
    rel = ctx.rel
    if tax is not None and ctx.rank_table is not None:
        ranks = [r for r in RANKS if r in tax.columns]
        common = [f for f in rel.index if f in tax.index]
        if len(ranks) >= 2 and common:
            hi, lo = ranks[0], ranks[-1]
            sub = rel.loc[common]
            grouped = sub.groupby([tax.loc[common, hi], tax.loc[common, lo]]).sum().sum(axis=1)
            flow = grouped.reset_index()
            flow.columns = ["source", "target", "weight"]
            flow = flow[flow["weight"] > 0]
            # 只保留流量最大的若干分支，避免图过于杂乱
            flow = flow.sort_values("weight", ascending=False).head(40)
            return flow

    # 退化方案：分组 → Top 类群
    comp = ctx.rank_table if ctx.rank_table is not None else rel
    top = comp.mean(axis=1).sort_values(ascending=False).head(8).index
    rows = []
    for g in sorted(ctx.groups.unique()):
        cols = [s for s in comp.columns if ctx.groups.get(s) == g]
        if not cols:
            continue
        sub = comp[cols].mean(axis=1)
        sub = sub[sub.index.isin(top)]
        total = float(sub.sum()) or 1.0
        for feat, v in sub.items():
            rows.append({"source": str(g), "target": str(feat)[:40],
                         "weight": float(v) / total})
    return pd.DataFrame(rows) if rows else None


def run_multivariate_tests(ctx: _Ctx):
    """envfit / Procrustes / Mantel（当没有环境表时跳过）。"""
    cfg, p = ctx.cfg, ctx.progress
    tests = cfg.get("stats_tests") or STAT_TEST_CHOICES
    if "envfit" not in tests:
        return
    if ctx.env_raw is None and ctx.env is None:
        return

    p.update(43, "多元统计检验", "envfit 因子拟合")
    env_for_fit = ctx.env_raw if ctx.env_raw is not None else ctx.env
    if env_for_fit is None or env_for_fit.empty:
        return

    try:
        coords, _ = dv.pcoa(_dist_for(ctx, "braycurtis"))
        common = [s for s in coords.index if s in env_for_fit.index]
        if len(common) >= 6:
            res = mv.envfit(coords.loc[common], env_for_fit.loc[common],
                            permutations=min(int(cfg.get("permutations", 999)), 999))
            if not res.empty:
                res.to_csv(ctx.tab_dir / "envfit.csv", index=False, encoding="utf-8-sig")
                ctx.add_table("envfit.csv", "环境/分组因子与排序结构的拟合结果。")
                ctx.plot("envfit", "20_envfit.png", "多元统计检验", pl.plot_envfit, res)
                ctx.result["envfit"] = res.to_dict("records")
    except Exception as exc:
        ctx.warn(f"envfit 失败：{exc}")


def run_procrustes(ctx: _Ctx):
    """Procrustes：比较两套排序结果是否一致。

    有 ≥2 个距离指标时比较两种距离的 PCoA；否则比较 PCoA 与 PCA。
    """
    from .ordination import pca_ordination

    try:
        metrics = [m for m in ctx.distances if m in dv.BETA_METRICS]
        if len(metrics) >= 2:
            m1, m2 = metrics[0], metrics[1]
            c1, _ = dv.pcoa(_dist_for(ctx, m1))
            c2, _ = dv.pcoa(_dist_for(ctx, m2))
            label_a, label_b = f"PCoA({dv.BETA_LABELS[m1]})", f"PCoA({dv.BETA_LABELS[m2]})"
        else:
            m1 = metrics[0] if metrics else "braycurtis"
            c1, _ = dv.pcoa(_dist_for(ctx, m1))
            pca_res = pca_ordination(ctx.table, n_components=4,
                                     transform=ctx.cfg.get("ordination_transform", "clr"))
            c2 = pca_res["coords"]
            label_a, label_b = f"PCoA({dv.BETA_LABELS[m1]})", "PCA(CLR)"

        common = [s for s in c1.index if s in c2.index]
        if len(common) < 4:
            return
        res = mv.procrustes(c1.loc[common, c1.columns[:2]],
                            c2.loc[common, c2.columns[:2]],
                            permutations=min(int(ctx.cfg.get("permutations", 999)), 999))
        ctx.plot("procrustes", "20b_procrustes.png", "多元统计检验",
                 pl.plot_procrustes, res, ctx.groups, label_a=label_a, label_b=label_b)
        ctx.result["procrustes"] = {
            "m2": res.get("m2"), "r": res.get("r"), "p": res.get("p"),
            "comparison": f"{label_a} vs {label_b}",
        }
        m2v = res.get("m2")
        pv = res.get("p")
        ctx.add_table(
            "procrustes.json",
            f"Procrustes 一致性（{label_a} vs {label_b}）："
            + (f"m² = {m2v:.4f}，p = {pv:.3f}" if m2v is not None and pv is not None else "结果不可用"),
        )
    except Exception as exc:
        ctx.warn(f"Procrustes 失败：{exc}")


def run_composition(ctx: _Ctx):
    cfg, p = ctx.cfg, ctx.progress
    p.update(66, "群落组成", "堆叠柱状图与热图")
    comp = ctx.rank_table if ctx.rank_table is not None else ctx.rel
    label = ctx.rank_label

    ctx.plot("composition_bar", "21_composition_bar.png", "群落组成",
             pl.plot_composition_bar, comp, ctx.groups, rank_label=label,
             top_n=int(cfg.get("top_n_composition", 10)))
    ctx.plot("composition_area", "22_composition_area.png", "群落组成",
             pl.plot_composition_area, comp, ctx.groups, rank_label=label,
             top_n=int(cfg.get("top_n_composition", 10)) + 2)
    ctx.plot("heatmap", "23_heatmap.png", "群落组成",
             pl.plot_heatmap, comp, ctx.groups,
             top_n=int(cfg.get("top_n_heatmap", 30)), rank_label=label)
    ctx.plot("top_taxa", "24_top_taxa_box.png", "群落组成",
             pl.plot_top_taxa_box, comp, ctx.groups, rank_label=label, top_n=9)
    ctx.plot("venn", "25_venn.png", "群落组成",
             pl.plot_venn, ctx.table, ctx.groups,
             threshold=float(cfg.get("venn_threshold", 0.5)), rank_label=label)
    ctx.plot("core_microbiome", "25b_core_microbiome.png", "群落组成",
             pl.plot_core_microbiome, ctx.table, ctx.groups)

    if ctx.groups.nunique() == 3:
        ctx.plot("ternary", "25c_ternary.png", "群落组成",
                 pl.plot_ternary, comp, ctx.groups, rank_label=label)

    comp_rel = to_relative(comp)
    comp_rel.to_csv(ctx.tab_dir / "composition_relative.csv", encoding="utf-8-sig")
    comp.to_csv(ctx.tab_dir / "composition_table.csv", encoding="utf-8-sig")
    ctx.add_table("composition_relative.csv", f"{label}水平相对丰度表。")
    ctx.add_table("composition_table.csv", f"{label}水平原始丰度表。")


def run_differential(ctx: _Ctx):
    cfg, p = ctx.cfg, ctx.progress
    cats = sorted(ctx.groups.unique())
    pairs = cfg.get("compare_pairs") or []
    if not pairs:
        pairs = [[cats[i], cats[j]] for i in range(len(cats)) for j in range(i + 1, len(cats))]
    pairs = [(a, b) for a, b in pairs if a in cats and b in cats]

    if not pairs:
        ctx.warn("少于两个分组，跳过差异分析。")
        return []
    if len(pairs) > 6:
        ctx.warn(f"比较组较多（{len(pairs)}），仅执行前 6 组。")
        pairs = pairs[:6]

    for k, (a, b) in enumerate(pairs):
        tag = f"{_safe(a)}_vs_{_safe(b)}"
        p.update(68 + k * 2, "差异分析", f"{a} vs {b}")
        try:
            # 16S 走 CLR 空间的 Wilcoxon 秩和检验（适合稀疏、组成型数据）
            diff = st.differential_abundance(ctx.table, ctx.groups, a, b, method="wilcoxon")
            diff["method"] = "Wilcoxon rank-sum + CLR"
        except Exception as exc:
            ctx.warn(f"差异分析 {a} vs {b} 失败：{exc}")
            continue

        ctx.diff_tables[tag] = diff
        diff.to_csv(ctx.tab_dir / f"diff_{tag}.csv", index=False, encoding="utf-8-sig")

        # 供自动结论生成器使用的摘要
        try:
            sig = diff[(diff["FDR"] < 0.05) & (diff["log2FC"].abs() >= 1.0)]
            ctx.result.setdefault("diff_summary", []).append({
                "group_a": a, "group_b": b,
                "method": str(diff.get("method", pd.Series(["Wilcoxon"])).iloc[0])
                if "method" in diff else "Wilcoxon rank-sum",
                "n_significant": int(len(sig)),
                "n_up": int((sig["log2FC"] > 0).sum()),
                "n_down": int((sig["log2FC"] < 0).sum()),
                "top_features": [
                    {
                        "feature": str(r["feature"]),
                        "log2FC": float(r["log2FC"]),
                        "p": float(r["p"]) if pd.notna(r["p"]) else None,
                        "FDR": float(r["FDR"]) if pd.notna(r["FDR"]) else None,
                        "enriched_in": a if r["log2FC"] > 0 else b,
                    }
                    for _, r in diff.head(5).iterrows()
                ],
            })
        except Exception as exc:
            ctx.warn(f"差异摘要生成失败（不影响出图）：{exc}")

        ctx.plot("volcano", f"26_volcano_{tag}.png", f"差异分析：{a} vs {b}",
                 pl.plot_volcano, diff, group_a=a, group_b=b)
        ctx.plot("ma_plot", f"27_ma_{tag}.png", f"差异分析：{a} vs {b}",
                 pl.plot_ma, diff, group_a=a, group_b=b)
        ctx.plot("diff_bar", f"28_diffbar_{tag}.png", f"差异分析：{a} vs {b}",
                 pl.plot_diff_bar, diff, group_a=a, group_b=b)
        ctx.plot("forest", f"29_forest_{tag}.png", f"差异分析：{a} vs {b}",
                 pl.plot_forest, diff, group_a=a, group_b=b)
        ctx.plot("diff_heatmap", f"29b_diffheat_{tag}.png", f"差异分析：{a} vs {b}",
                 pl.plot_diff_heatmap, ctx.table, ctx.groups, diff,
                 group_a=a, group_b=b)

        # 差异特征箱线图（最常被要求的附图）
        try:
            sub = ctx.table[[s for s in ctx.table.columns
                             if ctx.groups.get(s) in (a, b)]]
            feats = diff.head(int(cfg.get("top_n_diff_features", 9)))["feature"].tolist()
            if feats:
                ctx.plot("feature_box", f"30_featurebox_{tag}.png", f"差异分析：{a} vs {b}",
                         pl.plot_feature_box, sub, ctx.groups.loc[sub.columns], feats,
                         label_a=a, label_b=b, top_n=len(feats))
        except Exception as exc:
            ctx.warn(f"差异特征箱线图（{a} vs {b}）失败：{exc}")

        if cfg.get("run_lda", True):
            try:
                lda = st.lda_effect_size(ctx.table, ctx.groups, a, b)
                if not lda.empty:
                    lda.to_csv(ctx.tab_dir / f"lda_{tag}.csv", index=False, encoding="utf-8-sig")
                    ctx.plot("lda", f"31_lda_{tag}.png", f"差异分析：{a} vs {b}",
                             pl.plot_lda, lda, group_a=a, group_b=b)
            except Exception as exc:
                ctx.warn(f"LDA（{a} vs {b}）失败：{exc}")

    if ctx.diff_tables:
        ctx.add_table("diff_*.csv", "两两差异分析结果（含 log2FC、p、FDR、效应量）。")
        ctx.add_table("lda_*.csv", "LEfSe 风格 LDA 效应量。")

    if len(cats) > 2:
        try:
            mg = st.multi_group_test(ctx.table, ctx.groups)
            mg.to_csv(ctx.tab_dir / "diff_multigroup_kruskal.csv", index=False, encoding="utf-8-sig")
            ctx.add_table("diff_multigroup_kruskal.csv", "多组 Kruskal-Wallis 整体检验。")
        except Exception as exc:
            ctx.warn(f"多组检验失败：{exc}")

    return pairs


def run_machine(ctx: _Ctx):
    cfg, p = ctx.cfg, ctx.progress
    p.update(83, "机器学习", "随机森林分类")
    try:
        rf = st.random_forest_analysis(ctx.table, ctx.groups, seed=42)
    except Exception as exc:
        ctx.warn(f"随机森林失败：{exc}")
        return

    if not rf.get("ok"):
        ctx.warn(f"随机森林未执行：{rf.get('reason')}")
        return

    rf["importance"].to_csv(ctx.tab_dir / "rf_importance.csv", index=False, encoding="utf-8-sig")
    ctx.plot("rf_importance", "32_rf_importance.png", "机器学习",
             pl.plot_rf_importance, rf["importance_top"])

    if rf.get("roc"):
        ctx.plot("roc", "33_roc.png", "机器学习",
                 pl.plot_roc, rf["roc"], accuracy=rf["accuracy"])
    if rf.get("confusion") is not None:
        ctx.plot("confusion", "34_confusion.png", "机器学习",
                 pl.plot_confusion, rf["confusion"], accuracy=rf["accuracy"])

    ctx.result["random_forest"] = {
        "accuracy": rf["accuracy"],
        "accuracy_std": rf["accuracy_std"],
        "n_splits": rf["n_splits"],
        "n_samples": rf["n_samples"],
        "classes": rf["classes"],
        "auc": (rf.get("roc") or {}).get("auc"),
        "accuracy_folds": rf["accuracy_folds"],
        "confusion": rf["confusion"].to_dict() if rf.get("confusion") is not None else None,
        "top_features": [
            {"feature": str(r["feature"]), "importance": float(r["importance"])}
            for _, r in rf["importance"].head(10).iterrows()
        ],
    }
    ctx.add_table("rf_importance.csv", "随机森林特征重要性。")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run_analysis(config: dict, output_root: str | Path, progress_callback=None) -> dict:
    cfg = {**default_config(), **(config or {})}
    p = Progress(progress_callback)
    started = datetime.now()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out_dir = Path(output_root) / run_id
    fig_dir, tab_dir = out_dir / "figures", out_dir / "tables"
    for d in (fig_dir, tab_dir):
        d.mkdir(parents=True, exist_ok=True)

    meta_info = ANALYSIS_TYPES.get(cfg["analysis_type"], ANALYSIS_TYPES["amplicon16s"])
    style = Style.from_dict(cfg.get("style"))

    result: dict = {
        "run_id": run_id,
        "output_dir": str(out_dir),
        "analysis_type": cfg["analysis_type"],
        "analysis_label": meta_info["label"],
        "started_at": started.isoformat(timespec="seconds"),
        "figures": [],
        "tables": [],
        "sections": [],
        "warnings": [],
        "log": [],
        "style": style.to_dict(),
    }

    ctx = _Ctx(cfg=cfg, style=style, out_dir=out_dir, fig_dir=fig_dir, tab_dir=tab_dir,
               progress=p, result=result)

    try:
        # ---- 1. 读取 ----
        table, meta, tax, env_raw, info_out = load_inputs(cfg, p)
        ctx.table, ctx.meta, ctx.tax = table, meta, tax
        ctx.env_raw = env_raw
        result["input_info"] = info_out

        group_col = cfg.get("group_column") or str(meta.columns[0])
        if group_col not in meta.columns:
            group_col = str(meta.columns[0])
        ctx.groups = meta[group_col].astype(str)
        ctx.env = prepare_env(env_raw, ctx.groups)

        result["group_column"] = group_col
        result["groups"] = {g: int((ctx.groups == g).sum()) for g in sorted(ctx.groups.unique())}
        result["metadata_columns"] = [str(c) for c in meta.columns]
        p.update(17, "数据校验", f"分组列：{group_col}，共 {len(result['groups'])} 组")

        # ---- 2. 预处理 ----
        flt = cfg.get("filter", {}) or {}
        table_f, flt_stats = filter_features(
            table,
            min_count=float(flt.get("min_count", 0) or 0),
            min_prevalence=float(flt.get("min_prevalence", 0) or 0),
            min_total_frac=float(flt.get("min_total_frac", 0) or 0),
        )
        ctx.table = table_f
        result["filter_stats"] = flt_stats
        result["input_info"]["n_features_used"] = int(table_f.shape[0])

        # ---- 相对丰度自动识别 ----
        ab_info = detect_abundance_type(table_f)
        result["input_info"]["abundance_type"] = ab_info
        result["abundance_type"] = ab_info
        p.update(19, "预处理", ab_info["reason"])
        if ab_info["is_relative"]:
            ctx.warn(
                f"输入数据被识别为相对丰度表（{ab_info['reason']}）。"
                "Chao1 等依赖原始计数的指数与稀释曲线在此类数据上不具备严格意义，"
                "程序仍会计算，但解读时请注意；抽平功能已自动跳过。"
            )

        table_div = table_f
        if cfg.get("rarefy"):
            if ab_info["is_relative"]:
                p.update(19, "预处理", "输入为相对丰度，跳过抽平")
                result["rarefaction"] = {"skipped": True, "reason": "输入为相对丰度表"}
            else:
                try:
                    table_div, used_depth = rarefy(table_f, cfg.get("rarefy_depth"), seed=42)
                    result["rarefaction"] = {"used_depth": int(used_depth)}
                    p.update(19, "预处理", f"抽平至 {used_depth} reads/样本")
                except Exception as exc:
                    ctx.warn(f"抽平失败，已跳过：{exc}")
        ctx.table_div = table_div
        ctx.rel = to_relative(table_f)

        rank = cfg.get("rank") or meta_info["default_rank"]
        ctx.rank_label = "Feature"
        if tax is not None:
            rank = rank if rank in RANKS else meta_info["default_rank"]
            try:
                ctx.rank_table = aggregate_by_taxonomy(table_f, tax, rank)
                ctx.rank_label = rank
                result["rank"] = rank
            except Exception as exc:
                ctx.warn(f"分类级聚合失败：{exc}")

        # ---- 3. 各模块 ----
        if cfg.get("run_alpha", True):
            run_alpha(ctx)
        run_depth(ctx)

        # ---- 先组间 ----
        beta_metrics = []
        if cfg.get("run_beta", True):
            beta_metrics = run_beta(ctx)

        if cfg.get("run_multivariate", True):
            run_multivariate_tests(ctx)
            run_procrustes(ctx)

        if cfg.get("run_ordination", True):
            run_ordination(ctx, beta_metrics)

        # ---- 再组内 ----
        if cfg.get("run_within_group", True):
            run_grouping(ctx)

        # ---- 完整假设检验 ----
        if cfg.get("run_hypothesis", True):
            run_hypothesis(ctx)

        if cfg.get("run_composition", True):
            run_composition(ctx)

        if cfg.get("run_diff", True):
            run_differential(ctx)

        if cfg.get("run_rf", True):
            run_machine(ctx)

        # ---- 高级图形 ----
        if cfg.get("run_advanced", True):
            run_advanced_plots(ctx)

        # ---- 4. Excel ----
        p.update(93, "导出结果", "写出 Excel 汇总")
        try:
            export_excel(ctx)
            ctx.add_table("analysis_summary.xlsx", "全部结果表的 Excel 汇总（多工作表）。")
        except Exception as exc:
            ctx.warn(f"Excel 导出失败：{exc}")

        # ---- 5. 自动结论 ----
        p.update(95, "生成结论", "根据统计结果生成中文描述")
        try:
            from ..narrative import build_narrative, narrative_markdown, narrative_plain

            nar = build_narrative(result)
            result["narrative"] = nar
            (out_dir / "结论描述.md").write_text(
                narrative_markdown(nar), encoding="utf-8")
            (out_dir / "conclusions.txt").write_text(
                narrative_plain(nar), encoding="utf-8")
            ctx.add_table("结论描述.md / conclusions.txt",
                          "根据本次统计结果自动生成的中文结论描述，可直接引用或改写。")
        except Exception as exc:
            ctx.warn(f"自动结论生成失败：{exc}")

    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        result["log"] = p.logs
        _write_json(result, out_dir)
        raise

    p.update(97, "生成报告", "渲染 HTML")
    result["log"] = p.logs
    result["finished_at"] = datetime.now().isoformat(timespec="seconds")
    result["duration_sec"] = round((datetime.now() - started).total_seconds(), 1)

    from ..report import build_report

    build_report(result, out_dir / "report.html", out_dir)
    result["report"] = str(out_dir / "report.html")

    _write_json(result, out_dir)
    p.update(100, "完成", f"共生成 {len(result['figures'])} 张图")
    result["log"] = p.logs
    return result


def _safe(name: str) -> str:
    import re

    return re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", str(name))[:40]


def export_excel(ctx: _Ctx):
    result = ctx.result
    path = ctx.tab_dir / "analysis_summary.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({
            "项目": ["分析类型", "分组列", "样本数", "特征数（原始）", "特征数（过滤后）",
                     "分组构成", "抽平", "分析时间", "耗时(秒)"],
            "值": [
                result.get("analysis_label"), result.get("group_column"),
                result.get("input_info", {}).get("n_samples"),
                result.get("input_info", {}).get("n_features"),
                result.get("input_info", {}).get("n_features_used"),
                "; ".join(f"{k}={v}" for k, v in (result.get("groups") or {}).items()),
                json.dumps(result.get("rarefaction", {}), ensure_ascii=False)
                if result.get("rarefaction") else "未抽平",
                result.get("finished_at"), result.get("duration_sec"),
            ],
        }).to_excel(writer, sheet_name="00_概览", index=False)

        pd.DataFrame({"Group": ctx.groups}).to_excel(writer, sheet_name="01_分组表")

        alpha = result.get("alpha_table")
        if alpha is not None:
            a = alpha.copy()
            a.insert(0, "Group", ctx.groups.reindex(a.index).astype(str))
            a.round(4).to_excel(writer, sheet_name="02_Alpha多样性")

        if result.get("alpha_tests"):
            pd.DataFrame(result["alpha_tests"]).T.reset_index().rename(
                columns={"index": "metric"}).to_excel(writer, sheet_name="03_Alpha检验", index=False)

        ctx.table.round(4).to_excel(writer, sheet_name="04_丰度表_过滤后")
        ctx.rel.round(6).to_excel(writer, sheet_name="05_相对丰度")
        if ctx.rank_table is not None:
            to_relative(ctx.rank_table).round(6).to_excel(
                writer, sheet_name=f"06_{ctx.rank_label}水平"[:31])

        for tag, df in ctx.diff_tables.items():
            df.round(6).to_excel(writer, sheet_name=f"07_差异_{tag}"[:31], index=False)

        if result.get("beta"):
            rows = []
            for metric, info in result["beta"].items():
                perm = info.get("permanova") or {}
                anos = info.get("anosim") or {}
                mrpp = info.get("mrpp") or {}
                pd_res = info.get("permdisp") or {}
                rows.append({
                    "距离指标": info.get("label", metric),
                    "PERMANOVA_R2": perm.get("R2"), "PERMANOVA_F": perm.get("F"),
                    "PERMANOVA_p": perm.get("p"),
                    "ANOSIM_R": anos.get("R"), "ANOSIM_p": anos.get("p"),
                    "MRPP_delta": mrpp.get("delta"), "MRPP_A": mrpp.get("chance_corrected_within_group_agreement"),
                    "MRPP_p": mrpp.get("p"),
                    "PERMDISP_F": pd_res.get("F"), "PERMDISP_p": pd_res.get("p"),
                })
            pd.DataFrame(rows).round(6).to_excel(writer, sheet_name="08_Beta检验", index=False)

        if result.get("envfit"):
            pd.DataFrame(result["envfit"]).round(5).to_excel(
                writer, sheet_name="09_envfit", index=False)

        if result.get("random_forest"):
            rf = result["random_forest"]
            rows = [
                {"指标": "交叉验证准确率", "值": rf.get("accuracy")},
                {"指标": "准确率标准差", "值": rf.get("accuracy_std")},
                {"指标": "折数", "值": rf.get("n_splits")},
                {"指标": "样本数", "值": rf.get("n_samples")},
                {"指标": "AUC", "值": rf.get("auc")},
            ]
            pd.DataFrame(rows).to_excel(writer, sheet_name="10_随机森林", index=False)
            if rf.get("confusion"):
                pd.DataFrame(rf["confusion"]).to_excel(writer, sheet_name="11_混淆矩阵")

        try:
            style_df = pd.DataFrame(
                [{"样式项": k, "值": json.dumps(v, ensure_ascii=False)}
                 for k, v in sorted(ctx.style.diff_from_default().items())]
            )
            if not style_df.empty:
                style_df.to_excel(writer, sheet_name="99_样式设置", index=False)
        except Exception:
            pass


def _write_json(result: dict, out_dir: Path):
    payload = {k: v for k, v in result.items() if k not in ("traceback",)}
    with open(out_dir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
