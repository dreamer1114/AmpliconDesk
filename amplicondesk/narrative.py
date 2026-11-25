"""自动结论生成器：把统计结果翻译成可以直接写进论文的中文描述。

设计原则：
  * 每一句都必须由**实际数值**驱动，不编造。
  * 明确写出"是否显著""谁高谁低""用了什么检验""为什么用这个检验"。
  * 显著的结论和不显著的结论都要说，避免只报阳性结果。
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

__all__ = ["build_narrative", "narrative_markdown", "narrative_plain"]


def _p(p) -> str:
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "p 值不可用"
    if not math.isfinite(v):
        return "p 值不可用"
    if v < 0.001:
        return "p < 0.001"
    return f"p = {v:.3f}"


def _stars(p) -> str:
    try:
        v = float(p)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(v):
        return ""
    return "***" if v < 0.001 else "**" if v < 0.01 else "*" if v < 0.05 else ""


def _sig_word(p, alpha: float = 0.05) -> str:
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "无法判断"
    if not math.isfinite(v):
        return "无法判断"
    if v < 0.001:
        return "差异极显著"
    if v < 0.01:
        return "差异高度显著"
    if v < alpha:
        return "差异显著"
    if v < 0.1:
        return "差异接近显著（边际显著）"
    return "差异不显著"


def _fmt(v, nd=3) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(f):
        return "-"
    if abs(f) >= 1000:
        return f"{f:,.0f}"
    return f"{f:.{nd}f}"


def _higher_lower(mean_a, mean_b, name_a, name_b, what="该指标") -> str:
    try:
        a, b = float(mean_a), float(mean_b)
    except (TypeError, ValueError):
        return ""
    if not (math.isfinite(a) and math.isfinite(b)) or a == b:
        return ""
    hi, lo = (name_a, name_b) if a > b else (name_b, name_a)
    hi_v, lo_v = (a, b) if a > b else (b, a)
    ratio = (hi_v / lo_v) if lo_v > 0 else float("inf")
    if math.isfinite(ratio) and ratio >= 1.05:
        return f"其中 {hi} 的{what}更高（{_fmt(hi_v)} vs {_fmt(lo_v)}，约为其 {ratio:.2f} 倍）"
    return f"{hi} 略高（{_fmt(hi_v)} vs {_fmt(lo_v)}）"


# --------------------------------------------------------------------------
def build_narrative(res: dict) -> dict:
    """根据分析结果生成结构化中文结论。

    返回 {"headline", "sections": [(标题, [句子,...]), ...], "generated_at"}
    """
    info = res.get("input_info") or {}
    groups = res.get("groups") or {}
    group_col = res.get("group_column") or "分组"
    sections: list[tuple[str, list[str]]] = []

    # ---------- 数据概况 ----------
    n_s = info.get("n_samples", "-")
    n_f = info.get("n_features_used") or info.get("n_features", "-")
    grp_txt = "、".join(f"{k}（n={v}）" for k, v in groups.items())
    ab = res.get("abundance_type") or info.get("abundance_type") or {}
    ov = [f"本次分析共纳入 {n_s} 个样本、{n_f} 个特征，按「{group_col}」分为 {len(groups)} 组：{grp_txt}。"]
    if ab.get("reason"):
        ov.append(f"数据构成判定：{ab['reason']}。")
    if info.get("transposed"):
        ov.append("原始表格为「样本 × 特征」布局，程序已自动转置为标准丰度表。")
    only_t = info.get("n_only_in_table") or 0
    only_m = info.get("n_only_in_metadata") or 0
    if only_t or only_m:
        ov.append(
            f"丰度表与分组表的样本名未完全一致：{only_t} 个样本仅出现在丰度表中、"
            f"{only_m} 个仅出现在分组表中，分析按两者的交集进行。"
        )
    rar = res.get("rarefaction") or {}
    if rar.get("used_depth"):
        ov.append(f"为消除测序深度差异，已将所有样本抽平至 {rar['used_depth']:,} reads。")
    elif rar.get("skipped"):
        ov.append("输入为相对丰度数据，未执行抽平。")
    sections.append(("数据概况", ov))

    # ---------- Alpha 多样性 ----------
    at = res.get("alpha_tests") or {}
    if at:
        sents = []
        sig = [(k, v) for k, v in at.items()
               if v.get("p") is not None and np.isfinite(v.get("p", np.nan)) and float(v["p"]) < 0.05]
        ns = [k for k, v in at.items()
              if v.get("p") is not None and np.isfinite(v.get("p", np.nan)) and float(v["p"]) >= 0.05]
        if sig:
            sents.append(
                f"Alpha 多样性分析显示，{('、'.join(k for k, _ in sig))} "
                f"等指标在「{group_col}」各分组间存在显著差异。"
            )
            for k, v in sig[:4]:
                sents.append(
                    f"　· {k}：{v.get('test', '检验')}，{_p(v.get('p'))}{_stars(v.get('p'))}"
                    f"（{_sig_word(v.get('p'))}）。"
                )
        if ns:
            sents.append(
                f"其余指标（{'、'.join(ns)}）在各分组间{_sig_word(0.5)}，"
                "即组间差异未达到统计学显著水平（p ≥ 0.05）。"
            )
        if not sig and not ns:
            sents.append("Alpha 多样性检验未能给出有效结果（样本量或分组数不足）。")
        sections.append(("Alpha 多样性", sents))

    # ---------- Beta 多样性 ----------
    beta = res.get("beta") or {}
    if beta:
        sents = []
        for metric, b in beta.items():
            label = b.get("label", metric)
            perm = b.get("permanova") or {}
            anos = b.get("anosim") or {}
            mrpp = b.get("mrpp") or {}
            pdp = b.get("permdisp") or {}
            if perm.get("p") is None or not np.isfinite(perm.get("p", np.nan)):
                continue
            r2 = perm.get("R2")
            sents.append(
                f"基于 {label} 距离的 PERMANOVA 检验表明，分组对群落结构"
                f"{'有显著影响' if perm['p'] < 0.05 else '无显著影响'}"
                f"（R² = {_fmt(r2)}，F = {_fmt(perm.get('F'), 2)}，{_p(perm['p'])}{_stars(perm['p'])}），"
                f"即分组因素可解释约 {(_fmt(float(r2) * 100, 1) if r2 is not None else '-')}% 的组间变异。"
            )
            if anos.get("p") is not None and np.isfinite(anos.get("p", np.nan)):
                sents.append(
                    f"　· ANOSIM 结果{'一致' if (anos['p'] < 0.05) == (perm['p'] < 0.05) else '不一致'}"
                    f"（R = {_fmt(anos.get('R'))}，{_p(anos['p'])}）。"
                )
            if mrpp.get("p") is not None and np.isfinite(mrpp.get("p", np.nan)):
                a_val = mrpp.get("chance_corrected_within_group_agreement")
                sents.append(
                    f"　· MRPP 检验{'支持' if mrpp['p'] < 0.05 else '未支持'}组间差异"
                    f"（组内一致度 A = {_fmt(a_val)}，{_p(mrpp['p'])}）。"
                )
            if pdp.get("p") is not None and np.isfinite(pdp.get("p", np.nan)):
                if pdp["p"] < 0.05:
                    sents.append(
                        f"　· **PERMDISP 提示注意**：各组内部离散度不齐"
                        f"（F = {_fmt(pdp.get('F'), 2)}，{_p(pdp['p'])}）。"
                        "此时组间差异可能部分来自组内离散程度不同，而非质心位置不同，"
                        "解读 PERMANOVA 结果时应谨慎。"
                    )
                else:
                    sents.append(
                        f"　· PERMDISP 检验显示各组离散度齐性（F = {_fmt(pdp.get('F'), 2)}，{_p(pdp['p'])}），"
                        "因此上述组间差异可以较有把握地归因于群落组成本身的差异。"
                    )
        if sents:
            sections.append(("Beta 多样性", sents))

    # ---------- 排序分析 ----------
    ord_res = res.get("ordination") or {}
    if ord_res:
        sents = []
        if "pcoa" in ord_res and isinstance(ord_res["pcoa"], dict):
            for metric, explained in ord_res["pcoa"].items():
                if isinstance(explained, list) and len(explained) >= 2:
                    sents.append(
                        f"PCoA（{metric}）前两个主坐标轴分别解释了 {explained[0] * 100:.1f}% 和 "
                        f"{explained[1] * 100:.1f}% 的群落变异，两轴累计 "
                        f"{(explained[0] + explained[1]) * 100:.1f}%。"
                    )
        if "nmds" in ord_res and isinstance(ord_res["nmds"], dict):
            for metric, stress in ord_res["nmds"].items():
                if isinstance(stress, (int, float)) and math.isfinite(stress):
                    quality = ("排序结果可信" if stress < 0.2 else
                               "排序结果尚可参考" if stress < 0.3 else "排序结果可信度较低")
                    sents.append(f"NMDS（{metric}）的 stress = {stress:.3f}，{quality}。")
        if "plsda" in ord_res and isinstance(ord_res["plsda"], dict):
            acc = ord_res["plsda"].get("accuracy")
            if acc is not None:
                sents.append(
                    f"PLS-DA 交叉验证准确率为 {float(acc):.1%}，"
                    f"{'说明分组具有良好的可分性' if float(acc) > 0.8 else '说明分组可分性一般'}。"
                )
        if sents:
            sections.append(("排序分析", sents))

    # ---------- 多元统计检验 ----------
    envfit = res.get("envfit") or []
    procrustes = res.get("procrustes") or {}
    if envfit or procrustes:
        sents = []
        if envfit:
            sig_env = [v for v in envfit
                       if v.get("p_adj") is not None and float(v["p_adj"]) < 0.05]
            if sig_env:
                top = sorted(sig_env, key=lambda v: -float(v.get("r2") or 0))[:5]
                sents.append(
                    "envfit 分析显示，以下因子与排序结构显著相关："
                    + "；".join(f"{v['variable']}（R² = {_fmt(v.get('r2'))}，"
                               f"校正后 {_p(v.get('p_adj'))}）" for v in top)
                    + "。"
                )
            else:
                sents.append("envfit 分析未发现与排序结构显著相关的环境/分组因子。")
        if procrustes:
            m2, r, p = procrustes.get("m2"), procrustes.get("r"), procrustes.get("p")
            if m2 is not None:
                sents.append(
                    f"Procrustes 检验比较了两套排序结果（{procrustes.get('comparison', '')}）："
                    f"m² = {_fmt(m2, 4)}，r = {_fmt(r)}，{_p(p)}，"
                    f"{'两套排序高度一致' if r is not None and float(r) > 0.8 else '两套排序一致性一般'}。"
                )
        sections.append(("多元统计检验", sents))

    # ---------- 差异分析 ----------
    diff_tables = res.get("diff_summary") or []
    if diff_tables:
        sents = []
        for d in diff_tables:
            a, b = d.get("group_a", "A"), d.get("group_b", "B")
            n_sig = d.get("n_significant", 0)
            n_up = d.get("n_up", 0)
            n_down = d.get("n_down", 0)
            method = d.get("method", "")
            if n_sig > 0:
                sents.append(
                    f"在 {a} 与 {b} 的差异分析中共检出 {n_sig} 个差异显著特征"
                    f"（FDR < 0.05 且 |log2FC| ≥ 1），其中在 {a} 中富集 {n_up} 个、在 {b} 中富集 {n_down} 个。"
                    + (f"所用检验方法为 {method}。" if method else "")
                )
                top = d.get("top_features") or []
                for t in top[:3]:
                    sents.append(
                        f"　· {t['feature']}：log2FC = {_fmt(t.get('log2FC'), 2)}，"
                        f"{_p(t.get('p'))}，FDR {_p(t.get('FDR'))}"
                        f"（在 {t.get('enriched_in', '')} 中富集）"
                    )
            else:
                sents.append(
                    f"{a} 与 {b} 之间未检出差异显著的特征（FDR < 0.05 且 |log2FC| ≥ 1），"
                    f"即两组在这些特征上的丰度分布{_sig_word(0.5)}。"
                )
        if sents:
            sections.append(("差异分析", sents))

    # ---------- 组内差异 ----------
    wg = res.get("within_group") or {}
    if wg.get("by_group"):
        sents = []
        by = wg["by_group"]
        metric = wg.get("metric", "")
        meds = {g: v.get("median") for g, v in by.items() if v.get("median") is not None}
        if meds:
            harsh = max(meds, key=lambda g: meds[g])
            mild = min(meds, key=lambda g: meds[g])
            sents.append(
                f"组内异质性分析（{metric} 距离）显示，{harsh} 组内部样本差异最大"
                f"（组内距离中位数 {_fmt(meds[harsh])}），{mild} 组最为均一"
                f"（{_fmt(meds[mild])}）。"
            )
            cvs = {g: v.get("cv") for g, v in by.items() if v.get("cv") is not None}
            if cvs:
                hi = max(cvs, key=lambda g: cvs[g])
                sents.append(
                    f"各组组内距离的变异系数以 {hi} 组最高（CV = {_fmt(cvs[hi])}），"
                    "该组可能存在亚群结构或离群样本；"
                    "若其组内离散度明显高于其它组，解读组间差异时需谨慎。"
                )
        if sents:
            sections.append(("组内差异", sents))

    # ---------- 假设检验方法说明 ----------
    hyp_rows = (res.get("hypothesis") or {}).get("rows") or []
    if hyp_rows:
        sents = []
        auto_rows = [r for r in hyp_rows if r.get("是否自动选择")]
        if auto_rows:
            sents.append(
                f"本次共完成 {len(hyp_rows)} 项统计检验，其中 {len(auto_rows)} 项由程序"
                "根据数据的正态性与方差齐性自动选择方法。"
            )
            shown = 0
            for r in auto_rows:
                if shown >= 6:
                    break
                if not r.get("选择理由"):
                    continue
                sents.append(f"　· {r.get('变量')}：{r['选择理由']}")
                shown += 1
        methods = (res.get("hypothesis") or {}).get("methods_used") or []
        if methods:
            sents.append("本次用到的检验方法：" + "、".join(methods) + "。")
        if sents:
            sections.append(("统计检验方法说明", sents))

    # ---------- 机器学习 ----------
    rf = res.get("random_forest") or {}
    if rf.get("accuracy") is not None:
        acc = float(rf["accuracy"])
        auc_v = rf.get("auc")
        sents = [
            f"随机森林分类模型在 {rf.get('n_splits')} 折分层交叉验证下的准确率为 {acc:.1%}"
            + (f"（标准差 {float(rf.get('accuracy_std', 0)):.1%}）" if rf.get("accuracy_std") is not None else "")
            + "。"
        ]
        if auc_v is not None:
            sents.append(
                f"ROC 曲线下面积 AUC = {_fmt(auc_v)}，"
                f"{'表明基于群落特征可有效区分各分组' if float(auc_v) > 0.8 else '表明群落特征对分组的区分能力有限'}。"
            )
        else:
            sents.append("（分组数超过 2 类，未计算 ROC/AUC。）" if len(rf.get("classes") or []) > 2 else "")
        top = rf.get("top_features") or []
        if top:
            sents.append("对分类贡献最大的特征依次为：" + "、".join(
                f"{t['feature']}（重要性 {_fmt(t.get('importance'), 4)}）" for t in top[:5]) + "。")
        sections.append(("机器学习", [s for s in sents if s]))

    # ---------- 汇总金句 ----------
    headline_bits = []
    sig_count = sum(1 for _, v in at.items()
                    if v.get("p") is not None and np.isfinite(v.get("p", np.nan)) and float(v["p"]) < 0.05)
    if sig_count:
        headline_bits.append(f"{sig_count} 项 Alpha 多样性指标存在组间显著差异")
    beta_sig = [b for b in beta.values()
                if (b.get("permanova") or {}).get("p") is not None
                and float(b["permanova"]["p"]) < 0.05]
    if beta_sig:
        headline_bits.append(f"{len(beta_sig)} 种距离指标下群落结构存在显著组间差异")
    if rf.get("accuracy") is not None and float(rf["accuracy"]) >= 0.8:
        headline_bits.append(f"随机森林以 {float(rf['accuracy']):.0%} 的准确率成功区分分组")
    headline = (
        f"综合结果表明：{'；'.join(headline_bits)}。"
        if headline_bits else
        "综合结果表明：本次分析未发现稳健的组间显著差异，建议检查分组设置或增加样本量。"
    )
    return {
        "headline": headline,
        "sections": sections,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "group_column": group_col,
        "groups": groups,
    }


# --------------------------------------------------------------------------
def narrative_markdown(nar: dict) -> str:
    lines = [f"# 分析结论", "", f"> {nar['headline']}", ""]
    for title, sents in nar["sections"]:
        lines.append(f"## {title}")
        lines.append("")
        for s in sents:
            if s.startswith("　·"):
                lines.append(s.replace("　·", "-"))
            else:
                lines.append(s)
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def narrative_plain(nar: dict) -> str:
    lines = [nar["headline"], ""]
    for title, sents in nar["sections"]:
        lines.append(f"【{title}】")
        for s in sents:
            lines.append(s.replace("　·", "  -"))
        lines.append("")
    return "\n".join(lines).strip() + "\n"
