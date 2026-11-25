"""HTML 报告生成：把分析结果渲染成一个自包含的可视化报告。"""

from __future__ import annotations

import base64
import html
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SECTION_ORDER = [
    "数据概览",
    "测序深度",
    "Alpha 多样性",
    "组间差异",
    "多元统计检验",
    "排序分析",
    "组内差异",
    "群落组成",
    "差异分析",
    "机器学习",
    "高级图形",
]

CSS = """
:root{
  --bg:#f5f7fa; --card:#ffffff; --ink:#1c2430; --muted:#6b7785;
  --line:#e3e8ef; --accent:#2f6fd0; --accent2:#0f9d76; --warn:#c9761b;
  --shadow:0 1px 3px rgba(16,24,40,.06),0 6px 20px rgba(16,24,40,.06);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",Roboto,Helvetica,Arial,sans-serif;}
a{color:var(--accent);text-decoration:none}
.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 90px}
header.hero{background:linear-gradient(135deg,#1e3a8a 0%,#2f6fd0 55%,#0f9d76 120%);
  color:#fff;border-radius:16px;padding:30px 32px;box-shadow:var(--shadow)}
header.hero h1{margin:0 0 6px;font-size:26px;letter-spacing:.3px}
header.hero .sub{opacity:.9;font-size:14px}
.badges{margin-top:16px;display:flex;flex-wrap:wrap;gap:9px}
.badge{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.28);
  padding:5px 12px;border-radius:999px;font-size:12.5px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:14px;margin:22px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:15px 17px;box-shadow:var(--shadow)}
.card .k{font-size:12px;color:var(--muted);letter-spacing:.4px;text-transform:uppercase}
.card .v{font-size:23px;font-weight:650;margin-top:5px;color:var(--accent)}
.card .v.small{font-size:17px}
.card .n{font-size:12px;color:var(--muted);margin-top:3px}
nav.toc{position:sticky;top:0;z-index:9;background:rgba(245,247,250,.92);backdrop-filter:blur(6px);
  border-bottom:1px solid var(--line);margin:22px -22px 26px;padding:11px 22px;
  display:flex;gap:8px;flex-wrap:wrap}
nav.toc a{font-size:13px;padding:5px 12px;border-radius:8px;background:#fff;border:1px solid var(--line);color:var(--ink)}
nav.toc a:hover{border-color:var(--accent);color:var(--accent)}
section{margin:34px 0 0}
h2{font-size:19px;margin:0 0 6px;padding-left:11px;border-left:4px solid var(--accent)}
h2 .cnt{font-size:12px;color:var(--muted);font-weight:400;margin-left:8px}
.sec-note{color:var(--muted);font-size:13px;margin:6px 0 16px;padding-left:13px}
.fig{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;
  margin:18px 0;box-shadow:var(--shadow)}
.fig .head{padding:13px 18px 0;display:flex;justify-content:space-between;align-items:baseline;gap:12px}
.fig .title{font-weight:600;font-size:14.5px}
.fig .file{font-size:11.5px;color:var(--muted);font-family:ui-monospace,Consolas,monospace}
.fig img{display:block;width:100%;height:auto;padding:12px 16px 4px}
.fig .cap{padding:6px 18px 16px;color:var(--muted);font-size:13px;border-top:1px solid #f0f3f7;margin-top:8px;padding-top:12px}
table{border-collapse:collapse;width:100%;font-size:13.5px;background:#fff}
th,td{border-bottom:1px solid var(--line);padding:8px 12px;text-align:left}
th{background:#f8fafc;font-weight:600;color:#334155}
tr:hover td{background:#fbfdff}
.tbl-wrap{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:auto;box-shadow:var(--shadow);margin:14px 0}
.alert{border-radius:12px;padding:14px 18px;margin:16px 0;font-size:13.5px;border:1px solid}
.alert.warn{background:#fff9f0;border-color:#f2d9b5;color:#8a5a12}
.alert.err{background:#fff5f5;border-color:#f3cccc;color:#8f2a2a}
.alert.ok{background:#f1fbf6;border-color:#c6ead9;color:#0d6b50}
.alert ul{margin:8px 0 0;padding-left:20px}
.nar-headline{font-size:15px;line-height:1.75}
.nar-block{background:#fff;border:1px solid var(--line);border-radius:12px;
  padding:16px 20px 10px;margin:14px 0;box-shadow:var(--shadow)}
.nar-block h3{margin:0 0 10px;font-size:15px;color:var(--accent);
  padding-left:9px;border-left:3px solid var(--accent)}
.nar-block ul{margin:0;padding-left:20px}
.nar-block li{margin-bottom:8px;line-height:1.75;font-size:14px}
.nar-block li.nar-sub{list-style:circle;color:#4a5666;font-size:13.5px;margin-left:14px}
.nar-note{font-size:12.5px;color:var(--muted);margin-top:6px}
.tag{display:inline-block;font-size:11.5px;padding:2px 8px;border-radius:6px;background:#eef4ff;color:#2f6fd0;margin-right:6px}
.sig{color:#c0392b;font-weight:600}
footer{margin-top:50px;color:var(--muted);font-size:12.5px;text-align:center;border-top:1px solid var(--line);padding-top:18px}
details{background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px 18px;margin:14px 0}
summary{cursor:pointer;font-weight:600;font-size:14px}
pre{background:#0f172a;color:#e2e8f0;padding:14px;border-radius:10px;overflow:auto;font-size:12px;line-height:1.5}
@media print{nav.toc{display:none}.fig{break-inside:avoid}}
"""


def _img_tag(path: Path, embed: bool = True) -> str:
    if embed:
        try:
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return f'<img alt="{html.escape(path.name)}" src="data:image/png;base64,{data}">'
        except Exception:
            pass
    return f'<img alt="{html.escape(path.name)}" src="figures/{html.escape(path.name)}">'


def _stat_card(key: str, value, note: str = "") -> str:
    cls = "v small" if isinstance(value, str) and len(str(value)) > 12 else "v"
    note_html = f'<div class="n">{html.escape(str(note))}</div>' if note else ""
    return (
        f'<div class="card"><div class="k">{html.escape(key)}</div>'
        f'<div class="{cls}">{html.escape(str(value))}</div>{note_html}</div>'
    )


def _fmt_p(p) -> str:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "&lt; 0.001"
    return f"{p:.3f}"


def _stars(p) -> str:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return " ***"
    if p < 0.01:
        return " **"
    if p < 0.05:
        return " *"
    return ""


def _fmt3(v) -> str:
    """数值格式化到 3 位小数，非数值返回 -。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(f):
        return "-"
    return f"{f:.3f}"


def _sig_cls(p) -> str:
    try:
        f = float(p)
    except (TypeError, ValueError):
        return ""
    return "sig" if np.isfinite(f) and f < 0.05 else ""


def build_report(result: dict, report_path: Path, out_dir: Path, embed: bool = True) -> Path:
    res = result
    overview_cards = []

    info = res.get("input_info", {}) or {}
    overview_cards.append(_stat_card("分析类型", res.get("analysis_label", "-")))
    overview_cards.append(_stat_card("样本数", info.get("n_samples", "-")))
    overview_cards.append(_stat_card("特征数", info.get("n_features", "-")))
    overview_cards.append(_stat_card("分组数", len(res.get("groups") or {})))
    overview_cards.append(_stat_card("分组列", res.get("group_column", "-")))

    beta = res.get("beta") or {}
    if beta:
        best = next(iter(beta.values()))
        perm = best.get("permanova") or {}
        if perm.get("R2") is not None:
            overview_cards.append(
                _stat_card("PERMANOVA R²", f"{perm['R2']:.3f}", f"p = {_fmt_p(perm.get('p')).replace('&lt;','<')}")
            )
    rf = res.get("random_forest") or {}
    if rf.get("accuracy") is not None:
        note = f"AUC = {rf['auc']:.3f}" if rf.get("auc") else f"{rf.get('n_splits')} 折交叉验证"
        overview_cards.append(_stat_card("RF 准确率", f"{rf['accuracy']:.1%}", note))
    overview_cards.append(_stat_card("生成图表", len(res.get("figures") or [])))
    overview_cards.append(_stat_card("耗时", f"{res.get('duration_sec', 0)} 秒"))

    # ---- 分组构成表 ----
    groups = res.get("groups") or {}
    group_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td>"
        f"<td>{v / max(1, sum(groups.values())):.1%}</td></tr>"
        for k, v in groups.items()
    )

    # ---- 分析参数 ----
    param_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in [
            ("表格方向自动识别", "已转置（原表为 样本×特征）" if info.get("transposed") else "未转置（原表为 特征×样本）"),
            ("方向判定依据", info.get("orient_reason") or "-"),
            ("分类注释表", "已提供" if info.get("has_taxonomy") else "未提供"),
            ("环境因子表", f"{len(info.get('env_columns', []))} 个变量 / {info.get('n_env_samples', 0)} 个样本"
                          if info.get("env_columns") else "未提供"),
            ("使用的分类级", res.get("rank", "-")),
            ("特征过滤", json.dumps(res.get("filter_stats", {}), ensure_ascii=False)),
            ("抽平", json.dumps(res.get("rarefaction", {}), ensure_ascii=False) if res.get("rarefaction") else "未抽平"),
            ("起始时间", res.get("started_at", "-")),
            ("结束时间", res.get("finished_at", "-")),
        ]
    )

    # ---- 统计检验汇总 ----
    stats_blocks = []

    beta_rows = []
    for metric, b in (res.get("beta") or {}).items():
        perm = b.get("permanova") or {}
        anos = b.get("anosim") or {}
        mrpp = b.get("mrpp") or {}
        pdp = b.get("permdisp") or {}
        beta_rows.append(
            f"<tr><td>{html.escape(str(b.get('label', metric)))}</td>"
            f"<td>{_fmt3(perm.get('R2'))}</td><td>{_fmt3(perm.get('F'))}</td>"
            f"<td class='{_sig_cls(perm.get('p'))}'>{_fmt_p(perm.get('p'))}{_stars(perm.get('p'))}</td>"
            f"<td>{_fmt3(anos.get('R'))}</td><td class='{_sig_cls(anos.get('p'))}'>{_fmt_p(anos.get('p'))}</td>"
            f"<td>{_fmt3(mrpp.get('chance_corrected_within_group_agreement'))}</td>"
            f"<td class='{_sig_cls(mrpp.get('p'))}'>{_fmt_p(mrpp.get('p'))}</td>"
            f"<td>{_fmt3(pdp.get('F'))}</td>"
            f"<td class='{_sig_cls(pdp.get('p'))}'>{_fmt_p(pdp.get('p'))}</td></tr>"
        )
    if beta_rows:
        stats_blocks.append((
            "Beta 多样性多元检验",
            "<tr><th>距离指标</th><th>PERMANOVA R²</th><th>PERMANOVA F</th><th>PERMANOVA p</th>"
            "<th>ANOSIM R</th><th>ANOSIM p</th><th>MRPP A</th><th>MRPP p</th>"
            "<th>PERMDISP F</th><th>PERMDISP p</th></tr>",
            "".join(beta_rows),
        ))

    if res.get("ordination"):
        ord_rows = []
        for method, val in (res.get("ordination") or {}).items():
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, list):
                        txt = "、".join(f"{x * 100:.1f}%" for x in v2[:3])
                    else:
                        txt = f"{v2:.3f}"
                    ord_rows.append(
                        f"<tr><td>{html.escape(method.upper())}</td>"
                        f"<td>{html.escape(str(k2))}</td><td>{html.escape(txt)}</td></tr>")
        if ord_rows:
            stats_blocks.append((
                "排序分析摘要",
                "<tr><th>方法</th><th>距离/对象</th><th>解释度或 stress</th></tr>",
                "".join(ord_rows),
            ))

    if res.get("envfit"):
        env_rows = "".join(
            f"<tr><td>{html.escape(str(r.get('variable')))}</td>"
            f"<td>{html.escape(str(r.get('type', '-')))}</td>"
            f"<td>{_fmt3(r.get('r2'))}</td>"
            f"<td class='{_sig_cls(r.get('p'))}'>{_fmt_p(r.get('p'))}</td>"
            f"<td class='{_sig_cls(r.get('p_adj'))}'>{_fmt_p(r.get('p_adj'))}</td></tr>"
            for r in res["envfit"]
        )
        stats_blocks.append((
            "envfit 因子拟合",
            "<tr><th>变量</th><th>类型</th><th>R²</th><th>p</th><th>p (校正)</th></tr>",
            env_rows,
        ))

    if res.get("alpha_tests"):
        a_rows = "".join(
            f"<tr><td>{html.escape(str(m))}</td><td>{html.escape(str(t.get('test', '-')))}</td>"
            f"<td class='{_sig_cls(t.get('p'))}'>{_fmt_p(t.get('p'))}{_stars(t.get('p'))}</td></tr>"
            for m, t in res["alpha_tests"].items()
        )
        stats_blocks.append((
            "Alpha 多样性组间检验",
            "<tr><th>指标</th><th>检验方法</th><th>p 值</th></tr>",
            a_rows,
        ))

    if res.get("random_forest"):
        rf = res["random_forest"]
        cm = rf.get("confusion")
        cm_html = ""
        if cm:
            cm_df = pd.DataFrame(cm)
            head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cm_df.columns)
            body = "".join(
                "<tr><td>" + html.escape(str(idx)) + "</td>"
                + "".join(f"<td>{int(v)}</td>" for v in row) + "</tr>"
                for idx, row in cm_df.iterrows()
            )
            cm_html = f"<table class='tbl'><tr><th></th>{head}</tr>{body}</table>"
        stats_blocks.append((
            "随机森林",
            "<tr><th>指标</th><th>值</th></tr>",
            f"<tr><td>交叉验证准确率</td><td>{rf.get('accuracy', 0):.1%} "
            f"(± {rf.get('accuracy_std', 0):.1%}, {rf.get('n_splits')} 折)</td></tr>"
            f"<tr><td>样本数</td><td>{rf.get('n_samples')}</td></tr>"
            + (f"<tr><td>AUC</td><td>{rf['auc']:.3f}</td></tr>" if rf.get("auc") else "")
            + (f"<tr><td>混淆矩阵</td><td>{cm_html}</td></tr>" if cm_html else ""),
        ))

    stats_html = "".join(
        f'<section><h2>{html.escape(title)}</h2>'
        f'<div class="tbl-wrap"><table class="tbl">{head}{body}</table></div></section>'
        for title, head, body in stats_blocks
    )

    # ---- 样式设置 ----
    from .style import STYLE_DEFAULTS, SCHEMA_BY_KEY, Style

    style_obj = Style.from_dict(res.get("style"))
    style_diff = style_obj.diff_from_default()
    fig_overrides = style_obj.figure_overrides
    style_items = []
    for k, v in sorted(style_diff.items()):
        if k.startswith("fig."):
            continue
        label = SCHEMA_BY_KEY.get(k, {}).get("label", k)
        style_items.append((f"{label}（{k}）", json.dumps(v, ensure_ascii=False)))
    for fid, ov in fig_overrides.items():
        for k, v in ov.items():
            label = SCHEMA_BY_KEY.get(k, {}).get("label", k)
            style_items.append((f"[单图 {fid}] {label}", json.dumps(v, ensure_ascii=False)))
    style_rows = (
        "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td></tr>"
                for a, b in style_items)
        or '<tr><td colspan="2">本次使用默认样式（未做任何自定义）。</td></tr>'
    )

    # ---- 自动结论描述 ----
    nar = res.get("narrative") or {}
    nar_html = ""
    if nar:
        blocks = []
        for title, sents in (nar.get("sections") or []):
            items = []
            for s in sents:
                sub = s.strip()
                cls = "nar-sub" if sub.startswith("·") else "nar-item"
                sub = sub.lstrip("·").strip()
                sub = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(sub))
                items.append(f'<li class="{cls}">{sub}</li>')
            blocks.append(
                f'<div class="nar-block"><h3>{html.escape(title)}</h3>'
                f'<ul>{"".join(items)}</ul></div>'
            )
        headline = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(nar.get("headline", "")))
        nar_html = (
            f'<section id="conclusion"><h2>分析结论</h2>'
            f'<div class="sec-note">以下文字由程序根据本次实际统计结果自动生成，'
            f'每条结论都对应上面的统计表，可直接引用或按需改写。</div>'
            f'<div class="alert ok nar-headline">{headline}</div>'
            f'{"".join(blocks)}'
            f'<div class="nar-note">完整版见结果目录下的 '
            f'<code>结论描述.md</code> 与 <code>conclusions.txt</code>。</div>'
            f'</section>'
        )

    only_meta = info.get("samples_only_in_metadata") or []
    only_table = info.get("samples_only_in_table") or []
    match_note = ""
    if only_meta or only_table:
        match_note = (
            f'<div class="alert warn"><b>样本匹配提示</b>：'
            f'{len(only_table)} 个丰度表样本未出现在分组表中，'
            f'{len(only_meta)} 个分组表样本未出现在丰度表中，已按交集分析。'
            f'<ul><li>仅在丰度表：{html.escape(", ".join(map(str, only_table[:20]))) or "无"}</li>'
            f'<li>仅在分组表：{html.escape(", ".join(map(str, only_meta[:20]))) or "无"}</li></ul></div>'
        )

    # ---- 统计结论 ----
    concl = []
    if res.get("alpha_tests"):
        sig = [
            f"{m}（{r['test']}，p = {float(r['p']):.3f}）"
            for m, r in res["alpha_tests"].items()
            if r.get("p") is not None and np.isfinite(r.get("p", np.nan)) and float(r["p"]) < 0.05
        ]
        if sig:
            concl.append("Alpha 多样性在分组间存在显著差异：" + "；".join(sig) + "。")
        else:
            concl.append("Alpha 多样性在各分组间均无显著差异（p ≥ 0.05）。")
    for metric, b in beta.items():
        perm = b.get("permanova") or {}
        anos = b.get("anosim") or {}
        mrpp = b.get("mrpp") or {}
        if perm.get("p") is not None and np.isfinite(perm.get("p", np.nan)):
            parts = [
                f"PERMANOVA：R² = {perm['R2']:.3f}，F = {perm['F']:.2f}，"
                f"p = {perm['p']:.3f}{_stars(perm['p'])}"
            ]
            if anos.get("p") is not None and np.isfinite(anos.get("p", np.nan)):
                parts.append(f"ANOSIM R = {anos.get('R', float('nan')):.3f}，p = {anos['p']:.3f}")
            if mrpp.get("p") is not None and np.isfinite(mrpp.get("p", np.nan)):
                parts.append(f"MRPP A = {mrpp.get('chance_corrected_within_group_agreement', float('nan')):.3f}，"
                             f"p = {mrpp['p']:.3f}")
            concl.append(f"{b['label']} 距离下 " + "；".join(parts) + "。")

    pd_res = next((v.get("permdisp") for v in beta.values()
                   if (v.get("permdisp") or {}).get("p") is not None), None)
    if pd_res and np.isfinite(pd_res.get("p", np.nan)):
        if pd_res["p"] < 0.05:
            concl.append(
                f"PERMDISP 检验显示各组离散度**不齐**（F = {pd_res['F']:.2f}，p = {pd_res['p']:.3f}），"
                "组间差异可能部分来自组内离散程度不同，解读 PERMANOVA 时需谨慎。"
            )
        else:
            concl.append(
                f"PERMDISP 检验显示各组离散度齐性（F = {pd_res['F']:.2f}，p = {pd_res['p']:.3f}），"
                "组间差异可较有把握地归因于群落组成差异。"
            )

    rf = res.get("random_forest") or {}
    if rf.get("accuracy") is not None:
        concl.append(
            f"随机森林在 {rf['n_splits']} 折交叉验证下的分类准确率为 {rf['accuracy']:.1%}"
            + (f"，AUC = {rf['auc']:.3f}。" if rf.get("auc") else "。")
        )

    concl_html = "".join(f"<li>{html.escape(c)}</li>" for c in concl) or \
        "<li>本次分析未产生可自动归纳的统计结论。</li>"

    # ---- 图表分节 ----
    figs = res.get("figures") or []
    by_section: dict[str, list] = {}
    for f in figs:
        by_section.setdefault(f.get("section", "其他"), []).append(f)

    def section_key(name):
        for i, s in enumerate(SECTION_ORDER):
            if name.startswith(s):
                return (i, name)
        return (99, name)

    sections_html = []
    nav_html = []
    for sec in sorted(by_section, key=section_key):
        items = by_section[sec]
        nav_html.append(f'<a href="#s-{_slug(sec)}">{html.escape(sec)}</a>')
        cards = []
        for f in items:
            p = Path(f["path"])
            title = f.get("title") or _pretty_title(p.name)
            cards.append(
                f'<div class="fig"><div class="head"><div class="title">{html.escape(title)}</div>'
                f'<div class="file">{html.escape(p.name)}</div></div>'
                f'{_img_tag(p, embed)}'
                f'<div class="cap">{html.escape(f.get("caption") or "")}</div></div>'
            )
        sections_html.append(
            f'<section id="s-{_slug(sec)}"><h2>{html.escape(sec)}'
            f'<span class="cnt">{len(items)} 图</span></h2>'
            f'<div class="sec-note">{html.escape(_SECTION_NOTES.get(_base_section(sec), ""))}</div>'
            f'{"".join(cards)}</section>'
        )

    warnings = res.get("warnings") or []
    warn_html = ""
    if warnings:
        warn_html = (
            '<div class="alert warn"><b>运行提示（%d 条）</b><ul>%s</ul></div>'
            % (len(warnings), "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings))
        )

    tables = res.get("tables") or []
    tables_html = "".join(
        f'<tr><td><code>{html.escape(str(t["name"]))}</code></td><td>{html.escape(str(t.get("caption","")))}</td></tr>'
        for t in tables
    )

    log_html = "".join(html.escape(str(x)) + "\n" for x in (res.get("log") or []))

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(res.get('analysis_label','分析'))}报告 · {html.escape(res.get('run_id',''))}</title>
<style>{CSS}</style></head><body><div class="wrap">
<header class="hero">
  <h1>{html.escape(res.get('analysis_label','生物信息分析'))}报告</h1>
  <div class="sub">Run ID：{html.escape(res.get('run_id','-'))} · 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
  <div class="badges">
    <span class="badge">样本 {info.get('n_samples','-')}</span>
    <span class="badge">特征 {info.get('n_features','-')}</span>
    <span class="badge">分组 {len(groups)}</span>
    <span class="badge">分组列：{html.escape(str(res.get('group_column','-')))}</span>
    <span class="badge">图表 {len(figs)}</span>
  </div>
</header>

<div class="cards">{''.join(overview_cards)}</div>

<div class="alert ok"><b>主要结论</b><ul>{concl_html}</ul></div>
{match_note}
{warn_html}

<nav class="toc"><a href="#conclusion">分析结论</a><a href="#overview">数据概览</a><a href="#stats">统计检验</a>{''.join(nav_html)}<a href="#style">样式设置</a><a href="#tables">结果文件</a></nav>

{nar_html}

<section id="overview"><h2>数据概览</h2>
  <div class="sec-note">输入数据的基本构成与分组情况。</div>
  <div class="tbl-wrap"><table>
    <tr><th>分组</th><th>样本数</th><th>占比</th></tr>{group_rows}
  </table></div>
  <details><summary>分析参数与数据校验明细</summary>
    <div class="tbl-wrap"><table><tr><th>项目</th><th>值</th></tr>{param_rows}</table></div>
  </details>
</section>

<div id="stats">{stats_html}</div>

{''.join(sections_html)}

<section id="style"><h2>本次使用的绘图样式</h2>
  <div class="sec-note">以下为与默认值不同的样式设置。可在界面「绘图样式」面板中调整并保存为模板。</div>
  <div class="tbl-wrap"><table><tr><th>样式项</th><th>值</th></tr>{style_rows}</table></div>
</section>

<section id="tables"><h2>结果文件</h2>
  <div class="sec-note">所有中间结果表保存在 <code>tables/</code>，图片原图保存在 <code>figures/</code>（300 dpi，可直接用于发表）。</div>
  <div class="tbl-wrap"><table><tr><th>文件</th><th>说明</th></tr>{tables_html}
  <tr><td><code>analysis_summary.xlsx</code></td><td>全部结果表的 Excel 汇总（多工作表）</td></tr>
  <tr><td><code>result.json</code></td><td>结构化结果，便于二次开发调用</td></tr>
  </table></div>
</section>

<details><summary>运行日志</summary><pre>{log_html}</pre></details>

<footer>由 AmpliconDesk 本地分析流程生成 · 结果目录：{html.escape(str(out_dir))}</footer>
</div></body></html>"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html_doc, encoding="utf-8")
    return report_path


_SECTION_NOTES = {
    "Alpha 多样性": "衡量单个样本内部的物种丰富度与均匀度。两组比较用 Mann-Whitney U 检验，多组用 Kruskal-Wallis 检验。",
    "组间差异": "先确认「分组之间有没有差异」：距离矩阵、PERMANOVA、ANOSIM、MRPP、PERMDISP、"
              "Mantel，以及组间距离与组内距离的直接对比。",
    "排序分析": "降维与约束排序，用于观察样本聚类结构与主要变异来源。PCoA/NMDS 基于距离矩阵，PCA 基于方差，"
              "RDA/db-RDA/CCA 为约束排序，t-SNE/UMAP 用于非线性降维，PLS-DA 用于判别分析。",
    "多元统计检验": "检验环境/分组因子与群落结构的关系强度，以及不同排序结果之间是否一致。",
    "组内差异": "再确认「每组内部是否一致」。组内距离越小、变异系数越低，说明该组样本越同质；"
              "若组内本身就散得很开，组间差异的解读就要打折扣。",
    "群落组成": "展示各样本/分组在指定分类级上的相对丰度构成。",
    "差异分析": "逐特征检验组间丰度差异，输出倍数变化、显著性、效应量与置信区间。",
    "机器学习": "用随机森林评估分组是否可由群落特征区分，并给出关键判别特征。",
    "高级图形": "小提琴图、云雨图、密度图、蜂群图、棒棒糖图、气泡图、桑基图与树状图，"
              "用不同视觉形式呈现同一批数据的分布与构成。",
    "测序深度": "评估当前测序深度是否足以覆盖样本多样性。",
    "数据概览": "输入数据的基本构成与分组情况，含相对丰度自动识别结果。",
}
_BASE = list(_SECTION_NOTES)


def _base_section(name: str) -> str:
    for b in _BASE:
        if name.startswith(b):
            return b
    return name


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^\w\u4e00-\u9fff]+", "-", str(text)).strip("-")


_TITLE_MAP = [
    ("01_alpha_boxplot", "Alpha 多样性箱线图"),
    ("02_alpha_heatmap", "Alpha 多样性热图"),
    ("03_alpha_rarefaction", "Alpha 多样性稀释曲线"),
    ("04_rarefaction", "稀释曲线"),
    ("05_accumulation", "物种累积曲线"),
    ("06_rank_abundance", "等级丰度曲线"),
    ("07_pcoa", "PCoA 主坐标分析"),
    ("07b_scree", "碎石图"),
    ("08_distance", "组内/组间距离分布"),
    ("09_permdisp", "PERMDISP 离散度检验"),
    ("10_mantel", "Mantel 检验"),
    ("11_nmds", "NMDS 非度量多维标度"),
    ("12_pca", "PCA 主成分分析"),
    ("13_tsne", "t-SNE 降维"),
    ("14_umap", "UMAP 降维"),
    ("15_plsda", "PLS-DA 判别分析"),
    ("16_rda", "RDA 冗余分析"),
    ("17_dbrda", "db-RDA 基于距离的冗余分析"),
    ("18_cca", "CCA 典范对应分析"),
    ("19_cap", "CAP 典范分析主坐标"),
    ("20_envfit", "envfit 因子拟合"),
    ("21_composition_bar", "群落组成堆叠柱状图"),
    ("22_composition_area", "群落组成面积图"),
    ("23_heatmap", "丰度热图"),
    ("24_top_taxa_box", "优势类群组间箱线图"),
    ("25_venn", "共享/特有特征图"),
    ("26_volcano", "差异丰度火山图"),
    ("27_ma", "MA 图"),
    ("28_diffbar", "差异特征 log2FC"),
    ("29_forest", "差异特征森林图"),
    ("30_featurebox", "差异特征丰度箱线图"),
    ("31_lda", "LEfSe 风格 LDA 效应量"),
    ("32_rf_importance", "随机森林特征重要性"),
    ("33_roc", "ROC 曲线"),
    ("34_confusion", "混淆矩阵"),
    ("42_violin", "小提琴图"),
    ("43_raincloud", "云雨图"),
    ("44_density", "密度图"),
    ("45_beeswarm", "蜂群图"),
    ("46_lollipop", "棒棒糖图"),
    ("47_bubble", "气泡图"),
    ("48_sankey", "桑基图"),
    ("49_group_flow", "分组流向图"),
    ("50_voronoi", "树状图"),
]


def _pretty_title(filename: str) -> str:
    stem = Path(filename).stem
    # 按前缀长度降序匹配，避免 "07_pcoa" 抢走 "07b_scree"
    for key, label in sorted(_TITLE_MAP, key=lambda kv: -len(kv[0])):
        if stem.startswith(key):
            rest = stem[len(key):].strip("_").replace("_", " ")
            return f"{label}（{rest}）" if rest else label
    return stem
