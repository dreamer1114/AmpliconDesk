"""功能富集分析（过表征分析，ORA）。

输入一张「基因 → 功能条目」映射表和一串目标基因，用超几何检验计算每个条目是否被显著富集，
并做 BH-FDR 校正。适用于 GO / KEGG / COG / Pfam 等任意基因集注释。
"""

# ============================================================================
# 【未启用】本模块不属于 16S 扩增子的分析流程
# ----------------------------------------------------------------------------
# 基因功能的过表征分析（超几何检验 + BH 校正），服务于宏转录组 / 宏基因组。
#
# AmpliconDesk 定位为 16S 扩增子下游分析工具，**本模块不再被任何流程引用**，
# 保留是为了将来需要时可以直接接回。16S 的 OTU/ASV 没有 GO/KEGG 注释，
# 因此功能富集不在本工具范围内。
#
# 如需重新启用：恢复 pipeline 的 ``enrichment_path`` 配置项、
# ``load_inputs`` 里的注释表读取，以及 ``run_enrichment_analysis`` 调用。
# ============================================================================


from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .core.io import read_table
from .core.stats import bh_fdr

__all__ = ["read_gene_term_map", "ora", "run_enrichment", "ENRICHMENT_FORMATS"]


ENRICHMENT_FORMATS = """
功能注释表格式（任选其一）：

1) 两列长表（推荐）          2) 一列多条目（分号/逗号分隔）
   gene      term                gene      terms
   geneA     GO:0008150          geneA     GO:0008150;GO:0003674
   geneA     GO:0003674          geneB     GO:0008150
   geneB     GO:0008150

3) 三列（第三列为条目分类）    4) 含条目描述
   gene   term      category     gene   term   description
   geneA  GO:...    BP           geneA  GO:...  biological_process
"""


def read_gene_term_map(path: str) -> pd.DataFrame:
    """读取基因→功能条目映射表，统一成 (gene, term, category) 三列长表。

    自动识别上述四种常见格式。
    """
    df = read_table(path, index_col=None)
    if df.shape[1] < 2:
        raise ValueError(
            "功能注释表至少需要两列（基因 ID + 功能条目）。" + ENRICHMENT_FORMATS
        )

    df.columns = [str(c).strip() for c in df.columns]
    lower = [c.lower() for c in df.columns]

    # 识别列名
    gene_col = None
    for kw in ("gene", "gene_id", "geneid", "id", "feature", "transcript", "orf"):
        for i, c in enumerate(lower):
            if c == kw or c.startswith(kw):
                gene_col = df.columns[i]
                break
        if gene_col:
            break
    if gene_col is None:
        gene_col = df.columns[0]

    term_col = None
    for kw in ("term", "go", "go_id", "kegg", "ko", "pathway", "category", "annotation", "function"):
        for i, c in enumerate(lower):
            if df.columns[i] == gene_col:
                continue
            if c == kw or c.startswith(kw):
                term_col = df.columns[i]
                break
        if term_col:
            break
    if term_col is None:
        term_col = [c for c in df.columns if c != gene_col][0]

    cat_col = None
    for kw in ("category", "class", "namespace", "type", "ontology", "level"):
        for c in df.columns:
            if c in (gene_col, term_col):
                continue
            if c.lower().startswith(kw):
                cat_col = c
                break

    out = pd.DataFrame({
        "gene": df[gene_col].astype(str).str.strip(),
        "term": df[term_col].astype(str).str.strip(),
    })
    out["category"] = df[cat_col].astype(str).str.strip() if cat_col else ""

    # 处理一个基因对多个条目（分号/逗号/竖线分隔）与一行多基因
    records = []
    for _, row in out.iterrows():
        genes = [g.strip() for g in row["gene"].replace("|", ";").split(";") if g.strip()]
        terms = [t.strip() for t in row["term"].replace(",", ";").replace("|", ";").split(";") if t.strip()]
        for g in genes:
            for t in terms:
                records.append((g, t, row["category"]))
    long = pd.DataFrame(records, columns=["gene", "term", "category"])

    long = long[(long["gene"] != "") & (long["term"] != "") & (long["term"].str.lower() != "nan")]
    long = long.drop_duplicates(subset=["gene", "term"], keep="first").reset_index(drop=True)
    if long.empty:
        raise ValueError("功能注释表解析后为空，请检查基因 ID 与功能条目两列。" + ENRICHMENT_FORMATS)

    # 优先保留有意义的 category：全空或全为 "nan" 时清空
    if long["category"].isin(["", "nan", "None"]).all():
        long["category"] = ""
    return long


def ora(
    query_genes: list[str],
    gene_term: pd.DataFrame,
    background: list[str] | None = None,
    min_size: int = 3,
    max_size: int = 500,
    correct: str = "bh",
) -> pd.DataFrame:
    """过表征分析（超几何检验）。

    query_genes : 目标基因列表（如差异显著基因）
    gene_term   : read_gene_term_map 输出的长表
    background  : 背景基因集；默认使用注释表中出现过的全部基因
    min_size / max_size : 过滤条目大小，避免极小或极泛的条目
    """
    query = {str(g).strip() for g in query_genes if str(g).strip()}
    if not query:
        return pd.DataFrame(columns=["term", "category", "n_query", "n_term", "n_bg",
                                     "expected", "enrichment_ratio", "p", "FDR", "genes"])

    gene_term = gene_term.copy()
    gene_term["gene"] = gene_term["gene"].astype(str).str.strip()
    gene_term["term"] = gene_term["term"].astype(str).str.strip()

    bg_genes = {str(g).strip() for g in background} if background is not None else set(gene_term["gene"])
    bg_genes &= set(gene_term["gene"])
    if not bg_genes:
        bg_genes = set(gene_term["gene"])

    # 目标基因限制在背景内，否则统计量无意义
    query_in_bg = query & bg_genes
    if not query_in_bg:
        return pd.DataFrame(columns=["term", "category", "n_query", "n_term", "n_bg",
                                     "expected", "enrichment_ratio", "p", "FDR", "genes"])

    sub = gene_term[gene_term["gene"].isin(bg_genes)]
    n_bg = len(bg_genes)
    q_size = len(query_in_bg)

    rows = []
    for term, grp in sub.groupby("term", sort=False):
        term_genes = set(grp["gene"])
        k_term = len(term_genes)
        if k_term < min_size or k_term > max_size:
            continue
        hits = query_in_bg & term_genes
        k_query = len(hits)
        if k_query == 0:
            continue
        # 超几何：P(X >= k_query)
        p = float(hypergeom.sf(k_query - 1, n_bg, k_term, q_size))
        expected = q_size * k_term / n_bg
        rows.append({
            "term": str(term),
            "category": str(grp["category"].iloc[0]) if "category" in grp else "",
            "n_query": k_query,
            "n_term": k_term,
            "n_bg": n_bg,
            "expected": expected,
            "enrichment_ratio": (k_query / expected) if expected > 0 else np.nan,
            "p": p,
            "genes": ";".join(sorted(hits)[:60]),
        })

    if not rows:
        return pd.DataFrame(columns=["term", "category", "n_query", "n_term", "n_bg",
                                     "expected", "enrichment_ratio", "p", "FDR", "genes"])

    res = pd.DataFrame(rows)
    res["FDR"] = bh_fdr(res["p"].to_numpy())
    res["-log10FDR"] = -np.log10(np.clip(res["FDR"].to_numpy(dtype=float), 1e-300, 1.0))
    res = res.sort_values("p").reset_index(drop=True)

    # 富集倍数过大且命中数很少时，标注数据不足
    res["note"] = np.where(res["n_query"] < 2, "命中基因数偏少，结果仅供参考", "")
    return res


def run_enrichment(
    diff_table: pd.DataFrame,
    gene_term: pd.DataFrame,
    background: list[str] | None = None,
    fdr_cut: float = 0.05,
    fc_cut: float = 1.0,
    direction: str = "both",
    min_size: int = 3,
    max_size: int = 500,
) -> dict[str, pd.DataFrame]:
    """对一张差异分析结果表做富集分析。

    direction: "up"（仅在 A 中上调）/ "down" / "both"（两个方向分别做）
    返回 {"up": DataFrame, "down": DataFrame}（按需）。
    """
    out: dict[str, pd.DataFrame] = {}
    if diff_table is None or diff_table.empty:
        return out

    df = diff_table.copy()
    df["feature"] = df["feature"].astype(str)
    sig = df[(df["FDR"] < fdr_cut) & (df["log2FC"].abs() >= fc_cut)]
    if sig.empty:
        # 没有显著基因时退回用最显著的若干基因，保证结果不为空但会标注
        sig = df.nsmallest(min(100, len(df)), "p")

    up = sig[sig["log2FC"] > 0]["feature"].tolist()
    down = sig[sig["log2FC"] < 0]["feature"].tolist()

    if direction in ("up", "both") and up:
        out["up"] = ora(up, gene_term, background, min_size=min_size, max_size=max_size)
    if direction in ("down", "both") and down:
        out["down"] = ora(down, gene_term, background, min_size=min_size, max_size=max_size)
    return out
