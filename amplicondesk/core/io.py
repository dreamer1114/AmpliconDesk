"""表格读取、方向识别与样本匹配。

设计原则：
  * 只依赖 pandas / numpy / openpyxl。
  * 输入格式尽量宽容：csv / tsv / txt / xlsx / xls 均可，分隔符自动嗅探。
  * 丰度表方向（行=特征 还是 行=样本）自动判断，不要求用户手动指定。
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

TABLE_EXTS = (".csv", ".tsv", ".txt", ".tab", ".xlsx", ".xls", ".xlsm", ".gz")


# --------------------------------------------------------------------------
# 基础读取
# --------------------------------------------------------------------------
def _sniff_sep(sample_text: str) -> str:
    """在候选分隔符中挑一个最像表头的。"""
    lines = [ln for ln in sample_text.splitlines() if ln.strip()][:20]
    if not lines:
        return ","
    candidates = [",", "\t", ";", "|"]
    best, best_score = ",", -1
    for sep in candidates:
        counts = [len(next(csv.reader([ln], delimiter=sep))) for ln in lines]
        if not counts or max(counts) < 2:
            continue
        # 列数一致且列数多者优先
        consistency = counts.count(max(set(counts), key=counts.count)) / len(counts)
        score = max(counts) * consistency
        if score > best_score:
            best, best_score = sep, score
    return best


def read_table(path: str | Path, index_col: int | None = 0, sep: str | None = None) -> pd.DataFrame:
    """读取任意分隔符或 Excel 表格，返回 DataFrame。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, index_col=index_col, engine=None)
        return _tidy(df)

    # 文本类
    raw = path.read_bytes()
    if suffix == ".gz":
        import gzip

        raw = gzip.decompress(raw)
    for enc in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover
        text = raw.decode("utf-8", errors="replace")

    if sep is None:
        sep = _sniff_sep(text)
    df = pd.read_csv(
        _io_from_text(text),
        sep=sep,
        index_col=index_col,
        engine="python" if len(sep) > 1 else "c",
    )
    return _tidy(df)


def _io_from_text(text: str):
    import io

    return io.StringIO(text)


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """统一收尾：清掉全空行列、去首尾空格。"""
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.index = [str(i).strip() for i in df.index]
    df.columns = [str(c).strip() for c in df.columns]
    # 去掉 index/样本名重复出现的列
    dup = [c for c in df.columns if c.lower() in ("index", "sample", "sampleid", "sample_id")]
    if dup:
        df = df.drop(columns=dup)
    # 全空列名
    df = df.loc[:, [c for c in df.columns if c != "" and c.lower() != "nan"]]
    return df


# --------------------------------------------------------------------------
# 样本名匹配
# --------------------------------------------------------------------------
def _key(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(name)).lower()


def match_samples(
    table_names: list[str], meta_names: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """把表里的样本名和分组表里的样本名对上（忽略大小写/空格/下划线/连字符）。

    返回 (共有样本的 table 侧名字, 仅在表中的, 仅在分组表中的)。
    """
    meta_lookup: dict[str, str] = {}
    for m in meta_names:
        meta_lookup.setdefault(_key(m), m)

    common_table: list[str] = []
    only_table: list[str] = []
    for t in table_names:
        if _key(t) in meta_lookup:
            common_table.append(t)
        else:
            only_table.append(t)

    matched_meta_keys = {_key(t) for t in common_table}
    only_meta = [m for m in meta_names if _key(m) not in matched_meta_keys]
    return common_table, only_table, only_meta


def meta_name_for(table_name: str, meta_names: list[str]) -> str | None:
    lookup = {_key(m): m for m in meta_names}
    return lookup.get(_key(table_name))


# --------------------------------------------------------------------------
# 丰度表方向识别
# --------------------------------------------------------------------------
def orient_feature_table(
    df: pd.DataFrame, meta_names: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """保证返回 行=特征、列=样本 的矩阵。

    判定依据（按优先级）：
      1. 与分组表样本名的重合度；
      2. 若无法判断，则认为列数 >= 行数时行为特征（组学表通常特征远多于样本）；
      3. 数值性强弱作为兜底参考。
    """
    info: dict = {"transposed": False, "reason": ""}
    df = df.copy()

    def numeric_ratio(frame: pd.DataFrame) -> float:
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            return 0.0
        sub = frame.iloc[: min(500, frame.shape[0]), : min(500, frame.shape[1])]
        vals = pd.to_numeric(sub.stack(), errors="coerce")
        return float(vals.notna().mean())

    if meta_names:
        col_common, _, _ = match_samples(list(df.columns), meta_names)
        idx_common, _, _ = match_samples(list(df.index), meta_names)
        if len(col_common) > len(idx_common):
            info["reason"] = f"列名与分组表匹配 {len(col_common)} 个样本"
            return df, info
        if len(idx_common) > len(col_common):
            info.update(transposed=True, reason=f"行名与分组表匹配 {len(idx_common)} 个样本")
            return df.T, info

    # 无分组表时：样本数一般少于特征数
    n_rows, n_cols = df.shape
    if n_cols > n_rows:
        info["reason"] = f"列数({n_cols}) > 行数({n_rows})，判定行为特征"
        return df, info

    # 行数 >= 列数：看看哪一侧看起来更像"样本名"（非重复、非纯数字递增）
    def looks_like_features(labels: list[str]) -> int:
        pat = re.compile(r"(otu|asv|species|genus|k__|d__|ko\d|kog|gene|tax)", re.I)
        return sum(1 for x in labels[:200] if pat.search(str(x)))

    if looks_like_features(list(df.index)) > looks_like_features(list(df.columns)):
        info["reason"] = "行名含特征类关键词"
        return df, info

    if numeric_ratio(df) > 0.6:
        info["reason"] = "数值占比高且列数<=行数，按行列式处理"
        return df, info

    info.update(transposed=True, reason="兜底：转置为 行=特征")
    return df.T, info


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """把能转成数值的列转成数值，无法转的列保留为字符串。"""
    out = df.copy()
    for col in out.columns:
        conv = pd.to_numeric(out[col], errors="coerce")
        if conv.notna().sum() >= max(1, int(0.5 * len(conv))):
            out[col] = conv
    return out


def read_group_table(path: str | Path) -> pd.DataFrame:
    """读取分组/元数据表，首列作为样本名。"""
    df = read_table(path, index_col=0)
    if df.shape[1] == 0:
        raise ValueError("分组表至少需要一列分组信息。")
    return df


def read_taxonomy(path: str | Path) -> pd.DataFrame:
    """读取分类注释表。支持两种形态：
    * 多列：Kingdom/Phylum/... 或任意名前缀列；
    * 单列 'taxonomy'：以 ; 分隔的 k__Bacteria;p__Firmicutes;...
    """
    df = read_table(path, index_col=0)
    lower = {c.lower(): c for c in df.columns}
    if "taxonomy" in lower:
        col = lower["taxonomy"]
        ranks = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
        split = df[col].astype(str).str.split(r"\s*;\s*", regex=True)
        out = pd.DataFrame(
            [s + [""] * (7 - len(s)) if len(s) < 7 else s[:7] for s in split],
            index=df.index,
            columns=ranks,
        )
        for r in ranks:
            out[r] = out[r].str.replace(r"^[a-zA-Z]__", "", regex=True).str.strip()
        return out

    prefix_map = {
        "k": "Kingdom", "d": "Kingdom", "p": "Phylum", "c": "Class",
        "o": "Order", "f": "Family", "g": "Genus", "s": "Species",
    }
    ranks = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    cols = list(df.columns)
    out = pd.DataFrame(index=df.index)
    for r in ranks:
        hit = next((c for c in cols if c.lower().startswith(r.lower()[:3])), None)
        if hit is None:
            hit = next((c for c in cols if prefix_map.get(c[:1].lower()) == r), None)
        out[r] = df[hit].astype(str).str.strip() if hit is not None else "Unclassified"
    for r in ranks:
        out[r] = out[r].replace({"": "Unclassified", "nan": "Unclassified", "NA": "Unclassified"})
        out[r] = out[r].fillna("Unclassified")
    return out
