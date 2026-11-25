"""生成用于演示的 16S 模拟数据：OTU 表、分类注释表、分组表、环境因子表。

用法（在 AmpliconDesk 目录下执行）：
    python tools/make_demo_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "examples" / "demo_data"

PHYLA = ["Firmicutes", "Bacteroidetes", "Proteobacteria", "Actinobacteria", "Verrucomicrobia", "Fusobacteria"]
GENERA = [
    "Bacteroides", "Prevotella", "Faecalibacterium", "Roseburia", "Clostridium",
    "Escherichia", "Bifidobacterium", "Akkermansia", "Lactobacillus", "Streptococcus",
    "Ruminococcus", "Blautia", "Coprococcus", "Dialister", "Veillonella",
    "Alistipes", "Parabacteroides", "Dorea", "Subdoligranulum", "Megamonas",
]


def make_otu_table(n_samples=24, n_features=180, seed=7):
    """构造带分组效应的 OTU 表。"""
    rng = np.random.default_rng(seed)
    groups = ["Control", "Treatment"] * (n_samples // 2)
    samples = [f"{g[0]}{i // 2 + 1:02d}_{g}" for i, g in enumerate(groups)]

    base = rng.lognormal(mean=4.0, sigma=1.6, size=n_features)
    mat = np.zeros((n_features, n_samples))

    # 前 25 个特征在 Treatment 组中富集，接下来 25 个在 Control 组中富集
    for j, s in enumerate(samples):
        effect = np.ones(n_features)
        if groups[j] == "Treatment":
            effect[:25] = rng.uniform(2.0, 6.0, 25)
            effect[25:50] = rng.uniform(0.15, 0.5, 25)
        else:
            effect[:25] = rng.uniform(0.2, 0.6, 25)
            effect[25:50] = rng.uniform(2.0, 5.0, 25)
        depth = rng.uniform(0.6, 1.6)
        lam = base * effect * depth
        mat[:, j] = rng.poisson(np.clip(lam, 0.01, None))

    # 制造稀疏性：大部分低丰度特征在多数样本中为 0
    mask = rng.random(mat.shape) < 0.35
    low = mat < np.percentile(mat, 60)
    mat[mask & low] = 0

    feats = [f"OTU_{i + 1:04d}" for i in range(n_features)]
    table = pd.DataFrame(mat.astype(int), index=feats, columns=samples)
    table = table.loc[table.sum(axis=1) > 0]
    return table, pd.DataFrame({"Group": groups}, index=samples)


def make_taxonomy(features, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for f in features:
        phylum = rng.choice(PHYLA, p=[0.34, 0.26, 0.2, 0.1, 0.06, 0.04])
        if phylum == "Firmicutes":
            genus = rng.choice(["Faecalibacterium", "Roseburia", "Clostridium", "Lactobacillus",
                                "Streptococcus", "Ruminococcus", "Blautia", "Coprococcus",
                                "Dialister", "Veillonella", "Dorea", "Subdoligranulum"],
                               p=[.16, .12, .1, .08, .1, .1, .12, .08, .05, .04, .03, .02])
        elif phylum == "Bacteroidetes":
            genus = rng.choice(["Bacteroides", "Prevotella", "Alistipes", "Parabacteroides"],
                               p=[.45, .3, .15, .1])
        elif phylum == "Proteobacteria":
            genus = rng.choice(["Escherichia", "Shigella", "Klebsiella", "Pseudomonas"],
                               p=[.5, .2, .2, .1])
        elif phylum == "Actinobacteria":
            genus = rng.choice(["Bifidobacterium", "Collinsella", "Atopobium"], p=[.6, .3, .1])
        elif phylum == "Verrucomicrobia":
            genus = "Akkermansia"
        else:
            genus = rng.choice(["Fusobacterium", "Leptotrichia"], p=[.7, .3])
        rows.append(
            {
                "Kingdom": "Bacteria",
                "Phylum": phylum,
                "Class": f"{phylum[:5]}_class",
                "Order": f"{phylum[:5]}_order",
                "Family": f"{genus[:4]}_family",
                "Genus": genus,
                "Species": f"{genus} sp.{rng.integers(1, 40):02d}",
            }
        )
    return pd.DataFrame(rows, index=features)
def make_env_table(samples, seed=51):
    """环境因子表（含连续变量与分类变量），用于 RDA/CCA/envfit/Mantel 演示。"""
    rng = np.random.default_rng(seed)
    n = len(samples)
    ph = rng.normal(7.2, 0.45, n)
    temp = rng.normal(21.5, 3.2, n)
    # 让温度和 pH 与样本顺序（即分组）有一定的关联，保证约束排序有信号
    temp = temp + np.linspace(-1.2, 1.2, n)
    depth = rng.gamma(2.2, 1.4, n)
    site = ["SiteA" if i % 2 == 0 else "SiteB" for i in range(n)]
    return pd.DataFrame(
        {"pH": np.round(ph, 2), "Temperature": np.round(temp, 2),
         "Depth": np.round(depth, 2), "Site": site},
        index=samples,
    )


def main():
    """生成 16S 演示数据。"""
    OUT.mkdir(parents=True, exist_ok=True)

    otu, meta16s = make_otu_table()
    tax = make_taxonomy(otu.index)
    otu.to_csv(OUT / "16s_otu_table.csv", encoding="utf-8-sig")
    tax.to_csv(OUT / "16s_taxonomy.csv", encoding="utf-8-sig")
    meta16s.to_csv(OUT / "16s_metadata.csv", encoding="utf-8-sig")
    make_env_table(list(otu.columns)).to_csv(OUT / "16s_env.csv", encoding="utf-8-sig")
    otu.T.to_excel(OUT / "16s_otu_table_transposed.xlsx")

    # 同一张表的 5 种文件格式，用于验证格式自动识别
    fmt_dir = OUT / "formats"
    fmt_dir.mkdir(exist_ok=True)
    for ext in ("csv", "tsv", "txt"):
        otu.to_csv(fmt_dir / f"t.{ext}", sep={"csv": ",", "tsv": "\t", "txt": "\t"}[ext],
                   encoding="utf-8-sig")
    otu.T.to_excel(fmt_dir / "t.xlsx")
    otu.T.to_excel(fmt_dir / "t.xls")

    print(f"已生成 16S 演示数据到：{OUT}")
    for p in sorted(OUT.iterdir()):
        if p.is_file():
            print(f"  {p.name:44s} {p.stat().st_size / 1024:8.1f} KB")
    print(f"  formats/                                  "
          f"{len(list(fmt_dir.iterdir()))} 个文件（csv / tsv / txt / xls / xlsx）")
    print()
    print(f"OTU 表     : {otu.shape[0]:>5} 特征 × {otu.shape[1]:>2} 样本  "
          f"{meta16s['Group'].value_counts().to_dict()}")
    print(f"文库大小   : {otu.sum(axis=0).min():,} ~ {otu.sum(axis=0).max():,}")


if __name__ == "__main__":
    sys.exit(main())
