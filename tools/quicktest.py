"""快速集成自检：用演示数据跑全量分析，检查图数与错误。"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amplicondesk import run_analysis  # noqa: E402

DEMO = ROOT / "examples" / "demo_data"
OUT = ROOT / "output"


def show(tag, cfg):
    print("=" * 78)
    print(tag)
    print("=" * 78)
    t0 = time.time()
    try:
        res = run_analysis(cfg, output_root=str(OUT))
    except Exception as exc:
        import traceback

        print(f"!! 失败：{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None
    d = Path(res["output_dir"])
    print(f"耗时 {time.time() - t0:.1f}s | 图 {len(res['figures'])} 张 | 表 {len(res['tables'])} 个")
    print(f"报告 {d / 'report.html'}")
    if res["warnings"]:
        print(f"警告 {len(res['warnings'])} 条：")
        for w in res["warnings"][:25]:
            print("   -", w)
    else:
        print("警告：无 ✓")
    secs = {}
    for f in res["figures"]:
        secs[f["section"]] = secs.get(f["section"], 0) + 1
    print("各分区图数：")
    for k, v in secs.items():
        print(f"   {k:24s} {v}")
    return res


def main():
    a = show("用例 1：16S 全量（排序 + 多元统计 + 组内 + 群落 + 差异 + 机器学习 + 高级图形 + 环境因子）", {
        "analysis_type": "amplicon16s",
        "abundance_path": str(DEMO / "16s_otu_table.csv"),
        "metadata_path": str(DEMO / "16s_metadata.csv"),
        "taxonomy_path": str(DEMO / "16s_taxonomy.csv"),
        "env_path": str(DEMO / "16s_env.csv"),
        "group_column": "Group",
        "rarefy": True,
        "permutations": 299,
        "beta_metrics": ["braycurtis", "jaccard"],
        "ordinations": ["pcoa", "nmds", "pca", "tsne", "umap", "plsda", "rda", "dbrda", "cca", "cap"],
        "stats_tests": ["permanova", "anosim", "mrpp", "permdisp", "mantel", "envfit"],
    })

    b = show("用例 2：不抽平 + 少量排序（验证未抽平路径与精简配置）", {
        "analysis_type": "amplicon16s",
        "abundance_path": str(DEMO / "16s_otu_table.csv"),
        "metadata_path": str(DEMO / "16s_metadata.csv"),
        "taxonomy_path": str(DEMO / "16s_taxonomy.csv"),
        "group_column": "Group",
        "rarefy": False,
        "permutations": 299,
        "beta_metrics": ["braycurtis", "aitchison"],
        "ordinations": ["pcoa", "pca", "umap"],
        "stats_tests": ["permanova", "anosim", "mrpp", "permdisp"],
    })

    c = show("用例 3：TSV 格式 + 自动识别相对丰度", {
        "analysis_type": "amplicon16s",
        "abundance_path": str(DEMO / "formats" / "t.tsv"),
        "metadata_path": str(DEMO / "16s_metadata.csv"),
        "taxonomy_path": str(DEMO / "16s_taxonomy.csv"),
        "group_column": "Group",
        "permutations": 199,
        "ordinations": ["pcoa", "pca"],
        "stats_tests": ["permanova", "mrpp"],
    })

    print()
    print("=" * 78)
    ok = bool(a) and bool(b) and bool(c)
    print("结论：", "全部跑通 ✅" if ok else "存在失败 ❌")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
