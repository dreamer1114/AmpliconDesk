"""端到端自检：用演示数据跑通全流程并校验产物。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amplicondesk import run_analysis  # noqa: E402

DEMO = ROOT / "examples" / "demo_data"
OUT = ROOT / "output"


def check(name, path, min_bytes=1000):
    p = Path(path)
    ok = p.exists() and p.stat().st_size >= min_bytes
    print(f"  {'OK ' if ok else 'FAIL'} {name:52s} {p.stat().st_size/1024:9.1f} KB" if p.exists()
          else f"  FAIL {name:52s} 缺失")
    return ok


def main():
    all_ok = True

    print("=" * 78)
    print("用例 1：16S 扩增子（CSV 丰度表 + 分类表，含抽平）")
    print("=" * 78)
    t0 = time.time()
    res = run_analysis(
        {
            "analysis_type": "amplicon16s",
            "abundance_path": str(DEMO / "16s_otu_table.csv"),
            "metadata_path": str(DEMO / "16s_metadata.csv"),
            "taxonomy_path": str(DEMO / "16s_taxonomy.csv"),
            "group_column": "Group",
            "rarefy": True,
            "permutations": 499,
            "run_rf": True,
        },
        output_root=str(OUT),
        progress_callback=lambda pct, stage, msg: print(f"  [{pct:5.1f}%] {stage} {msg}"),
    )
    d = Path(res["output_dir"])
    print(f"\n耗时 {time.time()-t0:.1f}s | 图 {len(res['figures'])} 张 | 表 {len(res['tables'])} 个")
    print(f"分组: {res['groups']}")
    print(f"PERMANOVA: { {k: (v['permanova']['R2'], v['permanova']['p']) for k, v in res['beta'].items()} }")
    print(f"RF: {res.get('random_forest')}")
    print(f"警告: {res['warnings']}")
    print("产物校验：")
    all_ok &= check("report.html", d / "report.html", 5000)
    all_ok &= check("analysis_summary.xlsx", d / "tables" / "analysis_summary.xlsx")
    all_ok &= check("alpha_diversity.csv", d / "tables" / "alpha_diversity.csv")
    for f in res["figures"]:
        all_ok &= check(Path(f["path"]).name, f["path"], 3000)

    print()
    print("=" * 78)
    print("用例 2：宏基因组（转置 Excel 表，无分类表，自动方向识别）")
    print("=" * 78)
    t0 = time.time()
    res2 = run_analysis(
        {
            "analysis_type": "metagenome",
            "abundance_path": str(DEMO / "16s_otu_table_transposed.xlsx"),
            "metadata_path": str(DEMO / "16s_metadata.csv"),
            "group_column": "Group",
            "permutations": 499,
            "filter": {"min_count": 1, "min_prevalence": 0.2},
        },
        output_root=str(OUT),
        progress_callback=lambda pct, stage, msg: None,
    )
    d2 = Path(res2["output_dir"])
    print(f"耗时 {time.time()-t0:.1f}s | 转置识别: {res2['input_info']['transposed']} ({res2['input_info']['orient_reason']})")
    print(f"图 {len(res2['figures'])} 张 | 警告 {res2['warnings']}")
    print("产物校验：")
    all_ok &= check("report.html", d2 / "report.html", 5000)
    all_ok &= check("analysis_summary.xlsx", d2 / "tables" / "analysis_summary.xlsx")

    print()
    print("=" * 78)
    print("总结果：", "全部通过 ✅" if all_ok else "存在失败项 ❌")
    print("=" * 78)
    print(f"\n打开报告：{d / 'report.html'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
