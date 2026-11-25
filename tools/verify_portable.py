"""便携版验收测试：用内置 Python 跑一次真实分析，确认整个环境可用。

在便携版里这样调用（AmpliconDesk 目录下）：
    runtime\\python.exe app\\verify_portable.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amplicondesk import run_analysis  # noqa: E402
from amplicondesk.style import RESEARCH_PALETTES  # noqa: E402


def main() -> int:
    print("=" * 66)
    print("AmpliconDesk 便携版验收测试")
    print("=" * 66)
    print(f"Python   : {sys.version.split()[0]}")
    print(f"解释器   : {sys.executable}")
    print(f"科研配色 : {len(RESEARCH_PALETTES)} 套")
    print()

    demo = ROOT / "examples" / "demo_data"
    if not demo.exists():
        print("找不到演示数据 examples/demo_data")
        return 1

    t0 = time.time()
    res = run_analysis(
        {
            "analysis_type": "amplicon16s",
            "abundance_path": str(demo / "16s_otu_table.csv"),
            "metadata_path": str(demo / "16s_metadata.csv"),
            "taxonomy_path": str(demo / "16s_taxonomy.csv"),
            "env_path": str(demo / "16s_env.csv"),
            "group_column": "Group",
            "permutations": 199,
            "beta_metrics": ["braycurtis"],
            "ordinations": ["pcoa", "pca", "umap"],
            "stats_tests": ["permanova", "anosim", "mrpp", "permdisp", "envfit"],
            "run_hypothesis": True,
            "run_within_group": True,
            "run_advanced": True,
            "style": {
                "values": {
                    "font.family": "Times New Roman",
                    "font.title.style": "italic",
                    "export.formats": "png+pdf",
                    "palette.preset": "方案16",
                    "element.frame": False,
                    "element.line_width": 2.2,
                }
            },
        },
        output_root=str(ROOT / "output"),
    )

    out = Path(res["output_dir"])
    png = sorted((out / "figures").glob("*.png"))
    pdf = sorted((out / "figures").glob("*.pdf"))

    print(f"耗时       : {time.time() - t0:.1f}s")
    print(f"图表       : {len(res['figures'])} 张（PNG {len(png)} / PDF {len(pdf)}）")
    print(f"结果表     : {len(res['tables'])} 个")
    print(f"警告       : {res['warnings'] or '无'}")
    print(f"自动结论   : {res['narrative']['headline'][:120]}")
    print()

    checks = [
        ("生成图表 ≥ 20 张", len(png) >= 20),
        ("PNG 与 PDF 数量一致", len(png) == len(pdf) and len(png) > 0),
        ("生成 HTML 报告", (out / "report.html").exists()),
        ("生成中文结论", (out / "conclusions.txt").exists() and (out / "结论描述.md").exists()),
        ("生成 Excel 汇总", (out / "tables" / "analysis_summary.xlsx").exists()),
        ("生成假设检验表", (out / "tables" / "hypothesis_tests.csv").exists()),
        ("包含高级图形", any("violin" in p.name or "raincloud" in p.name for p in png)),
        ("包含组内差异分析", any("within" in p.name for p in png)),
    ]
    all_ok = True
    for name, ok in checks:
        all_ok &= ok
        print(f"  {'✓' if ok else '✗'} {name}")

    print()
    print(f"结果目录：{out}")
    print("结论：", "便携版可正常运行 ✅" if all_ok else "存在问题 ❌")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
