"""检查所有生成图是否存在"内容贴边"（= 可能被 tight bbox 裁掉）的情况。"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amplicondesk import run_analysis  # noqa: E402

STYLE = {
    "font.scale": 1.15, "font.title.color": "#1B4F9C", "font.title.size": 15,
    "palette.preset": "nejm", "legend.position": "outside right", "legend.ncol": 2,
    "element.point_size": 110, "figure.dpi": 200,
    "fig.volcano.font.title.text": "火山图（自定义）",
}

CFG = {
    "analysis_type": "amplicon16s",
    "abundance_path": str(ROOT / "examples/demo_data/16s_otu_table.csv"),
    "metadata_path": str(ROOT / "examples/demo_data/16s_metadata.csv"),
    "taxonomy_path": str(ROOT / "examples/demo_data/16s_taxonomy.csv"),
    "group_column": "Group", "permutations": 199,
    "ordinations": ["pcoa"], "stats_tests": ["permanova"],
    "run_alpha": False, "run_beta": True, "run_ordination": True, "run_multivariate": False,
    "run_composition": False, "run_diff": True, "run_lda": False, "run_rf": False,
    "run_advanced": False,
    "style": {"values": STYLE},
}


def edges_of(path: Path, thresh: int = 200) -> list[str]:
    a = np.array(Image.open(path).convert("L"))
    hits = []
    if (a[:, 0] < thresh).any():
        hits.append("left")
    if (a[:, -1] < thresh).any():
        hits.append("right")
    if (a[0, :] < thresh).any():
        hits.append("top")
    if (a[-1, :] < thresh).any():
        hits.append("bottom")
    return hits


def main() -> int:
    res = run_analysis(CFG, output_root=str(ROOT / "output"))
    print("警告:", res["warnings"] or "无")
    print()
    bad = []
    for f in res["figures"]:
        p = Path(f["path"])
        a = Image.open(p)
        w, h = a.size
        hits = edges_of(p)
        if hits:
            bad.append((p.name, hits))
        print(f"  {p.name:46s} {w:5d}x{h:<5d} 贴边={'/'.join(hits) or '无'}")
    print()
    if bad:
        print(f"❌ {len(bad)} 张图有贴边（可能被裁）:")
        for name, hits in bad:
            print(f"   {name}: {hits}")
        return 1
    print("✅ 无任何图贴边，内容完整")
    return 0


if __name__ == "__main__":
    sys.exit(main())
