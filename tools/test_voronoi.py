"""验证自实现的维诺树状图（乘方图）在真实数据上的收敛性与耗时。"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from amplicondesk.core.io import read_table, read_taxonomy
from amplicondesk.core.preprocess import aggregate_by_taxonomy
from amplicondesk.plots.treemap import (plot_treemap_fallback, plot_voronoi_treemap,
                                   voronoi_treemap_geometry)
from amplicondesk.style import Style


def main() -> int:
    t = read_table(Path("examples/demo_data/16s_otu_table.csv"), index_col=0)
    tax = read_taxonomy(Path("examples/demo_data/16s_taxonomy.csv"))
    g = aggregate_by_taxonomy(t, tax, "Genus")
    rel = g.div(g.sum(axis=0), axis=1)
    m = rel.mean(axis=1).sort_values(ascending=False)

    print("=== 收敛性与耗时（真实 16S 属水平数据）===")
    print(f"{'n':>4} {'耗时':>9} {'最大面积误差':>12}  结论")
    ok = 0
    for n in (4, 6, 8, 12, 18, 25):
        w = m.head(n).to_numpy(dtype=float)
        t0 = time.time()
        try:
            geo = voronoi_treemap_geometry(w, n_iter=150)
            dt = time.time() - t0
            err = geo["max_rel_error"]
            good = err < 0.12
            ok += good
            print(f"{n:>4} {dt:>8.2f}s {err:>11.1%}  {'✓ 收敛' if good else '误差偏大'}"
                  f"  迭代 {geo['iterations']} 次")
        except Exception as exc:
            print(f"{n:>4} {time.time() - t0:>8.2f}s {'--':>12}  失败 {str(exc)[:50]}")

    print()
    print("=== 极端权重（人为构造 1000:1）===")
    for factor in (10, 100, 1000):
        w = np.array([1.0, 1.0, 1.0, factor])
        t0 = time.time()
        try:
            geo = voronoi_treemap_geometry(w, n_iter=200)
            print(f"  比值 {factor:>5}: 耗时 {time.time() - t0:5.2f}s  "
                  f"误差 {geo['max_rel_error']:.1%}")
        except Exception as exc:
            print(f"  比值 {factor:>5}: 失败 {str(exc)[:60]}")

    print()
    print("=== 出图测试 ===")
    st = Style()
    for n in (8, 18):
        try:
            info = plot_voronoi_treemap(g, Path(f"preview/_voro_{n}.png"), st,
                                        rank_label="属", top_n=n)
            p = Path(info["path"])
            print(f"  维诺 top_n={n:2d}  OK  {p.stat().st_size / 1024:.0f} KB"
                  f"  误差 {info['voronoi']['max_rel_error']:.1%}")
        except Exception as exc:
            print(f"  维诺 top_n={n:2d}  失败 {str(exc)[:70]}")

    info = plot_treemap_fallback(g, Path("preview/_treemap.png"), st, rank_label="属", top_n=25)
    print(f"  矩形树图（兜底）OK  {Path(info['path']).stat().st_size / 1024:.0f} KB")

    print()
    print("结论：", "维诺树状图可用 ✅" if ok >= 5 else f"仅 {ok}/6 收敛 ⚠")
    return 0 if ok >= 5 else 1


if __name__ == "__main__":
    sys.exit(main())
