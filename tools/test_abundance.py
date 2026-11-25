"""验证相对丰度自动识别与转换。"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from amplicondesk.core.preprocess import detect_abundance_type, ensure_relative


def main() -> int:
    rng = np.random.default_rng(0)
    idx = [f"f{i}" for i in range(30)]
    cols = list("ABCDEFGH")

    cases = {
        "原始计数": pd.DataFrame(rng.poisson(50, (30, 8)), index=idx, columns=cols),
        "比例(和=1)": pd.DataFrame(rng.dirichlet(np.ones(30), 8).T, index=idx, columns=cols),
        "百分比(和=100)": pd.DataFrame(rng.dirichlet(np.ones(30), 8).T * 100, index=idx, columns=cols),
        "小数计数": pd.DataFrame(np.round(rng.lognormal(3, 1, (30, 8)), 2), index=idx, columns=cols),
        "已是相对丰度(未完全归一)": pd.DataFrame(
            rng.dirichlet(np.ones(30), 8).T * rng.uniform(0.9, 1.1, 8)[None, :],
            index=idx, columns=cols),
    }

    expect = {
        "原始计数": False,
        "比例(和=1)": True,
        "百分比(和=100)": True,
        "小数计数": False,
        "已是相对丰度(未完全归一)": True,
    }

    ok = True
    for name, df in cases.items():
        info = detect_abundance_type(df)
        rel, _ = ensure_relative(df, info)
        colsum = float(rel.sum(axis=0).iloc[0])
        good = (info["is_relative"] == expect[name]) and abs(colsum - 1.0) < 1e-9
        ok &= good
        print(f"{'✓' if good else '✗'} {name:24s} 类型={info['type']:18s} "
              f"相对丰度={info['is_relative']!s:5s} 列和中位={info['column_sum_median']:>10.3f} "
              f"整数比={info['integer_ratio']:.2f} 转换后列和={colsum:.6f}")
        print(f"    {info['reason']}")

    print()
    print("结论：", "全部通过 ✅" if ok else "存在不符合预期的判定 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
