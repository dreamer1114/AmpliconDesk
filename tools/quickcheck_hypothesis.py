"""临时验证脚本：交叉核对 hypothesis.py 中自实现的统计量（非交付文件，验完删除）。"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, ".")
from amplicondesk.core import hypothesis as H  # noqa: E402

warnings.simplefilter("error")
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- 1. Dunn 交叉核对
print("=" * 70)
print("1) Dunn 事后检验：与独立按定义实现的结果比对")


def dunn_reference(groups: dict[str, np.ndarray], method: str = "bh") -> pd.DataFrame:
    """按 Pohlert (2014) / Dunn (1964) 定义重新实现一遍（写法与模块内不同）。"""
    keys = list(groups)
    pooled = np.concatenate([groups[k] for k in keys])
    N = pooled.size
    ranks = st.rankdata(pooled)
    # 用 scipy 自带的 ties 计算方式（与 kruskal 内部一致）
    _, cnt = np.unique(pooled, return_counts=True)
    ties = float((cnt ** 3 - cnt).sum())
    tie_term = ties / (12.0 * (N - 1))
    pos = 0
    mr, n_i = {}, {}
    for k in keys:
        m = groups[k].size
        mr[k] = ranks[pos:pos + m].mean()
        n_i[k] = m
        pos += m
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            diff = mr[a] - mr[b]
            # 与模块不同的等价写法：Var(R̄a-R̄b) = (N(N+1)/12 - tie_term)*(na+nb)/(na*nb*N)
            var = (N * (N + 1) / 12.0 - tie_term) * (1.0 / n_i[a] + 1.0 / n_i[b])
            z = diff / np.sqrt(var)
            rows.append(dict(group1=a, group2=b, z=z, p=2 * st.norm.sf(abs(z)), var=var))
    out = pd.DataFrame(rows)
    if method == "bh":
        out["p_adj"] = H.bh_fdr(out["p"].to_numpy())
    return out


g = {
    "A": rng.normal(0.0, 1.0, 17),
    "B": rng.exponential(1.0, 23),
    "C": rng.normal(2.0, 1.0, 11),
    "D": np.round(rng.normal(1.0, 1.0, 19), 1),  # 人为制造并列
}
mine = H.dunn_posthoc(g, p_adjust="bh")
ref = dunn_reference(g)
cmp = mine.merge(ref[["group1", "group2", "z", "p", "var"]], on=["group1", "group2"], suffixes=("_mod", "_ref"))
print(cmp.to_string(index=False))
for col in ("z", "p"):
    print(f"  {col} 最大绝对误差 = {np.abs(cmp[f'{col}_mod'] - cmp[f'{col}_ref']).max():.3e}")
print(f"  模块 z={mine['z'].to_numpy()}")
print(f"  参考 z={ref['z'].to_numpy()}")
assert np.allclose(mine["z"], ref["z"], atol=1e-12)
assert np.allclose(mine["p"], ref["p"], atol=1e-12)
print("  OK")

# ---------------------------------------------------------------- 2. Tukey 交叉核对
print("=" * 70)
print("2) Tukey HSD：scipy.stats.tukey_hsd 官方实现 vs 自实现 studentized range 回退")
g2 = {
    "A": rng.normal(0.0, 1.0, 21),
    "B": rng.normal(0.6, 1.0, 18),
    "C": rng.normal(1.4, 1.0, 25),
}
official = H.tukey_hsd(g2)

# ---- 回退路径（自实现 studentized range 分布）----
print("  studentized range CDF 自实现 vs scipy.stats.studentized_range：")
import time as _time

t0 = _time.perf_counter()
for kk in (2, 3, 4):
    for dff in (10.0, 61.0):
        for qq in (1.5, 2.5, 3.4, 5.0):
            mine_cdf = H._studentized_range_cdf(qq, kk, dff)
            ref_cdf = float(st.studentized_range.cdf(qq, kk, dff))
            if abs(mine_cdf - ref_cdf) > 1e-6:
                print(f"    MISMATCH q={qq} k={kk} df={dff}: 自实现={mine_cdf:.8f} scipy={ref_cdf:.8f} "
                      f"diff={abs(mine_cdf - ref_cdf):.2e}")
                raise AssertionError("自实现 studentized range CDF 与 scipy 不一致")
t1 = _time.perf_counter()
print(f"    24 组 (q,k,df) 全部一致（误差 < 1e-6），首次建表总耗时 {t1 - t0:.2f}s")
for prob, kk, dff in ((0.95, 3, 60.0), (0.95, 4, 20.0), (0.99, 6, 12.0)):
    mine_q = H._studentized_range_ppf(prob, kk, dff)
    ref_q = float(st.studentized_range.ppf(prob, kk, dff))
    print(f"    ppf({prob}, k={kk}, df={dff:.0f}) = {mine_q:.5f} vs scipy {ref_q:.5f}（差 {abs(mine_q - ref_q):.2e}）")

# 仅使用回退路径重算 Tukey（临时移除 scipy.stats.tukey_hsd）
_saved = getattr(st, "tukey_hsd", None)
H._SR_CACHE.clear()
if _saved is not None:
    del st.tukey_hsd
try:
    fallback = H.tukey_hsd(g2)
finally:
    if _saved is not None:
        st.tukey_hsd = _saved
print("  回退路径 Tukey HSD：")
print(fallback.to_string(index=False))
print(f"  p 最大差异 = {np.abs(official['p'].to_numpy() - fallback['p'].to_numpy()).max():.3e}")
print(f"  CI 最大差异 = {np.abs(official['ci_low'].to_numpy() - fallback['ci_low'].to_numpy()).max():.3e} / "
      f"{np.abs(official['ci_high'].to_numpy() - fallback['ci_high'].to_numpy()).max():.3e}")
assert np.allclose(official["p"].to_numpy(), fallback["p"].to_numpy(), atol=1e-4)
assert np.allclose(official["meandiff"].to_numpy(), fallback["meandiff"].to_numpy(), atol=1e-9)

# ---------------------------------------------------------------- 3. 秩和 CI 下标
print("=" * 70)
print("3) Mann-Whitney U 下尾临界值：与独立组合计数实现比对")


def rank_sum_critical_ref(n1: int, n2: int, target: float) -> int | None:
    """独立实现：枚举 1..N 中取 a 个数的所有组合，统计 W 分布后平移到 U。"""
    from itertools import combinations
    from math import comb

    a, b = min(n1, n2), max(n1, n2)
    N = a + b
    offset = a * (a + 1) // 2
    M = a * b
    counts = [0] * (M + 1)
    for combo in combinations(range(1, N + 1), a):
        counts[sum(combo) - offset] += 1
    total = comb(N, a)
    assert sum(counts) == total
    cum = 0.0
    best = None
    for u in range(M + 1):
        cum += counts[u] / total
        if cum <= target + 1e-15:
            best = u
    return best


for (n1, n2) in [(4, 6), (5, 5), (6, 8), (3, 25), (10, 10)]:
    for target in (0.025, 0.05):
        mine_w = H._rank_sum_tail_prob(n1, n2, target)
        ref_w = rank_sum_critical_ref(n1, n2, target)
        flag = "OK" if mine_w == ref_w else "MISMATCH"
        print(f"  n1={n1:>2} n2={n2:>2} target={target}: 模块 k={mine_w} 参考 k={ref_w}  {flag}")
        assert mine_w == ref_w, f"U 下尾临界值不一致 n1={n1} n2={n2}"
print("  OK（组合枚举法穷举验证）")

# ---------------------------------------------------------------- 4. HL CI 覆盖率
print("=" * 70)
print("4) Hodges-Lehmann 95% CI 实际覆盖率（两样本同分布 → 真实偏移 0）")
for (n1, n2) in [(6, 6), (10, 10), (20, 20), (15, 25), (30, 30), (8, 40)]:
    NF = 1500 if n1 * n2 > 2500 else 3000
    cover = 0
    widths = []
    for _ in range(NF):
        x = rng.exponential(1.0, n1)
        y = rng.exponential(1.0, n2)
        _, lo, hi = H._hl_ci(x, y)
        if lo <= 0.0 <= hi:
            cover += 1
        widths.append(hi - lo)
    cov = cover / NF
    print(f"  n1={n1:>2} n2={n2:>2}: 覆盖率={cov:.4f}（理论≈0.95）平均宽度={np.mean(widths):.3f} "
          f"k={H._mw_ci_index(n1, n2, 0.05)}")
    assert 0.91 <= cov <= 0.98, f"HL CI 覆盖率异常 n1={n1} n2={n2}: {cov}"

# 平移不变性：y 整体 +3，CI 端点应整体 -3（θ = x - y）
x_t = rng.normal(0, 1, 25)
y_t = rng.normal(1.5, 1, 25)
lo1, hi1 = H._hl_ci(x_t, y_t)[1:]
lo2, hi2 = H._hl_ci(x_t, y_t + 3.0)[1:]
print(f"  平移不变性：CI(y)=[{lo1:.4f}, {hi1:.4f}]，CI(y+3)=[{lo2:.4f}, {hi2:.4f}]，"
      f"位移 {lo1 - lo2:.6f} / {hi1 - hi2:.6f}（应 = +3）")
assert abs((lo1 - lo2) - 3.0) < 1e-9 and abs((hi1 - hi2) - 3.0) < 1e-9
print("  OK")

# ---------------------------------------------------------------- 5. 配对 Walsh CI
print("=" * 70)
print("5) Wilcoxon 符号秩 Walsh 平均 CI 覆盖率（对称分布，真实中位数 0）")
for n in (10, 20, 30, 45):
    cover = 0
    for _ in range(3000):
        d = rng.normal(0.0, 1.0, n)
        walsh = np.sort(0.5 * (d[:, None] + d[None, :])[np.triu_indices(n)])
        idx = H._signed_rank_ci_index(n, 0.05)
        lo, hi = walsh[idx], walsh[walsh.size - 1 - idx]
        if lo <= 0.0 <= hi:
            cover += 1
    cov = cover / 3000
    print(f"  n={n:>2}: 覆盖率={cov:.4f}（理论≈0.95）下标={H._signed_rank_ci_index(n, 0.05)}")
    assert 0.91 <= cov <= 0.98, f"配对 Walsh CI 覆盖率异常 n={n}: {cov}"
print("  OK")

# ---------------------------------------------------------------- 6. 卡方/Fisher 与 scipy 一致性
print("=" * 70)
print("6) 卡方/Yates/Fisher 与 scipy 直接调用一致性")
ct = np.array([[20, 10], [8, 22]], dtype=float)
mine = H.compare_categorical(pd.DataFrame(ct))
chi_y = st.chi2_contingency(ct, correction=True)
odds, p_f = st.fisher_exact(ct)
print(f"  模块 chi2={mine['statistic']:.6f} p={mine['p']:.6f}  vs scipy Yates chi2={chi_y[0]:.6f} p={chi_y[1]:.6f}")
assert np.isclose(mine["statistic"], chi_y[0]) and np.isclose(mine["p"], chi_y[1])

small = np.array([[9, 1], [1, 7]], dtype=float)
mine_s = H.compare_categorical(pd.DataFrame(small))
print(f"  小期望表 → {mine_s['test']} OR={mine_s['statistic']:.4f} p={mine_s['p']:.6f} "
      f"(scipy fisher OR={odds:.4f})")
assert mine_s["test_key"] == "fisher"
assert np.isclose(mine_s["p"], st.fisher_exact(small)[1])

big = np.array([[30, 10, 5], [8, 25, 12], [5, 8, 30]], dtype=float)
mine_b = H.compare_categorical(pd.DataFrame(big))
chi_b = st.chi2_contingency(big, correction=False)
print(f"  3x3 → chi2={mine_b['statistic']:.6f} dof={mine_b['dof']:.0f} p={mine_b['p']:.6f} "
      f"vs scipy chi2={chi_b[0]:.6f} dof={chi_b[2]} p={chi_b[1]:.6f}")
assert np.isclose(mine_b["statistic"], chi_b[0]) and np.isclose(mine_b["p"], chi_b[1])
print("  OK")

# ---------------------------------------------------------------- 7. 自动选路的一致性扫描
print("=" * 70)
print("7) 自动选路扫描：300 次随机数据（正态/偏态 × 方差齐/不齐）")


def shapiro_p(a):
    return float(st.shapiro(a)[1])


mis_student = mis_welch = mis_mw = 0
for _ in range(300):
    n1, n2 = rng.integers(25, 45), rng.integers(25, 45)
    kind = rng.integers(0, 4)
    if kind == 0:
        x, y = rng.normal(0, 1, n1), rng.normal(0.5, 1, n2)
    elif kind == 1:
        x, y = rng.normal(0, 1, n1), rng.normal(0.5, 3, n2)
    else:
        x, y = rng.exponential(1, n1), rng.exponential(2, n2)
    res = H.compare_two_groups(x, y)
    pn = min(res["normality"]["per_group"]["a"]["p"], res["normality"]["per_group"]["b"]["p"])
    normal = pn > 0.05
    lev = res["variance"]
    eqvar = bool(lev.get("ok") and lev.get("p", 0) > 0.05)
    if normal and eqvar:
        expect = "student"
    elif normal and not eqvar:
        expect = "welch"
    else:
        expect = "mannwhitney"
    got = res["test_key"]
    if got != expect:
        print(f"    MISMATCH expect={expect} got={got} shapiro={pn:.4f} levene={lev.get('p')}")
        if expect == "student":
            mis_student += 1
        elif expect == "welch":
            mis_welch += 1
        else:
            mis_mw += 1
print(f"  300 次扫描中选路不一致：Student={mis_student} Welch={mis_welch} MW={mis_mw}")
assert mis_student + mis_welch + mis_mw == 0
print("  OK")

print("=" * 70)
print("全部交叉验证通过")
