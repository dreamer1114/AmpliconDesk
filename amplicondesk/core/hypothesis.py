"""假设检验模块：正态性/方差齐性诊断、两组与多组比较、分类变量关联检验。

设计目标
--------
给"科研用户一键出图"提供统计引擎：调用方只需要把 `值 + 分组` 丢进来，
本模块自动完成

1. 前提假设诊断（Shapiro-Wilk / D'Agostino / Anderson-Darling / KS 正态性；
   Levene / Bartlett / Fligner 方差齐性）；
2. 依据诊断结果**自动选择**检验方法（t 检验族 / Wilcoxon 族 / ANOVA / Kruskal-Wallis）；
3. 事后多重比较（Tukey HSD / Dunn），统一走 :func:`amplicondesk.core.stats.bh_fdr` 校正；
4. 效应量、置信区间与**可直接展示给科研用户的中文结论**（``reason`` 字段）。

实现约定
--------
* 只依赖 numpy / scipy / pandas，不依赖 statsmodels、不调用 R。
* 所有退化输入（样本数不足、零方差、全 NaN、组数不足）都不抛异常，
  而是返回带中文说明的结果字典，并置 ``ok=False``。
* 本模块**不绘图**，绘图由 ``amplicondesk.plots`` 负责。
* 导入时无任何副作用。
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.special import gamma as _gamma_fn
from scipy.special import ndtr as _ndtr

if __package__ in (None, ""):  # 直接运行脚本时把项目根目录加入 sys.path，便于绝对导入
    import sys
    from pathlib import Path

    _root = str(Path(__file__).resolve().parents[2])
    if _root not in sys.path:
        sys.path.insert(0, _root)

try:
    from .stats import bh_fdr
except ImportError:  # pragma: no cover - 仅在以脚本方式直接运行时触发
    from amplicondesk.core.stats import bh_fdr  # type: ignore[no-redef]

__all__ = [
    "check_normality",
    "check_variance",
    "compare_two_groups",
    "compare_multiple_groups",
    "compare_categorical",
    "compare_paired",
    "effect_size_cohens_d",
    "effect_size_hedges_g",
    "effect_size_cliffs_delta",
    "effect_size_rank_biserial",
    "describe_groups",
    "dunn_posthoc",
    "tukey_hsd",
    "auto_test",
    "significance_stars",
]

# --------------------------------------------------------------------------
# 全局常量
# --------------------------------------------------------------------------
_MIN_N: int = 3                 # 单组最少样本数
_SHAPIRO_MAX_N: int = 5000      # Shapiro-Wilk 的有效样本上限
_HL_MAX_PAIRS: int = 4_000_000  # Hodges-Lehmann 精确区间的成对差值上限（约 32 MB）
_EPS: float = 1e-12


# ==========================================================================
# 内部工具：数据清洗与安全调用
# ==========================================================================
def _as_float(values: Any) -> np.ndarray:
    """把任意一维输入转成 float 数组（非有限值统一变 NaN，不做删除）。"""
    if values is None:
        return np.array([], dtype=float)
    if isinstance(values, (pd.Series, pd.Index)):
        arr = values.to_numpy()
    elif isinstance(values, pd.DataFrame):
        arr = values.to_numpy()
    else:
        arr = np.asarray(values)
    arr = np.asarray(arr, dtype=float).ravel()
    out = np.array(arr, dtype=float, copy=True)
    out[~np.isfinite(out)] = np.nan
    return out


def _finite(values: Any) -> np.ndarray:
    """取输入中的有限值（丢弃 NaN / inf）。"""
    arr = _as_float(values)
    return arr[np.isfinite(arr)]


def _n_missing(values: Any) -> int:
    """统计输入中非有限值（NaN / inf）的个数。"""
    arr = _as_float(values)
    return int(np.count_nonzero(~np.isfinite(arr)))


def _clean_groups(groups_dict: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """把 ``{组名: 一维数据}`` 规范成 ``{str: float array}``（保留 NaN 以便统计缺失）。"""
    return {str(k): _as_float(v) for k, v in groups_dict.items()}


def _all_same(arr: np.ndarray) -> bool:
    """数组是否所有（有限）取值相同（即零方差）。"""
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        return True
    return bool(np.ptp(fin) <= 0)


class _Guard:
    """把 scipy 抛出的异常/警告收集成中文说明，避免整个流程崩溃。"""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def add(self, message: str) -> None:
        if message and message not in self.messages:
            self.messages.append(message)

    def run(self, func, *args, **kwargs):
        """执行统计函数，失败时返回 ``(nan, nan)`` 并记录中文说明。"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 需要兜住 scipy 的各种报错
            self.add(f"统计计算未能完成（{type(exc).__name__}: {exc}）。")
            return float("nan"), float("nan")

    @property
    def note(self) -> str:
        return " ".join(self.messages)


def _degenerate(messages: str, **extra: Any) -> dict:
    """构造统一的退化结果字典。"""
    out: dict[str, Any] = {
        "ok": False,
        "test": "未执行",
        "statistic": float("nan"),
        "p": float("nan"),
        "reason": messages,
        "note": messages,
    }
    out.update(extra)
    return out


def _p_adjust(p: np.ndarray, method: str = "bh") -> np.ndarray:
    """多重检验校正：``bh`` 走 :func:`amplicondesk.core.stats.bh_fdr`，``bonferroni`` 走简单放大。"""
    p = np.asarray(p, dtype=float)
    if str(method).lower() in ("bonferroni", "bonf", "holm-bonferroni"):
        n = int(np.isfinite(p).sum())
        return np.clip(p * max(n, 1), 0.0, 1.0)
    return np.asarray(bh_fdr(p), dtype=float)


def _fmt_p(p: float) -> str:
    """把 p 值格式化成用户可读文本。"""
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.3f}"


def significance_stars(p: float, thresholds: Sequence[float] = (0.001, 0.01, 0.05)) -> str:
    """把 p 值转成显著性星号。

    Args:
        p: p 值。
        thresholds: 依次对应 ``***`` / ``**`` / ``*`` 的阈值，默认 ``(0.001, 0.01, 0.05)``。

    Returns:
        ``"***"`` / ``"**"`` / ``"*"`` / ``"ns"``；p 非法时返回 ``"ns"``。

    Examples:
        >>> significance_stars(0.0004)
        '***'
        >>> significance_stars(0.2)
        'ns'
    """
    t = list(thresholds) if thresholds is not None else [0.001, 0.01, 0.05]
    if p is None or not np.isfinite(p):
        return "ns"
    stars = ["***", "**", "*"]
    for thr, mark in zip(t, stars):
        if p < thr:
            return mark
    return "ns"


# ==========================================================================
# 正态性检验
# ==========================================================================
def check_normality(
    values: Any,
    group: Any = None,
    method: str = "shapiro",
    alpha: float = 0.05,
) -> dict:
    """正态性检验（Shapiro-Wilk / D'Agostino / Anderson-Darling / KS）。

    当 ``group`` 为 ``None`` 时对整列做一次检验；否则按分组标签分别检验。

    Args:
        values: 一维数值序列（list / numpy 数组 / pandas Series）。
        group: 与 ``values`` 等长的分组标签；``None`` 表示不分组。
        method: 检验方法，``"shapiro"``（默认，n < 5000，n > 5000 自动回退
            D'Agostino）、``"dagostino"``（偏度-峰度检验）、``"anderson"``
            （Anderson-Darling）、``"ks"``（Kolmogorov-Smirnov，参数由样本估计）。
        alpha: 显著性水平，``p > alpha`` 判为正态。

    Returns:
        字典，含 ``per_group``（组名 → 单组结果）、``overall``、``all_normal``、
        ``method``、``alpha``、``note``（中文说明）以及 ``ok``。

    Notes:
        ``anderson`` 的 ``statistic`` 为 A² 统计量、``p`` 取
        ``scipy.stats.anderson`` 在方法内部拟合得到的近似 p 值（scipy ≥ 1.17），
        判正态仍统一用 ``p > alpha``，以保证四种方法口径一致。
    """
    method = str(method).lower()
    if method not in ("shapiro", "dagostino", "anderson", "ks"):
        return _degenerate(
            f"不支持的正态性检验方法：{method}（可选 shapiro / dagostino / anderson / ks）。"
        )

    if group is None:
        groups: dict[str, np.ndarray] = {"all": _as_float(values)}
    else:
        g_arr = np.asarray(group).ravel()
        v_arr = _as_float(values)
        if g_arr.size != v_arr.size:
            return _degenerate("数据长度与分组标签长度不一致，无法做正态性检验。")
        g_txt = np.array([str(x) for x in g_arr], dtype=object)
        groups = {name: v_arr[g_txt == name] for name in pd.unique(pd.Series(g_txt))}

    per_group: dict[str, dict] = {}
    notes: list[str] = []
    for name, raw in groups.items():
        fin = raw[np.isfinite(raw)]
        n = int(fin.size)
        n_missing = int(raw.size - n)
        info = _run_normality(fin, method, n)
        info["n_missing"] = n_missing
        info["normal"] = bool(info["ok"] and np.isfinite(info["p"]) and info["p"] > alpha)
        per_group[name] = info
        if info.get("note"):
            notes.append(f"组「{name}」：{info['note']}")
        elif n_missing > 0:
            notes.append(f"组「{name}」有 {n_missing} 个缺失/非有限值，已剔除。")
        # 统一用当次调用的 alpha 判定正态（覆盖 _run_normality 的默认 0.05）
        info["normal"] = bool(info["ok"] and np.isfinite(info["p"]) and info["p"] > alpha)

    overall = per_group.get("all") if group is None else None
    valid = [g for g in per_group.values() if g.get("ok")]
    all_normal = bool(len(valid) == len(per_group) and len(valid) > 0 and all(g["normal"] for g in valid))

    if group is None:
        if not valid:
            note = overall["note"] if overall else "无法进行正态性检验。"
        elif overall["normal"]:
            note = f"整体数据符合正态分布（{overall['test']} p = {_fmt_p(overall['p'])} > {alpha}）。"
        else:
            note = f"整体数据不符合正态分布（{overall['test']} p = {_fmt_p(overall['p'])} ≤ {alpha}）。"
    else:
        if not valid:
            note = "所有分组都无法完成正态性检验：" + (" ".join(notes) if notes else "有效样本不足。")
        elif all_normal:
            note = f"各组数据均符合正态分布（{method} 检验 p > {alpha}）。"
        else:
            bad = [k for k, v in per_group.items() if v.get("ok") and not v["normal"]]
            if not bad:
                bad = [k for k, v in per_group.items() if not v.get("ok")]
            note = f"以下分组不符合正态分布或无法检验：{'、'.join(map(str, bad))}；其余组符合正态分布。"
    if notes:
        note = note + " " + " ".join(notes)

    return {
        "per_group": per_group,
        "overall": overall,
        "all_normal": all_normal,
        "method": method,
        "alpha": float(alpha),
        "note": note,
        "ok": bool(valid),
    }


def _run_normality(fin: np.ndarray, method: str, n: int) -> dict:
    """在已剔除缺失值的数组上执行指定的正态性检验。"""
    if n < _MIN_N:
        return {
            "statistic": float("nan"),
            "p": float("nan"),
            "normal": False,
            "ok": False,
            "n": n,
            "test": method,
            "note": f"有效样本数不足（n = {n} < {_MIN_N}），无法进行正态性检验。",
        }
    if _all_same(fin):
        return {
            "statistic": float("nan"),
            "p": float("nan"),
            "normal": False,
            "ok": False,
            "n": n,
            "test": method,
            "note": "所有取值相同（零方差），正态性检验无定义。",
        }

    guard = _Guard()
    if method == "shapiro":
        if n >= _SHAPIRO_MAX_N:
            return {
                "statistic": float("nan"),
                "p": float("nan"),
                "normal": False,
                "ok": False,
                "n": n,
                "test": "Shapiro-Wilk",
                "note": (
                    f"样本数 n = {n} ≥ {_SHAPIRO_MAX_N}，Shapiro-Wilk 不再适用，"
                    "请改用 method='dagostino' 或 'anderson'。"
                ),
            }
        stat, p = guard.run(st.shapiro, fin)
        label = "Shapiro-Wilk"
    elif method == "dagostino":
        stat, p = guard.run(st.normaltest, fin)
        label = "D'Agostino-Pearson"
    elif method == "anderson":
        label = "Anderson-Darling"

        def _anderson() -> tuple[float, float]:
            res = st.anderson(fin, dist="norm")
            pval = getattr(res, "pvalue", None)
            if pval is None:  # 老版本 scipy：用 5% 临界值粗略换算
                stat_, crit_, sig_ = float(res.statistic), np.asarray(res.critical_values), np.asarray(res.significance_level)
                pval = float(np.interp(stat_, crit_, sig_)) / 100.0
            return float(res.statistic), float(pval)

        stat, p = guard.run(_anderson)
    else:  # ks
        label = "Kolmogorov-Smirnov"

        def _ks() -> tuple[float, float]:
            mu = float(np.mean(fin))
            sd = float(np.std(fin, ddof=1))
            res = st.kstest(fin, "norm", args=(mu, sd))
            return float(res.statistic), float(res.pvalue)

        stat, p = guard.run(_ks)

    ok = bool(np.isfinite(stat) and np.isfinite(p))
    return {
        "statistic": float(stat),
        "p": float(p),
        "normal": bool(ok and p > 0.05),
        "ok": ok,
        "n": n,
        "test": label,
        "note": guard.note,
    }


# ==========================================================================
# 方差齐性检验
# ==========================================================================
def check_variance(
    values: Any = None,
    groups: Any = None,
    method: str = "levene",
    alpha: float = 0.05,
) -> dict:
    """方差齐性检验（Levene / Bartlett / Fligner-Killeen）。

    Args:
        values: 分组数据。可以是 ``{组名: 一维数据}`` 的字典，也可以是
            "值 + 分组标签" 中的"值"（此时需同时给出 ``groups``）。
        groups: 与 ``values`` 等长的分组标签；``values`` 已是字典时可省略。
        method: ``"levene"``（默认，Brown-Forsythe 中位数版本，稳健）、
            ``"bartlett"``（要求各组正态，对偏态极敏感）、``"fligner"``
            （Fligner-Killeen，非参数，最稳健）。
        alpha: 显著性水平，``p > alpha`` 判为方差齐。

    Returns:
        字典，含 ``statistic``、``p``、``equal_variance``、``method``、``alpha``、
        ``note``（中文说明）、``ok``、``n_groups``；退化情形 ``ok=False``。
    """
    method = str(method).lower()
    if method not in ("levene", "bartlett", "fligner"):
        return _degenerate(f"不支持的方差齐性检验方法：{method}（可选 levene / bartlett / fligner）。")

    data = _resolve_groups(values, groups)
    samples = [fin for fin in (arr[np.isfinite(arr)] for arr in data.values())]
    names = list(data.keys())

    if len(samples) < 2:
        return _degenerate(
            f"只有 {len(samples)} 组有效数据，方差齐性检验至少需要 2 组。",
            method=method,
            alpha=float(alpha),
            n_groups=len(samples),
        )
    if any(s.size < _MIN_N for s in samples):
        bad = [names[i] for i, s in enumerate(samples) if s.size < _MIN_N]
        return _degenerate(
            f"以下组有效样本数不足（n < {_MIN_N}），无法做方差齐性检验：{'、'.join(map(str, bad))}。",
            method=method,
            alpha=float(alpha),
            n_groups=len(samples),
        )
    const = [names[i] for i, s in enumerate(samples) if _all_same(s)]
    if len(const) >= 2:
        return _degenerate(
            f"以下组取值完全相同（零方差），方差齐性检验无定义：{'、'.join(map(str, const))}。",
            method=method,
            alpha=float(alpha),
            n_groups=len(samples),
        )

    guard = _Guard()
    labels = {"levene": "Levene (Brown-Forsythe, 中位数中心)", "bartlett": "Bartlett", "fligner": "Fligner-Killeen"}
    if method == "levene":
        stat, p = guard.run(st.levene, *samples, center="median")
    elif method == "bartlett":
        stat, p = guard.run(st.bartlett, *samples)
    else:
        stat, p = guard.run(st.fligner, *samples)

    ok = bool(np.isfinite(stat) and np.isfinite(p))
    equal = bool(ok and p > alpha)
    if not ok:
        note = guard.note or "方差齐性检验未能完成。"
    elif equal:
        note = (
            f"各组方差齐性（{labels[method]} 检验：统计量 = {stat:.3f}，p = {_fmt_p(p)} > {alpha}），"
            "可使用方差齐性假设下的参数检验。"
        )
    else:
        note = (
            f"各组方差不齐（{labels[method]} 检验：统计量 = {stat:.3f}，p = {_fmt_p(p)} ≤ {alpha}），"
            "应使用 Welch 校正或非参数检验。"
        )
    return {
        "statistic": float(stat),
        "p": float(p),
        "equal_variance": equal,
        "method": method,
        "method_name": labels[method],
        "alpha": float(alpha),
        "n_groups": len(samples),
        "note": note,
        "ok": ok,
    }


def _resolve_groups(values: Any, groups: Any = None) -> dict[str, np.ndarray]:
    """把"值 + 标签"或"字典"统一规范为 ``{组名: float array}``。"""
    if isinstance(values, Mapping):
        return {str(k): _as_float(v) for k, v in values.items()}
    if values is None:
        return {}
    v = _as_float(values)
    if groups is None:
        return {"all": v}
    g = np.asarray(groups).ravel()
    if g.size != v.size:
        raise ValueError("数据长度与分组标签长度不一致。")
    g = np.array([str(x) for x in g], dtype=object)
    return {name: v[g == name] for name in pd.unique(pd.Series(g))}


# ==========================================================================
# 效应量
# ==========================================================================
def effect_size_cohens_d(a: Any, b: Any, paired: bool = False) -> float:
    """Cohen's d 效应量。

    Args:
        a: 第一组数据。
        b: 第二组数据。
        paired: ``True`` 时按配对样本计算（分母为差值的标准差，即 d_z）。

    Returns:
        Cohen's d；样本不足或分母为 0 时返回 ``nan``。符号以 "a 减 b" 为正方向。
    """
    x = _finite(a)
    y = _finite(b)
    if paired:
        if x.size != y.size or x.size < 2:
            return float("nan")
        d = x - y
        sd = float(np.std(d, ddof=1))
        if not np.isfinite(sd) or sd <= _EPS:
            return float("nan")
        return float(np.mean(d) / sd)
    if x.size < 2 or y.size < 2:
        return float("nan")
    n1, n2 = x.size, y.size
    sp2 = ((n1 - 1) * float(np.var(x, ddof=1)) + (n2 - 1) * float(np.var(y, ddof=1))) / (n1 + n2 - 2)
    if not np.isfinite(sp2) or sp2 <= _EPS:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / np.sqrt(sp2))


def effect_size_hedges_g(a: Any, b: Any) -> float:
    """Hedges' g（Cohen's d 的小样本无偏校正，独立样本）。"""
    x = _finite(a)
    y = _finite(b)
    d = effect_size_cohens_d(x, y, paired=False)
    n1, n2 = x.size, y.size
    if not np.isfinite(d) or n1 < 2 or n2 < 2:
        return float("nan")
    df = n1 + n2 - 2
    if df <= 1:
        return float("nan")
    j = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(d * j)


def effect_size_cliffs_delta(a: Any, b: Any) -> float:
    """Cliff's delta 非参数效应量，范围 ``[-1, 1]``，符号以 "a 减 b" 为正方向。

    定义：``delta = (#(a > b) - #(a < b)) / (n1 * n2)``。
    """
    x = _finite(a)
    y = _finite(b)
    if x.size == 0 or y.size == 0:
        return float("nan")
    diff = x[:, None] - y[None, :]
    gt = int(np.count_nonzero(diff > 0))
    lt = int(np.count_nonzero(diff < 0))
    return float((gt - lt) / (x.size * y.size))


def effect_size_rank_biserial(a: Any, b: Any) -> float:
    """秩双列相关（rank-biserial correlation），范围 ``[-1, 1]``。

    采用 Mann-Whitney 形式 ``r = 2·U_a/(n1·n2) - 1``，其中 ``U_a`` 为
    "a 组观测值大于 b 组观测值"的成对计数。该式等价于 ``2·AUC - 1``
    （AUC 即 a 大于 b 的概率），**正号表示 a 组取值整体偏大**，两组完全分离时
    ``|r| = 1``，与 :func:`effect_size_cliffs_delta` 符号方向一致。
    （推导：``U_a = #(a > b)``，故 ``U_a - U_b = n1·n2·delta``，代入
    ``U_a + U_b = n1·n2`` 即得 ``r = delta``。）
    """
    x = _finite(a)
    y = _finite(b)
    if x.size == 0 or y.size == 0:
        return float("nan")
    guard = _Guard()
    res = guard.run(st.mannwhitneyu, x, y, alternative="two-sided")
    if not np.isfinite(res[1]):
        return float("nan")
    u_a = float(res[0])
    return float(2.0 * u_a / (x.size * y.size) - 1.0)


def _cohen_d_note(d: float) -> str:
    """把 |d| 翻译成中文效应量描述。"""
    if not np.isfinite(d):
        return "效应量无法计算"
    ad = abs(d)
    if ad < 0.2:
        level = "极小"
    elif ad < 0.5:
        level = "小"
    elif ad < 0.8:
        level = "中等"
    else:
        level = "大"
    return f"效应量为{level}（|d| = {ad:.2f}）"


# ==========================================================================
# 描述统计
# ==========================================================================
def _describe_one(arr: np.ndarray, alpha: float = 0.05) -> dict:
    """单组描述统计。"""
    fin = arr[np.isfinite(arr)]
    n = int(fin.size)
    row: dict[str, Any] = {
        "n": n,
        "n_missing": int(arr.size - n),
        "mean": float("nan"),
        "sd": float("nan"),
        "median": float("nan"),
        "q1": float("nan"),
        "q3": float("nan"),
        "IQR": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
        "se": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
    }
    if n == 0:
        return row
    row["mean"] = float(np.mean(fin))
    row["median"] = float(np.median(fin))
    row["min"] = float(np.min(fin))
    row["max"] = float(np.max(fin))
    if n >= 2:
        sd = float(np.std(fin, ddof=1))
        row["sd"] = sd
        row["se"] = float(sd / np.sqrt(n))
        q1, q3 = np.quantile(fin, [0.25, 0.75])
        row["q1"] = float(q1)
        row["q3"] = float(q3)
        row["IQR"] = float(q3 - q1)
        tcrit = float(st.t.ppf(1.0 - alpha / 2.0, n - 1))
        if np.isfinite(tcrit):
            half = tcrit * row["se"]
            row["ci_low"] = float(row["mean"] - half)
            row["ci_high"] = float(row["mean"] + half)
    else:
        row["sd"] = float("nan")
        row["q1"] = float(fin[0])
        row["q3"] = float(fin[0])
        row["IQR"] = 0.0
    return row


def describe_groups(groups_dict: Mapping[str, Any], alpha: float = 0.05) -> pd.DataFrame:
    """计算每组的描述统计。

    Args:
        groups_dict: ``{组名: 一维数据}``。
        alpha: 置信水平（默认 95% 置信区间，基于 t 分布）。

    Returns:
        DataFrame，列为 ``group / n / n_missing / mean / sd / median / q1 / q3 /
        IQR / min / max / se / ci_low / ci_high``，索引为 ``group``。
    """
    data = _clean_groups(groups_dict) if isinstance(groups_dict, Mapping) else {}
    rows = []
    for name, arr in data.items():
        row = _describe_one(arr, alpha=alpha)
        row["group"] = str(name)
        rows.append(row)
    cols = [
        "group", "n", "n_missing", "mean", "sd", "median", "q1", "q3",
        "IQR", "min", "max", "se", "ci_low", "ci_high",
    ]
    if not rows:
        return pd.DataFrame(columns=cols).set_index("group")
    return pd.DataFrame(rows)[cols].set_index("group")


# ==========================================================================
# 两组比较
# ==========================================================================
def _mean_diff_ci(x: np.ndarray, y: np.ndarray, equal_var: bool, alpha: float = 0.05) -> tuple[float, float]:
    """均值差 (a - b) 的置信区间（Student 或 Welch 标准误）。"""
    n1, n2 = x.size, y.size
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    v1 = float(np.var(x, ddof=1))
    v2 = float(np.var(y, ddof=1))
    diff = float(np.mean(x) - np.mean(y))
    if equal_var:
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = float(np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2)))
        df = n1 + n2 - 2
    else:
        a, b = v1 / n1, v2 / n2
        se = float(np.sqrt(a + b))
        denom = (a * a) / (n1 - 1) + (b * b) / (n2 - 1)
        df = float((a + b) ** 2 / denom) if denom > 0 else float(n1 + n2 - 2)
    if not np.isfinite(se) or se <= 0 or not np.isfinite(df) or df <= 0:
        return float("nan"), float("nan")
    tcrit = float(st.t.ppf(1.0 - alpha / 2.0, df))
    if not np.isfinite(tcrit):
        return float("nan"), float("nan")
    return float(diff - tcrit * se), float(diff + tcrit * se)


def _hl_ci(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Hodges-Lehmann 位置偏移估计及其置信区间（Mann-Whitney 配套）。

    做法遵循 Mann-Whitney 的秩反演（Moses 1963；Hollander & Wolfe,
    *Nonparametric Statistical Methods*, 3rd ed., §4.2）：把全部 ``M = n1·n2``
    个成对差值 ``D = x_i - y_j`` 升序排列，则偏移 ``θ = x - y`` 的 ``(1-α)``
    置信区间为对称的次序统计量

    .. math::
        \\left[\\;D_{(k)},\\;D_{(M-1-k)}\\;\\right]

    其中 ``k`` 由 Mann-Whitney ``U`` 统计量的**精确**零分布下尾 ``α/2`` 分位点给出
    （并列时用正态近似），即 ``k`` 是满足 ``P(U ≤ k) ≤ α/2`` 的最大整数——
    注意 ``U`` 的**下**尾对应 ``D`` 的**上**尾，端点必须按 ``D_{(k)}``/``D_{(M-1-k)}``
    取。点估计为全部 ``D`` 的中位数。
    ``M`` 过大（> 4e6）时退化为按正态近似确定的新增区间并在两组的次序统计量上
    取端点，避免构造巨大的差值矩阵。

    Returns:
        ``(点估计, CI 下限, CI 上限)``；无法计算时元素为 ``nan``。
    """
    x = np.sort(x[np.isfinite(x)])
    y = np.sort(y[np.isfinite(y)])
    n1, n2 = int(x.size), int(y.size)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")
    M = n1 * n2
    k = _mw_ci_index(n1, n2, alpha)
    if k is None:
        k = 0
    k = int(min(max(k, 0), M - 1))
    if M > _HL_MAX_PAIRS:
        # 大样本回退：用两组的次序统计量近似（区间会略偏宽，属保守方向）
        i = int(min(max(round(k / n2), 0), n1 - 1))
        j = int(min(max(round(k / n1), 0), n2 - 1))
        lo = float(x[i] - y[j])
        hi = float(x[n1 - 1 - i] - y[n2 - 1 - j])
        if lo > hi:
            lo, hi = hi, lo
        return float(np.median(x) - np.median(y)), lo, hi
    diffs = (x[:, None] - y[None, :]).ravel()
    hl = float(np.median(diffs))
    if M < 2:
        return hl, hl, hl
    part = np.partition(diffs, [k, M - 1 - k])
    return hl, float(part[k]), float(part[M - 1 - k])


def _mw_ci_index(n1: int, n2: int, alpha: float = 0.05) -> int:
    """求 Mann-Whitney/Wilcoxon 秩和检验 CI 所需的临界值 ``k``。

    返回满足 ``P(W ≤ k) ≤ alpha/2`` 的最大 ``k``，其中 ``W`` 为 Wilcoxon 秩和
    统计量（取值 ``0..n1·n2``）。区间的两个端点随后由
    :func:`_hl_ci` 用该 ``k`` 作用于两组的次序统计量得到。
    """
    total = n1 * n2
    if total <= 1:
        return 0
    target = alpha / 2.0
    exact = _rank_sum_tail_prob(n1, n2, target)
    if exact is not None:
        return int(min(max(exact, 0), total - 1))
    # 正态近似（带连续性校正）
    mu = n1 * n2 / 2.0
    sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if not np.isfinite(sd) or sd <= 0:
        return 0
    q = float(st.norm.ppf(target))
    w = mu + q * sd + 0.5
    return int(min(max(np.floor(w), 0), total - 1))


def _rank_sum_tail_prob(n1: int, n2: int, target: float, max_total: int = 5000) -> int | None:
    """精确计算 Mann-Whitney ``U`` 统计量的下尾临界值。

    先求较小样本组的 Wilcoxon 秩和 ``W`` 的精确分布：``P(W = w)`` 等于从
    ``{1..N}`` 中取 ``a`` 个数之和为 ``w`` 的组合数除以 ``C(N, a)``
    （Mann & Whitney 1947；Hollander & Wolfe, 3rd ed., §4.1），生成函数为
    ``\\prod_{i=1}^{N}(1 + x\\,q^i)`` 的 ``x^a`` 系数。随后用恒等式

    .. math::
        U = W - \\frac{a(a+1)}{2}

    把秩和分布平移到 ``U ∈ [0, a·b]``，再返回满足 ``P(U ≤ k) ≤ target``
    的最大 ``k``。

    为保证性能，``n1·n2`` 超过 ``max_total`` 时返回 ``None``，交由正态近似处理。

    Returns:
        下尾临界值 ``k``；无法用精确法计算时返回 ``None``。
    """
    if n1 < 1 or n2 < 1:
        return None
    if n1 * n2 > max_total:
        return None
    a, b = min(n1, n2), max(n1, n2)
    N = a + b
    offset = a * (a + 1) // 2
    size = a * N - offset + 1  # U 的取值范围为 0..a*b
    # dp[j][s] = 从 1..rank 中取 j 个数、秩和为 s 的方案数；j 逆序原地更新
    dp = np.zeros((a + 1, size + offset + 1), dtype=float)
    dp[0, 0] = 1.0
    for rank in range(1, N + 1):
        for j in range(min(a, rank), 0, -1):
            if rank < dp.shape[1]:
                dp[j, rank:] += dp[j - 1, : dp.shape[1] - rank]
    pmf_w = dp[a]
    pmf_u = pmf_w[offset: offset + size]
    denom = float(np.sum(pmf_u))
    if denom <= 0:
        return None
    cdf = np.cumsum(pmf_u) / denom
    below = np.nonzero(cdf <= target)[0]
    if below.size == 0:
        return None
    return int(below[-1])


def _paired_ci(diff: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """配对差值的 Hodges-Lehmann 估计与 Wilcoxon 符号秩 CI（Walsh 平均法）。

    参考 Hollander & Wolfe, *Nonparametric Statistical Methods*, 3rd ed., §3.2：
    CI 由全部 Walsh 平均 ``(d_i + d_j)/2 (i ≤ j)`` 的次序统计量给出，
    下标由符号秩统计量的精确零分布确定。
    """
    d = diff[np.isfinite(diff)]
    n = int(d.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        return float(d[0]), float(d[0]), float(d[0])
    walsh = np.sort(0.5 * (d[:, None] + d[None, :])[np.triu_indices(n)])
    total = walsh.size
    hl = float(np.median(walsh))
    idx = _signed_rank_ci_index(n, alpha)
    lo = float(walsh[idx])
    hi = float(walsh[total - 1 - idx])
    return hl, lo, hi


def _signed_rank_ci_index(n: int, alpha: float = 0.05, max_total: int = 4000) -> int:
    """符号秩检验 CI 的次序统计量下标（0 基）。

    ``W+`` 在零假设下为 ``1..n`` 的等概率子集之和，其递推分布给出下尾临界值
    ``w``；CI 下限为第 ``w+1`` 个 Walsh 次序统计量（0 基下标 ``w``）。
    """
    if n < 2:
        return 0
    if n * (n + 1) // 2 > max_total:
        return _signed_rank_ci_index_approx(n, alpha)
    target = alpha / 2.0
    size = n * (n + 1) // 2 + 1
    pmf = np.zeros(size, dtype=float)
    pmf[0] = 1.0
    for rank in range(1, n + 1):
        shifted = np.zeros(size, dtype=float)
        shifted[rank:] = pmf[:-rank] if rank < size else 0.0
        pmf = pmf + shifted
    cdf = np.cumsum(pmf) / float(np.sum(pmf))
    below = np.nonzero(cdf <= target)[0]
    if below.size == 0:
        return 0
    return int(min(max(below[-1], 0), n * (n + 1) // 2))


def _signed_rank_ci_index_approx(n: int, alpha: float = 0.05) -> int:
    """大样本下符号秩 CI 下标的正态近似（带连续性校正）。"""
    mu = n * (n + 1) / 4.0
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    q = float(st.norm.ppf(alpha / 2.0))
    w = mu + q * sd
    return int(min(max(np.floor(w), 0), n * (n + 1) // 2))


def compare_two_groups(
    a: Any,
    b: Any,
    paired: bool = False,
    equal_var: bool | None = None,
    alpha: float = 0.05,
    auto: bool = True,
    normality: Mapping[str, Any] | None = None,
    variance: Mapping[str, Any] | None = None,
) -> dict:
    """两组比较，``auto=True`` 时按前提假设自动选择检验方法。

    自动决策规则（会写入 ``reason``）：

    1. ``paired=True``：对差值做正态性检验，正态 → 配对 t 检验，否则 → Wilcoxon 符号秩检验；
    2. 两组均正态 且 方差齐 → Student t 检验；
    3. 两组均正态 但方差不齐 → Welch t 检验；
    4. 至少一组不正态 → Mann-Whitney U 检验（即 Wilcoxon 秩和检验）。

    Args:
        a: 第一组数据。
        b: 第二组数据。
        paired: 是否为配对样本（此时 ``a`` 与 ``b`` 必须等长且一一对应）。
        equal_var: 方差齐性假设；``None`` 表示由数据自动判断，``True/False``
            表示用户强制指定（会体现在 ``reason`` 中）。
        alpha: 显著性水平，同时用于前提假设判定与置信区间。
        auto: ``False`` 时按 ``equal_var`` / 正态性给"保守"选择而不做自动推理
            （仍保证不崩溃）。
        normality: 预先算好的正态性结果（如 ``check_normality`` 的返回），
            给出时可复用，避免重复计算。需含 ``all_normal`` 或在 ``per_group``
            中提供 `a` / `b` 组的结果。
        variance: 预先算好的方差齐性结果（``check_variance`` 的返回）。

    Returns:
        字典，含 ``test``、``statistic``、``p``、``df``、``effect_size``、
        ``effect_name``、``method_auto``、``reason``、``normality``、``variance``、
        ``n1``/``n2``、``median1``/``median2``、``mean1``/``mean2``、
        ``ci_low``/``ci_high``、``ok`` 等键。
    """
    x_all = _as_float(a)
    y_all = _as_float(b)
    simple = {
        "n1": int(x_all.size),
        "n2": int(y_all.size),
        "mean1": float("nan"),
        "mean2": float("nan"),
        "median1": float("nan"),
        "median2": float("nan"),
        "df": None,
        "effect_size": float("nan"),
        "effect_name": "",
        "method_auto": bool(auto),
        "normality": dict(normality) if normality else None,
        "variance": dict(variance) if variance else None,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "ci_name": "",
        "paired": bool(paired),
    }

    x = x_all[np.isfinite(x_all)]
    y = y_all[np.isfinite(y_all)]
    dropped = [n for n, arr in (("a", x_all), ("b", y_all)) if _n_missing(arr) > 0]
    miss_note = f"（{'、'.join(dropped)} 组存在缺失/非有限值，已剔除）" if dropped else ""

    if x.size < _MIN_N or y.size < _MIN_N:
        return _degenerate(
            f"有效样本数不足（a: n = {x.size}，b: n = {y.size}，每组至少需要 {_MIN_N} 个）{miss_note}，"
            "无法进行两组比较。",
            **simple,
        )
    if paired and x_all.size != y_all.size:
        return _degenerate("配对检验要求两组样本数一致，当前长度不同。", **simple)
    if paired and x.size != y.size:
        return _degenerate("配对检验要求两组一一对应；剔除缺失值后长度不一致，请先成对剔除缺失。", **simple)

    simple["mean1"] = float(np.mean(x))
    simple["mean2"] = float(np.mean(y))
    simple["median1"] = float(np.median(x))
    simple["median2"] = float(np.median(y))

    zero_var = []
    if _all_same(x):
        zero_var.append("a")
    if _all_same(y):
        zero_var.append("b")

    if paired:
        diff = x - y
        norm = dict(normality) if normality else check_normality(diff, method="shapiro", alpha=alpha)
        normal_ok = bool(norm.get("all_normal"))
        var_info = dict(variance) if variance else {
            "ok": False,
            "p": float("nan"),
            "equal_variance": False,
            "method": "n/a",
            "note": "配对设计不涉及组间方差齐性检验。",
            "statistic": float("nan"),
        }
        if zero_var:
            return _degenerate(
                f"差分数据遇到零方差（{'、'.join(zero_var)} 组所有取值相同），配对检验无定义。",
                **simple,
            )
        chosen = "paired_t" if normal_ok else "wilcoxon"
        if normal_ok:
            reason = (
                f"配对差值经 {norm['method']} 检验符合正态分布"
                f"（p = {_fmt_p(norm['overall']['p'])} > {alpha}），采用配对样本 t 检验。"
            )
        else:
            reason = (
                f"配对差值经 {norm['method']} 检验不符合正态分布"
                f"（p = {_fmt_p(norm['overall']['p'])} ≤ {alpha}），采用 Wilcoxon 符号秩检验。"
            )
        return _two_group_parametric_or_not(
            x, y, chosen, reason, var_info, norm, simple, alpha, miss_note,
            diff_for_ci=diff, paired=True,
        )

    # ---- 独立样本 ----
    if normality is not None and "all_normal" in normality:
        normal_ok = bool(normality.get("all_normal"))
        norm_info = dict(normality)
    else:
        norm_info = check_normality(
            np.concatenate([x, y]),
            group=np.array(["a"] * x.size + ["b"] * y.size),
            method="shapiro",
            alpha=alpha,
        )
        normal_ok = bool(norm_info.get("all_normal"))
    norm_info = dict(norm_info)
    norm_info.setdefault("method", "shapiro")

    var_info = dict(variance) if variance else check_variance({"a": x, "b": y}, method="levene", alpha=alpha)
    equal_variance = var_info.get("equal_variance")
    if not var_info.get("ok"):
        equal_variance = False  # 方差检验不可用时保守选择 Welch
    if equal_var is not None:
        equal_variance = bool(equal_var)

    if normal_ok and equal_variance:
        chosen = "student"
    elif normal_ok and not equal_variance:
        chosen = "welch"
    else:
        chosen = "mannwhitney"

    p_a = norm_info.get("per_group", {}).get("a", {}).get("p", float("nan"))
    p_b = norm_info.get("per_group", {}).get("b", {}).get("p", float("nan"))
    norm_label = str(norm_info.get("per_group", {}).get("a", {}).get("test") or "Shapiro-Wilk")
    if chosen == "mannwhitney":
        if norm_info.get("all_normal") is False and np.isfinite([p_a, p_b]).all() and min(p_a, p_b) <= alpha:
            reason = (
                f"数据经 {norm_label} 检验不符合正态分布（a 组 p = {_fmt_p(p_a)}，b 组 p = {_fmt_p(p_b)}），"
                "因此采用 Mann-Whitney U 检验（Wilcoxon 秩和检验）。"
            )
        else:
            reason = (
                "数据不满足参数检验的正态性假设（存在样本数不足或零方差的分组），"
                "因此采用 Mann-Whitney U 检验（Wilcoxon 秩和检验）。"
            )
    elif chosen == "welch":
        reason = (
            f"两组数据均符合正态分布（p > {alpha}）但方差不齐"
            f"（{var_info.get('method_name', 'Levene')} 检验 p = {_fmt_p(var_info.get('p'))} ≤ {alpha}），"
            "采用 Welch t 检验（不假设方差齐性）。"
        )
    else:
        reason = (
            f"两组数据均符合正态分布（p > {alpha}）且方差齐性"
            f"（{var_info.get('method_name', 'Levene')} 检验 p = {_fmt_p(var_info.get('p'))}），"
            "采用 Student t 检验。"
        )
    if equal_var is not None:
        reason += f"（方差齐性由用户指定为 equal_var={equal_var}）"
    if not auto:
        reason += "（auto=False，按当前假设执行，未自动改写方法。）"
    if miss_note:
        reason += miss_note

    if zero_var and len(zero_var) == 2:
        deg = dict(simple)
        deg.pop("normality", None)
        deg.pop("variance", None)
        return _degenerate(
            f"两组数据均取单一常数值（{'、'.join(zero_var)} 组方差为 0）{miss_note}，"
            "组间差异检验无定义。",
            normality=norm_info,
            variance=var_info,
            **deg,
        )

    return _two_group_parametric_or_not(
        x, y, chosen, reason, var_info, norm_info, simple, alpha, miss_note, paired=False
    )


def _two_group_parametric_or_not(
    x: np.ndarray,
    y: np.ndarray,
    chosen: str,
    reason: str,
    var_info: Mapping[str, Any],
    norm_info: Mapping[str, Any],
    simple: dict,
    alpha: float,
    miss_note: str,
    diff_for_ci: np.ndarray | None = None,
    paired: bool = False,
) -> dict:
    """按 ``chosen`` 执行具体检验并汇总输出（两组比较的统一出口）。"""
    guard = _Guard()
    out = dict(simple)
    out["normality"] = norm_info
    out["variance"] = var_info
    out["method_auto"] = True
    out["paired"] = bool(paired)
    out["ok"] = True

    ci_low, ci_high, ci_name = float("nan"), float("nan"), ""
    if chosen == "student":
        res = guard.run(st.ttest_ind, x, y, equal_var=True)
        stat, p = float(res[0]), float(res[1])
        df = float(x.size + y.size - 2)
        out.update(
            test="Student t 检验",
            test_key="student",
            statistic=stat,
            p=p,
            df=df,
            effect_size=effect_size_cohens_d(x, y),
            effect_name="Cohen's d",
            hedges_g=effect_size_hedges_g(x, y),
            cliffs_delta=effect_size_cliffs_delta(x, y),
            rank_biserial=effect_size_rank_biserial(x, y),
        )
        ci_low, ci_high = _mean_diff_ci(x, y, equal_var=True, alpha=alpha)
        ci_name = "均值差 (a - b) 的 95% 置信区间"
    elif chosen == "welch":
        res = guard.run(st.ttest_ind, x, y, equal_var=False)
        stat, p = float(res[0]), float(res[1])
        a_, b_ = np.var(x, ddof=1) / x.size, np.var(y, ddof=1) / y.size
        denom = (a_ * a_) / (x.size - 1) + (b_ * b_) / (y.size - 1)
        df = float((a_ + b_) ** 2 / denom) if denom > 0 else float("nan")
        out.update(
            test="Welch t 检验",
            test_key="welch",
            statistic=stat,
            p=p,
            df=df,
            effect_size=effect_size_cohens_d(x, y),
            effect_name="Cohen's d",
            hedges_g=effect_size_hedges_g(x, y),
            cliffs_delta=effect_size_cliffs_delta(x, y),
            rank_biserial=effect_size_rank_biserial(x, y),
        )
        ci_low, ci_high = _mean_diff_ci(x, y, equal_var=False, alpha=alpha)
        ci_name = "均值差 (a - b) 的 95% 置信区间（Welch）"
    elif chosen == "paired_t":
        d = x - y
        res = guard.run(st.ttest_rel, x, y)
        stat, p = float(res[0]), float(res[1])
        df = float(d.size - 1)
        out.update(
            test="配对样本 t 检验",
            test_key="paired_t",
            statistic=stat,
            p=p,
            df=df,
            effect_size=effect_size_cohens_d(x, y, paired=True),
            effect_name="Cohen's d (d_z, 配对)",
            hedges_g=float("nan"),
            cliffs_delta=effect_size_cliffs_delta(x, y),
            rank_biserial=effect_size_rank_biserial(x, y),
        )
        se = float(np.std(d, ddof=1) / np.sqrt(d.size))
        tcrit = float(st.t.ppf(1.0 - alpha / 2.0, d.size - 1)) if d.size > 1 else float("nan")
        if np.isfinite(se) and np.isfinite(tcrit):
            ci_low, ci_high = float(np.mean(d) - tcrit * se), float(np.mean(d) + tcrit * se)
        ci_name = "配对差值均值的 95% 置信区间"
    elif chosen == "wilcoxon":
        d = x - y
        res = guard.run(st.wilcoxon, x, y, alternative="two-sided")
        stat, p = float(res[0]), float(res[1])
        r = _signed_rank_r(x, y)
        out.update(
            test="Wilcoxon 符号秩检验",
            test_key="wilcoxon",
            statistic=stat,
            p=p,
            df=None,
            effect_size=r,
            effect_name="秩双列相关 r",
            hedges_g=float("nan"),
            cliffs_delta=effect_size_cliffs_delta(x, y),
            rank_biserial=r,
        )
        hl, ci_low, ci_high = _paired_ci(d, alpha=alpha)
        out["hodges_lehmann"] = hl
        ci_name = "配对差值中位数的 95% 置信区间（Hodges-Lehmann）"
    else:  # mannwhitney
        res = guard.run(st.mannwhitneyu, x, y, alternative="two-sided")
        stat, p = float(res[0]), float(res[1])
        r = effect_size_rank_biserial(x, y)
        out.update(
            test="Mann-Whitney U 检验",
            test_key="mannwhitney",
            statistic=stat,
            p=p,
            df=None,
            effect_size=r,
            effect_name="秩双列相关 r",
            hedges_g=float("nan"),
            cliffs_delta=effect_size_cliffs_delta(x, y),
            rank_biserial=r,
        )
        hl, ci_low, ci_high = _hl_ci(x, y, alpha=alpha)
        out["hodges_lehmann"] = hl
        ci_name = "位置偏移 Hodges-Lehmann 的 95% 置信区间"

    out["ci_low"] = float(ci_low)
    out["ci_high"] = float(ci_high)
    out["ci_name"] = ci_name
    out["significant"] = bool(np.isfinite(out["p"]) and out["p"] < alpha)
    out["stars"] = significance_stars(out["p"])
    out["alpha"] = float(alpha)

    eff_txt = _cohen_d_note(out["effect_size"]) if out["effect_name"].startswith("Cohen") else (
        f"效应量 {out['effect_name']} = {out['effect_size']:.2f}" if np.isfinite(out["effect_size"]) else "效应量无法计算"
    )
    if guard.messages:
        out["note"] = guard.note
        out["reason"] = reason + " " + guard.note
    else:
        out["note"] = ""
        if np.isfinite(out["p"]):
            concl = "差异具有统计学意义" if out["significant"] else "差异未达统计学显著"
            out["reason"] = f"{reason} 结果：统计量 = {out['statistic']:.3f}，p = {_fmt_p(out['p'])}（{concl}，{eff_txt}）。"
        else:
            out["reason"] = reason
    return out


def _signed_rank_r(x: np.ndarray, y: np.ndarray) -> float:
    """Wilcoxon 符号秩检验的秩双列相关：``r = (W+ - W-) / (W+ + W-)``。"""
    d = x - y
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if d.size == 0:
        return float("nan")
    ranks = st.rankdata(np.abs(d))
    w_plus = float(np.sum(ranks[d > 0]))
    w_minus = float(np.sum(ranks[d < 0]))
    total = w_plus + w_minus
    if total <= 0:
        return float("nan")
    return float((w_plus - w_minus) / total)


def compare_paired(a: Any, b: Any, alpha: float = 0.05) -> dict:
    """配对样本比较（自动选择 Wilcoxon 符号秩检验 / 配对 t 检验）。

    Args:
        a: 配对样本的第一组。
        b: 配对样本的第二组（与 ``a`` 一一对应，等长）。
        alpha: 显著性水平。

    Returns:
        与 :func:`compare_two_groups` 相同结构的结果字典，另加
        ``mean_diff``、``median_diff``、``diff_normality``（差值正态性）。
    """
    out = compare_two_groups(a, b, paired=True, alpha=alpha)
    x = _finite(a)
    y = _finite(b)
    if out.get("ok") and x.size == y.size and x.size > 0:
        d = x - y
        out["mean_diff"] = float(np.mean(d))
        out["median_diff"] = float(np.median(d))
        out["diff_normality"] = out.get("normality")
    else:
        out.setdefault("mean_diff", float("nan"))
        out.setdefault("median_diff", float("nan"))
        out.setdefault("diff_normality", out.get("normality"))
    return out


# ==========================================================================
# Tukey HSD（含 studentized range 分布的自实现回退）
# ==========================================================================
#: studentized range CDF 的网格缓存：``{(k, df): (q_grid, cdf_grid, q_max)}``。
#: 网格只需构建一次，之后所有 p 值/分位点都退化成插值，避免反复做数值积分。
_SR_CACHE: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """标准正态密度（数组版本，避免逐点调用 scipy 的开销）。"""
    return np.exp(-0.5 * np.square(x)) / np.sqrt(2.0 * np.pi)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """标准正态分布函数（数组版本，基于 ``scipy.special.ndtr``）。"""
    return _ndtr(x)


#: 标准正态极差 CDF 的插值网格缓存：``{k: (z_grid, Fr_grid)}``。
_NR_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}
#: 网格上界：z ≥ 25 时 ``F_R(z) > 1 - 1e-12``，直接取 1 的误差远小于浮点分辨率。
_NR_ZMAX: float = 25.0
_NR_STEP: float = 0.001


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """标准正态密度（数组版本，避免逐点调用 scipy 的开销）。"""
    return np.exp(-0.5 * np.square(x)) / np.sqrt(2.0 * np.pi)


def _normal_range_cdf_direct(z: float, k: int, *, step: float = 0.002, bound: float = 12.0) -> float:
    """标准正态下 k 个独立观测的极差分布 ``P(max - min ≤ z)``（单点直接求积）。

    由对称形式（等价于对最大值密度 ``k·φ(x)·[Φ(x+z) − Φ(x)]^{k−1}`` 积分）得到

    .. math::
        P(R \\le z) = k\\int_{-\\infty}^{\\infty} \\phi(x)
        \\left[\\Phi(x+z) - \\Phi(x)\\right]^{k-1} dx

    数值上用 ``x ∈ [-bound, bound + z]`` 上的高密度梯形法（步长 ``step``）计算：
    kernel 仅在 ``x ≳ -bound`` 且 ``x ≲ bound + z`` 处非零，区间外正态尾部贡献 < 1e-30。
    k = 2 时该式严格等于 ``2Φ(z/√2) − 1``，模块自检以此为对照；生产路径使用
    :func:`_normal_range_cdf`（同一公式的 banded 前缀和加速版本）。
    """
    if not np.isfinite(z) or z <= 0:
        return 0.0
    if k < 2:
        return float("nan")
    x = np.arange(-bound, bound + z + step, step)
    kernel = _ndtr(x + z) - _ndtr(x)
    np.clip(kernel, 0.0, None, out=kernel)
    integrand = float(k) * _norm_pdf(x) * np.power(kernel, k - 1.0)
    return float(min(max(np.trapezoid(integrand, x), 0.0), 1.0))


def _normal_range_cdf(z: Any, k: int) -> Any:
    """标准正态极差 CDF ``P(max - min ≤ z)``（缓存查找表 + 线性插值）。

    与 :func:`_normal_range_cdf_direct` 完全同一公式，只是预先在
    ``z ∈ [0, _NR_ZMAX]`` 上以步长 :data:`_NR_STEP` 求值并缓存，之后任意 ``z``
    都退化为一次 :func:`numpy.interp`。该 CDF 在区间内光滑（二阶导有界），
    步长 0.002 时的线性插值误差 < 1e-4（模块自检与直接求积逐点对照）。

    Args:
        z: 极差阈值（标量或任意形状数组）。
        k: 观测个数（≥ 2）。

    Returns:
        与 ``z`` 同形的 CDF；标量输入返回 ``float``。``z ≥ _NR_ZMAX`` 时为 1.0。
    """
    scalar = np.isscalar(z) or (isinstance(z, np.ndarray) and z.ndim == 0)
    z_arr = np.asarray(z, dtype=float)
    flat = np.atleast_1d(z_arr).ravel().astype(float)
    if k < 2:
        out = np.full(flat.shape, np.nan)
        return float(out[0]) if scalar else out.reshape(z_arr.shape)
    grid = _NR_CACHE.get(k)
    if grid is None:
        z_grid = np.arange(0.0, _NR_ZMAX + _NR_STEP, _NR_STEP)
        fr = np.array([_normal_range_cdf_direct(float(v), k) for v in z_grid])
        fr = np.clip(np.maximum.accumulate(fr), 0.0, 1.0)
        grid = (z_grid, fr)
        _NR_CACHE[k] = grid
    z_grid, fr = grid
    out = np.interp(flat, z_grid, fr)
    if scalar:
        return float(out[0])
    return out.reshape(z_arr.shape)


def _studentized_range_grid(k: int, df: float) -> tuple[np.ndarray, np.ndarray]:
    """构建 ``(q, CDF)`` 网格并缓存，供 :func:`_studentized_range_cdf` 插值使用。

    推导：设各组均值独立同分布 ``N(μ, σ²/n)``、``s² ~ σ²χ²_df/df`` 且二者独立，
    则 ``Q = R/(s/σ)``，其中 ``R`` 为标准正态 k 个观测的极差，于是

    .. math::
        P(Q \\le q) = \\int_0^{\\infty} F_R(q s)\\, g_{df}(s)\\, ds

    ``g_df`` 为 ``χ_df/√df`` 的密度：由 ``a ~ χ_df`` 经 ``s = a/√df`` 变量替换得
    ``g(s) = √df · [2^{1-df/2}/Γ(df/2)] · (s√df)^{df-1} · exp(-df s²/2)``。
    """
    # q 方向也做插值查询，步长 0.05 时线性插值误差 < 1e-5（p 值精度足够）。
    q_grid = np.linspace(1e-6, 40.0, 801)
    # χ_df/√df 的密度在 s ≈ 1 处极窄（σ ≈ 1/√(2df)），求积窗口按 χ² 的分位数取：
    # 用生存函数反函数 chi2.isf（比 ppf(1-1e-14) 数值稳定，后者在 df 稍大时
    # 会因 1-1e-14 被舍入成 1 而返回 inf）。
    a_max = float(np.sqrt(st.chi2.isf(1e-14, df)))
    a_grid = np.linspace(0.0, a_max, 4001)
    s_grid = np.maximum(a_grid / np.sqrt(df), 1e-6)  # 下限用于避开 0 附近的比值噪声
    # χ_df 的密度（a 为自变量）：f(a) = 2^{1-df/2}/Γ(df/2) · a^{df-1} · e^{-a²/2}
    log_dens = (
        (1.0 - df / 2.0) * np.log(2.0)
        - np.log(_gamma_fn(df / 2.0))
        + (df - 1.0) * np.log(np.maximum(a_grid, 1e-300))
        - a_grid ** 2 / 2.0
    )
    with np.errstate(under="ignore"):
        dens = np.exp(log_dens)
    fr = _normal_range_cdf(q_grid[:, None] * s_grid[None, :], k)
    fr = np.asarray(fr, dtype=float).reshape(q_grid.size, s_grid.size)
    cdf_grid = np.clip(np.trapezoid(fr * dens[None, :], a_grid, axis=1), 0.0, 1.0)
    cdf_grid = np.maximum.accumulate(cdf_grid)
    return q_grid, cdf_grid
    fr = _normal_range_cdf(q_grid[:, None] * s_grid[None, :], k)
    fr = np.asarray(fr, dtype=float).reshape(q_grid.size, s_grid.size)
    cdf_grid = np.clip(np.trapezoid(fr * dens[None, :], s_grid, axis=1), 0.0, 1.0)
    cdf_grid = np.maximum.accumulate(cdf_grid)
    return q_grid, cdf_grid
    fr = _normal_range_cdf(q_grid[:, None] * s_grid[None, :], k)
    fr = np.asarray(fr, dtype=float).reshape(q_grid.size, s_grid.size)
    cdf_grid = np.clip((fr * dens[None, :]).sum(axis=1), 0.0, 1.0)
    cdf_grid = np.maximum.accumulate(cdf_grid)
    return q_grid, cdf_grid


def _studentized_range_cdf(q: float, k: int, df: float) -> float:
    """studentized range 分布的 CDF ``P(Q ≤ q)``（Tukey HSD 的 p 值来源）。

    仅在本机 scipy 缺少 :func:`scipy.stats.tukey_hsd` 时作为回退路径使用。
    公式与数值实现见 :func:`_normal_range_cdf` / :func:`_studentized_range_grid`，
    已与 ``scipy.stats.studentized_range.cdf`` 逐点核对（误差 < 1e-6，见自检）。

    Args:
        q: studentized range 统计量
            ``q = |mean_i - mean_j| / sqrt(MSE/2 · (1/n_i + 1/n_j))``。
        k: 组数（≥ 2）。
        df: 误差自由度（> 0）。

    Returns:
        CDF 值；输入非法时返回 ``nan``。
    """
    if k < 2 or not np.isfinite(df) or df <= 0 or not np.isfinite(q):
        return float("nan")
    if q <= 0:
        return 0.0
    k = int(k)
    df = float(df)
    key = (k, df)
    grid = _SR_CACHE.get(key)
    if grid is None:
        grid = _studentized_range_grid(k, df)
        _SR_CACHE[key] = grid
    q_grid, cdf_grid = grid
    if q >= q_grid[-1]:
        return 1.0
    return float(np.interp(q, q_grid, cdf_grid))


def _studentized_range_sf(q: float, k: int, df: float) -> float:
    """studentized range 分布的生存函数 ``P(Q > q)``，即 Tukey 两两比较的 p 值。"""
    cdf = _studentized_range_cdf(q, k, df)
    if not np.isfinite(cdf):
        return float("nan")
    return float(min(max(1.0 - cdf, 0.0), 1.0))


def _studentized_range_ppf(prob: float, k: int, df: float, lo: float = 0.0, hi: float = 40.0) -> float:
    """studentized range 分布分位点（二分法；用于回退路径的临界值）。"""
    if not np.isfinite(prob) or prob <= 0:
        return 0.0
    if prob >= 1:
        return float(hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = _studentized_range_cdf(mid, k, df)
        if not np.isfinite(cdf):
            return float("nan")
        if cdf < prob:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def tukey_hsd(groups_dict: Mapping[str, Any]) -> pd.DataFrame:
    """Tukey HSD 事后检验（ANOVA 之后的两两比较）。

    优先使用 :func:`scipy.stats.tukey_hsd`（scipy ≥ 1.11），其 p 值已由
    studentized range 分布给出；若本机 scipy 缺少该函数，则用
    studentized range 分布（见 :func:`_studentized_range_cdf`）自行计算
    ``p = P(Q > q)``，其中 ``q = |mean_i - mean_j| / sqrt(MSE / n_h)``
    采用 Tukey-Kramer 不等样本量形式。

    Args:
        groups_dict: ``{组名: 一维数据}``。

    Returns:
        DataFrame，列为 ``group1 / group2 / meandiff / statistic / p / p_adj /
        ci_low / ci_high / significant / stars``；输入组数 < 2 时返回空表。
    """
    cols = ["group1", "group2", "meandiff", "statistic", "p", "p_adj", "ci_low", "ci_high", "significant", "stars"]
    data = _clean_groups(groups_dict) if isinstance(groups_dict, Mapping) else {}
    names = list(data.keys())
    samples = {k: v[np.isfinite(v)] for k, v in data.items()}
    if len(names) < 2 or any(s.size < 2 for s in samples.values()):
        return pd.DataFrame(columns=cols)

    arrs = [samples[k] for k in names]
    if all(_all_same(s) for s in arrs):
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    k = len(arrs)
    used_scipy = hasattr(st, "tukey_hsd")
    if used_scipy:
        guard = _Guard()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                res = st.tukey_hsd(*arrs)
        except Exception as exc:  # noqa: BLE001
            guard.add(f"scipy.stats.tukey_hsd 调用失败（{type(exc).__name__}: {exc}）。")
            used_scipy = False
        else:
            ci = res.confidence_interval(confidence_level=0.95)
            ci_low = np.asarray(ci.low)
            ci_high = np.asarray(ci.high)
            for i in range(k):
                for j in range(i + 1, k):
                    rows.append(
                        {
                            "group1": names[i],
                            "group2": names[j],
                            "meandiff": float(res.statistic[i, j]),
                            "statistic": float(res.statistic[i, j]),
                            "p": float(res.pvalue[i, j]),
                            "ci_low": float(ci_low[i, j]),
                            "ci_high": float(ci_high[i, j]),
                        }
                    )

    if not used_scipy:
        n_i = np.array([s.size for s in arrs], dtype=float)
        means = np.array([float(np.mean(s)) for s in arrs])
        variances = np.array([float(np.var(s, ddof=1)) for s in arrs])
        df_within = float(n_i.sum() - k)
        sse = float(np.sum((n_i - 1.0) * variances))
        if df_within <= 0 or sse <= 0:
            return pd.DataFrame(columns=cols)
        mse = sse / df_within
        qcrit = _studentized_range_ppf(0.95, k, df_within)
        for i in range(k):
            for j in range(i + 1, k):
                diff = float(means[i] - means[j])
                se = float(np.sqrt(mse / 2.0 * (1.0 / n_i[i] + 1.0 / n_i[j])))
                q_stat = abs(diff) / se if se > 0 else float("inf")
                p = _studentized_range_sf(q_stat, k, df_within)
                half = qcrit * se if np.isfinite(qcrit) else float("nan")
                rows.append(
                    {
                        "group1": names[i],
                        "group2": names[j],
                        "meandiff": diff,
                        "statistic": q_stat,
                        "p": p,
                        "ci_low": diff - half,
                        "ci_high": diff + half,
                    }
                )

    out = pd.DataFrame(rows, columns=["group1", "group2", "meandiff", "statistic", "p", "ci_low", "ci_high"])
    # scipy 的 Tukey HSD 已按 studentized range 分布做过全部两两比较的多重性校正，
    # 因此 p_adj 与 p 相同；此处仍保留独立的 p_adj 列以统一接口。
    out["p_adj"] = _p_adjust(out["p"].to_numpy(), method="bh")
    out["significant"] = out["p_adj"] < 0.05
    out["stars"] = [significance_stars(v) for v in out["p"]]
    return out[cols].reset_index(drop=True)


# ==========================================================================
# Dunn 事后检验（Kruskal-Wallis 之后）
# ==========================================================================
def dunn_posthoc(groups_dict: Mapping[str, Any], p_adjust: str = "bh") -> pd.DataFrame:
    """Dunn 事后检验（Kruskal-Wallis 之后的两两比较）。

    公式来源：Dunn, O. J. (1964). *Multiple comparisons using rank sums*.
    Technometrics 6(3): 241-252；并列校正沿用 scipy ``kruskal`` 的
    ``ties = 1 - Σ(t³ - t)/(N³ - N)``（Hollander & Wolfe, 3rd ed., §6.3）。

    .. math::
        z_{ij} = \\frac{\\bar R_i - \\bar R_j}{\\sqrt{\\frac{N(N+1)}{12} - ties}
                  \\sqrt{\\frac{1}{n_i} + \\frac{1}{n_j}}},\\quad
        \\mathrm{Var}(\\bar R_i - \\bar R_j) = \\frac{N(N+1)/12 - ties}{n_i n_j}
        \\cdot \\frac{n_i + n_j}{N}

    即分母等价于 ``sqrt(tie_var * (1/n_i + 1/n_j))``，``p`` 取标准正态双尾。

    Args:
        groups_dict: ``{组名: 一维数据}``。
        p_adjust: 多重校正方法，``"bh"``（默认，走
            :func:`amplicondesk.core.stats.bh_fdr`）或 ``"bonferroni"``。

    Returns:
        DataFrame，列为 ``group1 / group2 / z / p / p_adj / significant / stars``；
        可用组不足 2 时返回空表。
    """
    cols = ["group1", "group2", "z", "p", "p_adj", "significant", "stars"]
    data = _clean_groups(groups_dict) if isinstance(groups_dict, Mapping) else {}
    names = list(data.keys())
    samples = {k: v[np.isfinite(v)] for k, v in data.items()}
    samples = {k: v for k, v in samples.items() if v.size > 0}
    if len(samples) < 2:
        return pd.DataFrame(columns=cols)

    all_vals = np.concatenate([samples[k] for k in samples])
    n_total = int(all_vals.size)
    if n_total <= len(samples):
        return pd.DataFrame(columns=cols)

    pooled = np.concatenate([samples[k] for k in samples])
    ranks = st.rankdata(pooled)
    idx = 0
    mean_ranks: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for k in samples:
        m = samples[k].size
        mean_ranks[k] = float(np.mean(ranks[idx: idx + m]))
        sizes[k] = int(m)
        idx += m

    _, counts = np.unique(pooled, return_counts=True)
    ties = float(np.sum(counts ** 3 - counts))
    tie_var = n_total * (n_total + 1) / 12.0 - ties / (12.0 * (n_total - 1))
    if tie_var <= 0:
        return pd.DataFrame(columns=cols)

    keys = list(samples.keys())
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ni, nj = sizes[keys[i]], sizes[keys[j]]
            z = (mean_ranks[keys[i]] - mean_ranks[keys[j]]) / np.sqrt(
                tie_var * (1.0 / ni + 1.0 / nj)
            )
            p = float(2.0 * st.norm.sf(abs(z)))
            rows.append({"group1": keys[i], "group2": keys[j], "z": float(z), "p": p})

    out = pd.DataFrame(rows, columns=["group1", "group2", "z", "p"])
    out["p_adj"] = _p_adjust(out["p"].to_numpy(), method=p_adjust)
    out["significant"] = out["p_adj"] < 0.05
    out["stars"] = [significance_stars(v) for v in out["p"]]
    return out[cols].reset_index(drop=True)


# ==========================================================================
# 多组比较
# ==========================================================================
def _epsilon_squared_kw(h: float, n: int, k: int) -> float:
    """Kruskal-Wallis 的效应量 ε²（epsilon-squared）。"""
    if not np.isfinite(h) or n <= k or k < 2:
        return float("nan")
    return float(max(0.0, (h - k + 1.0) / (n - k)))


def compare_multiple_groups(
    groups_dict: Mapping[str, Any],
    alpha: float = 0.05,
    auto: bool = True,
    posthoc: bool = True,
) -> dict:
    """多组比较（单因素 ANOVA / Kruskal-Wallis），可带事后检验。

    ``auto=True`` 时：

    * 全部组的 Shapiro-Wilk 均不拒绝正态 **且** Levene 检验方差齐 → 单因素 ANOVA，
      事后用 Tukey HSD（scipy ``tukey_hsd``，缺失时自实现）；
    * 否则 → Kruskal-Wallis，事后用 Dunn 检验（BH/Bonferroni 校正）。

    Args:
        groups_dict: ``{组名: 一维 array}``。
        alpha: 显著性水平。
        auto: 是否自动选择方法；``False`` 时只要全部组正态就用 ANOVA，否则 KW。
        posthoc: 是否同时做事后多重比较（组数 < 3 时为 ``None``）。

    Returns:
        字典，含 ``test``、``statistic``、``p``、``df_between``、``df_within``、
        ``effect_size``、``effect_name``、``method_auto``、``reason``、
        ``normality``（含 ``per_group`` / ``overall``）、``variance``、
        ``groups_summary``（DataFrame）、``posthoc``（DataFrame 或 ``None``）、``ok``。
    """
    data = _clean_groups(groups_dict) if isinstance(groups_dict, Mapping) else {}
    samples = {k: v[np.isfinite(v)] for k, v in data.items()}
    samples = {k: v for k, v in samples.items() if v.size > 0}
    names = list(samples.keys())
    missing_note = ""
    dropped = [k for k, v in data.items() if _n_missing(v) > 0]
    if dropped:
        missing_note = f"（{'、'.join(map(str, dropped))} 组存在缺失/非有限值，已剔除）"

    base = {
        "df_between": None,
        "df_within": None,
        "effect_size": float("nan"),
        "effect_name": "",
        "method_auto": bool(auto),
        "normality": None,
        "variance": None,
        "groups_summary": describe_groups(samples) if samples else None,
        "posthoc": None,
        "groups_summary_note": "每组描述统计见 groups_summary。",
        "alpha": float(alpha),
    }

    if len(samples) < 2:
        return _degenerate(
            f"有效组数不足（当前 {len(samples)} 组，至少需要 2 组）{missing_note}，无法进行多组比较。",
            **base,
        )
    short = [k for k, v in samples.items() if v.size < _MIN_N]
    if short:
        return _degenerate(
            f"以下组有效样本数不足（n < {_MIN_N}）：{'、'.join(map(str, short))}{missing_note}，"
            "无法进行多组比较。",
            **base,
        )
    const_groups = [k for k, v in samples.items() if _all_same(v)]
    if const_groups and len(const_groups) == len(samples):
        return _degenerate(
            f"所有组取值完全相同（零方差）：{'、'.join(map(str, const_groups))}"
            f"{missing_note}，组间差异检验无定义。",
            **base,
        )

    all_vals = np.concatenate([samples[k] for k in names])
    labels = np.concatenate([np.full(samples[k].size, str(k), dtype=object) for k in names])
    norm_info = check_normality(all_vals, group=labels, method="shapiro", alpha=alpha)
    var_info = check_variance(samples, method="levene", alpha=alpha)

    all_normal = bool(norm_info.get("all_normal"))
    equal_var = bool(var_info.get("ok") and var_info.get("equal_variance"))
    use_anova = (all_normal and equal_var) if auto else all_normal
    n_total = int(all_vals.size)
    k = len(samples)

    guard = _Guard()
    posthoc_df: pd.DataFrame | None = None
    result: dict[str, Any]
    if use_anova:
        res = guard.run(st.f_oneway, *[samples[n] for n in names])
        f_stat, p = float(res[0]), float(res[1])
        df_between = float(k - 1)
        df_within = float(n_total - k)
        ss_between = float(sum(samples[n].size * (np.mean(samples[n]) - np.mean(all_vals)) ** 2 for n in names))
        ss_total = float(np.sum((all_vals - np.mean(all_vals)) ** 2))
        eta2 = float(ss_between / ss_total) if ss_total > 0 else float("nan")
        if all_normal and equal_var:
            reason = (
                f"各组数据均符合正态分布（Shapiro-Wilk p > {alpha}）且方差齐性"
                f"（{var_info.get('method_name', 'Levene')} p = {_fmt_p(var_info.get('p'))} > {alpha}），"
                "采用单因素方差分析（ANOVA）。"
            )
        elif all_normal:
            reason = (
                f"各组数据均符合正态分布（Shapiro-Wilk p > {alpha}），但方差不齐"
                f"（p = {_fmt_p(var_info.get('p'))}），按 auto=False 的设定仍采用单因素方差分析（ANOVA）；"
                "如需更稳健的结果可改用 Welch ANOVA 或 Kruskal-Wallis。"
            )
        else:
            reason = (
                "各组数据符合正态分布，采用单因素方差分析（ANOVA）。"
            )
        if guard.messages:
            reason = reason + " " + guard.note
        result = {
            "test": "单因素方差分析（ANOVA）",
            "test_key": "anova",
            "statistic": f_stat,
            "p": p,
            "df_between": df_between,
            "df_within": df_within,
            "effect_size": eta2,
            "effect_name": "η² (eta squared)",
            "method_auto": bool(auto),
            "reason": reason,
            "normality": norm_info,
            "variance": var_info,
            "n_groups": k,
            "n_total": n_total,
            "ok": bool(np.isfinite(p)),
            "note": guard.note,
        }
        if posthoc and k >= 3:
            posthoc_df = tukey_hsd(samples)
    else:
        res = guard.run(st.kruskal, *[samples[n] for n in names])
        h_stat, p = float(res[0]), float(res[1])
        if not all_normal and not equal_var:
            reason = (
                f"数据不满足参数检验前提（Shapiro-Wilk 提示非正态，或 Levene 提示方差不齐"
                f"（p = {_fmt_p(var_info.get('p'))}）），采用 Kruskal-Wallis 秩和检验。"
            )
        elif not all_normal:
            bad = [
                key for key, val in (norm_info.get("per_group") or {}).items()
                if not val.get("normal")
            ]
            reason = (
                f"以下组不符合正态分布：{'、'.join(map(str, bad))}，因此采用 Kruskal-Wallis 秩和检验。"
            )
        else:
            reason = (
                f"各组数据均正态，但方差不齐（Levene p = {_fmt_p(var_info.get('p'))} ≤ {alpha}），"
                "采用 Kruskal-Wallis 秩和检验（非参数替代方案）。"
            )
        if guard.messages:
            reason = reason + " " + guard.note
        result = {
            "test": "Kruskal-Wallis 秩和检验",
            "test_key": "kruskal",
            "statistic": h_stat,
            "p": p,
            "df_between": float(k - 1),
            "df_within": None,
            "effect_size": _epsilon_squared_kw(h_stat, n_total, k),
            "effect_name": "ε² (epsilon squared)",
            "method_auto": bool(auto),
            "reason": reason,
            "normality": norm_info,
            "variance": var_info,
            "n_groups": k,
            "n_total": n_total,
            "ok": bool(np.isfinite(p)),
            "note": guard.note,
        }
        if posthoc and k >= 3:
            posthoc_df = dunn_posthoc(samples, p_adjust="bh")

    result["groups_summary"] = describe_groups(samples)
    result["posthoc"] = posthoc_df
    result["stars"] = significance_stars(result.get("p", float("nan")))
    if missing_note:
        result["reason"] = result["reason"] + missing_note
    return result


# ==========================================================================
# 分类变量关联检验
# ==========================================================================
def compare_categorical(
    table: Any,
    row_var: str | None = None,
    col_var: str | None = None,
    correction: bool = True,
) -> dict:
    """分类变量关联检验（列联表）：自动选择卡方检验或 Fisher 精确检验。

    选择规则：

    * 期望频数 < 5 的格子超过 20%，或总样本量 < 40 → Fisher 精确检验；
      （2×2 用 ``scipy.stats.fisher_exact``；行列数更大时用
      ``scipy.stats.fisher_exact`` 的蒙特卡洛版本，10 万次抽样。）
    * 否则 → 卡方检验；2×2 表默认启用 Yates 连续性校正（``correction=True``），
      更大表用 Pearson 卡方。

    Args:
        table: 两种用法之一：① ``pd.DataFrame`` 原始观测数据（分类列），
            此时需给出 ``row_var`` / ``col_var``；② 已汇总的列联表
            （``pd.crosstab`` 结果或同形 DataFrame）。
        row_var: 行变量列名（用法 ① 必填）。
        col_var: 列变量列名（用法 ① 必填）。
        correction: 2×2 卡方是否做 Yates 连续性校正，默认 ``True``。

    Returns:
        字典，含 ``test``、``statistic``、``p``、``dof``、``expected``（DataFrame）、
        ``table``（DataFrame）、``effect_size``（Cramér's V）、``reason``、``ok`` 等。
    """
    if isinstance(table, pd.DataFrame) and row_var and col_var:
        if row_var not in table.columns or col_var not in table.columns:
            return _degenerate(f"数据中找不到列 {row_var} 或 {col_var}，无法构建列联表。")
        sub = table[[row_var, col_var]].dropna()
        if sub.empty:
            return _degenerate("剔除缺失值后没有可用数据，无法做分类变量关联检验。")
        crosstab = pd.crosstab(sub[row_var].astype(str), sub[col_var].astype(str))
    elif isinstance(table, pd.DataFrame):
        crosstab = table.copy()
    else:
        try:
            crosstab = pd.DataFrame(table)
        except Exception as exc:  # noqa: BLE001
            return _degenerate(f"无法把输入解析为列联表：{exc}")

    if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
        return _degenerate("列联表至少需要 2 行 2 列，当前结构无法做关联检验。")
    try:
        obs = crosstab.to_numpy(dtype=float)
    except Exception:  # noqa: BLE001
        return _degenerate("列联表包含非数值内容，请确认输入是频数表。")
    if not np.all(np.isfinite(obs)):
        return _degenerate("列联表包含缺失或非有限值，无法做关联检验。")
    if obs.sum() <= 0:
        return _degenerate("列联表总频数为 0，无法做关联检验。")
    if np.any(obs < 0):
        return _degenerate("列联表出现负频数，无法做关联检验。")

    # 去掉全零行列，避免卡方检验报错
    r_keep = obs.sum(axis=1) > 0
    c_keep = obs.sum(axis=0) > 0
    if not r_keep.all() or not c_keep.all():
        crosstab = crosstab.loc[crosstab.index[r_keep], crosstab.columns[c_keep]]
        obs = crosstab.to_numpy(dtype=float)
    if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
        return _degenerate("剔除全零行列后列联表不足 2×2，无法做关联检验。")

    n_total = float(obs.sum())
    row_tot = obs.sum(axis=1, keepdims=True)
    col_tot = obs.sum(axis=0, keepdims=True)
    expected = row_tot @ col_tot / n_total
    expected_df = pd.DataFrame(expected, index=crosstab.index, columns=crosstab.columns)
    small_frac = float(np.count_nonzero(expected < 5)) / expected.size
    use_fisher = bool(small_frac > 0.20 or n_total < 40)

    guard = _Guard()
    stat = float("nan")
    dof: float | None = None
    if use_fisher:
        if obs.shape == (2, 2):
            res = guard.run(st.fisher_exact, obs)
            odds, p = float(res[0]), float(res[1])
            stat = odds
            test_name = "Fisher 精确检验（2×2）"
            stat_note = "统计量为优势比（odds ratio）"
        else:
            def _fisher_mc() -> tuple[float, float]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = st.fisher_exact(
                        obs,
                        method="monte-carlo",
                        n_resamples=100_000,
                        rng=np.random.default_rng(20240501),
                    )
                return float(res.statistic), float(res.pvalue)
            stat, p = guard.run(_fisher_mc)
            test_name = "Fisher 精确检验（蒙特卡洛，100000 次重抽样）"
            stat_note = "统计量为优势比（odds ratio）"
        p_val = p
        if np.isfinite(small_frac) and small_frac > 0.20:
            reason = (
                f"列联表中期望频数小于 5 的格子占 {small_frac:.0%}（> 20%），"
                f"卡方近似不可靠，因此采用{test_name}。"
            )
        else:
            reason = f"总样本量偏小（N = {int(n_total)} < 40），因此采用{test_name}。"
    else:
        res = guard.run(st.chi2_contingency, obs, correction=bool(correction) and obs.shape == (2, 2))
        stat, p, dof = float(res[0]), float(res[1]), float(res[2])
        p_val = p
        yates = bool(correction) and obs.shape == (2, 2)
        test_name = "卡方检验" + ("（Yates 连续性校正）" if yates else "")
        stat_note = "统计量为卡方值"
        reason = (
            f"列联表期望频数小于 5 的格子占 {small_frac:.0%}（≤ 20%）且总样本量 N = {int(n_total)} ≥ 40，"
            f"卡方近似可靠，因此采用{test_name}。"
        )

    if not np.isfinite(p_val):
        reason = reason + " " + (guard.note or "关联检验未能完成。")
    v = cramers_v(obs, stat if not use_fisher or obs.shape != (2, 2) else float("nan"), p_val)
    if np.isfinite(v):
        reason = reason + f" 效应量 Cramér's V = {v:.3f}。"

    return {
        "test": test_name,
        "test_key": "fisher" if use_fisher else "chi2",
        "statistic": float(stat),
        "statistic_note": stat_note,
        "p": float(p_val),
        "dof": dof,
        "expected": expected_df,
        "table": pd.DataFrame(obs.astype(int), index=crosstab.index, columns=crosstab.columns),
        "effect_size": v,
        "effect_name": "Cramér's V",
        "small_expected_frac": small_frac,
        "n_total": int(n_total),
        "correction": bool(correction),
        "reason": reason,
        "note": guard.note,
        "ok": bool(np.isfinite(p_val)),
        "significant": bool(np.isfinite(p_val) and p_val < 0.05),
        "stars": significance_stars(p_val),
    }


def cramers_v(observed: np.ndarray, chi2: float = float("nan"), p: float = float("nan")) -> float:
    """Cramér's V 效应量（分类变量关联强度）。

    ``V = sqrt(χ² / (N · min(r-1, c-1)))``；2×2 表时等价于 φ 系数。
    若未提供有效 ``chi2``（例如 Fisher 路径），则自行按 Pearson 卡方重算。

    Args:
        observed: 列联表频数矩阵。
        chi2: 已知的卡方统计量（可选）。
        p: 保留参数以便调用方透传（不参与计算）。

    Returns:
        Cramér's V，范围 ``[0, 1]``；无法计算时返回 ``nan``。
    """
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 2 or obs.size == 0:
        return float("nan")
    n = float(obs.sum())
    if n <= 0:
        return float("nan")
    k = min(obs.shape[0] - 1, obs.shape[1] - 1)
    if k <= 0:
        return float("nan")
    if not np.isfinite(chi2):
        row = obs.sum(axis=1, keepdims=True)
        col = obs.sum(axis=0, keepdims=True)
        exp = row @ col / n
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = float(np.nansum((obs - exp) ** 2 / np.where(exp > 0, exp, np.nan)))
        if not np.isfinite(chi2):
            return float("nan")
    return float(np.sqrt(max(chi2, 0.0) / (n * k)))


# ==========================================================================
# 统一入口
# ==========================================================================
def auto_test(
    values: Any,
    group: Any,
    paired: bool = False,
    alpha: float = 0.05,
) -> dict:
    """统一入口：按组数自动走两组或（多组）多组流程，返回结构一致的结果。

    Args:
        values: 一维数值序列。
        group: 等长的分组标签。
        paired: 是否配对（仅两组有意义）。
        alpha: 显著性水平。

    Returns:
        字典，至少含 ``n_groups``、``test``、``p``、``statistic``、``reason``、
        ``summary``（DataFrame）、``posthoc``（DataFrame 或 ``None``）、``normality``、
        ``variance``、``effect_size``、``effect_name``；退化情形 ``ok=False``。
    """
    v = _as_float(values)
    if group is None:
        return _degenerate("缺少分组标签，无法确定检验流程。", n_groups=0, posthoc=None, summary=None)
    g_raw = np.asarray(group).ravel()
    if g_raw.size != v.size:
        return _degenerate("数据长度与分组标签长度不一致。", n_groups=0, posthoc=None, summary=None)

    # 只保留 (值, 组) 都有效的样本
    valid = np.isfinite(v) & np.array([x is not None and str(x) != "nan" for x in g_raw], dtype=bool)
    v_ok = v[valid]
    g_txt = np.array([str(x) for x in g_raw[valid]], dtype=object)
    n_missing = int(v.size - v_ok.size)
    miss_note = f"（已剔除 {n_missing} 个缺失/非有限值样本）" if n_missing > 0 else ""

    names = list(pd.unique(pd.Series(g_txt))) if g_txt.size else []
    groups = {name: v_ok[g_txt == name] for name in names}
    n_groups = len(groups)

    if n_groups < 2:
        summary = describe_groups(groups) if groups else None
        return {
            "ok": False,
            "n_groups": n_groups,
            "test": "未执行",
            "test_key": "none",
            "statistic": float("nan"),
            "p": float("nan"),
            "df": None,
            "effect_size": float("nan"),
            "effect_name": "",
            "reason": f"当前只有 {n_groups} 个有效分组（至少需要 2 组）{miss_note}，无法进行组间检验。",
            "summary": summary,
            "posthoc": None,
            "normality": None,
            "variance": None,
            "alpha": float(alpha),
        }

    if n_groups == 2 and not paired:
        a_name, b_name = names[0], names[1]
        res = compare_two_groups(
            groups[a_name], groups[b_name], paired=False, alpha=alpha, auto=True
        )
        summary = describe_groups(groups)
        out = dict(res)
        out.update(
            n_groups=2,
            group_a=a_name,
            group_b=b_name,
            summary=summary,
            posthoc=None,
            alpha=float(alpha),
        )
        if miss_note:
            out["reason"] = str(out.get("reason", "")) + miss_note
        return out

    if paired and n_groups == 2:
        a_name, b_name = names[0], names[1]
        res = compare_paired(groups[a_name], groups[b_name], alpha=alpha)
        summary = describe_groups(groups)
        out = dict(res)
        out.update(
            n_groups=2,
            group_a=a_name,
            group_b=b_name,
            summary=summary,
            posthoc=None,
            alpha=float(alpha),
        )
        if miss_note:
            out["reason"] = str(out.get("reason", "")) + miss_note
        return out

    res = compare_multiple_groups(groups, alpha=alpha, auto=True, posthoc=True)
    out = dict(res)
    out["summary"] = res.get("groups_summary")
    out["n_groups"] = n_groups
    if paired:
        out["reason"] = str(out.get("reason", "")) + "（组数超过 2，已按独立样本的多组流程处理。）"
    if miss_note:
        out["reason"] = str(out.get("reason", "")) + miss_note
    return out


# ==========================================================================
# 自检（python amplicondesk/core/hypothesis.py）
# ==========================================================================
def _selftest() -> int:
    """模块自检：一次性验证自动选路、效应量、事后检验、第一类错误率与退化输入。"""
    warnings.simplefilter("error")  # 自检期间任何 RuntimeWarning 都视为失败
    rng = np.random.default_rng(20240517)
    failures: list[str] = []
    lines: list[str] = []

    def report(name: str, ok: bool, detail: str) -> None:
        mark = "PASS" if ok else "FAIL"
        lines.append(f"[{mark}] {name}: {detail}")
        if not ok:
            failures.append(name)

    def normals(mu: float, sd: float, n: int) -> np.ndarray:
        return rng.normal(mu, sd, n)

    def assert_normal(arr: np.ndarray, name: str) -> None:
        p = float(st.shapiro(arr)[1])
        assert p > 0.10, f"{name} 的 Shapiro p = {p:.4f}，构造数据并非正态"

    # ---------------- 1. Student t 路径 ----------------
    a = normals(0.0, 1.0, 40)
    b = normals(1.0, 1.0, 40)
    assert_normal(a, "t 路径 a 组")
    assert_normal(b, "t 路径 b 组")
    r1 = compare_two_groups(a, b, alpha=0.05)
    nrm = r1["normality"]["per_group"]
    ok = (
        np.isfinite(r1["p"]) and r1["p"] < 0.05
        and ("Student" in str(r1["test"]))
        and float(np.mean(a) - np.mean(b)) < 0
        and r1["effect_size"] < 0
    )
    report(
        "Student t 路径",
        ok,
        f"test={r1['test']} p={r1['p']:.3e} d={r1['effect_size']:.3f} "
        f"shapiro_a={nrm['a']['p']:.3f} shapiro_b={nrm['b']['p']:.3f} levene={r1['variance']['p']:.3f} "
        f"CI=[{r1['ci_low']:.3f}, {r1['ci_high']:.3f}]",
    )
    lines.append(f"      reason: {r1['reason']}")

    # ---------------- 2. Welch 路径 ----------------
    a2 = normals(0.0, 1.0, 60)
    b2 = normals(0.8, np.sqrt(10.0), 60)
    assert_normal(a2, "Welch 路径 a 组")
    assert_normal(b2, "Welch 路径 b 组")
    r2 = compare_two_groups(a2, b2, alpha=0.05)
    is_welch = ("Welch" in str(r2["test"])) or (
        r2.get("test_key") == "welch"
    )
    ok = is_welch and ("方差齐性" in r2["reason"]) and np.isfinite(r2["p"])
    report(
        "Welch 路径",
        ok,
        f"test={r2['test']} p={r2['p']:.3e} levene_p={r2['variance']['p']:.3e} "
        f"var_ratio={np.var(b2, ddof=1) / np.var(a2, ddof=1):.2f} df={r2['df']:.2f}",
    )
    lines.append(f"      reason: {r2['reason']}")
    if not (np.isfinite(r2["variance"]["p"]) and r2["variance"]["p"] <= 0.05):
        lines.append("      提示：本次抽样 Levene 未检出方差不齐，Welch 判定依赖 normality+variance 组合。")

    # ---------------- 3. Mann-Whitney 路径 ----------------
    a3 = rng.exponential(1.0, 60)
    b3 = rng.exponential(2.5, 60)
    r3 = compare_two_groups(a3, b3, alpha=0.05)
    ok = ("Mann-Whitney" in str(r3["test"])) and ("Mann-Whitney" in r3["reason"]) and np.isfinite(r3["p"])
    report(
        "Mann-Whitney 路径",
        ok,
        f"test={r3['test']} p={r3['p']:.3e} shapiro_a={r3['normality']['per_group']['a']['p']:.3e} "
        f"shapiro_b={r3['normality']['per_group']['b']['p']:.3e} r_rb={r3['effect_size']:.3f} "
        f"HL={r3.get('hodges_lehmann', float('nan')):.3f} CI=[{r3['ci_low']:.3f}, {r3['ci_high']:.3f}]",
    )
    lines.append(f"      reason: {r3['reason']}")

    # ---------------- 4. 配对流程 ----------------
    base = normals(0.0, 1.0, 45)
    diff_n = normals(0.6, 1.0, 45)
    p1 = compare_paired(base, base + diff_n, alpha=0.05)
    ok1 = ("配对样本 t 检验" in str(p1["test"])) and np.isfinite(p1["p"])
    report(
        "配对流程（差值为正态 → 配对 t）",
        ok1,
        f"test={p1['test']} p={p1['p']:.3e} diff_shapiro={p1['diff_normality']['overall']['p']:.3f} "
        f"mean_diff={p1['mean_diff']:.3f} dz={p1['effect_size']:.3f} CI=[{p1['ci_low']:.3f}, {p1['ci_high']:.3f}]",
    )
    lines.append(f"      reason: {p1['reason']}")

    diff_s = rng.exponential(1.0, 60) + 0.4
    base_pair = normals(0.5, 1.0, 60)
    p2 = compare_paired(base_pair, base_pair + diff_s, alpha=0.05)
    ok2 = ("Wilcoxon 符号秩" in str(p2["test"])) and np.isfinite(p2["p"])
    report(
        "配对流程（差值偏态 → Wilcoxon 符号秩）",
        ok2,
        f"test={p2['test']} p={p2['p']:.3e} diff_shapiro={p2['diff_normality']['overall']['p']:.3e} "
        f"median_diff={p2['median_diff']:.3f} r={p2['effect_size']:.3f} "
        f"HL={p2.get('hodges_lehmann', float('nan')):.3f}",
    )
    lines.append(f"      reason: {p2['reason']}")

    # ---------------- 5. ANOVA 路径 ----------------
    g1 = normals(0.0, 1.0, 35)
    g2 = normals(0.2, 1.0, 35)
    g3 = normals(1.5, 1.0, 35)
    for nm, g in (("A", g1), ("B", g2), ("C", g3)):
        assert_normal(g, f"ANOVA 路径 {nm} 组")
    groups3 = {"A": g1, "B": g2, "C": g3}
    r5 = compare_multiple_groups(groups3, alpha=0.05, posthoc=True)
    ph = r5["posthoc"]
    ok5 = (
        r5["test_key"] == "anova"
        and ph is not None and len(ph) == 3
        and np.all(ph["p_adj"].to_numpy() >= ph["p"].to_numpy() - 1e-12)
        and isinstance(r5["groups_summary"], pd.DataFrame)
        and r5["groups_summary"].shape[0] == 3
    )
    report(
        "ANOVA 路径",
        bool(ok5),
        f"test={r5['test']} p={r5['p']:.3e} F={r5['statistic']:.3f} "
        f"df=({r5['df_between']:.0f}, {r5['df_within']:.0f}) eta2={r5['effect_size']:.3f} "
        f"levene_p={r5['variance']['p']:.3f} posthoc_pairs={0 if ph is None else len(ph)}",
    )
    if ph is not None:
        lines.append(
            "      Tukey: "
            + "; ".join(
                f"{row.group1}-{row.group2} diff={row.meandiff:.3f} p={row.p:.3g} p_adj={row.p_adj:.3g} "
                f"CI=[{row.ci_low:.3f},{row.ci_high:.3f}]"
                for row in ph.itertuples()
            )
        )
    lines.append(f"      reason: {r5['reason']}")

    # ---------------- 6. Kruskal-Wallis 路径 ----------------
    k1 = rng.exponential(1.0, 35)
    k2 = rng.exponential(2.0, 35)
    k3 = rng.exponential(4.0, 35)
    groups_kw = {"Ctl": k1, "Low": k2, "High": k3}
    r6 = compare_multiple_groups(groups_kw, alpha=0.05, posthoc=True)
    ph6 = r6["posthoc"]
    ok6 = (
        r6["test_key"] == "kruskal"
        and ph6 is not None and len(ph6) == 3
        and np.all(ph6["p_adj"].to_numpy() >= ph6["p"].to_numpy() - 1e-12)
    )
    report(
        "Kruskal-Wallis 路径",
        bool(ok6),
        f"test={r6['test']} p={r6['p']:.3e} H={r6['statistic']:.3f} eps2={r6['effect_size']:.3f} "
        f"posthoc_pairs={0 if ph6 is None else len(ph6)}",
    )
    if ph6 is not None:
        lines.append(
            "      Dunn: "
            + "; ".join(
                f"{row.group1}-{row.group2} z={row.z:.3f} p={row.p:.3g} p_adj={row.p_adj:.3g}"
                for row in ph6.itertuples()
            )
        )
    lines.append(f"      reason: {r6['reason']}")

    # ---------------- 7 & 8. 第一类错误率 ----------------
    def type1_rate_t(n_iter: int = 300, n: int = 30, seed: int = 11) -> float:
        r = np.random.default_rng(seed)
        cnt = 0
        for _ in range(n_iter):
            x = r.normal(0, 1, n)
            y = r.normal(0, 1, n)
            if float(st.ttest_ind(x, y, equal_var=True)[1]) < 0.05:
                cnt += 1
        return cnt / n_iter

    def type1_rate_kw(n_iter: int = 300, n: int = 25, seed: int = 22) -> float:
        r = np.random.default_rng(seed)
        cnt = 0
        for _ in range(n_iter):
            x = r.normal(0, 1, n)
            y = r.normal(0, 1, n)
            z = r.normal(0, 1, n)
            if float(st.kruskal(x, y, z)[1]) < 0.05:
                cnt += 1
        return cnt / n_iter

    rate_t = type1_rate_t()
    ok7 = 0.03 <= rate_t <= 0.08
    report("第一类错误率（t 检验，300 次）", ok7, f"p<0.05 比例 = {rate_t:.4f}（要求 0.03~0.08）")

    rate_kw = type1_rate_kw()
    ok8 = 0.03 <= rate_kw <= 0.08
    report("第一类错误率（Kruskal-Wallis，300 次）", ok8, f"p<0.05 比例 = {rate_kw:.4f}（要求 0.03~0.08）")

    # ---------------- 9. 卡方 / Fisher ----------------
    ct = pd.DataFrame([[20, 10], [8, 22]], index=["暴露", "对照"], columns=["病例", "健康"])
    r9 = compare_categorical(ct, correction=True)
    r9b = compare_categorical(ct, correction=False)
    ok9 = (
        r9["test_key"] == "chi2" and "Yates" in r9["test"]
        and r9["dof"] == 1
        and np.isfinite(r9["p"]) and np.isfinite(r9["effect_size"])
        and abs(r9["p"] - r9b["p"]) > 1e-6
    )
    report(
        "卡方检验 + Yates 校正",
        bool(ok9),
        f"test={r9['test']} chi2={r9['statistic']:.4f} dof={r9['dof']:.0f} p={r9['p']:.4f} "
        f"V={r9['effect_size']:.3f} | Yates=False p={r9b['p']:.4f}（应当更小）",
    )
    lines.append(f"      reason: {r9['reason']}")

    ct_small = pd.DataFrame([[9, 1], [1, 7]], index=["A", "B"], columns=["+", "-"])
    r10 = compare_categorical(ct_small)
    ok10 = (
        r10["test_key"] == "fisher"
        and r10["p"] < 0.05
        and r10["small_expected_frac"] > 0.20
        and np.isfinite(r10["effect_size"])
    )
    report(
        "期望频数过小 → Fisher",
        bool(ok10),
        f"test={r10['test']} p={r10['p']:.4f} 期望<5 比例={r10['small_expected_frac']:.0%} "
        f"OR={r10['statistic']:.3f} V={r10['effect_size']:.3f}",
    )
    lines.append(f"      reason: {r10['reason']}")

    ct_big = pd.DataFrame(
        np.array([[30, 10, 5], [8, 25, 12], [5, 8, 30]]),
        index=["G1", "G2", "G3"], columns=["T1", "T2", "T3"],
    )
    r10b = compare_categorical(ct_big)
    ok10b = r10b["test_key"] == "chi2" and r10b["dof"] == 4
    report(
        "3×3 列联表 → Pearson 卡方",
        bool(ok10b),
        f"test={r10b['test']} chi2={r10b['statistic']:.3f} dof={r10b['dof']:.0f} p={r10b['p']:.4f} V={r10b['effect_size']:.3f}",
    )

    # ---------------- 10. Dunn 检验 ----------------
    d_groups = {
        "G1": rng.exponential(1.0, 20),
        "G2": rng.exponential(1.0, 20),
        "G3": rng.exponential(1.0, 20),
        "G4": rng.exponential(1.0, 20),
    }
    dunn = dunn_posthoc(d_groups, p_adjust="bh")
    expect_pairs = 4 * 3 // 2
    ordered = dunn.sort_values("p")
    mono_ok = bool(np.all(np.diff(ordered["p_adj"].to_numpy()) >= -1e-12))
    ok11 = (
        len(dunn) == expect_pairs
        and np.all(dunn["p_adj"].to_numpy() >= dunn["p"].to_numpy() - 1e-12)
        and mono_ok
    )
    report(
        "Dunn 事后检验",
        bool(ok11),
        f"对数={len(dunn)}（期望 C(4,2)={expect_pairs}）p_adj≥p 成立，BH 单调={mono_ok}，"
        f"最小 p={dunn['p'].min():.4f} 最小 p_adj={dunn['p_adj'].min():.4f}",
    )

    # ---------------- 11. 退化输入 ----------------
    degen: list[tuple[str, bool, str]] = []

    def check_degen(name: str, func, expect_ok_false: bool = True) -> dict:
        try:
            res = func()
        except Exception as exc:  # noqa: BLE001
            degen.append((name, False, f"抛出异常 {type(exc).__name__}: {exc}"))
            return {}
        msg = str(res.get("reason") or res.get("note") or "")
        has_cn = any("\u4e00" <= ch <= "\u9fff" for ch in msg)
        ok = has_cn and (not expect_ok_false or res.get("ok") is False)
        degen.append((name, ok, msg[:60].replace("\n", " ")))
        return res

    zero_a = np.zeros(10)
    zero_b = np.ones(10)
    check_degen("零方差（两组）", lambda: compare_two_groups(zero_a, zero_b))
    check_degen("单组", lambda: compare_multiple_groups({"A": np.arange(10.0)}))
    check_degen("全 NaN", lambda: compare_two_groups(np.full(10, np.nan), np.arange(10.0)))
    check_degen("样本 < 3", lambda: compare_two_groups(np.array([1.0, 2.0]), np.array([3.0, 4.0])))
    check_degen("正态性：全 NaN", lambda: check_normality(np.full(8, np.nan)))
    check_degen("正态性：样本 < 3", lambda: check_normality(np.array([1.0, 2.0])))
    check_degen("方差齐性：单组", lambda: check_variance({"A": np.arange(10.0)}))
    check_degen("多组：样本 < 3", lambda: compare_multiple_groups({"A": np.arange(10.0), "B": np.array([1.0])}))
    check_degen("多组：零方差", lambda: compare_multiple_groups({"A": zero_a, "B": zero_b}))
    check_degen("auto_test：全 NaN", lambda: auto_test(np.full(10, np.nan), np.array(["A"] * 5 + ["B"] * 5)))
    check_degen("auto_test：单组", lambda: auto_test(np.arange(6.0), np.array(["A"] * 6)))
    check_degen("列联表：单列", lambda: compare_categorical(pd.DataFrame([[3], [4]])))
    check_degen("列联表：全零", lambda: compare_categorical(pd.DataFrame([[0, 0], [0, 0]])))
    check_degen("配对：长度不一致", lambda: compare_paired(np.arange(6.0), np.arange(5.0)))

    ok12 = all(ok for _, ok, _ in degen)
    detail = "; ".join(f"{nm}{'✓' if ok else '✗'}" for nm, ok, _ in degen)
    report("退化输入（14 项）", ok12, detail)
    for nm, ok, msg in degen:
        if not ok:
            lines.append(f"      ✗ {nm}: {msg}")

    # ---------------- 12. 其他小接口 ----------------
    ok13 = (
        significance_stars(0.0005) == "***"
        and significance_stars(0.005) == "**"
        and significance_stars(0.04) == "*"
        and significance_stars(0.5) == "ns"
        and significance_stars(float("nan")) == "ns"
    )
    report(
        "significance_stars",
        ok13,
        " / ".join(
            [
                significance_stars(0.0005),
                significance_stars(0.005),
                significance_stars(0.04),
                significance_stars(0.5),
                significance_stars(float("nan")),
            ]
        ),
    )

    desc = describe_groups(groups3)
    ok14 = (
        list(desc.columns) == ["n", "n_missing", "mean", "sd", "median", "q1", "q3", "IQR", "min", "max", "se", "ci_low", "ci_high"]
        and desc.shape[0] == 3
        and np.all(desc["n"] == 35)
        and np.all(desc["ci_low"] < desc["mean"]) and np.all(desc["mean"] < desc["ci_high"])
    )
    report(
        "describe_groups",
        bool(ok14),
        f"shape={desc.shape} 列={len(desc.columns)} 首组 mean={desc['mean'].iloc[0]:.3f} "
        f"CI=[{desc['ci_low'].iloc[0]:.3f}, {desc['ci_high'].iloc[0]:.3f}] 各列齐全={list(desc.columns) == ['n','n_missing','mean','sd','median','q1','q3','IQR','min','max','se','ci_low','ci_high']}",
    )

    eff = {
        "d": effect_size_cohens_d(g1, g3),
        "g": effect_size_hedges_g(g1, g3),
        "delta": effect_size_cliffs_delta(g1, g3),
        "r": effect_size_rank_biserial(g1, g3),
        "dz": effect_size_cohens_d(base, base + diff_n, paired=True),
        "d_rev": effect_size_cohens_d(g3, g1),
    }
    r_pair = compare_two_groups(g1, g3)
    ok15 = (
        np.isfinite(eff["d"]) and eff["d"] < 0 and abs(eff["g"]) < abs(eff["d"])
        and np.isfinite(eff["delta"]) and eff["delta"] < 0
        and np.isfinite(eff["r"]) and eff["r"] < 0
        and np.isclose(eff["r"], eff["delta"], atol=1e-9)  # rank-biserial 与 Cliff's delta 同号同向
        and np.isfinite(eff["dz"]) and eff["dz"] < 0
        and np.isfinite(eff["d_rev"]) and eff["d_rev"] > 0
        and abs(eff["d"] + eff["d_rev"]) < 1e-9  # 交换两组后符号取反
        and np.isclose(eff["d"], r_pair["effect_size"], atol=1e-9)  # 与 compare_two_groups 口径一致
        and np.isclose(eff["delta"], r_pair["cliffs_delta"], atol=1e-9)
    )
    report(
        "效应量",
        bool(ok15),
        f"d={eff['d']:.3f} g={eff['g']:.3f} cliffs_delta={eff['delta']:.3f} "
        f"r_rb={eff['r']:.3f}（与 delta 一致）d_z={eff['dz']:.3f} d(交换两组)={eff['d_rev']:.3f}",
    )

    r16 = auto_test(np.concatenate([g1, g3]), np.array(["A"] * 35 + ["C"] * 35))
    ok16 = r16["n_groups"] == 2 and r16["ok"] and isinstance(r16["summary"], pd.DataFrame)
    report(
        "auto_test（两组）",
        bool(ok16),
        f"n_groups={r16['n_groups']} test={r16['test']} p={r16['p']:.3e} summary_shape={r16['summary'].shape}",
    )

    # ---------------- 11b. studentized range 回退实现 ----------------
    # k = 2 时正态极差 CDF 有闭式解 2Φ(z/√2) - 1，用它校验求积内核；
    # 若本机 scipy 提供 studentized_range，再逐点核对整体实现。
    close_errs = [
        abs(float(_normal_range_cdf(z, 2)) - (2.0 * float(st.norm.cdf(z / np.sqrt(2.0))) - 1.0))
        for z in (0.05, 0.3, 1.0, 1.5, 3.0, 6.0)
    ]
    kernel_err = max(close_errs)
    checks = [
        abs(float(_normal_range_cdf(z, kk)) - _normal_range_cdf_direct(z, kk)) < 1e-6
        for kk in (2, 3, 4)
        for z in (0.3, 1.0, 2.0, 4.0)
    ]
    ok18 = bool(kernel_err < 1e-9 and all(checks))
    detail18 = f"正态极差闭式解最大误差={kernel_err:.2e}，与直接求积一致={all(checks)}"
    if hasattr(st, "studentized_range"):
        sr_errs = [
            abs(_studentized_range_cdf(qq, kk, dff) - float(st.studentized_range.cdf(qq, kk, dff)))
            for kk in (2, 3, 4)
            for dff in (10.0, 61.0)
            for qq in (1.5, 2.5, 3.4)
        ]
        sr_err = max(sr_errs)
        ok18 = ok18 and sr_err < 1e-6
        detail18 += f"，studentized range 与 scipy 最大误差={sr_err:.2e}"
    report("studentized range 回退实现", ok18, detail18)

    r17 = auto_test(
        np.concatenate([g1, g2, g3]), np.array(["A"] * 35 + ["B"] * 35 + ["C"] * 35)
    )
    ok17 = (
        r17["n_groups"] == 3 and r17["test_key"] == "anova"
        and r17["posthoc"] is not None and len(r17["posthoc"]) == 3
        and r17["summary"] is not None and r17["summary"].shape[0] == 3
    )
    report(
        "auto_test（三组）",
        bool(ok17),
        f"n_groups={r17['n_groups']} test={r17['test']} p={r17['p']:.3e} "
        f"posthoc_pairs={0 if r17['posthoc'] is None else len(r17['posthoc'])}",
    )

    # ---------------- 输出 ----------------
    print("=" * 78)
    print("amplicondesk.core.hypothesis 自检")
    print("=" * 78)
    for line in lines:
        print(line)
    print("-" * 78)
    if failures:
        print(f"自检结果：{len(failures)} 项未通过 → {'、'.join(failures)}")
    else:
        print("自检结果：全部通过 ✅")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
