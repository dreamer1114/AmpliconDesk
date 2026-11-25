"""机器学习相关绘图：特征重要性、ROC 曲线、混淆矩阵。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..style import Style
from .base import finish_axes, new_figure, place_title, resolve_legend, save, style_axes

__all__ = ["plot_rf_importance", "plot_roc", "plot_confusion"]


def plot_rf_importance(importance: pd.DataFrame, out_path: str | Path, style: Style,
                       top_n: int = 20) -> dict:
    df = importance.head(top_n).iloc[::-1]
    fig, ax = new_figure(style, 7.4, max(3.6, 0.34 * len(df) + 1.8))
    ax.barh([str(f)[:46] for f in df["feature"]], df["importance"],
            color=style.single_color(2),
            edgecolor="white" if style.flag("element.bar_edge", True) else "none", linewidth=0.5)
    style_axes(ax, style)
    ax.set_xlabel("Gini importance", fontsize=style.font_size("axis_label"))
    place_title(ax, style, f"随机森林特征重要性 Top {len(df)}")
    ax.tick_params(axis="y", labelsize=style.font_size("tick_label") * 0.9)
    fig.tight_layout()
    return save(fig, out_path, style,
                "随机森林基尼重要性排序，表示各特征对样本分类的贡献程度。",
                "随机森林特征重要性")


def plot_roc(roc: dict, out_path: str | Path, style: Style,
             accuracy: float | None = None) -> dict:
    fpr = np.asarray(roc["fpr"])
    tpr = np.asarray(roc["tpr"])
    fig, ax = new_figure(style, 5.8, 5.4)
    line_color = style.single_color(3)
    ax.plot(fpr, tpr, color=line_color, lw=2.2, label=f"AUC = {roc['auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="#888888", ls="--", lw=1.0, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.12, color=line_color)
    style_axes(ax, style)
    ax.set_xlabel("False Positive Rate", fontsize=style.font_size("axis_label"))
    ax.set_ylabel("True Positive Rate", fontsize=style.font_size("axis_label"))
    title = f"ROC 曲线（正类：{roc['positive_class']}）"
    if accuracy is not None:
        title += f"\n交叉验证准确率 = {accuracy:.1%}"
    place_title(ax, style, title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    resolve_legend(style, ax, preferred="lower right")
    fig.tight_layout()
    return save(fig, out_path, style,
                "基于交叉验证预测概率绘制的 ROC 曲线，AUC 越接近 1 表示分组的可分性越强。",
                "ROC 曲线")


def plot_confusion(cm: pd.DataFrame, out_path: str | Path, style: Style,
                   accuracy: float | None = None, title: str = "混淆矩阵") -> dict:
    """混淆矩阵热图。cm 为 DataFrame（真实 × 预测）。"""
    fig, ax = new_figure(style, 5.4, 4.8)
    data = cm.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=style.cmap("continuous"), aspect="auto")
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels([str(c) for c in cm.columns], rotation=style.num("element.axis_rotate", 45))
    ax.set_yticklabels([str(i) for i in cm.index])
    ax.set_xlabel("预测类别", fontsize=style.font_size("axis_label"))
    ax.set_ylabel("真实类别", fontsize=style.font_size("axis_label"))

    vmax = data.max() if data.size else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(data[i, j])}", ha="center", va="center",
                    fontsize=style.font_size("annotation"),
                    color="white" if data[i, j] > vmax * 0.6 else style.font_color("annotation"))

    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("样本数")
    head = title + (f"（准确率 {accuracy:.1%}）" if accuracy is not None else "")
    place_title(ax, style, head)
    fig.tight_layout()
    return save(fig, out_path, style,
                "对角线为分类正确的样本数，非对角线为误判。基于分层交叉验证的预测结果统计。",
                title)
