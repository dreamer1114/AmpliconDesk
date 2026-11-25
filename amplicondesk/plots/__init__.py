"""AmpliconDesk 绘图模块。

所有绘图函数统一签名：``f(数据..., out_path, style, ...) -> {"path","name","caption","title"}``
"""

from __future__ import annotations

from .alpha import (
    plot_accumulation, plot_alpha_heatmap, plot_alpha_panel, plot_alpha_rarefaction,
    plot_rank_abundance, plot_rarefaction, plot_sample_quality,
)
from .base import (
    conf_ellipse, finish_axes, new_figure, panel_grid, place_annotation, place_axis_labels,
    place_suptitle, place_title, resolve_legend, save, scatter_groups, style_axes,
)
from .advanced import (
    plot_beeswarm, plot_bubble, plot_density, plot_lollipop, plot_raincloud,
    plot_sankey, plot_stacked_flow, plot_violin, plot_voronoi_treemap,
)
from .composition import (
    plot_composition_area, plot_composition_bar, plot_core_microbiome, plot_heatmap,
    plot_ternary, plot_top_taxa_box, plot_venn,
)
from .differential import (
    plot_diff_bar, plot_diff_heatmap, plot_feature_box, plot_forest, plot_lda, plot_ma,
    plot_volcano,
)
from .expression import (
    plot_correlation_heatmap, plot_enrichment_bar, plot_enrichment_bubble,
    plot_expression_boxplot, plot_expression_pca, plot_size_factors,
)
from .machine import plot_confusion, plot_rf_importance, plot_roc
from .treemap import (
    plot_treemap_auto, plot_treemap_fallback, plot_voronoi_treemap,
    voronoi_treemap_geometry,
)
from .grouping import (
    plot_between_vs_within, plot_group_overview, plot_within_group_distance,
    plot_within_group_diversity,
)
from .ordination import (
    plot_cap, plot_cca, plot_dbrda, plot_distance_boxplot, plot_envfit, plot_mantel,
    plot_nmds, plot_pca, plot_pcoa, plot_permdisp, plot_plsda, plot_plsda_vip,
    plot_procrustes, plot_rda, plot_scree, plot_tsne, plot_umap,
)

__all__ = [
    # base
    "save", "new_figure", "panel_grid", "style_axes", "finish_axes",
    "place_title", "place_suptitle", "place_axis_labels", "place_annotation",
    "conf_ellipse", "scatter_groups", "resolve_legend",
    # alpha
    "plot_alpha_panel", "plot_alpha_heatmap", "plot_rarefaction",
    "plot_alpha_rarefaction", "plot_accumulation", "plot_rank_abundance",
    "plot_sample_quality",
    # ordination
    "plot_pcoa", "plot_nmds", "plot_pca", "plot_rda", "plot_dbrda", "plot_cca",
    "plot_cap", "plot_tsne", "plot_umap", "plot_plsda", "plot_plsda_vip",
    "plot_scree", "plot_distance_boxplot", "plot_permdisp", "plot_envfit", "plot_mantel",
    "plot_procrustes",
    # composition
    "plot_composition_bar", "plot_composition_area", "plot_heatmap",
    "plot_top_taxa_box", "plot_venn", "plot_core_microbiome", "plot_ternary",
    # differential
    "plot_volcano", "plot_ma", "plot_diff_bar", "plot_forest", "plot_feature_box",
    "plot_lda", "plot_diff_heatmap",
    # machine
    "plot_rf_importance", "plot_roc", "plot_confusion",
    # grouping（组间 / 组内）
    "plot_group_overview", "plot_between_vs_within", "plot_within_group_distance",
    "plot_within_group_diversity",
    # advanced（高级图形）
    "plot_violin", "plot_raincloud", "plot_density", "plot_beeswarm",
    "plot_lollipop", "plot_bubble", "plot_sankey", "plot_voronoi_treemap",
    "plot_stacked_flow",
    # treemap（面积正比于丰度，维诺 + 矩形兜底）
    "plot_treemap_auto", "plot_treemap_fallback", "voronoi_treemap_geometry",
    # expression
    "plot_expression_boxplot", "plot_correlation_heatmap", "plot_expression_pca",
    "plot_size_factors", "plot_enrichment_bubble", "plot_enrichment_bar",
]
