"""AmpliconDesk —— 本地可运行的生物信息下游分析流程。

主要入口：
    from amplicondesk import run_analysis
    result = run_analysis(config, output_root="output")
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["run_analysis", "default_config", "ANALYSIS_TYPES", "__version__"]

from .core.pipeline import ANALYSIS_TYPES, default_config, run_analysis
