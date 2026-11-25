"""AmpliconDesk 本地服务：把分析流程包装成一个可在浏览器里操作的「小软件」。

所有请求都只在本机 127.0.0.1 上处理，不联网、不外传数据。
"""

from __future__ import annotations

import io
import json
import os
import platform
import re
import subprocess
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from .. import __version__
from ..core.io import orient_feature_table, read_group_table, read_table
from ..core.pipeline import ANALYSIS_TYPES, default_config, run_analysis

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
DEMO_DIR = BASE_DIR / "examples" / "demo_data"
THEME_DIR = BASE_DIR / "themes"
for _d in (UPLOAD_DIR, OUTPUT_DIR, THEME_DIR):
    _d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def _safe_name(name: str) -> str:
    name = Path(str(name)).name
    name = re.sub(r"[^\w\u4e00-\u9fff.\-]+", "_", name).strip("._")
    return name or "file"


def _read_preview(path: Path, n_rows: int = 8, n_cols: int = 12) -> dict:
    """读取表格头部，用于界面预览。"""
    try:
        df = read_table(path, index_col=0)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    head = df.iloc[:n_rows, :n_cols]
    return {
        "ok": True,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "index_name": str(df.index.name or ""),
        "columns": [str(c) for c in head.columns],
        "rows": [[str(i)] + [str(v) for v in row] for i, row in zip(head.index, head.to_numpy())],
        "truncated_cols": bool(df.shape[1] > n_cols),
        "truncated_rows": bool(df.shape[0] > n_rows),
    }


def _inspect_inputs(abundance_path: str, metadata_path: str, taxonomy_path: str = "") -> dict:
    """体检：方向识别、样本匹配、分组列候选。"""
    ab = Path(abundance_path)
    md = Path(metadata_path)
    if not ab.exists():
        return {"ok": False, "error": f"丰度表不存在：{ab}"}
    if not md.exists():
        return {"ok": False, "error": f"分组表不存在：{md}"}

    table_raw = read_table(ab, index_col=0)
    meta = read_group_table(md)

    oriented, info = orient_feature_table(table_raw, list(meta.index))

    # 匹配样本
    key = lambda x: re.sub(r"[\s_\-]+", "", str(x)).lower()  # noqa: E731
    meta_lookup = {key(m): m for m in meta.index}
    matched = [c for c in oriented.columns if key(c) in meta_lookup]
    only_table = [c for c in oriented.columns if key(c) not in meta_lookup]
    matched_keys = {key(c) for c in matched}
    only_meta = [m for m in meta.index if key(m) not in matched_keys]

    group_candidates = []
    for col in meta.columns:
        vals = meta[col].astype(str)
        nun = int(vals.nunique())
        if nun < 2:
            continue
        counts = vals.value_counts()
        group_candidates.append(
            {
                "column": str(col),
                "n_levels": nun,
                "levels": [str(x) for x in counts.index[:30]],
                "counts": {str(k): int(v) for k, v in counts.items()},
                "looks_like_group": bool(2 <= nun <= max(12, len(meta) // 2)),
            }
        )
    group_candidates.sort(key=lambda d: (not d["looks_like_group"], d["n_levels"]))

    return {
        "ok": True,
        "transposed": bool(info.get("transposed")),
        "orient_reason": info.get("reason", ""),
        "n_samples_matched": len(matched),
        "n_features": int(oriented.shape[0]),
        "matched_samples": [str(s) for s in matched],
        "only_in_table": [str(s) for s in only_table[:30]],
        "only_in_metadata": [str(s) for s in only_meta[:30]],
        "n_only_in_table": len(only_table),
        "n_only_in_metadata": len(only_meta),
        "metadata_columns": [str(c) for c in meta.columns],
        "group_candidates": group_candidates,
        "taxonomy_ok": bool(taxonomy_path and Path(taxonomy_path).exists()),
        "abundance_preview": _read_preview(ab),
    }


# --------------------------------------------------------------------------
# 页面
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(app.template_folder, "index.html")


@app.get("/api/config")
def api_config():
    from ..style import (GROUPS, PALETTES, PRESETS, RESEARCH_PALETTES, STYLE_SCHEMA,
                         palette_names, research_palette_names)
    from .preview import PREVIEWABLE

    return jsonify(
        {
            "version": __version__,
            "analysis_types": [
                {"id": k, "label": v["label"], "note": v["note"],
                 "feature_label": v["feature_label"], "default_beta": v["default_beta"],
                 "default_ord": v.get("default_ord", [])}
                for k, v in ANALYSIS_TYPES.items()
            ],
            "beta_metrics": [
                {"id": "braycurtis", "label": "Bray-Curtis"},
                {"id": "jaccard", "label": "Jaccard"},
                {"id": "aitchison", "label": "Aitchison (CLR)"},
                {"id": "euclidean", "label": "Euclidean"},
            ],
            "ordinations": [
                {"id": "pcoa", "label": "PCoA 主坐标分析", "group": "距离排序"},
                {"id": "nmds", "label": "NMDS 非度量多维标度", "group": "距离排序"},
                {"id": "cap", "label": "CAP 典范分析主坐标", "group": "距离排序"},
                {"id": "pca", "label": "PCA 主成分分析", "group": "线性降维"},
                {"id": "tsne", "label": "t-SNE", "group": "非线性降维"},
                {"id": "umap", "label": "UMAP", "group": "非线性降维"},
                {"id": "plsda", "label": "PLS-DA 判别分析", "group": "判别分析"},
                {"id": "rda", "label": "RDA 冗余分析", "group": "约束排序"},
                {"id": "dbrda", "label": "db-RDA 基于距离的冗余分析", "group": "约束排序"},
                {"id": "cca", "label": "CCA 典范对应分析", "group": "约束排序"},
            ],
            "stats_tests": [
                {"id": "permanova", "label": "PERMANOVA 置换多元方差分析"},
                {"id": "anosim", "label": "ANOSIM 相似性分析"},
                {"id": "mrpp", "label": "MRPP 多元响应置换"},
                {"id": "permdisp", "label": "PERMDISP 离散度齐性检验"},
                {"id": "mantel", "label": "Mantel 检验（需环境因子表）"},
                {"id": "envfit", "label": "envfit 因子拟合（需环境因子表）"},
            ],
            "ranks": ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"],
            "hypothesis_methods": [
                {"id": "auto", "label": "自动判断（先查正态性与方差齐性）"},
                {"id": "t", "label": "Student t 检验"},
                {"id": "welch", "label": "Welch t 检验"},
                {"id": "paired_t", "label": "配对 t 检验"},
                {"id": "wilcoxon_rank", "label": "Wilcoxon 秩和检验（Mann-Whitney U）"},
                {"id": "wilcoxon_signed", "label": "Wilcoxon 符号秩检验"},
                {"id": "anova", "label": "单因素方差分析 ANOVA"},
                {"id": "kruskal", "label": "Kruskal-Wallis 检验"},
                {"id": "chisq", "label": "卡方检验"},
                {"id": "fisher", "label": "Fisher 精确检验"},
            ],
            "defaults": default_config(),
            "style_schema": STYLE_SCHEMA,
            "style_groups": GROUPS,
            "style_presets": list(PRESETS.keys()),
            "palette_names": palette_names(),
            "research_palettes": RESEARCH_PALETTES,
            "research_palette_names": research_palette_names(),
            "palettes": {k: v for k, v in PALETTES.items()},
            "previewable": PREVIEWABLE,
            "platform": platform.system(),
            "output_dir": str(OUTPUT_DIR),
            "demo_available": DEMO_DIR.exists(),
        }
    )


@app.get("/api/fonts")
def api_fonts():
    """系统可用字体列表（含中文字体优先排序）。"""
    from ..style import all_fonts, available_cjk_fonts

    cjk = available_cjk_fonts()
    others = [f for f in all_fonts() if f not in cjk]
    return jsonify({"ok": True, "cjk": cjk, "all": cjk + others})


@app.get("/api/demo")
def api_demo():
    """返回内置演示数据路径。"""
    if not DEMO_DIR.exists():
        return jsonify({"ok": False, "error": "演示数据不存在，请先运行 amplicondesk/examples/make_demo_data.py"}), 404
    return jsonify(
        {
            "ok": True,
            "dir": str(DEMO_DIR),
            "datasets": [
                {
                    "id": "16s",
                    "label": "16S 扩增子（24 样本 / 180 OTU / 含分类表与环境因子）",
                    "analysis_type": "amplicon16s",
                    "abundance_path": str(DEMO_DIR / "16s_otu_table.csv"),
                    "metadata_path": str(DEMO_DIR / "16s_metadata.csv"),
                    "taxonomy_path": str(DEMO_DIR / "16s_taxonomy.csv"),
                    "env_path": str(DEMO_DIR / "16s_env.csv"),
                    "group_column": "Group",
                },
                {
                    "id": "transposed",
                    "label": "转置 Excel 表（测试方向自动识别）",
                    "analysis_type": "amplicon16s",
                    "abundance_path": str(DEMO_DIR / "16s_otu_table_transposed.xlsx"),
                    "metadata_path": str(DEMO_DIR / "16s_metadata.csv"),
                    "taxonomy_path": "",
                    "env_path": "",
                    "group_column": "Group",
                },
                {
                    "id": "formats",
                    "label": "同一张表的 5 种格式（csv / tsv / txt / xls / xlsx）",
                    "analysis_type": "amplicon16s",
                    "abundance_path": str(DEMO_DIR / "formats" / "t.tsv"),
                    "metadata_path": str(DEMO_DIR / "16s_metadata.csv"),
                    "taxonomy_path": str(DEMO_DIR / "16s_taxonomy.csv"),
                    "env_path": "",
                    "group_column": "Group",
                },
            ],
        }
    )


# --------------------------------------------------------------------------
# 样式预览
# --------------------------------------------------------------------------
@app.post("/api/style/preview")
def api_style_preview():
    from .preview import render_preview

    payload = request.get_json(silent=True) or {}
    cfg = payload.get("config") or {}
    if not cfg.get("abundance_path") or not cfg.get("metadata_path"):
        return jsonify({"ok": False, "error": "请先选择丰度表和分组表"}), 400

    figure_id = payload.get("figure_id") or "pcoa"
    force = bool(payload.get("force"))
    try:
        path, caption = render_preview(cfg, payload.get("style") or {}, figure_id, force=force)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({
        "ok": True,
        "url": f"/preview/{path.name}?v={int(path.stat().st_mtime)}",
        "name": path.name,
        "caption": caption,
    })


@app.post("/api/style/save")
def api_style_save():
    """把当前样式保存为一个命名主题文件。"""
    from ..style import Style

    payload = request.get_json(silent=True) or {}
    name = _safe_name(payload.get("name") or "theme")
    if not name.endswith(".json"):
        name += ".json"
    style = Style.from_dict(payload.get("style") or {})
    path = THEME_DIR / name
    style.save(path)
    return jsonify({"ok": True, "path": str(path), "name": name})


@app.get("/api/style/themes")
def api_style_themes():
    themes = []
    if THEME_DIR.exists():
        for f in sorted(THEME_DIR.glob("*.json")):
            themes.append({"name": f.stem, "path": str(f)})
    return jsonify({"ok": True, "themes": themes})


@app.get("/api/style/theme/<name>")
def api_style_theme(name: str):
    from ..style import Style

    path = (THEME_DIR / f"{_safe_name(name)}.json").resolve()
    if not str(path).startswith(str(THEME_DIR.resolve())) or not path.exists():
        return jsonify({"ok": False, "error": "主题不存在"}), 404
    return jsonify({"ok": True, "style": Style.load(path).to_dict()})


@app.post("/api/style/clear_cache")
def api_style_clear_cache():
    from .preview import clear_cache

    return jsonify({"ok": True, "cleared": clear_cache()})


@app.get("/preview/<path:filename>")
def serve_preview(filename: str):
    from .preview import PREVIEW_DIR

    target = (PREVIEW_DIR / _safe_name(filename)).resolve()
    if not str(target).startswith(str(PREVIEW_DIR.resolve())) or not target.is_file():
        abort(404)
    return send_file(target)


# --------------------------------------------------------------------------
# 文件
# --------------------------------------------------------------------------
@app.post("/api/upload")
def api_upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "没有收到文件"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "error": "文件名为空"}), 400

    token = uuid.uuid4().hex[:8]
    name = _safe_name(f.filename)
    dest = UPLOAD_DIR / f"{token}_{name}"
    f.save(dest)
    return jsonify(
        {
            "ok": True,
            "path": str(dest),
            "name": name,
            "size": dest.stat().st_size,
            "preview": _read_preview(dest),
        }
    )


@app.post("/api/metadata/from_text")
def api_metadata_from_text():
    """把界面上直接粘贴的分组内容保存成分组表文件。"""
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "分组内容为空"}), 400

    lines = [ln for ln in text.replace("\r", "").split("\n") if ln.strip()]
    if len(lines) < 2:
        return jsonify({"ok": False, "error": "至少需要表头和一行样本"}), 400

    sep = "\t" if "\t" in lines[0] else ("," if "," in lines[0] else None)
    if sep is None:
        # 支持空格分隔
        try:
            df = pd.read_csv(io.StringIO("\n".join(lines)), sep=r"\s+", index_col=0)
        except Exception as exc:
            return jsonify({"ok": False, "error": f"解析失败：{exc}"}), 400
    else:
        try:
            df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, index_col=0)
        except Exception as exc:
            return jsonify({"ok": False, "error": f"解析失败：{exc}"}), 400

    if df.shape[1] == 0:
        return jsonify({"ok": False, "error": "没有解析出分组列"}), 400

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    token = uuid.uuid4().hex[:8]
    dest = UPLOAD_DIR / f"{token}_pasted_metadata.csv"
    df.to_csv(dest, encoding="utf-8-sig")

    return jsonify(
        {
            "ok": True,
            "path": str(dest),
            "name": dest.name,
            "n_samples": int(df.shape[0]),
            "columns": [str(c) for c in df.columns],
            "group_candidates": [
                {
                    "column": str(c),
                    "n_levels": int(df[c].nunique()),
                    "levels": [str(x) for x in df[c].value_counts().index[:30]],
                    "counts": {str(k): int(v) for k, v in df[c].value_counts().items()},
                    "looks_like_group": bool(2 <= df[c].nunique() <= max(12, len(df) // 2)),
                }
                for c in df.columns
            ],
        }
    )


@app.post("/api/inspect")
def api_inspect():
    payload = request.get_json(silent=True) or {}
    try:
        res = _inspect_inputs(
            payload.get("abundance_path", ""),
            payload.get("metadata_path", ""),
            payload.get("taxonomy_path", ""),
        )
        if res.get("ok"):
            res["env_ok"] = bool(payload.get("env_path") and Path(payload["env_path"]).exists())
            if res["env_ok"]:
                try:
                    env = read_table(payload["env_path"], index_col=0)
                    res["env_columns"] = [str(c) for c in env.columns]
                    res["n_env_samples"] = int(env.shape[0])
                except Exception as exc:
                    res["env_ok"] = False
                    res["env_error"] = str(exc)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify(res), (200 if res.get("ok") else 400)


# --------------------------------------------------------------------------
# 运行
# --------------------------------------------------------------------------
@app.post("/api/run")
def api_run():
    cfg = request.get_json(silent=True) or {}
    if not cfg.get("abundance_path") or not cfg.get("metadata_path"):
        return jsonify({"ok": False, "error": "请先选择丰度表和分组表"}), 400

    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": "running",
            "percent": 0.0,
            "stage": "排队中",
            "message": "",
            "logs": [],
            "result": None,
            "error": None,
            "config": cfg,
        }

    def worker():
        def cb(percent, stage, message):
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job:
                    job["percent"] = percent
                    job["stage"] = stage
                    job["message"] = message

        try:
            res = run_analysis(cfg, output_root=str(OUTPUT_DIR), progress_callback=cb)
            with _jobs_lock:
                _jobs[job_id].update(
                    status="done", percent=100.0, stage="完成",
                    result=res, logs=res.get("log", []),
                )
        except Exception as exc:
            with _jobs_lock:
                _jobs[job_id].update(
                    status="error", stage="失败",
                    error=f"{exc}",
                    logs=[traceback.format_exc()],
                )

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.get("/api/job/<job_id>")
def api_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "任务不存在"}), 404
        payload = dict(job)
    if payload.get("result"):
        payload["result"] = _slim_result(payload["result"])
    return jsonify({"ok": True, "job": _jsonable(payload)})


def _jsonable(obj, depth: int = 0):
    """把结果中的 numpy / pandas 对象转成可 JSON 序列化的等价物。

    作为最后一道安全网：即便分析模块往结果里塞了 DataFrame，接口也不会 500。
    """
    if depth > 8:
        return str(obj)
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist(), depth + 1)
    if isinstance(obj, pd.DataFrame):
        return {"__dataframe__": True, "shape": [int(obj.shape[0]), int(obj.shape[1])],
                "columns": [str(c) for c in obj.columns[:40]]}
    if isinstance(obj, pd.Series):
        return _jsonable(obj.to_dict(), depth + 1)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v, depth + 1) for v in obj]
    return str(obj)


def _slim_result(res: dict) -> dict:
    """把结果压缩成前端友好的结构。"""
    if not res:
        return res
    out = {
        "run_id": res.get("run_id"),
        "output_dir": res.get("output_dir"),
        "analysis_label": res.get("analysis_label"),
        "group_column": res.get("group_column"),
        "groups": res.get("groups"),
        "input_info": res.get("input_info"),
        "filter_stats": res.get("filter_stats"),
        "alpha_tests": res.get("alpha_tests"),
        "beta": res.get("beta"),
        "ordination": res.get("ordination"),
        "envfit": res.get("envfit"),
        "procrustes": res.get("procrustes"),
        "within_group": res.get("within_group"),
        "abundance_type": res.get("abundance_type"),
        "diff_summary": res.get("diff_summary"),
        "narrative": res.get("narrative"),
        "random_forest": res.get("random_forest"),
        "warnings": res.get("warnings"),
        "duration_sec": res.get("duration_sec"),
        "report": res.get("report"),
        "tables": res.get("tables"),
        "style": res.get("style"),
        "figures": [
            {
                "name": Path(f["path"]).name,
                "section": f.get("section"),
                "caption": f.get("caption"),
                "title": f.get("title"),
                "url": f"/output/{res.get('run_id')}/figures/{Path(f['path']).name}",
            }
            for f in (res.get("figures") or [])
        ],
    }
    return _jsonable(out)


@app.get("/api/runs")
def api_runs():
    """扫描输出目录，列出历史运行。"""
    runs = []
    if OUTPUT_DIR.exists():
        for d in sorted(OUTPUT_DIR.iterdir(), reverse=True):
            rj = d / "result.json"
            if not (d.is_dir() and rj.exists()):
                continue
            try:
                data = json.loads(rj.read_text(encoding="utf-8"))
            except Exception:
                continue
            runs.append(
                {
                    "run_id": data.get("run_id", d.name),
                    "analysis_label": data.get("analysis_label", "-"),
                    "finished_at": data.get("finished_at", ""),
                    "n_figures": len(data.get("figures") or []),
                    "n_samples": (data.get("input_info") or {}).get("n_samples"),
                    "groups": data.get("groups"),
                    "duration_sec": data.get("duration_sec"),
                    "report_url": f"/output/{d.name}/report.html",
                    "dir": str(d),
                }
            )
    return jsonify({"ok": True, "runs": runs[:50]})


@app.post("/api/open_folder")
def api_open_folder():
    """在文件管理器中打开结果目录（桌面软件的常见便利功能）。"""
    payload = request.get_json(silent=True) or {}
    target = payload.get("path") or str(OUTPUT_DIR)
    p = Path(target).resolve()
    allowed_roots = [OUTPUT_DIR.resolve(), DEMO_DIR.resolve(), UPLOAD_DIR.resolve(),
                     THEME_DIR.resolve()]
    if not any(str(p) == str(r) or str(p).startswith(str(r) + os.sep) for r in allowed_roots):
        return jsonify({"ok": False, "error": "不允许打开该路径"}), 403
    if not p.exists():
        return jsonify({"ok": False, "error": "路径不存在"}), 404
    try:
        if platform.system() == "Windows":
            os.startfile(str(p))  # noqa: S606
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/output/<run_id>/<path:subpath>")
def serve_output(run_id: str, subpath: str):
    run_dir = (OUTPUT_DIR / run_id).resolve()
    if not str(run_dir).startswith(str(OUTPUT_DIR.resolve())) or not run_dir.is_dir():
        abort(404)
    target = (run_dir / subpath).resolve()
    if not str(target).startswith(str(run_dir)) or not target.exists():
        abort(404)
    if target.is_dir():
        abort(404)
    return send_file(target)


@app.get("/uploads/<path:subpath>")
def serve_upload(subpath: str):
    target = (UPLOAD_DIR / subpath).resolve()
    if not str(target).startswith(str(UPLOAD_DIR.resolve())) or not target.is_file():
        abort(404)
    return send_from_directory(UPLOAD_DIR, subpath, as_attachment=True)


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"ok": False, "error": "文件超过 512 MB 上限"}), 413


@app.after_request
def no_cache(resp):
    if resp.mimetype in ("text/html", "application/json", "text/javascript", "text/css"):
        resp.headers["Cache-Control"] = "no-store"
    return resp
