"""回归测试样式预览接口：逐个图种请求，确认都能出图。

之前这个接口没被任何回归覆盖，`load_inputs` 返回值从 6 个减到 5 个时漏改了这里的解包，
直到截图时才发现「预览失败：not enough values to unpack」。
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8765"
ROOT = Path(__file__).resolve().parent.parent          # AmpliconDesk/
DEMO = ROOT / "examples" / "demo_data"

CONFIG = {
    "analysis_type": "amplicon16s",
    "abundance_path": str(DEMO / "16s_otu_table.csv"),
    "metadata_path": str(DEMO / "16s_metadata.csv"),
    "taxonomy_path": str(DEMO / "16s_taxonomy.csv"),
    "env_path": str(DEMO / "16s_env.csv"),
    "group_column": "Group",
    "compare_pairs": [["Control", "Treatment"]],
}

STYLE = {"values": {"font.family": "Times New Roman", "palette.preset": "方案16",
                    "export.formats": "png"}}


def post(path: str, payload: dict, timeout: int = 120):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # 400 的响应体里才有真正的错误原因，不能只看状态码
        body = e.read().decode("utf-8", "replace")
        try:
            return json.loads(body)
        except Exception:
            return {"ok": False, "error": f"HTTP {e.code}: {body[:200]}"}


def get(path: str, timeout: int = 60):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    print("=" * 70)
    print("样式预览接口回归")
    print("=" * 70)

    try:
        cfg = get("/api/config")
    except Exception as exc:
        print(f"连不上服务（{BASE}）：{exc}")
        return 1

    figures = [p["id"] for p in cfg.get("previewable", [])]
    if not figures:
        print("配置里没有可预览图，跳过")
        return 0

    print(f"从 /api/config 取到 {len(figures)} 个可预览图种")
    print()

    ok = fail = 0
    failures = []
    for fig in figures:
        r = post("/api/style/preview",
                 {"figure_id": fig, "style": STYLE, "config": CONFIG})
        if r.get("ok"):
            print(f"  ✓ {fig:22s} {r.get('url')}")
            ok += 1
        else:
            err = r.get("error") or "未知错误"
            print(f"  ✗ {fig:22s} {err}")
            failures.append((fig, err))
            fail += 1

    print()
    print(f"成功 {ok} / 失败 {fail}（共 {len(figures)} 个图种）")
    if failures:
        print()
        print("失败明细：")
        for fig, err in failures:
            print(f"  {fig}: {err}")
    print()
    print("结论：", "全部通过 ✅" if fail == 0 else "存在失败 ❌")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
