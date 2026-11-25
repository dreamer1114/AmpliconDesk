"""把用户提供的科研配色方案 Excel 提取成软件内置的 JSON 数据文件。

输出：  amplicondesk/data/research_palettes.json

只需要在配色方案更新时重新跑一次；日常运行直接读 JSON，不依赖 Excel。

用法：
    python tools/import_palettes.py 科研配色方案.xlsx
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import openpyxl

OUT = Path(__file__).resolve().parent.parent / "amplicondesk" / "data" / "research_palettes.json"

HEX_RE = re.compile(r"^#?[0-9A-Fa-f]{6}$")


def norm_hex(v) -> str | None:
    """把单元格值规范成 #RRGGBB。支持字符串与 openpyxl 的颜色对象。"""
    if v is None:
        return None
    s = str(v).strip()
    if HEX_RE.match(s):
        return "#" + s.lstrip("#").upper()
    # openpyxl 有时把单元格填充色存成 "FFRRGGBB"
    if re.fullmatch(r"[0-9A-Fa-f]{8}", s):
        return "#" + s[2:].upper()
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 Excel 导入科研配色方案")
    parser.add_argument("source", type=Path, help="科研配色方案 Excel 文件")
    args = parser.parse_args(argv)

    src = args.source.expanduser().resolve()
    if not src.is_file():
        print(f"找不到源文件：{src}")
        return 1

    wb = openpyxl.load_workbook(src, data_only=True)
    schemes: dict[str, dict] = {}

    for ws in wb.worksheets:
        for row in ws.iter_rows(min_row=2):
            name = row[0].value
            if name is None or not str(name).strip():
                continue
            name = str(name).strip()

            colors: list[str] = []
            for cell in row[1:]:
                hx = norm_hex(cell.value)
                if hx:
                    colors.append(hx)
                # 单元格没有文字但背景有填充色也收进来
                elif cell.fill is not None and cell.fill.fgColor is not None:
                    rgb = getattr(cell.fill.fgColor, "rgb", None)
                    hx2 = norm_hex(rgb) if isinstance(rgb, str) else None
                    if hx2 and hx2 not in ("#000000", "#FFFFFF"):
                        colors.append(hx2)

            # 去重但保留顺序
            seen = set()
            uniq = [c for c in colors if not (c in seen or seen.add(c))]
            if len(uniq) >= 2:
                schemes[name] = {"name": name, "colors": uniq,
                                 "source_sheet": ws.title}

    if not schemes:
        print("没有解析出任何配色方案，请检查 Excel 格式。")
        return 1

    payload = {
        "meta": {
            "source": src.name,
            "n_schemes": len(schemes),
            "description": "用户提供的科研配色方案，可在界面中整套使用或任选其中若干色。",
        },
        "schemes": schemes,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已写入 {OUT}")
    print(f"共 {len(schemes)} 套配色：")
    for k, v in schemes.items():
        print(f"  {k:8s} {len(v['colors']):2d} 色  {' '.join(v['colors'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
