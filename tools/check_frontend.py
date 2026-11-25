"""前端一致性检查：确认 JS 引用的 DOM id 在 HTML 中都存在。"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "amplicondesk" / "web"

# 这些元素由 JS 在运行时创建，不要求出现在 index.html 中
DYNAMIC_IDS = {
    "lightbox", "btnClearScope", "groupMapInput",
    # 科研成果配色选择器（JS 生成）
    "schemeSelect", "schemeSwatches", "btnUseWholeScheme", "btnUsePicked", "btnClearPicked",
    # 结论复制按钮（JS 生成）
    "btnCopyNar",
}

html = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
js = (WEB / "static" / "app.js").read_text(encoding="utf-8")
css = (WEB / "static" / "style.css").read_text(encoding="utf-8")

ids = set(re.findall(r'id="([^"]+)"', html))
used = set(re.findall(r'\$\("#([A-Za-z0-9_]+)"\)', js))
used |= set(re.findall(r'\$\$\("#([A-Za-z0-9_]+)', js))
selectors = set(re.findall(r'\$\("#([A-Za-z0-9_]+)"\)', js))

missing = sorted(used - ids - DYNAMIC_IDS)
unused = sorted(ids - used - {"toast"})

print(f"HTML id 数量    : {len(ids)}")
print(f"JS 引用的 id    : {len(used)}")
print(f"缺失的 id       : {missing or '无 ✓'}")
print(f"未被 JS 使用的  : {unused or '无'}")

# CSS 类名覆盖检查（跳过含模板表达式 / 动态拼接的类名）
cls_html = set(re.findall(r'class="([^"]+)"', html))
cls_js = set(re.findall(r'class="([^"]+)"', js))
flat = set()
for group in cls_html | cls_js:
    if "${" in group:
        continue
    flat |= {c for c in group.split() if c}
css_classes = set(re.findall(r"\.([a-zA-Z][\w-]*)", css))
missing_css = sorted(flat - css_classes)
print(f"HTML/JS 用到的类: {len(flat)}")
print(f"缺少样式的类    : {missing_css or '无 ✓'}")

# 模板占位符检查
placeholders = re.findall(r"\$\{([^}]+)\}", js)
print(f"JS 模板占位符   : {len(placeholders)} 处")

ok = not missing and not missing_css
print("\n结论:", "通过 ✅" if ok else "存在问题 ❌")
raise SystemExit(0 if ok else 1)
