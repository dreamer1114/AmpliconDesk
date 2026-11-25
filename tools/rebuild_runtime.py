"""重建/刷新本文件夹内置的 Python 运行时。

适用场景：你在本机升级了依赖，想把 AmpliconDesk/runtime 重新打一遍，
       或者把这个文件夹复制到新机器后想重建运行时。

做法（默认，适配开启了 Smart App Control 的机器）：
    1. 下载与本机同版本的 Python 嵌入式包，解压到 runtime/
    2. 打开 site-packages 支持
    3. **复制本机已安装并正常运行的依赖**（而不是 pip 下载）
    4. 解除下载文件的网络来源标记

之所以默认「复制本机依赖」而不是 `pip install`：Windows 的 Smart App Control
会拦截新下载的未签名扩展模块（报错"应用程序控制策略已阻止此文件"），
而本机已装好的依赖是被放行的。

用法：
    runtime\\python.exe tools\\rebuild_runtime.py             # 复制本机依赖（推荐）
    runtime\\python.exe tools\\rebuild_runtime.py --pip       # 改为在线 pip 安装
    runtime\\python.exe tools\\rebuild_runtime.py --verify    # 完成后跑验收
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # AmpliconDesk/
RUNTIME = ROOT / "runtime"
CACHE = ROOT / "_build"

PY_VERSION = "3.14.5"
PY_URL = f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip"
GETPIP_URL = "https://bootstrap.pypa.io/get-pip.py"

SYSTEM_SITE_CANDIDATES = [
    Path.home() / "AppData" / "Roaming" / "Python" / "Python314" / "site-packages",
    Path.home() / "AppData" / "Roaming" / "Python" / "Python313" / "site-packages",
    Path.home() / "AppData" / "Roaming" / "Python" / "Python312" / "site-packages",
]

REQUIREMENTS = [
    "flask>=3.0", "pandas>=2.2", "numpy>=1.26,<3", "scipy>=1.11",
    "scikit-learn>=1.4", "seaborn>=0.13", "matplotlib>=3.8", "openpyxl>=3.1",
    "xlrd>=2.0", "umap-learn>=0.5", "matplotlib-venn>=0.11",
]


def log(msg: str) -> None:
    print(f"[rebuild_runtime] {msg}", flush=True)


def _looks_complete(path: Path) -> bool:
    try:
        if path.stat().st_size < 1024:
            return False
    except OSError:
        return False
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as z:
                return z.testzip() is None
        except Exception:
            return False
    return True


def download(url: str, dest: Path, retries: int = 4) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and _looks_complete(dest):
        log(f"已存在且完整，跳过下载：{dest.name}")
        return dest
    for attempt in range(1, retries + 1):
        if dest.exists():
            dest.unlink(missing_ok=True)
        try:
            log(f"下载 {url}（第 {attempt}/{retries} 次）")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length") or 0)
                with open(dest, "wb") as f:
                    while True:
                        chunk = r.read(256 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            size = dest.stat().st_size
            if size > 1024 and (not total or size >= total * 0.99) and _looks_complete(dest):
                log(f"完成：{dest.name}（{size / 1024 / 1024:.1f} MB）")
                return dest
            log(f"文件不完整（{size}/{total}），重试")
        except Exception as exc:
            log(f"下载失败：{type(exc).__name__}: {exc}")
        time.sleep(2 * attempt)
    raise RuntimeError(f"多次尝试后仍无法下载 {url}")


def extract_zip(zpath: Path, target: Path) -> None:
    log(f"解压 {zpath.name}")
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"压缩包损坏：{bad}")
        z.extractall(target)


def find_system_site() -> Path | None:
    for p in SYSTEM_SITE_CANDIDATES:
        if p.exists() and any(p.iterdir()):
            return p
    return None


def prepare_runtime(from_system: bool) -> Path:
    """下载嵌入式 Python 并解压，开启 site-packages。"""
    if RUNTIME.exists():
        log("清空旧的 runtime")
        shutil.rmtree(RUNTIME, ignore_errors=True)
    RUNTIME.mkdir(parents=True)

    zpath = download(PY_URL, CACHE / Path(PY_URL).name)
    extract_zip(zpath, RUNTIME)

    pth = next(RUNTIME.glob("python*._pth"), None)
    if pth is None:
        raise RuntimeError("解压后找不到 python*._pth")
    lines = []
    for ln in pth.read_text(encoding="utf-8").splitlines():
        lines.append("import site" if ln.strip().startswith("#import site") else ln)
    for extra in ("Lib\\site-packages", ".."):
        if extra not in lines:
            lines.append(extra)
    pth.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"已配置 {pth.name}")
    return RUNTIME / "python.exe"


def install_from_system(python_exe: Path) -> None:
    src = find_system_site()
    if src is None:
        raise RuntimeError(
            "找不到本机 site-packages，请改用 --pip 方式安装依赖"
        )
    dst = RUNTIME / "Lib" / "site-packages"
    dst.mkdir(parents=True, exist_ok=True)
    log(f"复制本机依赖 {src} → {dst}（约 900 MB，需要几分钟）")
    n = 0
    for item in src.iterdir():
        if item.name == "__pycache__" or item.name.startswith("~"):
            continue
        target = dst / item.name
        try:
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(item, target)
            n += 1
        except Exception as exc:
            log(f"  跳过 {item.name}：{exc}")
    log(f"已复制 {n} 个条目")


def install_with_pip(python_exe: Path) -> None:
    getpip = download(GETPIP_URL, CACHE / "get-pip.py")
    log("安装 pip")
    subprocess.run([str(python_exe), str(getpip), "--no-warn-script-location"],
                   check=True, cwd=str(ROOT))
    log("安装 setuptools / wheel")
    subprocess.run([str(python_exe), "-m", "pip", "install", "--no-warn-script-location",
                    "--no-cache-dir", "--upgrade", "pip", "setuptools", "wheel"],
                   check=True, cwd=str(ROOT))
    log(f"安装 {len(REQUIREMENTS)} 个依赖（这一步最慢）")
    subprocess.run([str(python_exe), "-m", "pip", "install", "--no-warn-script-location",
                    "--no-cache-dir", "--prefer-binary", *REQUIREMENTS],
                   check=True, cwd=str(ROOT))


def unblock_tree(root: Path) -> None:
    """解除下载文件的网络来源标记，避免被应用程序控制策略拦截。"""
    if not root.exists():
        return
    log("解除下载文件的网络来源标记")
    ps = (
        "Get-ChildItem -LiteralPath $args[0] -Recurse -File "
        "-Include *.pyd,*.dll,*.exe -ErrorAction SilentlyContinue | "
        "Unblock-File -ErrorAction SilentlyContinue; "
        "Get-ChildItem -LiteralPath $args[0] -Recurse -File -ErrorAction SilentlyContinue | "
        "ForEach-Object { Remove-Item -LiteralPath $_.FullName -Stream Zone.Identifier "
        "-ErrorAction SilentlyContinue }"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps, str(root)],
                       capture_output=True, text=True, timeout=900)
    except Exception as exc:
        log(f"Unblock 步骤异常（可忽略）：{exc}")


def verify(python_exe: Path) -> bool:
    log("验证运行时")
    code = (
        "import sys;"
        "import flask,pandas,numpy,scipy,sklearn,seaborn,matplotlib,openpyxl;"
        "import umap,matplotlib_venn;"
        "sys.path.insert(0, r'{root}');"
        "import amplicondesk;"
        "from amplicondesk.style import RESEARCH_PALETTES;"
        "print('OK', sys.version.split()[0], len(RESEARCH_PALETTES))"
    ).format(root=str(ROOT))
    r = subprocess.run([str(python_exe), "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = "OK" in (r.stdout or "")
    log(("验证通过：" if ok else "验证失败：") + (r.stdout or "").strip()[:160])
    if not ok:
        log("stderr: " + (r.stderr or "").strip()[:600])
    return ok


def dir_size(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024


def main() -> int:
    ap = argparse.ArgumentParser(description="重建 AmpliconDesk 的内置 Python 运行时")
    ap.add_argument("--pip", action="store_true", help="用 pip 在线安装依赖（默认复制本机依赖）")
    ap.add_argument("--verify", action="store_true", help="完成后跑一次验收测试")
    args = ap.parse_args()

    python_exe = prepare_runtime(from_system=not args.pip)
    if args.pip:
        install_with_pip(python_exe)
    else:
        install_from_system(python_exe)
    unblock_tree(RUNTIME)

    ok = verify(python_exe)
    if args.verify and ok:
        subprocess.run([str(python_exe), str(ROOT / "tools" / "verify_portable.py")],
                       cwd=str(ROOT))

    print()
    print("=" * 62)
    print(f"运行时已重建：{RUNTIME}")
    print(f"大小：{dir_size(RUNTIME):.0f} MB    验证：{'通过 ✅' if ok else '失败 ❌'}")
    print("=" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
