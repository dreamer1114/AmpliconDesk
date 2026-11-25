@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo.
echo   ==================================================
echo     AmpliconDesk   16S 扩增子分析工作台
echo   ==================================================
echo.

rem ---- 优先使用本文件夹内置的 Python，没有则回退到系统 Python ----
set "PYEXE="
if exist "%~dp0runtime\python.exe" (
  set "PYEXE=%~dp0runtime\python.exe"
  set "PYTHONHOME=%~dp0runtime"
  set "PATH=%~dp0runtime;%~dp0runtime\Scripts;%PATH%"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo   [错误] 既没有内置运行时 ^(runtime\python.exe^)，也找不到系统 python。
    echo   请把整个 AmpliconDesk 文件夹完整复制过来，或安装 Python 3.10+。
    echo.
    pause
    exit /b 1
  )
  set "PYEXE=python"
)

rem ---- 环境自检 ----
"%PYEXE%" -c "import flask, pandas, numpy, scipy, sklearn, seaborn, matplotlib, openpyxl" >nul 2>nul
if errorlevel 1 (
  echo   [提示] 缺少依赖，正在尝试安装...
  "%PYEXE%" -m pip install -r "%~dp0requirements.txt"
  if errorlevel 1 (
    echo   [错误] 依赖安装失败，请手动执行：
    echo          "%PYEXE%" -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
  )
)

echo   正在启动，浏览器会自动打开...
echo   程序目录 : %~dp0
echo   结果目录 : %~dp0output
echo   按 Ctrl+C 退出
echo.

"%PYEXE%" "%~dp0run.py" %*

echo.
echo   服务已停止。
pause
