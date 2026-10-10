@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist .venv (
    echo 第一次运行，正在创建虚拟环境...
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)

.venv\Scripts\python.exe -m streamlit run app.py
pause
