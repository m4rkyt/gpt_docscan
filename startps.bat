@echo off
set "VENV_DIR=C:\docintel\.venv"
cd /d C:\docintel
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '.\.venv\Scripts\Activate.ps1'; python demo_header.py C:\docintel\docs\barc1.pdf --out_dir C:\docintel\out --dpi 220 --debug"

pause
