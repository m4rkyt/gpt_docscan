@echo off
echo ===============================
echo Starting Local Chatbot Environment
echo ===============================

REM Change to project directory
cd /d C:\docintel
REM Activate virtual environment
call .venv\Scripts\activate.bat

echo.
echo ✅ Virtual environment activated
echo 📂 Current directory:
cd
echo.
cmd