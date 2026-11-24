@echo off
REM Starts the app on http://127.0.0.1:8000 and opens the browser. Works offline.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8000"
cd backend
python -W ignore -m uvicorn app.main:app --host 127.0.0.1 --port 8000
