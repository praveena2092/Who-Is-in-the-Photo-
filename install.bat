@echo off
REM One-time setup (needs internet). After this, start.bat works offline.
cd /d "%~dp0"
python -m venv .venv || goto :error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r backend\requirements.txt || goto :error
python scripts\download_models.py || goto :error
cd frontend
call npm install || goto :error
call npm run build || goto :error
cd ..
echo.
echo All set. Double-click start.bat
pause
exit /b 0
:error
echo Setup failed. See the message above.
pause
exit /b 1
