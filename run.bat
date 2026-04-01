@echo off
cd /d "%~dp0"
echo Starting NOIMA...
python main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application failed to start.
    echo Did you forget to install the required libraries?
    echo Try running: pip install -r requirements.txt
    echo.
)
pause
