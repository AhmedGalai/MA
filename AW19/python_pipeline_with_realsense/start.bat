@echo off
REM AW19 System Startup Script (Windows)
REM Uses specific Python environment

set PYTHON=C:\Users\Lenovo\.guv\envs\masterarbeit\Scripts\python.exe

echo ========================================
echo AW19 - AVP Vision Processing System
echo ========================================
echo.
echo Python: %PYTHON%
echo.

REM Check if Python exists
if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    echo Please update the PYTHON variable in this script.
    echo.
    pause
    exit /b 1
)

echo Starting system components...
echo.

echo [1/4] Starting Main API (port 5000)...
echo     (This may take 10-30 seconds to load depth model)
start "AW19 - Main API" cmd /k "%PYTHON% main_api.py"
timeout /t 2 /nobreak >nul

REM Select default model when API is ready (can be overwritten later)
echo [2/4] Selecting default model (cube.ply) when ready...
"%PYTHON%" select_default_model.py --url http://localhost:5000 --model cube.ply --timeout 180

echo [3/4] Starting Screen Capture UI...
start "AW19 - Screen Capture" cmd /k "%PYTHON% screen_capture.py"
timeout /t 2 /nobreak >nul

echo [4/4] Starting Debug Viewer...
start "AW19 - Debug Viewer" cmd /k "%PYTHON% tk_debugging_client.py"

echo.
echo ========================================
echo All components started successfully!
echo ========================================
echo.
echo Services:
echo   - Main API:        http://localhost:5000
echo   - Screen Capture:  UI window
echo   - Debug Viewer:    UI window
echo.
echo To stop: Close each command window
echo.
echo Press any key to exit this launcher...
pause >nul
