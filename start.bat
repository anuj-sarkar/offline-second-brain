@echo off
REM start.bat - one-click launcher for Offline Second Brain
REM Double-click this file, or run `.\start.bat` from a terminal in the project root.

echo Starting Offline Second Brain...
echo.

REM --- Backend (FastAPI) in its own window ---
start "Second Brain - Backend" cmd /k "cd /d %~dp0 && venv\Scripts\activate && uvicorn backend.main:app --reload --port 8000"

REM Give the backend a few seconds head start before the frontend tries to talk to it
timeout /t 4 /nobreak >nul

REM --- Frontend (Vite dev server) in its own window ---
start "Second Brain - Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

REM Give Vite a moment to boot, then open the browser
timeout /t 4 /nobreak >nul
start http://localhost:5173

echo.
echo Backend and frontend are starting in two separate windows.
echo Ollama should already be running in the background (check your system tray).
echo.
echo To stop everything: close both of the new windows (or Ctrl+C in each).
echo This window can be closed - it's done its job.
pause
