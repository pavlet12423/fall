@echo off
rem ===============================================================
rem  run_probe.bat - launch probe.py on Windows.
rem  Creates/uses a virtualenv, installs numpy + opencv if needed,
rem  then starts the local server. Just double-click this file.
rem  (ASCII-only on purpose: batch files break on non-ASCII codepages)
rem ===============================================================
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

echo.
echo Project folder: %CD%
echo.

if not exist "probe.py" (
    echo ERROR: probe.py not found next to this file.
    echo Put run_probe.bat in the same folder as probe.py and data\
    goto :end
)

if not exist "data\frames\*.png" (
    echo ERROR: no frames in data\frames
    echo Download the repo archive and put the data folder next to probe.py
    goto :end
)

rem --- 1. find or create a virtualenv ------------------------------
set PY=.venv\Scripts\python.exe
if exist "..\venv\Scripts\python.exe" set PY=..\venv\Scripts\python.exe
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
    echo [1/4] Creating virtualenv .venv ...
    python -m venv .venv
    if errorlevel 1 goto :nopython
) else (
    echo [1/4] Using virtualenv: %PY%
)

rem --- 2. dependencies ---------------------------------------------
echo [2/4] Checking numpy and opencv ...
%PY% -c "import numpy, cv2" >nul 2>&1
if not errorlevel 1 (
    echo       Already installed.
    goto :run
)

echo [3/4] Downloading numpy and opencv, about 60 MB ...
%PY% -m pip install --upgrade pip
if errorlevel 1 goto :pipfail

%PY% -m pip install numpy opencv-python-headless
if errorlevel 1 goto :pipfail

rem --- 3. run -------------------------------------------------------
:run
echo [4/4] Starting probe.py ...
echo.
echo When you see "otkroyte http://127.0.0.1:8765" open that address
echo in your browser. Press Ctrl+C to stop.
echo.
%PY% probe.py --frames data\frames --fps 25
goto :end

:nopython
echo.
echo ERROR: python not found.
echo Install Python from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during setup.
goto :end

:pipfail
echo.
echo ERROR: could not download the packages.
echo Behind a proxy, run manually:
echo   %PY% -m pip install --proxy http://LOGIN:PASS@HOST:PORT numpy opencv-python-headless
echo Otherwise try again with:
echo   %PY% -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org numpy opencv-python-headless
goto :end

:end
echo.
pause
