@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

rem ---------------- Config ----------------
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

set "VENV_DIR=%PROJECT_DIR%.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
set "VENV_ACT=%VENV_DIR%\Scripts\activate.bat"
set "PORT=8000"
set "HOST=127.0.0.1"
set "UI_URL=http://%HOST%:%PORT%/"

set "DEFAULT_PY_PACKAGES=fastapi uvicorn[standard] python-dotenv azure-storage-blob pandas weasyprint jinja2"

echo.
echo [INFO] Project root: %PROJECT_DIR%

rem ---------------- System python check ----------------
where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found on PATH. Please install Python 3.8+ and re-run.
  pause
  exit /b 1
)
for /f "delims=" %%p in ('where python ^| find /i "python.exe"') do set "SYSTEM_PY=%%p"
echo [INFO] System python: %SYSTEM_PY%

rem ---------------- Create venv if missing ----------------
if not exist "%VENV_PY%" (
  echo [INFO] Creating virtualenv at "%VENV_DIR%" ...
  "%SYSTEM_PY%" -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Failed to create virtualenv with %SYSTEM_PY%.
    pause
    exit /b 1
  )
) else (
  echo [INFO] Virtualenv already exists at "%VENV_DIR%"
)

rem ---------------- Ensure pip in venv ----------------
if not exist "%VENV_PIP%" (
  echo [INFO] Bootstrapping pip in venv...
  "%VENV_PY%" -m ensurepip --upgrade
)

echo [INFO] Upgrading pip, setuptools, wheel in venv...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel

rem ---------------- Install Python dependencies ----------------
if exist "requirements.txt" (
  echo [INFO] Installing Python packages from requirements.txt ...
  "%VENV_PY%" -m pip install -r "requirements.txt"
  if errorlevel 1 echo [WARN] pip install -r requirements.txt returned non-zero.
) else (
  echo [INFO] No requirements.txt — installing defaults: %DEFAULT_PY_PACKAGES%
  "%VENV_PY%" -m pip install %DEFAULT_PY_PACKAGES%
  if errorlevel 1 echo [WARN] pip install default packages returned non-zero. (weasyprint may need additional system libs)
)

rem ---------------- Node/npm ----------------
where npm >nul 2>&1
if errorlevel 1 (
  echo [WARN] npm not found on PATH. Skipping npm install.
) else (
  if exist "package.json" (
    echo [INFO] package.json found — running npm install...
    npm install
    if errorlevel 1 echo [WARN] npm install returned non-zero.
  ) else (
    echo [INFO] No package.json — skipping npm install.
  )
)

rem ---------------- wkhtmltopdf detection (warn only) ----------------
set "WK_FOUND=0"
if exist "C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe" (
  set "WK_FOUND=1"
  set "WK_PATH=C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
)
if "%WK_FOUND%"=="0" if exist "C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe" (
  set "WK_FOUND=1"
  set "WK_PATH=C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe"
)
where wkhtmltopdf >nul 2>&1
if errorlevel 0 (
  for /f "delims=" %%x in ('where wkhtmltopdf') do if not defined WK_PATH set "WK_PATH=%%x" & set "WK_FOUND=1"
)
if "%WK_FOUND%"=="1" (
  echo [INFO] wkhtmltopdf detected: %WK_PATH%
) else (
  echo [WARN] wkhtmltopdf not found. WeasyPrint fallback will be used if available, otherwise HTML->PDF may fail.
)

rem ---------------- Create server runner cmd file ----------------
set "SRV_CMD=%PROJECT_DIR%server_run.cmd"
echo @echo off > "%SRV_CMD%"
echo cd /d "%PROJECT_DIR%" >> "%SRV_CMD%"
if exist "%VENV_ACT%" (
  echo call "%VENV_ACT%" >> "%SRV_CMD%"
) else (
  echo echo [WARN] venv activate script not found at "%VENV_ACT%" >> "%SRV_CMD%"
  echo echo [WARN] Activating venv failed; continuing with system python (may not be correct) >> "%SRV_CMD%"
)
echo echo [INFO] Running: python -m uvicorn api_server:app --reload --port %PORT% >> "%SRV_CMD%"
echo python -u -m uvicorn api_server:app --reload --port %PORT% >> "%SRV_CMD%"

rem ---------------- Start server in new persistent window ----------------
echo.
echo [INFO] Starting FastAPI server in a new window titled "PermitServer"...
rem Use start to open a visible window and run server_run.cmd inside it.
start "PermitServer" cmd /k "%SRV_CMD%"

rem ---------------- Wait a bit and open the UI ----------------
echo [INFO] Waiting up to 6 seconds for server to start...
timeout /t 6 >nul

echo [INFO] Opening browser at %UI_URL%
start "" "%UI_URL%"

echo.
echo [INFO] run_bot.bat finished. Server logs are in the 'PermitServer' window (or check server_run.cmd output).
echo [TIP] If browser shows an error, inspect the 'PermitServer' window for diagnostic output.
pause
ENDLOCAL
