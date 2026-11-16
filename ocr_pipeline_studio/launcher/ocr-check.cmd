@echo off
rem Diagnoses a PC where OCR finds no text on any page. Drag a thesis PDF onto
rem this file; it reads page 1 three ways and saves ocr-check-report.txt to the
rem Desktop. See pipeline\ocr_check.py.

setlocal
set "PY=%LOCALAPPDATA%\OCR Pipeline Studio\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo The app is not set up on this PC yet - run the Desktop shortcut first.
    pause
    exit /b 1
)
if "%~1"=="" (
    echo Drag a thesis PDF onto this file to check it.
    pause
    exit /b 1
)

pushd "%~dp0..\pipeline"
"%PY%" ocr_check.py "%~1" %2
popd
echo.
pause
