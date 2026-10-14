@echo off
rem First-run setup for OCR Pipeline Studio, run by launch.cmd when the venv is
rem missing or half-built. Everything here is deliberately visible and
rem deliberately re-runnable: each step is skipped if it is already done, so an
rem interrupted setup is fixed by double-clicking the shortcut again.
rem
rem Written with labels rather than nested if/else blocks throughout: cmd.exe
rem is unreliable about parentheses containing other parentheses, `choice` or
rem blank echoes, and this file has to work on someone else's machine first try.

setlocal
rem This file lives in launcher\; everything it acts on is one level up.
set "ROOT=%~dp0.."
rem pushd rather than cd: the project may be on a network share, and cd
rem refuses a \\server\share path where pushd maps it to a drive letter.
pushd "%ROOT%"

rem The venv is per PC, not in the project folder -- see launch.cmd.
set "LOCAL=%LOCALAPPDATA%\OCR Pipeline Studio"
if not exist "%LOCAL%" mkdir "%LOCAL%"
set "PYTHONPYCACHEPREFIX=%LOCAL%\pycache"
set "VENV=%LOCAL%\.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "STAMP=%VENV%\.setup-complete"
set "PADDLE_INDEX=https://www.paddlepaddle.org.cn/packages/stable/cu126/"
set "PYTHON_URL=https://www.python.org/downloads/release/python-3119/"

title OCR Pipeline Studio - setup

echo ============================================================
echo  OCR Pipeline Studio - first-time setup
echo ============================================================
echo.
echo This installs what the app needs. It downloads about 3 GB and
echo usually takes 10-20 minutes, mostly the OCR engine.
echo.
echo You only have to do this once. Afterwards the Desktop shortcut
echo opens the app straight away.
echo.

if not exist "%ROOT%\.venv\Scripts\python.exe" goto :nostray
echo Note: there is an old .venv inside the project folder, from before
echo setup moved to this PC's own storage. It is several gigabytes, no
echo longer used, and can be deleted once no one is still running the app
echo from it:
echo     %ROOT%\.venv
echo.
:nostray

rem ===============================================================
echo [1/4] Looking for Python 3.11...

rem 3.11 specifically: paddlepaddle and paddleocr publish no wheels for 3.12
rem or newer, so a newer Python cannot run the OCR engine at all.
call :findpython
if defined PY311 goto :havepython

echo.
echo      Python 3.11 is not installed. The app needs that exact version:
echo      the OCR engine publishes no packages for 3.12 or newer.
echo.

where winget >nul 2>&1
if errorlevel 1 goto :pythonbyhand

choice /c YN /n /m "      Install Python 3.11 now? [Y/N] "
if errorlevel 2 goto :pythondeclined

echo.
echo      Installing Python 3.11 through winget...
winget install --id Python.Python.3.11 --exact --scope user --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto :pythonbyhand

rem A fresh install may not be on this console's PATH yet.
call :findpython
if not defined PY311 goto :pythonreopen

:havepython
for /f "delims=" %%v in ('%PY311% --version 2^>^&1') do echo      Using %%v
echo.

rem ===============================================================
echo [2/4] Creating the environment...

if exist "%VPY%" goto :venvready
%PY311% -m venv "%VENV%"
if errorlevel 1 goto :venvfailed
echo      Done.
goto :venvdone
:venvready
echo      Already there, reusing it.
:venvdone
echo.

rem ===============================================================
echo [3/4] Installing the app's packages...
echo      (Flask, PyMuPDF, PaddleOCR and friends - a few minutes.)
echo.

"%VPY%" -m pip install --disable-pip-version-check -r "%ROOT%\requirements.txt"
if errorlevel 1 goto :pipfailed
echo.

rem ===============================================================
echo [4/4] Installing the OCR engine...
echo      (About 2.5 GB. This is the slow one.)
echo.

rem requirements.txt installs paddle on macOS only, because Windows wants the
rem CUDA build and PyPI does not carry it past 2.6.2. It comes from Paddle's
rem own index instead -- a CDN that is not always reachable, hence the
rem fallback to the slower PyPI wheel rather than failing the whole setup.
"%VPY%" -m pip install --disable-pip-version-check paddlepaddle-gpu==3.0.0 -i %PADDLE_INDEX%
if not errorlevel 1 goto :paddledone

echo.
echo      That index could not be reached. Falling back to the standard
echo      wheel, which works but runs OCR several times slower.
echo.
"%VPY%" -m pip install --disable-pip-version-check paddlepaddle==3.0.0
if errorlevel 1 goto :paddlefailed

:paddledone
echo.

rem Written last and only here: launch.cmd treats its absence as "unfinished".
echo Setup completed. Delete this file to make setup run again.> "%STAMP%"

echo ============================================================
echo  Setup finished.
echo ============================================================
echo.

rem ===============================================================
rem Ollama is a separate program, not a package, so setup only reports on it.
rem The app shows a live banner either way and runs fine without it.
where ollama >nul 2>&1
if errorlevel 1 goto :noollama

ollama list 2>nul | findstr /i "qwen2.5vl:3b" >nul
if not errorlevel 1 goto :ollamaready

echo Ollama is installed, but the vision model has not been pulled yet.
echo That download is about 3 GB.
echo.
choice /c YN /n /m "Pull qwen2.5vl:3b now? [Y/N] "
if errorlevel 2 goto :ollamaready
echo.
ollama pull qwen2.5vl:3b
echo.
goto :ollamaready

:noollama
echo The vision-model step also needs Ollama, which installs separately:
echo.
echo     https://ollama.com
echo.
echo The app runs without it; the OCR/VLM comparison does not.
echo.

:ollamaready
echo Starting OCR Pipeline Studio...
start "" "%VENV%\Scripts\pythonw.exe" "%ROOT%\app.py"

rem A breather so this window is not gone before the app's own window shows.
rem ping rather than timeout: timeout fails outright when stdin is redirected.
ping -n 4 127.0.0.1 >nul
rem Releases the drive letter pushd may have mapped for a network share.
popd
exit /b 0

rem ===============================================================
rem Failure exits. Each says what to do; all of them pause, because this
rem window closing with the error still in it is no help to anyone.

:findpython
rem Sets PY311 to something runnable, or leaves it empty.
set "PY311="
py -3.11 --version >nul 2>&1 && set "PY311=py -3.11"
if defined PY311 exit /b 0
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
exit /b 0

:pythonbyhand
echo.
echo      Install Python 3.11 from:
echo.
echo          %PYTHON_URL%
echo.
echo      Choose "Windows installer (64-bit)" and tick "Add python.exe to
echo      PATH". Then run this shortcut again.
goto :failed

:pythondeclined
echo.
echo      Nothing was installed. Run the shortcut again when you are ready.
goto :failed

:pythonreopen
echo.
echo      Python 3.11 installed, but this window cannot see it yet.
echo      Close this window and run the shortcut again.
goto :failed

:venvfailed
echo.
echo      Could not create the environment in:
echo          %VENV%
echo      Check there is disk space free - it needs about 5 GB.
goto :failed

:pipfailed
echo.
echo      Installing the packages failed. The message above says why; a
echo      dropped connection is the usual cause, and running the shortcut
echo      again carries on from where this stopped.
goto :failed

:paddlefailed
echo.
echo      The OCR engine could not be installed. Run the shortcut again to
echo      retry - nothing already downloaded is lost.
goto :failed

:failed
echo.
echo ------------------------------------------------------------
echo  Setup did not finish. Nothing is broken - running the
echo  shortcut again retries from where it stopped.
echo ------------------------------------------------------------
echo.
pause
popd
exit /b 1
