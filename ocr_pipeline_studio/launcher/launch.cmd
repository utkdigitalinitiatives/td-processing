@echo off
rem Starts OCR Pipeline Studio. This is what the desktop shortcut points at.
rem
rem Lives in launcher\, so everything it acts on is one level up.
rem
rem There is no "activate" step: activating a venv only puts its Scripts folder
rem on PATH, so calling the venv's own pythonw.exe is the same thing. pythonw
rem rather than python because it has no console window attached -- output goes
rem to workdir\launch.log instead, see app.py.
rem
rem Nothing installed yet? setup.cmd is run instead, so the desktop shortcut is
rem the only thing anyone has to touch.

set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\pythonw.exe"
set "STAMP=%ROOT%\.venv\.setup-complete"

if not exist "%PY%" goto :setup

rem The stamp is written by setup.cmd as its last act, so a venv without one is
rem a setup that died part way -- which looks identical to a finished one until
rem the app fails on a missing package.
if not exist "%STAMP%" goto :setup

rem start returns straight away, so this window closes and leaves the app.
start "" "%PY%" "%ROOT%\app.py"
exit /b 0

rem At top level rather than nested inside the checks above: cmd.exe is
rem unreliable about if/else inside a parenthesised block.
:setup
rem Setup takes ten minutes or more and is worth watching, while this window is
rem minimized by the shortcut -- so it gets a normal window of its own.
start "OCR Pipeline Studio setup" cmd /c "%~dp0setup.cmd"
exit /b 0
