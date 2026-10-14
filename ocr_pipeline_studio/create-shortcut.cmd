@echo off
rem Puts an "OCR Pipeline Studio" shortcut on the Desktop, pointing at
rem launcher\launch.cmd. Run once; after that use the shortcut.
rem
rem This one file stays in the project folder rather than in launcher\, because
rem it is the only one a person has to find and double-click themselves.
rem
rem A .cmd wrapping PowerShell rather than a .ps1, because Windows will not run
rem a .ps1 on double-click under the default execution policy -- and
rem double-clicking is the whole point of this file.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
 "$link = (New-Object -ComObject WScript.Shell).CreateShortcut(([Environment]::GetFolderPath('Desktop')) + '\OCR Pipeline Studio.lnk');" ^
 "$link.TargetPath = '%~dp0launcher\launch.cmd';" ^
 "$link.WorkingDirectory = (Resolve-Path '%~dp0').ProviderPath;" ^
 "$link.Description = 'OCR Pipeline Studio';" ^
 "$link.WindowStyle = 7;" ^
 "$link.Save()"

if errorlevel 1 (
    echo Could not create the shortcut.
    pause
    exit /b 1
)

echo Done. "OCR Pipeline Studio" is on your Desktop - double-click it to start.
echo.
pause
