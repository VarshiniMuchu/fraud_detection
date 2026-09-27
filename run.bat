@echo off
REM ============================================================
REM  run.bat - Fraud Detection Project Runner
REM  Usage: run.bat src\autoencoder.py
REM         run.bat src\experiments.py
REM         run.bat main.py
REM ============================================================

SET PYTHON=C:\Users\VARSHINI\AppData\Local\Programs\Python\Python313\python.exe

IF "%~1"=="" (
    echo [!] Usage: run.bat ^<script^>
    echo     Example: run.bat src\autoencoder.py
    pause
    exit /b 1
)

echo [*] Running: %PYTHON% %*
echo.
"%PYTHON%" %*
pause
