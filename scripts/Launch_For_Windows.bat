@echo off
SETLOCAL ENABLEEXTENSIONS

REM --- 1. Determine where Python 3 is located ---
set "python_path="
for /f "tokens=*" %%A in ('where /R . python ^| findstr /i "Python\\ \\d\.\d\.\d"') do (
    set "python_path=%%A"
)

REM Check if Python was found
if "%python_path%" == "" (
    echo ERROR: Could not locate a Python 3 interpreter.
    echo This script requires Python 3 to create a virtual environment and install dependencies.
    echo For instructions on how to install Python for windows, have a look at sites like: 
    echo https://realpython.com/installing-python/#windows-how-to-install-python-from-the-microsoft-store
    echo Please ensure Python 3 is installed and available in your PATH. & goto :end
)

echo Found Python at: %python_path%

REM --- 2. Extract the pip path from the python executable ---
set "pip_path=%python_path:Python=Scripts\pip%"

REM Verify that the found pip works (optional but good for debugging)
if not exist "%pip_path%.exe" (
    echo ERROR: Could not find 'pip' at the expected location: %pip_path%
    echo This might indicate a non-standard Python installation. & goto :end
)

echo Found pip at: %pip_path%

REM --- 3. Create Virtual Environment and Install Packages ---
set "venv_dir=%~dp0\venv"
if not exist "%venv_dir%" md "%venv_dir%"

REM Activate the virtual environment in this script's context
call "%venv_dir%\Scripts\activate.bat" >nul 2>&1

REM Now, pip is available in the activated environment
echo Creating virtual environment and installing packages...
"%pip_path%" install --ignore-installed --no-cache-dir -r "requirements.txt" >nul 2>&1

if errorlevel 1 (
    echo ERROR: Failed to install packages from requirements.txt.
    echo Please check that a requirements.txt file exists and is correctly formatted. & goto :end
)

echo Packages installed successfully.

REM --- 4. Launch Jupyter Notebook ---
echo Starting Jupyter Notebook...
jupyter notebook "lessons\" --port=8888

:end
pause
exit /B 0
