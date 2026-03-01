@echo off
REM ESG Investment Analytics - Setup Script for Windows
REM Automated project setup

echo ================================================================
echo   ESG Investment Analytics - Automated Setup
echo ================================================================
echo.

REM Check Python installation
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo Python %PYTHON_VERSION% found
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo pip upgraded
echo.

REM Install dependencies
echo Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo All dependencies installed
echo.

REM Create directory structure
echo Creating directory structure...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\models" mkdir data\models
if not exist "outputs" mkdir outputs
if not exist "notebooks" mkdir notebooks
echo Directory structure created
echo.

REM Create placeholder files
type nul > data\raw\.gitkeep
type nul > data\processed\.gitkeep
type nul > data\models\.gitkeep
type nul > outputs\.gitkeep
echo Git placeholder files created
echo.

REM Verify installation
echo Verifying installation...
python -c "import pandas, numpy, sklearn, streamlit, plotly, matplotlib, seaborn; print('All packages verified successfully!')"
if errorlevel 1 (
    echo ERROR: Package verification failed
    pause
    exit /b 1
)
echo.

echo ================================================================
echo   Setup Complete!
echo ================================================================
echo.
echo Next Steps:
echo   1. Activate virtual environment:
echo      venv\Scripts\activate
echo.
echo   2. Download datasets from Kaggle (optional):
echo      - ESG: https://www.kaggle.com/datasets/debashis74017/esg-scores-and-ratings
echo      - Place in: data\raw\
echo.
echo   3. Run the complete pipeline:
echo      python main.py
echo.
echo   4. Or launch the dashboard:
echo      streamlit run dashboard.py
echo.
echo Happy analyzing!
echo.
pause
