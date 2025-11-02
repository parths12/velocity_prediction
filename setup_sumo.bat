@echo off
echo Setting up SUMO environment for vehicle velocity prediction...
echo.

REM Check if SUMO_HOME is already set
if defined SUMO_HOME (
    echo SUMO_HOME is already set to: %SUMO_HOME%
    echo.
    goto :check_sumo
)

REM Try to find SUMO installation in common locations
set "POSSIBLE_SUMO_PATH=C:\Program Files (x86)\Eclipse SUMO"
if exist "%POSSIBLE_SUMO_PATH%" (
    set "SUMO_HOME=%POSSIBLE_SUMO_PATH%"
    echo Found SUMO installation at: %SUMO_HOME%
    goto :set_env
)

set "POSSIBLE_SUMO_PATH=C:\Program Files\Eclipse SUMO"
if exist "%POSSIBLE_SUMO_PATH%" (
    set "SUMO_HOME=%POSSIBLE_SUMO_PATH%"
    echo Found SUMO installation at: %SUMO_HOME%
    goto :set_env
)

REM Ask user for SUMO installation path
echo Could not find SUMO installation automatically.
echo.
set /p SUMO_PATH=Please enter the path to your SUMO installation (e.g., C:\Program Files (x86)\Eclipse SUMO): 

if not exist "%SUMO_PATH%" (
    echo Error: The specified path does not exist.
    goto :eof
)

set "SUMO_HOME=%SUMO_PATH%"

:set_env
REM Set SUMO_HOME for current session
setx SUMO_HOME "%SUMO_HOME%" /M
echo.
echo SUMO_HOME has been set to: %SUMO_HOME%
echo.

:check_sumo
REM Check if SUMO binaries exist
if not exist "%SUMO_HOME%\bin\sumo.exe" (
    echo Error: SUMO executable not found at %SUMO_HOME%\bin\sumo.exe
    echo Please make sure SUMO is installed correctly.
    goto :eof
)

echo SUMO installation verified successfully.
echo.
echo You may need to restart your command prompt or PowerShell for the changes to take effect.
echo.
echo To run the vehicle velocity prediction pipeline:
echo python main.py --sumo-config data/sumo_config.sumocfg --ego-id ego --gui
echo.

pause 