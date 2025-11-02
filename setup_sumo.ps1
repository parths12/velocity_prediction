# PowerShell script to set up SUMO environment for vehicle velocity prediction

Write-Host "Setting up SUMO environment for vehicle velocity prediction..." -ForegroundColor Green
Write-Host ""

# Check if SUMO_HOME is already set
if ($env:SUMO_HOME) {
    Write-Host "SUMO_HOME is already set to: $env:SUMO_HOME" -ForegroundColor Cyan
    Write-Host ""
} else {
    # Try to find SUMO installation in common locations
    $possiblePaths = @(
        "C:\Program Files (x86)\Eclipse SUMO",
        "C:\Program Files\Eclipse SUMO"
    )

    $sumoPath = $null
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            $sumoPath = $path
            Write-Host "Found SUMO installation at: $sumoPath" -ForegroundColor Cyan
            break
        }
    }

    # If not found, ask user for path
    if (-not $sumoPath) {
        Write-Host "Could not find SUMO installation automatically." -ForegroundColor Yellow
        Write-Host ""
        $sumoPath = Read-Host "Please enter the path to your SUMO installation (e.g., C:\Program Files (x86)\Eclipse SUMO)"

        if (-not (Test-Path $sumoPath)) {
            Write-Host "Error: The specified path does not exist." -ForegroundColor Red
            exit
        }
    }

    # Set SUMO_HOME for current session
    $env:SUMO_HOME = $sumoPath
    Write-Host ""
    Write-Host "SUMO_HOME has been set to: $env:SUMO_HOME for the current session" -ForegroundColor Green
    
    # Ask if user wants to set it permanently
    Write-Host ""
    $setPermanently = Read-Host "Do you want to set SUMO_HOME permanently? (y/n)"
    
    if ($setPermanently -eq "y" -or $setPermanently -eq "Y") {
        try {
            [System.Environment]::SetEnvironmentVariable("SUMO_HOME", $sumoPath, "Machine")
            Write-Host "SUMO_HOME has been set permanently." -ForegroundColor Green
        } catch {
            Write-Host "Error: Failed to set environment variable permanently. Try running as Administrator." -ForegroundColor Red
            Write-Host "You can set it manually by running the following command as Administrator:" -ForegroundColor Yellow
            Write-Host "[System.Environment]::SetEnvironmentVariable('SUMO_HOME', '$sumoPath', 'Machine')" -ForegroundColor Yellow
        }
    }
}

# Check if SUMO binaries exist
if (-not (Test-Path "$env:SUMO_HOME\bin\sumo.exe")) {
    Write-Host "Error: SUMO executable not found at $env:SUMO_HOME\bin\sumo.exe" -ForegroundColor Red
    Write-Host "Please make sure SUMO is installed correctly." -ForegroundColor Red
    exit
}

Write-Host "SUMO installation verified successfully." -ForegroundColor Green
Write-Host ""
Write-Host "You may need to restart your PowerShell for the changes to take effect." -ForegroundColor Yellow
Write-Host ""
Write-Host "To run the vehicle velocity prediction pipeline:" -ForegroundColor Cyan
Write-Host "python main.py --sumo-config data/sumo_config.sumocfg --ego-id ego --gui" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to continue" 