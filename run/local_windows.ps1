# PowerShell script to run experiments in parallel on Windows
# Usage: .\run\local_windows.ps1 -JsonFiles "experiments/production_dqn.json" -PythonFile "src/mainjson.py" -MaxThreads 7

param(
    [Parameter(Mandatory=$true)]
    [string[]]$JsonFiles,
    
    [Parameter(Mandatory=$true)]
    [string]$PythonFile,
    
    [int]$MaxThreads = -1,
    
    [int]$Start = 0,
    
    [int]$End = -1,
    
    [switch]$Overwrite = $false,
    
    [string]$CondaEnv = "deep_reinforcement_learning"
)

# Activate conda environment
Write-Host "Activating conda environment: $CondaEnv" -ForegroundColor Yellow
conda activate $CondaEnv
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to activate conda environment: $CondaEnv" -ForegroundColor Red
    Write-Host "Please ensure the environment exists: conda env list" -ForegroundColor Yellow
    exit 1
}

# Add the current directory to Python path
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

# Collect all pending experiments
$allCommands = @()

foreach ($jsonFile in $JsonFiles) {
    Write-Host "Processing: $jsonFile" -ForegroundColor Cyan
    
    # Get pending experiments using helper script
    $overwriteStr = if ($Overwrite.IsPresent) { "True" } else { "False" }
    $experimentInfo = python run\get_pending.py $jsonFile $overwriteStr $Start $End
    
    if ($LASTEXITCODE -eq 0 -and $experimentInfo) {
        $pendingIndices = $experimentInfo.Split(',') | Where-Object { $_ -ne '' }
        Write-Host "Pending experiments for ${jsonFile}: $($pendingIndices.Count)" -ForegroundColor Green
        
        foreach ($idx in $pendingIndices) {
            $cmd = @{
                Command = "python $PythonFile $jsonFile $idx"
                JsonFile = $jsonFile
                Index = $idx
            }
            $allCommands += $cmd
        }
    }
}

if ($allCommands.Count -eq 0) {
    Write-Host "No pending experiments found!" -ForegroundColor Yellow
    exit 0
}

Write-Host "`nTotal commands to execute: $($allCommands.Count)" -ForegroundColor Green

# Determine thread limit
if ($MaxThreads -le 0) {
    $MaxThreads = [Environment]::ProcessorCount
    Write-Host "Using all available cores: $MaxThreads" -ForegroundColor Yellow
} else {
    Write-Host "Using $MaxThreads parallel threads" -ForegroundColor Yellow
}

Write-Host "`nStarting parallel execution..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to cancel`n" -ForegroundColor Yellow

# Execute commands in parallel using PowerShell Jobs (PowerShell 5.1 compatible)
$startTime = Get-Date
$completedCount = 0
$failedCount = 0

$jobIndex = 0
foreach ($cmd in $allCommands) {
    # Wait if we've hit the max thread limit
    while (@(Get-Job -State Running).Count -ge $MaxThreads) {
        Start-Sleep -Milliseconds 500
        
        # Check for completed jobs
        $completed = Get-Job -State Completed
        foreach ($job in $completed) {
            $result = Receive-Job $job
            if ($result -match "Finished") {
                Write-Host "[COMPLETED $($completedCount + 1)/$($allCommands.Count)] $($job.Name)" -ForegroundColor Green
                $completedCount++
            }
            else {
                Write-Host "[FAILED] $($job.Name)" -ForegroundColor Red
                Write-Host $result -ForegroundColor DarkGray
                $failedCount++
            }
            Remove-Job $job
        }
        
        # Check for failed jobs
        $failed = Get-Job -State Failed
        foreach ($job in $failed) {
            Write-Host "[FAILED] $($job.Name)" -ForegroundColor Red
            $failedCount++
            Remove-Job $job
        }
    }
    
    # Start new job
    $jobName = "$($cmd.JsonFile)-$($cmd.Index)"
    Write-Host "[START] $jobName" -ForegroundColor Cyan
    
    $job = Start-Job -Name $jobName -ScriptBlock {
        param($pythonFile, $jsonFile, $idx, $condaEnv, $workDir)
        
        Set-Location $workDir
        $env:PYTHONPATH = "$workDir;$env:PYTHONPATH"
        
        # Run with conda environment
        $command = "conda activate $condaEnv 2>&1 | Out-Null; python $pythonFile $jsonFile $idx"
        $output = powershell -Command $command 2>&1
        $output
    } -ArgumentList $PythonFile, $cmd.JsonFile, $cmd.Index, $CondaEnv, $PWD
    
    $jobIndex++
}

# Wait for remaining jobs to complete
Write-Host "`nWaiting for remaining jobs to complete..." -ForegroundColor Yellow

while (@(Get-Job -State Running).Count -gt 0) {
    Start-Sleep -Seconds 2
    
    # Check for completed jobs
    $completed = Get-Job -State Completed
    foreach ($job in $completed) {
        $result = Receive-Job $job
        if ($result -match "Finished") {
            Write-Host "[COMPLETED $($completedCount + 1)/$($allCommands.Count)] $($job.Name)" -ForegroundColor Green
            $completedCount++
        }
        else {
            Write-Host "[FAILED] $($job.Name)" -ForegroundColor Red
            Write-Host $result -ForegroundColor DarkGray
            $failedCount++
        }
        Remove-Job $job
    }
    
    # Check for failed jobs
    $failed = Get-Job -State Failed
    foreach ($job in $failed) {
        Write-Host "[FAILED] $($job.Name)" -ForegroundColor Red
        $failedCount++
        Remove-Job $job
    }
}

# Final cleanup
Get-Job | Remove-Job -Force

$endTime = Get-Date
$duration = $endTime - $startTime

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "EXECUTION SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total experiments: $($allCommands.Count)"
Write-Host "Successful: $completedCount" -ForegroundColor Green
Write-Host "Failed: $failedCount" -ForegroundColor Red
Write-Host "Duration: $($duration.ToString('hh\:mm\:ss'))"
Write-Host "========================================`n" -ForegroundColor Cyan
