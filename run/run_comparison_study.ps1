# Master script to run all 5 DQN variant comparisons in parallel
# Each algorithm gets 39 seeds, 15 cores total, 13 hour runtime
# Total: 195 seed runs (39 seeds × 5 models)

param(
    [int]$CoresPerExperiment = 3
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DQN Variants Comparison Study" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "5 Algorithms × 39 Seeds = 195 Total Runs" -ForegroundColor Yellow
Write-Host "500,000 Episodes per Run" -ForegroundColor Yellow
Write-Host "Estimated Duration: 13 hours" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

$experiments = @(
    "experiments/comparison_dqn.json",
    "experiments/comparison_dqn_optimistic.json",
    "experiments/comparison_double_dqn.json",
    "experiments/comparison_dqn_lrdecay.json",
    "experiments/comparison_dqn_nstep.json"
)

Write-Host "Starting experiments in parallel..." -ForegroundColor Green
Write-Host "Press Ctrl+C in any window to stop that experiment`n" -ForegroundColor Yellow

$jobs = @()

foreach ($exp in $experiments) {
    $algoName = ($exp -split '/')[-1] -replace 'comparison_', '' -replace '.json', ''
    Write-Host "Launching: $algoName ($CoresPerExperiment cores)" -ForegroundColor Cyan
    
    # Start each experiment in a new PowerShell window
    $command = ".\run\local_windows.ps1 -JsonFiles '$exp' -PythonFile 'src/mainjson.py' -MaxThreads $CoresPerExperiment"
    
    $job = Start-Process -FilePath "powershell.exe" `
        -ArgumentList "-NoExit", "-Command", "& { Write-Host 'Running: $algoName' -ForegroundColor Cyan; $command }" `
        -PassThru
    
    $jobs += $job
    Start-Sleep -Seconds 2
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "All experiments launched!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Monitor progress in separate windows" -ForegroundColor Yellow
Write-Host "`nTo monitor results:"
Write-Host "  ls results/*/*.dw | Measure-Object | Select-Object -ExpandProperty Count" -ForegroundColor Gray
Write-Host "`nExpected final count: 195 files`n" -ForegroundColor Gray
