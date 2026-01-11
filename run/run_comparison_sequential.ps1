# Sequential execution of all 5 DQN variant comparisons
# Uses all 15 cores per algorithm, runs one at a time
# More efficient if you want to maximize throughput per algorithm

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DQN Variants Sequential Comparison" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running 5 algorithms sequentially" -ForegroundColor Yellow
Write-Host "15 cores per algorithm" -ForegroundColor Yellow
Write-Host "39 seeds per algorithm" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

$experiments = @(
    @{Name="Standard DQN"; File="experiments/comparison_dqn.json"},
    @{Name="Optimistic Q-Init"; File="experiments/comparison_dqn_optimistic.json"},
    @{Name="Double DQN"; File="experiments/comparison_double_dqn.json"},
    @{Name="Learning Rate Decay"; File="experiments/comparison_dqn_lrdecay.json"},
    @{Name="N-Step Returns"; File="experiments/comparison_dqn_nstep.json"}
)

$totalStart = Get-Date

for ($i = 0; $i -lt $experiments.Count; $i++) {
    $exp = $experiments[$i]
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "[$($i+1)/5] Running: $($exp.Name)" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    
    $start = Get-Date
    
    .\run\local_windows.ps1 -JsonFiles $exp.File -PythonFile "src/mainjson.py" -MaxThreads 15
    
    $end = Get-Date
    $duration = $end - $start
    
    Write-Host "`n✓ Completed: $($exp.Name)" -ForegroundColor Green
    Write-Host "Duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
    
    # Show progress
    $completed = $i + 1
    $remaining = $experiments.Count - $completed
    Write-Host "Progress: $completed/$($experiments.Count) algorithms complete" -ForegroundColor Cyan
    Write-Host "Remaining: $remaining algorithms`n" -ForegroundColor Yellow
}

$totalEnd = Get-Date
$totalDuration = $totalEnd - $totalStart

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "ALL EXPERIMENTS COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total Duration: $($totalDuration.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
Write-Host "`nResults Summary:" -ForegroundColor Cyan

foreach ($exp in $experiments) {
    $algo = ($exp.File -split '/')[-1] -replace 'comparison_', '' -replace '.json', ''
    $resultCount = (Get-ChildItem "results/$algo/*.dw" -ErrorAction SilentlyContinue).Count
    Write-Host "  $($exp.Name): $resultCount/39 seeds" -ForegroundColor $(if ($resultCount -eq 39) {"Green"} else {"Yellow"})
}

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Process results:" -ForegroundColor White
Write-Host "   python analysis/process_data.py experiments/comparison_*.json" -ForegroundColor Gray
Write-Host "2. Generate plots:" -ForegroundColor White
Write-Host "   python analysis/learning_curve.py y returns auc experiments/comparison_*.json" -ForegroundColor Gray
Write-Host "3. Calculate metrics:" -ForegroundColor White
Write-Host "   python analysis/convergence_metrics.py experiments/comparison_*.json" -ForegroundColor Gray
