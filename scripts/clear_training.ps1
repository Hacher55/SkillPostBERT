#Requires -Version 5.1
# clear_training.ps1  - removes fine-tuned model checkpoints so training can be
# rerun. Does NOT touch raw data, processed data, or results.
#
# Usage:
#   .\scripts\clear_training.ps1

Set-Location (Split-Path $PSScriptRoot -Parent)

$MODELS = "models"

if (-not (Test-Path $MODELS)) {
    Write-Host "models\ does not exist  - nothing to clear."
    exit 0
}

$items = Get-ChildItem $MODELS -Force
if ($items.Count -eq 0) {
    Write-Host "models\ is already empty."
    exit 0
}

Write-Host "Clearing $MODELS ..."
foreach ($item in $items) {
    Remove-Item $item.FullName -Recurse -Force
    Write-Host "  removed $($item.Name)"
}
Write-Host "Done. Re-run .\scripts\run_part1.ps1 to retrain."
