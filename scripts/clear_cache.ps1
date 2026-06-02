#Requires -Version 5.1
# clear_cache.ps1 — removes downloaded raw data so it will be re-fetched on the
# next run. Does NOT touch processed data, trained models, or results.
#
# Usage:
#   .\scripts\clear_cache.ps1

Set-Location (Split-Path $PSScriptRoot -Parent)

$RAW = "data\raw"

if (-not (Test-Path $RAW)) {
    Write-Host "data\raw\ does not exist — nothing to clear."
    exit 0
}

$items = Get-ChildItem $RAW -Force
if ($items.Count -eq 0) {
    Write-Host "data\raw\ is already empty."
    exit 0
}

Write-Host "Clearing $RAW ..."
foreach ($item in $items) {
    Remove-Item $item.FullName -Recurse -Force
    Write-Host "  removed $($item.Name)"
}
Write-Host "Done. Re-run .\scripts\run_part1.ps1 to re-download."
