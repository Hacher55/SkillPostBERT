#Requires -Version 5.1
# reset.ps1 — full clean slate. Removes all generated and downloaded artefacts:
#
#   data/raw/        downloaded Kaggle datasets
#   data/processed/  corpus, gold files
#   models/          fine-tuned checkpoints
#   results/         metrics JSON and figures
#
# After this you can rerun the entire pipeline from scratch via run_part1.ps1.
#
# Usage:
#   .\scripts\reset.ps1

Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "=================================================================="
Write-Host " SkillPostBERT -- full reset"
Write-Host ""
Write-Host " This will permanently delete:"
Write-Host "   data\raw\        downloaded Kaggle datasets"
Write-Host "   data\processed\  corpus + gold annotation files"
Write-Host "   models\          fine-tuned checkpoints"
Write-Host "   results\         metrics JSON + figures"
Write-Host ""
Write-Host " Press Ctrl-C within 8 seconds to abort ..."
Write-Host "=================================================================="
Start-Sleep -Seconds 8

$targets = @(
    @{ Path = "data\raw";        Label = "raw data" },
    @{ Path = "data\processed";  Label = "processed data" },
    @{ Path = "models";          Label = "model checkpoints" },
    @{ Path = "results";         Label = "results" }
)

foreach ($t in $targets) {
    if (-not (Test-Path $t.Path)) {
        Write-Host "  $($t.Path)\ — not found, skipping."
        continue
    }
    $items = Get-ChildItem $t.Path -Force
    if ($items.Count -eq 0) {
        Write-Host "  $($t.Path)\ — already empty."
        continue
    }
    Write-Host "  Clearing $($t.Path) ($($t.Label)) ..."
    foreach ($item in $items) {
        Remove-Item $item.FullName -Recurse -Force
    }
    Write-Host "    done ($($items.Count) item(s) removed)."
}

Write-Host ""
Write-Host "Reset complete. Run .\scripts\run_part1.ps1 to start fresh."
