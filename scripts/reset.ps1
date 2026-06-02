#Requires -Version 5.1
# reset.ps1 — full clean slate. Removes all generated and downloaded artifacts
# and the conda environment:
#
#   data/raw/        downloaded Kaggle datasets
#   data/processed/  corpus, gold files
#   models/          fine-tuned checkpoints
#   results/         metrics JSON and figures
#   conda env        SSE691NLP (or $env:CONDA_ENV)
#
# After this, rebuild the environment and rerun the pipeline:
#   .\scripts\setup_env.ps1
#   conda activate SSE691NLP
#   .\scripts\run_part1.ps1
#
# Knobs:
#   $env:CONDA_ENV — conda environment name to remove (default: SSE691NLP)
#
# Usage:
#   .\scripts\reset.ps1

Set-Location (Split-Path $PSScriptRoot -Parent)

$ENV_NAME = if ($env:CONDA_ENV) { $env:CONDA_ENV } else { "SSE691NLP" }

Write-Host "=================================================================="
Write-Host " SkillPostBERT -- full reset"
Write-Host ""
Write-Host " This will permanently delete:"
Write-Host "   data\raw\        downloaded Kaggle datasets"
Write-Host "   data\processed\  corpus + gold annotation files"
Write-Host "   models\          fine-tuned checkpoints"
Write-Host "   results\         metrics JSON + figures"
Write-Host "   conda env        $ENV_NAME"
Write-Host ""
Write-Host " Press Ctrl-C within 8 seconds to abort ..."
Write-Host "=================================================================="
Start-Sleep -Seconds 8

# ---- data and artefact directories ---------------------------------------- #
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

# ---- conda environment ---------------------------------------------------- #
Write-Host ""
$condaVer = $null
try { $condaVer = conda --version } catch {}

if (-not $condaVer) {
    Write-Host "  conda not found — skipping environment removal."
} else {
    $envList = (conda env list) -join "`n"
    if (-not ($envList -match "(?m)^\s*$([regex]::Escape($ENV_NAME))[\s/\\]")) {
        Write-Host "  conda env '$ENV_NAME' — not found, skipping."
    } elseif ($env:CONDA_DEFAULT_ENV -eq $ENV_NAME) {
        Write-Host "  conda env '$ENV_NAME' — currently active, skipping."
        Write-Host "    Run 'conda deactivate' then .\scripts\clear_env.ps1 to remove it."
    } else {
        Write-Host "  Removing conda environment '$ENV_NAME' ..."
        conda env remove -n $ENV_NAME -y
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    done."
        } else {
            Write-Host "    failed (exit $LASTEXITCODE) — remove manually with: conda env remove -n $ENV_NAME"
        }
    }
}

Write-Host ""
Write-Host "Reset complete."
Write-Host "Run .\scripts\setup_env.ps1 to rebuild the environment, then .\scripts\run_part1.ps1 to start fresh."
