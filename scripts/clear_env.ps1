#Requires -Version 5.1
# clear_env.ps1 — removes the SSE691NLP conda environment entirely.
#
# Use this to do a clean reinstall (e.g. to change Python version or fix a
# broken environment). After clearing, rerun setup_env to rebuild it.
#
# Knobs:
#   $env:CONDA_ENV — environment name to remove (default: SSE691NLP)
#
# Usage:
#   .\scripts\clear_env.ps1

Set-Location (Split-Path $PSScriptRoot -Parent)

$ENV_NAME = if ($env:CONDA_ENV) { $env:CONDA_ENV } else { "SSE691NLP" }

# ---- preflight: conda ----------------------------------------------------- #
$condaVer = $null
try { $condaVer = conda --version } catch {}
if (-not $condaVer) {
    Write-Host "ERROR: conda not found on PATH."
    exit 1
}

# ---- check the environment exists ----------------------------------------- #
$envList = (conda env list) -join "`n"
if (-not ($envList -match "(?m)^\s*$([regex]::Escape($ENV_NAME))[\s/\\]")) {
    Write-Host "Conda environment '$ENV_NAME' does not exist — nothing to remove."
    exit 0
}

# ---- guard: refuse to remove the active environment ----------------------- #
if ($env:CONDA_DEFAULT_ENV -eq $ENV_NAME) {
    Write-Host "ERROR: '$ENV_NAME' is currently active."
    Write-Host "  Run 'conda deactivate' first, then rerun this script."
    exit 1
}

Write-Host "=================================================================="
Write-Host " This will permanently remove the conda environment '$ENV_NAME'."
Write-Host " All installed packages will be deleted."
Write-Host ""
Write-Host " Rerun .\scripts\setup_env.ps1 afterwards to rebuild it."
Write-Host ""
Write-Host " Press Ctrl-C within 8 seconds to abort ..."
Write-Host "=================================================================="
Start-Sleep -Seconds 8

Write-Host ""
Write-Host "Removing environment '$ENV_NAME' ..."
conda env remove -n $ENV_NAME -y
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done. Run .\scripts\setup_env.ps1 to rebuild."
