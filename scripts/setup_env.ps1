#Requires -Version 5.1
# setup_env.ps1  - one-shot environment setup.
#
#   1. Create (or verify) the SSE691NLP conda environment (Python 3.10)
#   2. Install all Python dependencies from requirements.txt
#   3. Detect your NVIDIA GPU / CUDA driver and swap in a CUDA-enabled
#      PyTorch wheel automatically (skipped on non-NVIDIA hardware)
#
# After this script finishes, activate the environment and run the pipeline:
#   conda activate SSE691NLP
#   .\scripts\run_part1.ps1
#
# Knobs (set env vars before running):
#   $env:CONDA_ENV   - conda environment name  (default: SSE691NLP)
#   $env:PYTHON_VER  - Python version to create (default: 3.10)
#
# Usage:
#   .\scripts\setup_env.ps1

Set-Location (Split-Path $PSScriptRoot -Parent)

$ENV_NAME   = if ($env:CONDA_ENV)   { $env:CONDA_ENV }   else { "SSE691NLP" }
$PYTHON_VER = if ($env:PYTHON_VER)  { $env:PYTHON_VER }  else { "3.10" }

Write-Host "=================================================================="
Write-Host " SkillPostBERT -- environment setup"
Write-Host "   conda env:    $ENV_NAME"
Write-Host "   Python:       $PYTHON_VER"
Write-Host "=================================================================="

# ---- preflight: conda ----------------------------------------------------- #
Write-Host ""
Write-Host "[preflight] checking for conda ..."
$condaVer = $null
try { $condaVer = conda --version } catch {}
if (-not $condaVer) {
    Write-Host "  ERROR: conda not found on PATH."
    Write-Host "  Install Miniconda or Anaconda, then open a new terminal and rerun."
    Write-Host "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
}
Write-Host "  $condaVer"

# ---- 1. create / verify environment --------------------------------------- #
Write-Host ""
Write-Host "[1/3] setting up conda environment '$ENV_NAME' ..."

$envList = (conda env list) -join "`n"
$envExists = $envList -match "(?m)^\s*$([regex]::Escape($ENV_NAME))[\s/\\]"

if ($envExists) {
    Write-Host "  '$ENV_NAME' already exists  - skipping creation."
} else {
    Write-Host "  creating '$ENV_NAME' (Python $PYTHON_VER) ..."
    conda create -n $ENV_NAME python=$PYTHON_VER -y
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "  created."
}

# ---- 2. install Python dependencies --------------------------------------- #
Write-Host ""
Write-Host "[2/3] installing Python dependencies ..."
Write-Host "  (this may take a few minutes on the first install)"
conda run -n $ENV_NAME pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "  dependencies installed."

# ---- 3. CUDA-enabled PyTorch (NVIDIA only) -------------------------------- #
Write-Host ""
Write-Host "[3/3] checking for NVIDIA GPU ..."

$nvidiaSMI = $null
try { $nvidiaSMI = (nvidia-smi) -join " " } catch {}

if (-not $nvidiaSMI) {
    Write-Host "  No NVIDIA GPU detected  - CPU PyTorch is sufficient."
    Write-Host "  (Apple Silicon MPS is already included in the default wheel.)"
} else {
    $cudaMatch = [regex]::Match($nvidiaSMI, "CUDA Version:\s*(\d+)\.(\d+)")
    if (-not $cudaMatch.Success) {
        Write-Host "  NVIDIA GPU found but could not parse CUDA driver version."
        Write-Host "  Keeping CPU PyTorch  - swap manually if needed (see README)."
    } else {
        $major   = [int]$cudaMatch.Groups[1].Value
        $minor   = [int]$cudaMatch.Groups[2].Value
        $gpuName = $null
        try { $gpuName = (nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1) } catch {}
        Write-Host "  GPU:  $gpuName"
        Write-Host "  CUDA: $major.$minor (driver)"

        if ($major -gt 12 -or ($major -eq 12 -and $minor -ge 8)) {
            $wheel = "cu128"
        } elseif ($major -eq 12 -and $minor -ge 1) {
            $wheel = "cu121"
        } elseif ($major -eq 11 -and $minor -ge 8) {
            $wheel = "cu118"
        } else {
            $wheel = $null
            Write-Host "  CUDA $major.$minor is older than 11.8  - no matching wheel available."
            Write-Host "  Keeping CPU PyTorch."
        }

        if ($wheel) {
            Write-Host "  Installing CUDA-enabled PyTorch ($wheel) ..."
            conda run -n $ENV_NAME pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$wheel"
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            Write-Host "  CUDA PyTorch installed ($wheel)."
        }
    }
}

Write-Host ""
Write-Host "=================================================================="
Write-Host " Setup complete."
Write-Host ""
Write-Host " Next steps:"
Write-Host "   1. conda activate $ENV_NAME"
Write-Host "   2. Place kaggle.json at ~\.kaggle\kaggle.json  (see README Setup)"
Write-Host "   3. .\scripts\run_part1.ps1"
Write-Host ""
Write-Host " Tip: run .\scripts\check_env.ps1 (after activating) to verify."
Write-Host "=================================================================="
