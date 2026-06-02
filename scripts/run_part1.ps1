#Requires -Version 5.1
# run_part1.ps1  - everything up to the manual annotation checkpoint.
#
#   download data -> preprocess -> train BERT -> export a gold template
#
# After this finishes you HAND-CORRECT data\processed\gold.conll, then run
# scripts\run_part2.ps1. The correction step can't be automated  - it's what makes the
# BERT-vs-baseline comparison meaningful rather than circular.
#
# Usage:
#   .\scripts\run_part1.ps1
#
# Knobs (set env vars before running, e.g.):
#   $env:MODEL_NAME = "distilbert-base-uncased"; .\scripts\run_part1.ps1

Set-Location (Split-Path $PSScriptRoot -Parent)

# ---- configuration (single source of truth) ------------------------------ #
$MODEL_NAME     = if ($env:MODEL_NAME)     { $env:MODEL_NAME }     else { "bert-base-uncased" }
$MODEL_DIR      = if ($env:MODEL_DIR)      { $env:MODEL_DIR }      else { "models/bert-skills-ner" }
$PREPROCESS_MAX = if ($env:PREPROCESS_MAX) { $env:PREPROCESS_MAX } else { "2000" }
$GOLD_N         = if ($env:GOLD_N)         { $env:GOLD_N }         else { "60" }

Write-Host "=================================================================="
Write-Host " SkillPostBERT -- pipeline part 1"
Write-Host "   model:        $MODEL_NAME"
Write-Host "   output dir:   $MODEL_DIR"
Write-Host "   max/disc:     $PREPROCESS_MAX"
Write-Host "   gold sample:  $GOLD_N"
Write-Host "=================================================================="

# ---- preflight: required Python packages ---------------------------------- #
Write-Host ""
Write-Host "[preflight] checking dependencies ..."
python -c "import torch, transformers, datasets, pandas, seqeval, evaluate, yaml" *>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Missing dependencies. Install them first:"
    Write-Host "      pip install -r requirements.txt"
    exit 1
}
Write-Host "  ok."

# ---- preflight: Kaggle credentials (warn only) ---------------------------- #
$kaggleJson = Join-Path $HOME ".kaggle\kaggle.json"
if (-not (Test-Path $kaggleJson) -and -not $env:KAGGLE_USERNAME) {
    Write-Host "[preflight] WARNING: no Kaggle credentials found"
    Write-Host "  (~\.kaggle\kaggle.json missing and KAGGLE_USERNAME unset)."
    Write-Host "  download_data.py will fail without them -- see its docstring."
}

# ---- 1. download -------------------------------------------------------- #
Write-Host ""
Write-Host "[1/4] downloading datasets ..."
python -m src.download_data
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ---- 2. preprocess ------------------------------------------------------ #
Write-Host ""
Write-Host "[2/4] preprocessing (weak-labeling BIO tags) ..."
python -m src.preprocess --model $MODEL_NAME --max $PREPROCESS_MAX
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ---- 3. train ----------------------------------------------------------- #
Write-Host ""
Write-Host "[3/4] training ..."
$gpuOutput = python -c "from src.utils import get_hardware_profile; p = get_hardware_profile(); print(p['device_type'], p['device_name'])" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $gpuOutput) {
    $gpuOutput = "cpu CPU (no GPU detected)"
}
$gpuParts  = $gpuOutput.Trim() -split '\s+', 2
$GPU_TYPE  = $gpuParts[0]
$GPU_LABEL = if ($gpuParts.Count -gt 1) { $gpuParts[1] } else { $gpuParts[0] }

Write-Host "  $GPU_LABEL"
if ($GPU_TYPE -eq "cpu") {
    Write-Host "  WARNING: no GPU detected. Fine-tuning $MODEL_NAME on CPU is slow"
    Write-Host "  (potentially hours). Consider `$env:MODEL_NAME = 'distilbert-base-uncased'."
    Write-Host "  Press Ctrl-C within 8 seconds to abort ..."
    Start-Sleep -Seconds 8
}
python -m src.train --model $MODEL_NAME --output-dir $MODEL_DIR
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ---- 4. export gold template -------------------------------------------- #
Write-Host ""
Write-Host "[4/4] exporting gold annotation template ..."
python -m src.evaluate --export-gold --n $GOLD_N --model-name $MODEL_NAME
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "=================================================================="
Write-Host " Part 1 complete."
Write-Host ""
Write-Host " NEXT -- hand-correct the gold labels:"
Write-Host "   1. Open  data\processed\gold.conll  in a text editor."
Write-Host "   2. Fix the second column (the BIO tag) on each line:"
Write-Host "        - add skills the matcher missed   (O  -> B-<CAT>)"
Write-Host "        - remove false positives          (B-<CAT> -> O)"
Write-Host "        - fix wrong categories/boundaries"
Write-Host "   3. Then run:  .\scripts\run_part2.ps1"
Write-Host "=================================================================="
