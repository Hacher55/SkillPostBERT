#Requires -Version 5.1
# run_part2.ps1 — after you've hand-corrected data\processed\gold.conll.
#
#   apply corrections -> evaluate BERT vs baseline on gold -> cross-discipline
#   comparison (the figures/tables for your paper)
#
# Usage:
#   .\scripts\run_part2.ps1
#
# Must use the SAME MODEL_DIR you trained with in scripts\run_part1.ps1.

Set-Location (Split-Path $PSScriptRoot -Parent)

# ---- configuration (must match run_part1.ps1) ---------------------------- #
$MODEL_DIR  = if ($env:MODEL_DIR)   { $env:MODEL_DIR }   else { "models/bert-skills-ner" }
$GOLD_CONLL = if ($env:GOLD_CONLL)  { $env:GOLD_CONLL }  else { "data/processed/gold.conll" }
$GOLD_JSONL = if ($env:GOLD_JSONL)  { $env:GOLD_JSONL }  else { "data/processed/gold.jsonl" }

Write-Host "=================================================================="
Write-Host " SkillPostBERT -- pipeline part 2"
Write-Host "   model dir:  $MODEL_DIR"
Write-Host "   gold:       $GOLD_CONLL"
Write-Host "=================================================================="

# ---- preflight ----------------------------------------------------------- #
python -c "import torch, transformers, seqeval, evaluate, pandas, matplotlib" *>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Missing dependencies. Run: pip install -r requirements.txt"
    exit 1
}
if (-not (Test-Path $GOLD_CONLL)) {
    Write-Host "ERROR: $GOLD_CONLL not found. Run .\scripts\run_part1.ps1 first."
    exit 1
}
if (-not (Test-Path $MODEL_DIR -PathType Container)) {
    Write-Host "ERROR: trained model not found at $MODEL_DIR. Run .\scripts\run_part1.ps1 first."
    exit 1
}

# Sanity nudge: did you actually edit the gold file, or apply raw weak labels?
Write-Host ""
Write-Host "[reminder] make sure you HAND-CORRECTED $GOLD_CONLL before this step."
Write-Host "  Applying the un-edited template just re-scores the matcher against"
Write-Host "  itself. Press Ctrl-C within 5 seconds if you still need to annotate ..."
Start-Sleep -Seconds 5

# ---- 1. fold corrections back in ---------------------------------------- #
Write-Host ""
Write-Host "[1/3] applying corrected labels ..."
python -m src.evaluate --apply-conll $GOLD_CONLL
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ---- 2. head-to-head evaluation ----------------------------------------- #
Write-Host ""
Write-Host "[2/3] evaluating BERT vs keyword baseline on gold ..."
python -m src.evaluate --gold $GOLD_JSONL --model $MODEL_DIR
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# ---- 3. cross-discipline comparison ------------------------------------- #
Write-Host ""
Write-Host "[3/3] cross-discipline skill analysis (BERT predictions) ..."
python -m src.compare --source bert --model $MODEL_DIR
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "=================================================================="
Write-Host " Part 2 complete. Results in results/:"
Write-Host "   comparison.json                  BERT vs baseline metrics"
Write-Host "   top_skills_by_discipline.{csv,png}"
Write-Host "   category_mix.{csv,png}"
Write-Host "   skill_heatmap.png"
Write-Host "=================================================================="
