# Launch the interactive BIO tag editor in a browser.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

try { python -c "import flask" 2>$null } catch {}
if ($LASTEXITCODE -ne 0) {
    Write-Host "flask not found — installing..."
    pip install flask
}

python tools/bio_editor.py @args
