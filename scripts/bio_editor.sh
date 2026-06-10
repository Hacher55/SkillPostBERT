#!/usr/bin/env bash
# Launch the interactive BIO tag editor in a browser.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! python -c "import flask" 2>/dev/null; then
    echo "flask not found — installing..."
    pip install flask
fi

python tools/bio_editor.py "$@"
