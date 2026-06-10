"""Interactive BIO tag editor — click/drag words to annotate."""

import re
import sys
from pathlib import Path
from threading import Lock, Timer

try:
    from flask import Flask, jsonify, request
except ImportError:
    sys.exit("Flask not found.  Run:  pip install flask")

# ── Config ────────────────────────────────────────────────────────────────────

GOLD_CONLL = Path("data/processed/gold.conll")
PORT = 5050

LABELS = [
    "O",
    "B-TECHNICAL", "I-TECHNICAL",
    "B-TOOLS",     "I-TOOLS",
    "B-SOFT",      "I-SOFT",
    "B-CERT",      "I-CERT",
]
SPECIAL = frozenset({"[CLS]", "[SEP]", "[PAD]"})

_lock     = Lock()
_records: list[dict] = []
_modified: set[str]  = set()

# ── Data helpers ──────────────────────────────────────────────────────────────

def parse_conll(path: Path) -> list[dict]:
    records: list[dict] = []
    cur_id = cur_disc = None
    tokens: list[str] = []
    tags:   list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# id:"):
            if cur_id is not None:
                records.append({"id": cur_id, "discipline": cur_disc,
                                 "tokens": tokens[:], "tags": tags[:]})
            m = re.match(r"# id:\s*(\S+)\s+\((\w+)\)", line)
            cur_id   = m.group(1) if m else line.split("# id:")[1].strip().split()[0]
            cur_disc = m.group(2) if m else ""
            tokens, tags = [], []
        elif line.strip():
            parts = line.split("\t")
            if len(parts) >= 2:
                tokens.append(parts[0])
                tags.append(parts[1].strip())
    if cur_id is not None:
        records.append({"id": cur_id, "discipline": cur_disc,
                         "tokens": tokens[:], "tags": tags[:]})
    return records


def write_conll(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(f"# id: {r['id']}  ({r['discipline']})\n")
            for tok, tag in zip(r["tokens"], r["tags"]):
                f.write(f"{tok}\t{tag}\n")
            f.write("\n")


def word_spans(tokens: list[str], tags: list[str]) -> list[dict]:
    """Merge BERT ## subwords into display words; track which token indices each word covers."""
    words: list[dict] = []
    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        if tok in SPECIAL:
            continue
        if tok.startswith("##") and words:
            words[-1]["text"] += tok[2:]
            words[-1]["token_indices"].append(i)
        else:
            words.append({"text": tok, "token_indices": [i], "tag": tag})
    return words

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BIO Tag Editor</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, sans-serif; background: #f1f5f9; color: #1e293b; }

    #app { display: flex; min-height: 100vh; }

    /* ── Sidebar ── */
    #sidebar {
      width: 240px; min-width: 240px; background: #1e293b; color: #e2e8f0;
      padding: 20px 16px; display: flex; flex-direction: column; gap: 16px;
      position: sticky; top: 0; height: 100vh; overflow-y: auto;
    }
    #sidebar h2 { font-size: 1rem; font-weight: 700; color: #f8fafc; }

    #nav { display: flex; align-items: center; gap: 6px; }
    #nav button {
      flex: 1; padding: 7px 4px; border: 1px solid #475569; border-radius: 6px;
      background: #334155; color: #e2e8f0; cursor: pointer; font-size: 0.82rem; font-weight: 600;
    }
    #nav button:hover:not(:disabled) { background: #4b5f78; }
    #nav button:disabled { opacity: 0.35; cursor: not-allowed; }
    #record-counter { flex: 1; text-align: center; font-size: 0.8rem; color: #94a3b8; white-space: nowrap; }

    #jump { display: flex; gap: 4px; align-items: center; }
    #jump label { font-size: 0.75rem; color: #94a3b8; white-space: nowrap; }
    #jump-input {
      width: 54px; padding: 4px 6px; border: 1px solid #475569; border-radius: 4px;
      background: #334155; color: #e2e8f0; font-size: 0.82rem; text-align: center;
    }
    #jump-go {
      padding: 4px 10px; border: none; border-radius: 4px;
      background: #475569; color: #e2e8f0; cursor: pointer; font-size: 0.82rem;
    }
    #jump-go:hover { background: #64748b; }

    #stats { display: flex; flex-direction: column; gap: 6px; }
    .stat-label { font-size: 0.75rem; color: #94a3b8; }
    .stat-value { font-size: 0.85rem; font-weight: 600; color: #e2e8f0; }
    #progress-bar { height: 4px; background: #334155; border-radius: 2px; overflow: hidden; }
    #progress-fill { height: 100%; background: #38bdf8; transition: width 0.3s; width: 0%; }
    #modified-note { font-size: 0.75rem; color: #fb923c; min-height: 1em; }

    #btn-save {
      padding: 10px; border: none; border-radius: 8px;
      background: #0284c7; color: white; cursor: pointer; font-weight: 700; font-size: 0.85rem;
    }
    #btn-save:hover:not(:disabled) { background: #0369a1; }
    #btn-save:disabled { opacity: 0.4; cursor: not-allowed; }
    #save-msg { font-size: 0.75rem; color: #4ade80; text-align: center; min-height: 1.1em; }

    #shortcuts { border-top: 1px solid #334155; padding-top: 12px; }
    #shortcuts h3 {
      font-size: 0.68rem; text-transform: uppercase; letter-spacing: .06em;
      color: #64748b; margin-bottom: 8px;
    }
    #shortcuts table { width: 100%; border-collapse: collapse; }
    #shortcuts td { padding: 2px 0; font-size: 0.75rem; }
    #shortcuts td:first-child { color: #94a3b8; font-family: monospace; width: 36px; }
    #shortcuts td:last-child  { color: #cbd5e1; }

    #categories { border-top: 1px solid #334155; padding-top: 12px; }
    #categories h3 {
      font-size: 0.68rem; text-transform: uppercase; letter-spacing: .06em;
      color: #64748b; margin-bottom: 10px;
    }
    .cat-entry { margin-bottom: 8px; }
    .cat-entry dt {
      font-size: 0.75rem; font-weight: 700; margin-bottom: 2px;
    }
    .cat-entry dt.TECHNICAL { color: #60a5fa; }
    .cat-entry dt.TOOLS     { color: #4ade80; }
    .cat-entry dt.SOFT      { color: #fbbf24; }
    .cat-entry dt.CERT      { color: #c084fc; }
    .cat-entry dd { font-size: 0.72rem; color: #94a3b8; margin-left: 0; line-height: 1.35; }

    /* ── Main ── */
    #main { flex: 1; padding: 24px 28px; max-width: 860px; }

    #record-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 10px; }

    #legend { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
    .legend-item {
      font-size: 0.75rem; padding: 2px 10px; border-radius: 12px; font-weight: 600; border: 1px solid;
    }
    .legend-item.O         { color: #9ca3af; border-color: #9ca3af; background: #9ca3af18; }
    .legend-item.TECHNICAL { color: #2563eb; border-color: #2563eb; background: #2563eb14; }
    .legend-item.TOOLS     { color: #16a34a; border-color: #16a34a; background: #16a34a14; }
    .legend-item.SOFT      { color: #d97706; border-color: #d97706; background: #d9770614; }
    .legend-item.CERT      { color: #7c3aed; border-color: #7c3aed; background: #7c3aed14; }

    /* ── Annotation area ── */
    #annotation-area {
      background: white; border: 1px solid #e2e8f0; border-radius: 12px;
      padding: 20px 20px 16px; margin-bottom: 10px;
    }
    #token-view { line-height: 3.2; font-size: 1.05em; user-select: none; cursor: default; }

    .word {
      display: inline-block; padding: 2px 6px; margin: 2px 3px;
      border-radius: 5px; cursor: pointer; border: 1px solid transparent;
      transition: opacity 0.1s;
    }
    .word:hover { opacity: 0.72; }
    .word.O         { color: #6b7280; }
    .word.TECHNICAL { color: #2563eb; background: #2563eb14; border-color: #2563eb; font-weight: 600; }
    .word.TOOLS     { color: #16a34a; background: #16a34a14; border-color: #16a34a; font-weight: 600; }
    .word.SOFT      { color: #d97706; background: #d9770614; border-color: #d97706; font-weight: 600; }
    .word.CERT      { color: #7c3aed; background: #7c3aed14; border-color: #7c3aed; font-weight: 600; }
    .word.selecting {
      background: #fef08a !important; color: #713f12 !important;
      border-color: #eab308 !important; font-weight: 700 !important;
    }

    /* ── Picker ── */
    #picker {
      margin-top: 14px; padding: 10px 14px;
      background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
      display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
    }
    #picker.hidden { display: none; }
    #sel-count { font-size: 0.8rem; color: #64748b; white-space: nowrap; }

    .cat-btn {
      padding: 6px 14px; border: none; border-radius: 6px;
      font-size: 0.8rem; font-weight: 700; cursor: pointer; letter-spacing: .02em;
    }
    .cat-btn:hover { opacity: 0.82; }
    .cat-btn.TECHNICAL { background: #2563eb; color: white; }
    .cat-btn.TOOLS     { background: #16a34a; color: white; }
    .cat-btn.SOFT      { background: #d97706; color: white; }
    .cat-btn.CERT      { background: #7c3aed; color: white; }
    .cat-btn.O         { background: #6b7280; color: white; }
    .cat-btn.cancel    { background: #e2e8f0; color: #475569; }

    #instructions { font-size: 0.78rem; color: #94a3b8; }
  </style>
</head>
<body>
<div id="app">

  <aside id="sidebar">
    <h2>&#127991; BIO Tag Editor</h2>

    <div id="nav">
      <button id="btn-prev" onclick="navigate(-1)" disabled>&#8592; Prev</button>
      <span id="record-counter">&#x2013; / &#x2013;</span>
      <button id="btn-next" onclick="navigate(1)"  disabled>Next &#8594;</button>
    </div>

    <div id="jump">
      <label>Jump:&nbsp;<input type="number" id="jump-input" min="0" value="0"></label>
      <button id="jump-go" onclick="jumpTo()">Go</button>
    </div>

    <div id="stats">
      <div><span class="stat-label">Tagged: </span><span class="stat-value" id="tagged-count">&#x2013;</span></div>
      <div id="progress-bar"><div id="progress-fill"></div></div>
      <div id="modified-note"></div>
    </div>

    <button id="btn-save" onclick="saveToFile()" disabled>&#128190; Save to gold.conll</button>
    <div id="save-msg"></div>

    <div id="categories">
      <h3>Categories</h3>
      <dl>
        <div class="cat-entry">
          <dt class="TECHNICAL">TECHNICAL</dt>
          <dd>Domain knowledge, engineering methods, theory</dd>
        </div>
        <div class="cat-entry">
          <dt class="TOOLS">TOOLS</dt>
          <dd>Named software, languages, platforms, instruments</dd>
        </div>
        <div class="cat-entry">
          <dt class="SOFT">SOFT</dt>
          <dd>Transferable &amp; interpersonal skills</dd>
        </div>
        <div class="cat-entry">
          <dt class="CERT">CERT</dt>
          <dd>Certifications, licenses, formal credentials</dd>
        </div>
      </dl>
    </div>

    <div id="shortcuts">
      <h3>Shortcuts</h3>
      <table>
        <tr><td>&#8592;&#8594;</td><td>Prev / Next</td></tr>
        <tr><td>T</td><td>TECHNICAL</td></tr>
        <tr><td>W</td><td>TOOLS</td></tr>
        <tr><td>S</td><td>SOFT</td></tr>
        <tr><td>C</td><td>CERT</td></tr>
        <tr><td>O</td><td>Remove (O)</td></tr>
        <tr><td>Esc</td><td>Cancel</td></tr>
      </table>
    </div>
  </aside>

  <main id="main">
    <h2 id="record-title">Loading&#x2026;</h2>

    <div id="legend">
      <span class="legend-item O"         title="Not part of any skill mention">O &mdash; outside</span>
      <span class="legend-item TECHNICAL" title="Domain knowledge, engineering methods, theory">TECHNICAL</span>
      <span class="legend-item TOOLS"     title="Named software, languages, platforms, instruments">TOOLS</span>
      <span class="legend-item SOFT"      title="Transferable and interpersonal skills">SOFT</span>
      <span class="legend-item CERT"      title="Certifications, licenses, formal credentials">CERT</span>
    </div>

    <div id="annotation-area">
      <div id="token-view"><em style="color:#94a3b8">Loading record&hellip;</em></div>
      <div id="picker" class="hidden">
        <span id="sel-count"></span>
        <button class="cat-btn TECHNICAL" onclick="applyCategory('TECHNICAL')" title="Domain knowledge, engineering methods, theory [T]">TECHNICAL</button>
        <button class="cat-btn TOOLS"     onclick="applyCategory('TOOLS')"     title="Named software, languages, platforms, instruments [W]">TOOLS</button>
        <button class="cat-btn SOFT"      onclick="applyCategory('SOFT')"      title="Transferable and interpersonal skills [S]">SOFT</button>
        <button class="cat-btn CERT"      onclick="applyCategory('CERT')"      title="Certifications, licenses, formal credentials [C]">CERT</button>
        <button class="cat-btn O"         onclick="applyCategory('O')"         title="Remove — not part of any skill [O]">Remove (O)</button>
        <button class="cat-btn cancel"    onclick="cancelSelection()"          title="Esc">&#10005; Cancel</button>
      </div>
    </div>
    <p id="instructions">Click a word to select it &mdash; click and drag to select a span &mdash; then choose a category.</p>
  </main>

</div>

<script>
// ── State ─────────────────────────────────────────────────────────────────────
let currentIdx    = 0;
let totalRecords  = 0;
let currentRecord = null;   // {idx, id, discipline, tokens, tags, words}

let isDragging = false;
let selStart   = -1;
let selEnd     = -1;
let hasSel     = false;

// ── Fetch helper ──────────────────────────────────────────────────────────────
async function api(url, opts) {
  const res = await fetch(url, opts);
  return res.json();
}

// ── Record loading ────────────────────────────────────────────────────────────
async function loadRecord(idx) {
  const data = await api(`/api/record/${idx}`);
  if (data.error) { alert(data.error); return; }
  currentRecord = data;
  currentIdx    = data.idx;
  totalRecords  = data.total;
  cancelSelection();
  renderRecord();
  updateNav();
}

async function refreshStats() {
  const records    = await api('/api/records');
  const tagged     = records.filter(r => r.has_skills).length;
  const nModified  = records.filter(r => r.modified).length;
  document.getElementById('tagged-count').textContent = `${tagged} / ${records.length}`;
  document.getElementById('progress-fill').style.width =
    `${records.length ? (tagged / records.length * 100).toFixed(1) : 0}%`;
  const note = document.getElementById('modified-note');
  if (nModified > 0) {
    note.textContent = `${nModified} unsaved change(s)`;
    document.getElementById('btn-save').disabled = false;
  } else {
    note.textContent = '';
    document.getElementById('btn-save').disabled = true;
  }
}

// ── Render ────────────────────────────────────────────────────────────────────
const DISC = { se: 'Software', me: 'Mechanical', ee: 'Electrical' };

function tagToClass(tag) {
  if (tag === 'O') return 'O';
  const parts = tag.split('-');
  return parts.length > 1 ? parts[1] : 'O';
}

function renderRecord() {
  const disc = DISC[currentRecord.discipline] || currentRecord.discipline.toUpperCase();
  document.getElementById('record-title').textContent =
    `Record ${currentRecord.id} — ${disc} Engineering`;
  renderWords();
}

function renderWords() {
  const view = document.getElementById('token-view');
  const lo   = hasSel ? Math.min(selStart, selEnd) : -1;
  const hi   = hasSel ? Math.max(selStart, selEnd) : -1;

  // Build DOM in one shot to avoid reflow thrash
  const frag = document.createDocumentFragment();
  currentRecord.words.forEach((word, wIdx) => {
    const span = document.createElement('span');
    let cls = `word ${tagToClass(word.tag)}`;
    if (wIdx >= lo && wIdx <= hi) cls += ' selecting';
    span.className       = cls;
    span.textContent     = word.text;
    span.dataset.wordIdx = wIdx;
    frag.appendChild(span);
    frag.appendChild(document.createTextNode(' '));
  });
  view.replaceChildren(frag);
}

function updateNav() {
  document.getElementById('record-counter').textContent =
    `${currentIdx + 1} / ${totalRecords}`;
  document.getElementById('btn-prev').disabled = (currentIdx === 0);
  document.getElementById('btn-next').disabled = (currentIdx === totalRecords - 1);
  document.getElementById('jump-input').value  = currentIdx;
}

// ── Navigation ────────────────────────────────────────────────────────────────
async function navigate(delta) {
  const next = currentIdx + delta;
  if (next < 0 || next >= totalRecords) return;
  await loadRecord(next);
}

async function jumpTo() {
  const val = parseInt(document.getElementById('jump-input').value);
  if (isNaN(val) || val < 0 || val >= totalRecords) return;
  await loadRecord(val);
}

// ── Selection ─────────────────────────────────────────────────────────────────
const annotationArea = document.getElementById('annotation-area');

annotationArea.addEventListener('mousedown', e => {
  const span = e.target.closest('.word');
  if (!span) return;
  isDragging = true;
  hasSel     = false;
  selStart   = selEnd = parseInt(span.dataset.wordIdx);
  renderWords();
  e.preventDefault();
});

annotationArea.addEventListener('mousemove', e => {
  if (!isDragging) return;
  const span = e.target.closest('.word');
  if (!span) return;
  const idx = parseInt(span.dataset.wordIdx);
  if (idx !== selEnd) { selEnd = idx; hasSel = true; renderWords(); }
});

document.addEventListener('mouseup', () => {
  if (!isDragging) return;
  isDragging = false;
  hasSel     = true;
  renderWords();
  showPicker();
});

function showPicker() {
  const lo    = Math.min(selStart, selEnd);
  const hi    = Math.max(selStart, selEnd);
  const count = hi - lo + 1;
  document.getElementById('sel-count').textContent =
    `${count} word${count !== 1 ? 's' : ''} selected:`;
  document.getElementById('picker').classList.remove('hidden');
}

function hidePicker() {
  document.getElementById('picker').classList.add('hidden');
}

function cancelSelection() {
  isDragging = false;
  selStart = selEnd = -1;
  hasSel   = false;
  hidePicker();
  if (currentRecord) renderWords();
}

// ── Apply category ────────────────────────────────────────────────────────────
async function applyCategory(cat) {
  if (!currentRecord || selStart < 0) return;

  const lo = Math.min(selStart, selEnd);
  const hi = Math.max(selStart, selEnd);

  for (let w = lo; w <= hi; w++) {
    const word = currentRecord.words[w];
    word.token_indices.forEach((tokIdx, j) => {
      if (cat === 'O') {
        currentRecord.tags[tokIdx] = 'O';
      } else {
        currentRecord.tags[tokIdx] = (w === lo && j === 0) ? `B-${cat}` : `I-${cat}`;
      }
    });
    word.tag = (cat === 'O') ? 'O' : (w === lo ? `B-${cat}` : `I-${cat}`);
  }

  cancelSelection();

  await api(`/api/record/${currentIdx}/tags`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tags: currentRecord.tags }),
  });

  await refreshStats();
}

// ── Save to file ──────────────────────────────────────────────────────────────
async function saveToFile() {
  const result = await api('/api/save', { method: 'POST' });
  const msg = document.getElementById('save-msg');
  if (result.ok) {
    msg.textContent = 'Saved!';
    document.getElementById('btn-save').disabled = true;
    document.getElementById('modified-note').textContent = '';
    setTimeout(() => { msg.textContent = ''; }, 2500);
  } else {
    msg.textContent = 'Error saving.';
  }
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (!hasSel) {
    if (e.key === 'ArrowLeft')  { navigate(-1); return; }
    if (e.key === 'ArrowRight') { navigate(1);  return; }
    return;
  }
  switch (e.key.toLowerCase()) {
    case 't':      applyCategory('TECHNICAL'); break;
    case 'w':      applyCategory('TOOLS');     break;
    case 's':      applyCategory('SOFT');      break;
    case 'c':      applyCategory('CERT');      break;
    case 'o':      applyCategory('O');         break;
    case 'escape': cancelSelection();          break;
  }
  e.preventDefault();
});

// ── Init ──────────────────────────────────────────────────────────────────────
(async () => {
  await loadRecord(0);
  await refreshStats();
})();
</script>
</body>
</html>
"""

# ── Flask routes ──────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    return HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/records")
def api_records():
    with _lock:
        return jsonify([
            {
                "idx":        i,
                "id":         r["id"],
                "discipline": r["discipline"],
                "has_skills": any(t != "O" for t in r["tags"]),
                "modified":   r["id"] in _modified,
            }
            for i, r in enumerate(_records)
        ])


@app.route("/api/record/<int:idx>")
def api_record(idx: int):
    with _lock:
        if idx < 0 or idx >= len(_records):
            return jsonify({"error": "not found"}), 404
        r = _records[idx]
        return jsonify({
            "idx":        idx,
            "id":         r["id"],
            "discipline": r["discipline"],
            "tokens":     r["tokens"],
            "tags":       r["tags"],
            "words":      word_spans(r["tokens"], r["tags"]),
            "total":      len(_records),
        })


@app.route("/api/record/<int:idx>/tags", methods=["POST"])
def api_update_tags(idx: int):
    data     = request.get_json(force=True)
    new_tags = data.get("tags", [])
    with _lock:
        if idx < 0 or idx >= len(_records):
            return jsonify({"error": "not found"}), 404
        r = _records[idx]
        if len(new_tags) != len(r["tokens"]):
            return jsonify({"error": "tag count mismatch"}), 400
        invalid = [t for t in new_tags if t not in LABELS]
        if invalid:
            return jsonify({"error": f"invalid tags: {set(invalid)}"}), 400
        r["tags"] = new_tags
        _modified.add(r["id"])
    return jsonify({"ok": True})


@app.route("/api/save", methods=["POST"])
def api_save():
    with _lock:
        write_conll(GOLD_CONLL, _records)
        _modified.clear()
    return jsonify({"ok": True})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not GOLD_CONLL.exists():
        sys.exit(
            f"Error: {GOLD_CONLL} not found.\n"
            "Run ./scripts/run_part1.sh first to generate the gold set."
        )

    _records = parse_conll(GOLD_CONLL)
    print(f"Loaded {len(_records)} records from {GOLD_CONLL}")
    print(f"Editor running at  http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")

    import webbrowser
    Timer(0.8, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()

    app.run(port=PORT, debug=False, use_reloader=False)
