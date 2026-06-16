#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SkillPostBERT Hub — Cross-platform GUI command center.
Runs the full NLP pipeline for skill extraction from engineering job postings.

Usage: python gui.py
"""

import json
import os
import platform
import queue
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
SRC     = ROOT / "src"
SCRIPTS = ROOT / "scripts"
DATA    = ROOT / "data"
MODELS  = ROOT / "models"
RESULTS = ROOT / "results"
CONFIG  = ROOT / "configs" / "bert_base.yaml"
TOOLS   = ROOT / "tools"

ENV_NAME = "SSE691NLP"

IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"

# ── Color Palette (Catppuccin Mocha) ──────────────────────────────────────────
C = {
    "base":      "#1e1e2e",
    "mantle":    "#181825",
    "crust":     "#11111b",
    "surface0":  "#313244",
    "surface1":  "#45475a",
    "surface2":  "#585b70",
    "overlay0":  "#6c7086",
    "text":      "#cdd6f4",
    "subtext1":  "#bac2de",
    "blue":      "#89b4fa",
    "lavender":  "#b4befe",
    "green":     "#a6e3a1",
    "yellow":    "#f9e2af",
    "peach":     "#fab387",
    "red":       "#f38ba8",
    "teal":      "#94e2d5",
    "sky":       "#89dceb",
    "mauve":     "#cba6f7",
    "pink":      "#f5c2e7",
    "flamingo":  "#f2cdcd",
    "rosewater": "#f5e0dc",
}


# ══════════════════════════════════════════════════════════════════════════════
class SkillPostBERTHub(tk.Tk):
    """Main application window — central hub for the SkillPostBERT pipeline."""

    def __init__(self):
        super().__init__()
        self.title("SkillPostBERT Hub")
        self.geometry("1280x860")
        self.minsize(1000, 650)
        self.configure(bg=C["base"])

        self.output_q: queue.Queue = queue.Queue()
        self.current_proc: subprocess.Popen | None = None
        self._chart_photo = None   # keep reference to prevent GC
        self._hf_token: str = self._load_hf_token()

        self._setup_style()
        self._build_ui()
        self._poll_output()
        self.after(200, self._refresh_status)

    # ── Styling ───────────────────────────────────────────────────────────────

    def _setup_style(self):
        s = ttk.Style(self)
        base = "clam" if "clam" in s.theme_names() else s.theme_names()[0]
        s.theme_use(base)

        s.configure("TNotebook",           background=C["mantle"],  borderwidth=0)
        s.configure("TNotebook.Tab",       background=C["surface0"], foreground=C["subtext1"],
                                           padding=[14, 6], font=("Helvetica", 10))
        s.map("TNotebook.Tab",
              background=[("selected", C["blue"])],
              foreground=[("selected", C["base"])])

        s.configure("TFrame",             background=C["base"])
        s.configure("TLabel",             background=C["base"],    foreground=C["text"])
        s.configure("TButton",            background=C["surface0"], foreground=C["text"],
                                           padding=[8, 4], relief="flat")
        s.map("TButton",
              background=[("active", C["surface1"]), ("pressed", C["surface2"])])

        s.configure("TEntry",  fieldbackground=C["surface0"], foreground=C["text"],
                               insertcolor=C["text"], relief="flat")
        s.configure("TSpinbox", fieldbackground=C["surface0"], foreground=C["text"],
                                insertcolor=C["text"])
        s.configure("TCombobox", fieldbackground=C["surface0"], foreground=C["text"],
                                  selectbackground=C["blue"], selectforeground=C["base"])
        s.map("TCombobox", fieldbackground=[("readonly", C["surface0"])])

        s.configure("TLabelframe",       background=C["base"],    foreground=C["blue"],
                                          bordercolor=C["surface1"])
        s.configure("TLabelframe.Label", background=C["base"],    foreground=C["blue"],
                                          font=("Helvetica", 10, "bold"))
        s.configure("TScrollbar",        background=C["surface0"], troughcolor=C["base"],
                                          borderwidth=0, arrowsize=12)
        s.configure("TSeparator",        background=C["surface1"])
        s.configure("TProgressbar",      background=C["blue"],    troughcolor=C["surface0"])

    # ── Root Layout ───────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C["crust"], pady=8)
        hdr.pack(fill="x")

        tk.Label(hdr, text="SkillPostBERT Hub", bg=C["crust"], fg=C["blue"],
                 font=("Helvetica", 18, "bold")).pack(side="left", padx=18)
        tk.Label(hdr, text="NLP Pipeline Command Center", bg=C["crust"], fg=C["subtext1"],
                 font=("Helvetica", 11)).pack(side="left")

        # Status indicators (top-right)
        ind = tk.Frame(hdr, bg=C["crust"])
        ind.pack(side="right", padx=18)

        def _ind(text):
            lbl = tk.Label(ind, text=f"● {text}", bg=C["crust"], fg=C["yellow"],
                           font=("Helvetica", 10, "bold"))
            lbl.pack(side="left", padx=6)
            return lbl

        self._lbl_env   = _ind("ENV")
        self._lbl_data  = _ind("DATA")
        self._lbl_model = _ind("MODEL")

        # ── Split pane: notebook (top) + console (bottom) ─────────────────────
        pane = tk.PanedWindow(self, orient="vertical", bg=C["surface1"],
                              sashwidth=4, sashrelief="flat")
        pane.pack(fill="both", expand=True, padx=6, pady=6)

        self.notebook = ttk.Notebook(pane)
        pane.add(self.notebook, minsize=400)

        console_outer = tk.Frame(pane, bg=C["base"])
        pane.add(console_outer, minsize=160)

        # ── Console ───────────────────────────────────────────────────────────
        chdr = tk.Frame(console_outer, bg=C["mantle"], pady=4)
        chdr.pack(fill="x")

        tk.Label(chdr, text="Console", bg=C["mantle"], fg=C["blue"],
                 font=("Helvetica", 10, "bold")).pack(side="left", padx=10)

        self._proc_lbl = tk.Label(chdr, text="Idle", bg=C["mantle"], fg=C["green"],
                                   font=("Helvetica", 9))
        self._proc_lbl.pack(side="left", padx=6)

        for text, cmd in (("■ Stop", self._stop_proc), ("⎚ Clear", self._clear_console)):
            tk.Button(chdr, text=text, bg=C["surface0"], fg=C["text"],
                      font=("Helvetica", 9), relief="flat", padx=8, pady=2,
                      command=cmd).pack(side="right", padx=4)

        self.console = tk.Text(
            console_outer, bg=C["crust"], fg="#e6edf3",
            font=("Courier New" if IS_WIN else "Monaco", 10),
            wrap="word", state="disabled", relief="flat",
            insertbackground="white",
        )
        csb = ttk.Scrollbar(console_outer, command=self.console.yview)
        self.console.configure(yscrollcommand=csb.set)
        csb.pack(side="right", fill="y")
        self.console.pack(fill="both", expand=True)

        self.console.tag_configure("err",  foreground=C["red"])
        self.console.tag_configure("ok",   foreground=C["green"])
        self.console.tag_configure("info", foreground=C["blue"])
        self.console.tag_configure("warn", foreground=C["peach"])

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_bar = tk.Label(
            self, text="Ready", bg=C["mantle"], fg=C["subtext1"],
            font=("Helvetica", 9), anchor="w", padx=10, pady=2,
        )
        self._status_bar.pack(fill="x", side="bottom")

        # ── Tabs ──────────────────────────────────────────────────────────────
        self._build_pipeline_tab()
        self._build_environment_tab()
        self._build_results_tab()
        self._build_status_tab()
        self._build_settings_tab()

    # ══════════════════════════════════════════════════════════════════════════
    # Pipeline Tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_pipeline_tab(self):
        root_frame = ttk.Frame(self.notebook)
        self.notebook.add(root_frame, text="  Pipeline  ")

        canvas = tk.Canvas(root_frame, bg=C["base"], highlightthickness=0)
        vsb = ttk.Scrollbar(root_frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)

        def _mwheel(evt):
            canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")
        canvas.bind("<MouseWheel>", _mwheel)
        inner.bind("<MouseWheel>", _mwheel)

        col1 = ttk.Frame(inner)
        col1.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        col2 = ttk.Frame(inner)
        col2.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)

        # ── Phase 1 (left column) ─────────────────────────────────────────────

        ph1 = ttk.LabelFrame(col1, text="Phase 1 — Data Collection & Training")
        ph1.pack(fill="x", pady=(0, 8))

        self._btn(ph1, "▶  Run Full Phase 1", self._run_phase1,
                  bg=C["green"], fg=C["base"], bold=True).pack(fill="x", padx=10, pady=8)

        ttk.Separator(ph1).pack(fill="x", padx=10, pady=2)

        # Step 1 – Download
        s1 = ttk.LabelFrame(ph1, text="1.  Download Data")
        s1.pack(fill="x", padx=10, pady=6)
        self._btn(s1, "Download from Kaggle", self._run_download).pack(
            side="left", padx=8, pady=6)
        tk.Label(s1, text="Requires ~/.kaggle/kaggle.json",
                 bg=C["base"], fg=C["overlay0"], font=("Helvetica", 9)).pack(
            side="left", padx=4)

        # Step 2 – Preprocess
        s2 = ttk.LabelFrame(ph1, text="2.  Preprocess")
        s2.pack(fill="x", padx=10, pady=6)
        row = tk.Frame(s2, bg=C["base"])
        row.pack(fill="x", padx=8, pady=4)
        tk.Label(row, text="Max per discipline:", bg=C["base"], fg=C["text"]).pack(side="left")
        self._v_pp_max = tk.StringVar(value="2000")
        ttk.Spinbox(row, from_=100, to=10000, increment=100,
                    textvariable=self._v_pp_max, width=7).pack(side="left", padx=6)
        self._btn(s2, "Run Preprocess", self._run_preprocess).pack(
            side="left", padx=8, pady=6)

        # Step 3 – Train
        s3 = ttk.LabelFrame(ph1, text="3.  Train BERT")
        s3.pack(fill="x", padx=10, pady=6)
        grid = tk.Frame(s3, bg=C["base"])
        grid.pack(fill="x", padx=8, pady=4)

        self._v_model  = tk.StringVar(value="bert-base-uncased")
        self._v_epochs = tk.StringVar(value="4")
        self._v_lr     = tk.StringVar(value="2e-5")
        self._v_batch  = tk.StringVar(value="16")
        _rows = [
            ("Model",         ttk.Combobox(grid, textvariable=self._v_model,
                                           values=["bert-base-uncased", "distilbert-base-uncased"],
                                           width=24)),
            ("Epochs",        ttk.Spinbox(grid, from_=1, to=30,
                                          textvariable=self._v_epochs, width=6)),
            ("Learning Rate", ttk.Entry(grid, textvariable=self._v_lr, width=10)),
            ("Batch Size",    ttk.Spinbox(grid, from_=1, to=128,
                                          textvariable=self._v_batch, width=6)),
        ]
        for i, (lbl, wgt) in enumerate(_rows):
            tk.Label(grid, text=f"{lbl}:", bg=C["base"], fg=C["text"]).grid(
                row=i, column=0, sticky="w", padx=6, pady=3)
            wgt.grid(row=i, column=1, sticky="w", padx=6, pady=3)
        self._btn(s3, "Start Training", self._run_train,
                  bg=C["blue"], fg=C["base"], bold=True).pack(side="left", padx=8, pady=6)

        # Step 4 – Export Gold
        s4 = ttk.LabelFrame(ph1, text="4.  Export Gold Set")
        s4.pack(fill="x", padx=10, pady=6)
        row4 = tk.Frame(s4, bg=C["base"])
        row4.pack(fill="x", padx=8, pady=4)
        tk.Label(row4, text="Sample count (n):", bg=C["base"], fg=C["text"]).pack(side="left")
        self._v_gold_n = tk.StringVar(value="60")
        ttk.Spinbox(row4, from_=10, to=500, textvariable=self._v_gold_n, width=7).pack(
            side="left", padx=6)
        self._btn(s4, "Export Gold Set", self._run_export_gold).pack(
            side="left", padx=8, pady=6)

        # ── BIO Editor (right column top) ─────────────────────────────────────

        bio = ttk.LabelFrame(col2, text="BIO Tag Editor  —  Manual Annotation Review")
        bio.pack(fill="x", pady=(0, 8))

        tk.Label(bio, text=(
            "After exporting the gold set, review and correct\n"
            "BIO tags in the Flask editor before running Phase 2."
        ), bg=C["base"], fg=C["subtext1"], justify="left").pack(
            padx=10, pady=(6, 2), anchor="w")

        self._btn(bio, "▶  Launch BIO Editor (Flask)", self._launch_bio_editor,
                  bg=C["blue"], fg=C["base"], bold=True).pack(
            fill="x", padx=10, pady=4)

        self._btn(bio, "↗  Open localhost:5050 in Browser",
                  lambda: webbrowser.open("http://localhost:5050"),
                  bg=C["surface0"], fg=C["text"]).pack(fill="x", padx=10, pady=(0, 10))

        # ── Phase 2 (right column bottom) ─────────────────────────────────────

        ph2 = ttk.LabelFrame(col2, text="Phase 2 — Evaluation & Analysis")
        ph2.pack(fill="x", pady=(0, 8))

        self._btn(ph2, "▶  Run Full Phase 2", self._run_phase2,
                  bg=C["green"], fg=C["base"], bold=True).pack(fill="x", padx=10, pady=8)

        ttk.Separator(ph2).pack(fill="x", padx=10, pady=2)

        # Step 5 – Apply CoNLL
        s5 = ttk.LabelFrame(ph2, text="5.  Apply Corrections")
        s5.pack(fill="x", padx=10, pady=6)
        row5 = tk.Frame(s5, bg=C["base"])
        row5.pack(fill="x", padx=8, pady=4)
        tk.Label(row5, text="CoNLL file:", bg=C["base"], fg=C["text"]).pack(side="left")
        self._v_conll = tk.StringVar(value=str(DATA / "processed" / "gold.conll"))
        ttk.Entry(row5, textvariable=self._v_conll, width=28).pack(side="left", padx=4)
        self._btn(row5, "…", lambda: self._browse_file(
            self._v_conll, [("CoNLL", "*.conll"), ("All", "*.*")]
        ), small=True).pack(side="left")
        self._btn(s5, "Apply CoNLL", self._run_apply_conll).pack(
            side="left", padx=8, pady=6)

        # Step 6 – Evaluate
        s6 = ttk.LabelFrame(ph2, text="6.  Evaluate BERT vs Baseline")
        s6.pack(fill="x", padx=10, pady=6)
        row6 = tk.Frame(s6, bg=C["base"])
        row6.pack(fill="x", padx=8, pady=4)
        tk.Label(row6, text="Model path:", bg=C["base"], fg=C["text"]).pack(side="left")
        self._v_eval_model = tk.StringVar(value=str(MODELS / "bert-skills-ner"))
        ttk.Entry(row6, textvariable=self._v_eval_model, width=28).pack(side="left", padx=4)
        self._btn(row6, "…", lambda: self._browse_dir(self._v_eval_model),
                  small=True).pack(side="left")
        self._btn(s6, "Evaluate", self._run_evaluate).pack(
            side="left", padx=8, pady=6)

        # Step 7 – Compare
        s7 = ttk.LabelFrame(ph2, text="7.  Cross-Discipline Analysis")
        s7.pack(fill="x", padx=10, pady=6)
        row7 = tk.Frame(s7, bg=C["base"])
        row7.pack(fill="x", padx=8, pady=6)
        self._btn(row7, "Compare (Weak Labels)", self._run_compare_weak).pack(side="left", padx=4)
        self._btn(row7, "Compare (BERT)", self._run_compare_bert).pack(side="left", padx=4)

    # ══════════════════════════════════════════════════════════════════════════
    # Environment Tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_environment_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="  Environment  ")

        col1 = ttk.Frame(frame)
        col1.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        col2 = ttk.Frame(frame)
        col2.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # ── Check & Setup ─────────────────────────────────────────────────────
        check = ttk.LabelFrame(col1, text="Environment Check")
        check.pack(fill="x", pady=(0, 8))
        tk.Label(check, text="Verifies Python, packages, GPU, Kaggle credentials, and data dirs.",
                 bg=C["base"], fg=C["subtext1"], wraplength=340, justify="left").pack(
            padx=10, pady=(6, 2), anchor="w")
        self._btn(check, "↻  Run Environment Check", self._run_check_env,
                  bg=C["blue"], fg=C["base"], bold=True).pack(fill="x", padx=10, pady=(2, 10))

        setup = ttk.LabelFrame(col1, text="Setup Environment")
        setup.pack(fill="x", pady=(0, 8))
        tk.Label(setup, text="Creates conda env 'SSE691NLP' and installs all dependencies.",
                 bg=C["base"], fg=C["subtext1"], wraplength=340, justify="left").pack(
            padx=10, pady=(6, 2), anchor="w")
        self._btn(setup, "Run Setup  (create conda env)", self._run_setup_env,
                  bg=C["green"], fg=C["base"], bold=True).pack(fill="x", padx=10, pady=(2, 4))

        # Install requirements into the currently active Python (most common need)
        install_frame = ttk.LabelFrame(col1, text="Install Requirements")
        install_frame.pack(fill="x", pady=(0, 8))
        tk.Label(install_frame,
                 text="Installs requirements.txt into the active Python environment.\n"
                      "Use this when the env exists but packages are missing.",
                 bg=C["base"], fg=C["subtext1"], wraplength=340, justify="left").pack(
            padx=10, pady=(6, 2), anchor="w")
        self._btn(install_frame, "pip install -r requirements.txt", self._run_install_req,
                  bg=C["blue"], fg=C["base"], bold=True).pack(fill="x", padx=10, pady=(2, 10))

        # ── Hardware Info ─────────────────────────────────────────────────────
        hw = ttk.LabelFrame(col1, text="Hardware Info")
        hw.pack(fill="x", pady=(0, 8))
        self._hw_text = self._ro_text(hw, height=7)
        self._hw_text.pack(fill="x", padx=6, pady=4)
        self._btn(hw, "↻  Detect Hardware", self._refresh_hardware).pack(
            side="left", padx=8, pady=(0, 6))

        # ── Kaggle Credentials ────────────────────────────────────────────────
        kg = ttk.LabelFrame(col2, text="Kaggle Credentials")
        kg.pack(fill="x", pady=(0, 8))
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        found = kaggle_json.exists()
        tk.Label(kg,
                 text=f"kaggle.json: {'✓ Found' if found else '✗ Not found'}",
                 bg=C["base"], fg=C["green"] if found else C["red"],
                 font=("Helvetica", 10, "bold")).pack(padx=10, pady=(6, 2), anchor="w")
        tk.Label(kg, text=str(kaggle_json),
                 bg=C["base"], fg=C["subtext1"], font=("Courier New" if IS_WIN else "Monaco", 9)
                 ).pack(padx=10, anchor="w")
        self._btn(kg, "Open Folder", lambda: self._open_folder(kaggle_json.parent)
                  ).pack(padx=10, pady=(4, 10), anchor="w")

        # ── Python Info ───────────────────────────────────────────────────────
        py = ttk.LabelFrame(col2, text="Python / Conda Environment")
        py.pack(fill="x", pady=(0, 8))
        self._py_text = self._ro_text(py, height=6)
        self._py_text.pack(fill="x", padx=6, pady=4)
        self._update_py_info()

        # ── Danger Zone ───────────────────────────────────────────────────────
        dz = ttk.LabelFrame(col2, text="Cache & Cleanup")
        dz.pack(fill="x", pady=(0, 8))
        tk.Label(dz, text="⚠  These actions permanently delete files.",
                 bg=C["base"], fg=C["peach"], font=("Helvetica", 9, "bold")).pack(
            padx=10, pady=(6, 4), anchor="w")

        for label, cmd, color in [
            ("Clear Cache  (data/raw/)",     self._clear_cache,    C["peach"]),
            ("Clear Training  (models/)",    self._clear_training, C["peach"]),
            ("Clear Conda Environment",      self._clear_env,      C["red"]),
            ("Full Reset  (everything)",     self._full_reset,     C["red"]),
        ]:
            self._btn(dz, label, cmd, bg=color, fg=C["base"]).pack(
                fill="x", padx=10, pady=3)
        tk.Frame(dz, bg=C["base"]).pack(pady=2)

    # ══════════════════════════════════════════════════════════════════════════
    # Results Tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_results_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="  Results  ")

        sub = ttk.Notebook(frame)
        sub.pack(fill="both", expand=True, padx=5, pady=5)

        # ── Metrics ───────────────────────────────────────────────────────────
        mf = ttk.Frame(sub)
        sub.add(mf, text="  Metrics  ")

        mhdr = tk.Frame(mf, bg=C["base"])
        mhdr.pack(fill="x", padx=6, pady=6)
        self._btn(mhdr, "↻  Refresh", self._refresh_metrics).pack(side="left")
        tk.Label(mhdr, text=str(RESULTS / "comparison.json"),
                 bg=C["base"], fg=C["overlay0"], font=("Helvetica", 9)).pack(side="left", padx=10)

        self._metrics_text = self._ro_text(mf, font_size=11, mono=True)
        msb = ttk.Scrollbar(mf, command=self._metrics_text.yview)
        self._metrics_text.configure(yscrollcommand=msb.set)
        msb.pack(side="right", fill="y")
        self._metrics_text.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._refresh_metrics()

        # ── Charts ────────────────────────────────────────────────────────────
        cf = ttk.Frame(sub)
        sub.add(cf, text="  Charts  ")

        chdr = tk.Frame(cf, bg=C["base"])
        chdr.pack(fill="x", padx=6, pady=6)
        self._btn(chdr, "↻  Refresh", self._refresh_charts).pack(side="left", padx=(0, 8))
        self._v_chart = tk.StringVar()
        self._chart_cb = ttk.Combobox(chdr, textvariable=self._v_chart, width=40, state="readonly")
        self._chart_cb.pack(side="left")
        self._chart_cb.bind("<<ComboboxSelected>>", lambda _: self._show_chart())

        self._chart_canvas = tk.Canvas(cf, bg=C["surface0"], highlightthickness=0)
        self._chart_canvas.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._chart_lbl = tk.Label(self._chart_canvas, bg=C["surface0"], fg=C["subtext1"],
                                    font=("Helvetica", 11))
        self._chart_lbl.place(relx=0.5, rely=0.5, anchor="center")
        self._refresh_charts()

        # ── CSV Tables ────────────────────────────────────────────────────────
        csvf = ttk.Frame(sub)
        sub.add(csvf, text="  CSV Tables  ")

        csvhdr = tk.Frame(csvf, bg=C["base"])
        csvhdr.pack(fill="x", padx=6, pady=6)
        self._btn(csvhdr, "↻  Refresh", self._refresh_csv_list).pack(side="left", padx=(0, 8))
        self._v_csv = tk.StringVar()
        self._csv_cb = ttk.Combobox(csvhdr, textvariable=self._v_csv, width=40, state="readonly")
        self._csv_cb.pack(side="left")
        self._csv_cb.bind("<<ComboboxSelected>>", lambda _: self._show_csv())

        self._csv_text = self._ro_text(csvf, mono=True, font_size=10)
        csvsb = ttk.Scrollbar(csvf, command=self._csv_text.yview)
        self._csv_text.configure(yscrollcommand=csvsb.set)
        csvsb.pack(side="right", fill="y")
        self._csv_text.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._refresh_csv_list()

    # ══════════════════════════════════════════════════════════════════════════
    # Status Tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_status_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="  Status  ")

        col1 = ttk.Frame(frame)
        col1.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        col2 = ttk.Frame(frame)
        col2.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # ── File Status ───────────────────────────────────────────────────────
        ff = ttk.LabelFrame(col1, text="File Status")
        ff.pack(fill="both", expand=True, pady=(0, 8))

        fhdr = tk.Frame(ff, bg=C["base"])
        fhdr.pack(fill="x", padx=6, pady=4)
        self._btn(fhdr, "↻  Refresh", self._refresh_status).pack(side="left")

        self._file_text = self._ro_text(ff, mono=True, font_size=10)
        fsb = ttk.Scrollbar(ff, command=self._file_text.yview)
        self._file_text.configure(yscrollcommand=fsb.set)
        self._file_text.tag_configure("hdr",  foreground=C["blue"],  font=("Courier New" if IS_WIN else "Monaco", 10, "bold"))
        self._file_text.tag_configure("ok",   foreground=C["green"])
        self._file_text.tag_configure("miss", foreground=C["red"])
        self._file_text.tag_configure("warn", foreground=C["peach"])
        fsb.pack(side="right", fill="y")
        self._file_text.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # ── Corpus Stats ──────────────────────────────────────────────────────
        cs = ttk.LabelFrame(col2, text="Corpus Statistics")
        cs.pack(fill="x", pady=(0, 8))
        self._corpus_text = self._ro_text(cs, mono=True, height=8)
        self._corpus_text.pack(fill="x", padx=6, pady=4)
        self._btn(cs, "↻  Count Records", self._count_corpus).pack(
            side="left", padx=8, pady=(0, 6))

        # ── Model Info ────────────────────────────────────────────────────────
        mi = ttk.LabelFrame(col2, text="Model Checkpoints")
        mi.pack(fill="x", pady=(0, 8))
        self._model_info_text = self._ro_text(mi, mono=True, height=7)
        self._model_info_text.pack(fill="x", padx=6, pady=4)
        self._btn(mi, "↻  Check Model", self._check_model_info).pack(
            side="left", padx=8, pady=(0, 6))

    # ══════════════════════════════════════════════════════════════════════════
    # Settings Tab
    # ══════════════════════════════════════════════════════════════════════════

    def _build_settings_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="  Settings  ")

        col1 = ttk.Frame(frame)
        col1.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        col2 = ttk.Frame(frame)
        col2.pack(side="left", fill="x", padx=10, pady=10)

        # ── Config Editor ─────────────────────────────────────────────────────
        cf = ttk.LabelFrame(col1, text=f"Training Config  —  {CONFIG.name}")
        cf.pack(fill="both", expand=True, pady=(0, 8))

        cfhdr = tk.Frame(cf, bg=C["base"])
        cfhdr.pack(fill="x", padx=6, pady=4)
        self._btn(cfhdr, "↻  Load", self._load_config).pack(side="left", padx=(0, 4))
        self._btn(cfhdr, "💾  Save", self._save_config,
                  bg=C["green"], fg=C["base"], bold=True).pack(side="left")
        tk.Label(cfhdr, text=str(CONFIG), bg=C["base"], fg=C["overlay0"],
                 font=("Helvetica", 9)).pack(side="left", padx=10)

        self._config_text = tk.Text(
            cf, bg=C["crust"], fg="#e6edf3",
            font=("Courier New" if IS_WIN else "Monaco", 11),
            insertbackground="white", relief="flat",
        )
        cfvsb = ttk.Scrollbar(cf, command=self._config_text.yview)
        self._config_text.configure(yscrollcommand=cfvsb.set)
        cfvsb.pack(side="right", fill="y")
        self._config_text.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._load_config()

        # ── HuggingFace Token ─────────────────────────────────────────────────
        hf = ttk.LabelFrame(col2, text="HuggingFace Token  (HF_TOKEN)")
        hf.pack(fill="x", pady=(0, 8))

        tk.Label(hf,
                 text="Optional — prevents rate-limit warnings when\n"
                      "downloading models. Get yours at hf.co/settings/tokens.\n"
                      "Saved to .env (git-ignored).",
                 bg=C["base"], fg=C["subtext1"],
                 font=("Helvetica", 9), justify="left").pack(padx=10, pady=(6, 4), anchor="w")

        hf_row = tk.Frame(hf, bg=C["base"])
        hf_row.pack(fill="x", padx=10, pady=(0, 4))

        self._v_hf_token = tk.StringVar(value=self._hf_token)
        hf_entry = ttk.Entry(hf_row, textvariable=self._v_hf_token, show="•", width=34)
        hf_entry.pack(side="left", padx=(0, 6))

        self._hf_show = tk.BooleanVar(value=False)
        def _toggle_show():
            hf_entry.configure(show="" if self._hf_show.get() else "•")
        ttk.Checkbutton(hf_row, text="Show", variable=self._hf_show,
                        command=_toggle_show).pack(side="left")

        hf_btns = tk.Frame(hf, bg=C["base"])
        hf_btns.pack(fill="x", padx=10, pady=(0, 8))

        def _save_token():
            self._save_hf_token(self._v_hf_token.get())
            status = "✓ Saved" if self._hf_token else "✓ Cleared"
            messagebox.showinfo("HF Token", f"{status}\nAll pipeline runs will now include HF_TOKEN.")

        def _clear_token():
            self._v_hf_token.set("")
            self._save_hf_token("")
            messagebox.showinfo("HF Token", "Token cleared from .env.")

        self._btn(hf_btns, "Save Token", _save_token,
                  bg=C["green"], fg=C["base"], bold=True).pack(side="left", padx=(0, 6))
        self._btn(hf_btns, "Clear", _clear_token,
                  bg=C["surface0"], fg=C["text"]).pack(side="left")

        # Token source indicator
        src = "env var HF_TOKEN" if os.environ.get("HF_TOKEN") else \
              (".env file" if self._hf_token else "not set")
        self._hf_src_lbl = tk.Label(hf, text=f"Source: {src}",
                                     bg=C["base"], fg=C["overlay0"], font=("Helvetica", 9))
        self._hf_src_lbl.pack(padx=10, pady=(0, 4), anchor="w")

        # ── About ─────────────────────────────────────────────────────────────
        ab = ttk.LabelFrame(col2, text="About")
        ab.pack(fill="x", pady=(0, 8))
        tk.Label(ab, justify="left", bg=C["base"], fg=C["subtext1"],
                 font=("Helvetica", 10), text=(
            "SkillPostBERT\n"
            "ECE / SSE / CYS 691 — NLP Course Project\n\n"
            "Fine-tunes BERT for NER to extract skill\n"
            "mentions from engineering job postings.\n\n"
            "Skill categories:\n"
            "  TECHNICAL  domain knowledge & methods\n"
            "  TOOLS      software, languages, platforms\n"
            "  SOFT       communication, teamwork\n"
            "  CERT       licenses & certifications\n\n"
            "Disciplines compared:\n"
            "  ME  Mechanical Engineering\n"
            "  EE  Electrical Engineering\n"
            "  SE  Software Engineering\n\n"
            "Pipeline:\n"
            "  Phase 1: Download → Preprocess → Train\n"
            "           → Export Gold\n"
            "  [Manual: BIO Tag Editor review]\n"
            "  Phase 2: Apply → Evaluate → Compare"
        )).pack(padx=12, pady=10, anchor="w")

        # ── Python Info ───────────────────────────────────────────────────────
        py = ttk.LabelFrame(col2, text="Python / Runtime")
        py.pack(fill="x", pady=(0, 8))
        py_info = self._ro_text(py, mono=True, height=5)
        py_info.pack(fill="x", padx=6, pady=4)
        py_info.configure(state="normal")
        py_info.insert("end", "\n".join([
            f"Python    {sys.version.split()[0]}",
            f"Exec      {sys.executable}",
            f"Platform  {platform.system()} {platform.release()} {platform.machine()}",
            f"CONDA     {os.environ.get('CONDA_PREFIX', '(none)')}",
            f"ROOT      {ROOT}",
        ]))
        py_info.configure(state="disabled")

    # ══════════════════════════════════════════════════════════════════════════
    # Script Runners
    # ══════════════════════════════════════════════════════════════════════════

    def _python(self) -> str:
        """Return the best available python executable."""
        prefix = os.environ.get("CONDA_PREFIX", "")
        if prefix:
            p = Path(prefix) / ("python.exe" if IS_WIN else "bin/python")
            if p.exists():
                return str(p)
        return sys.executable

    def _run(self, args: list, label: str = "Running"):
        """Launch a subprocess and stream its output to the console."""
        if self.current_proc and self.current_proc.poll() is None:
            messagebox.showwarning("Busy", "A process is already running.\nStop it first.")
            return

        ts = datetime.now().strftime("%H:%M:%S")
        self._clog(f"\n[{ts}]  {label}\n", "info")
        self._clog(f"  cmd: {' '.join(str(a) for a in args)}\n", "info")
        self._clog("─" * 64 + "\n")
        self._proc_lbl.configure(text="Running…", fg=C["yellow"])
        self._status_bar.configure(text=f"Running: {label}")

        def worker():
            try:
                env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1",
                       **self._hf_env()}
                self.current_proc = subprocess.Popen(
                    args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=str(ROOT), env=env, bufsize=1,
                )
                for line in self.current_proc.stdout:
                    tag = self._line_tag(line)
                    self.output_q.put((line, tag))
                rc = self.current_proc.wait()
                if rc == 0:
                    self.output_q.put((f"\n✓ {label} completed\n", "ok"))
                    self.after(0, lambda: self._proc_lbl.configure(text="Done", fg=C["green"]))
                    self.after(0, lambda: self._status_bar.configure(text=f"Done: {label}"))
                    self.after(0, self._refresh_status)
                else:
                    self.output_q.put((f"\n✗ {label} exited with code {rc}\n", "err"))
                    self.after(0, lambda: self._proc_lbl.configure(text="Failed", fg=C["red"]))
                    self.after(0, lambda: self._status_bar.configure(text=f"Failed: {label}"))
            except Exception as exc:
                self.output_q.put((f"\nError: {exc}\n", "err"))
                self.after(0, lambda: self._proc_lbl.configure(text="Error", fg=C["red"]))

        threading.Thread(target=worker, daemon=True).start()

    def _run_seq(self, steps: list[tuple[list, str]]):
        """Run multiple (args, label) steps sequentially in a thread."""
        if self.current_proc and self.current_proc.poll() is None:
            messagebox.showwarning("Busy", "A process is already running.")
            return

        self._proc_lbl.configure(text="Running…", fg=C["yellow"])

        def worker():
            for args, label in steps:
                ts = datetime.now().strftime("%H:%M:%S")
                self.output_q.put((f"\n[{ts}]  ══  {label}  ══\n", "info"))
                env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1",
                       **self._hf_env()}
                self.current_proc = subprocess.Popen(
                    args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=str(ROOT), env=env, bufsize=1,
                )
                for line in self.current_proc.stdout:
                    self.output_q.put((line, self._line_tag(line)))
                if self.current_proc.wait() != 0:
                    self.output_q.put((f"\n✗ {label} failed — pipeline stopped.\n", "err"))
                    self.after(0, lambda: self._proc_lbl.configure(text="Failed", fg=C["red"]))
                    return
            self.output_q.put(("\n✓ All steps completed successfully!\n", "ok"))
            self.after(0, lambda: self._proc_lbl.configure(text="Done", fg=C["green"]))
            self.after(0, self._refresh_status)

        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _line_tag(line: str) -> str | None:
        lo = line.lower()
        if any(w in lo for w in ("error", "traceback", "exception", "failed", "✗")):
            return "err"
        if any(w in lo for w in ("warning", "warn")):
            return "warn"
        if any(w in lo for w in ("epoch", "f1", "precision", "recall", "saved",
                                   "complete", "done", "✓", "step")):
            return "ok"
        return None

    # ── Individual script launchers ───────────────────────────────────────────

    def _run_download(self):
        self._run([self._python(), "-m", "src.download_data"], "Download Data")

    def _run_preprocess(self):
        self._run([self._python(), "-m", "src.preprocess",
                   "--max", self._v_pp_max.get()], "Preprocess")

    def _run_train(self):
        self._run([
            self._python(), "-m", "src.train",
            "--model", self._v_model.get(),
            "--epochs", self._v_epochs.get(),
            "--learning-rate", self._v_lr.get(),
            "--train-batch-size", self._v_batch.get(),
        ], "Train BERT")

    def _run_export_gold(self):
        self._run([self._python(), "-m", "src.evaluate",
                   "--export-gold", "--n", self._v_gold_n.get()], "Export Gold Set")

    def _run_apply_conll(self):
        self._run([self._python(), "-m", "src.evaluate",
                   "--apply-conll", self._v_conll.get()], "Apply CoNLL Corrections")

    def _run_evaluate(self):
        self._run([
            self._python(), "-m", "src.evaluate",
            "--gold", str(DATA / "processed" / "gold.jsonl"),
            "--model", self._v_eval_model.get(),
        ], "Evaluate BERT vs Baseline")

    def _run_compare_weak(self):
        self._run([self._python(), "-m", "src.compare",
                   "--source", "weak"], "Compare (Weak Labels)")

    def _run_compare_bert(self):
        self._run([
            self._python(), "-m", "src.compare",
            "--source", "bert", "--model", self._v_eval_model.get(),
        ], "Compare (BERT)")

    def _run_phase1(self):
        if not messagebox.askyesno(
            "Run Full Phase 1",
            "This will run:\n  Download → Preprocess → Train → Export Gold\n\n"
            "Training can take a long time. Continue?",
        ):
            return
        py = self._python()
        self._run_seq([
            ([py, "-m", "src.download_data"], "Download Data"),
            ([py, "-m", "src.preprocess", "--max", self._v_pp_max.get()], "Preprocess"),
            ([py, "-m", "src.train",
              "--model", self._v_model.get(),
              "--epochs", self._v_epochs.get(),
              "--learning-rate", self._v_lr.get(),
              "--train-batch-size", self._v_batch.get()], "Train BERT"),
            ([py, "-m", "src.evaluate",
              "--export-gold", "--n", self._v_gold_n.get()], "Export Gold Set"),
        ])

    def _run_phase2(self):
        py = self._python()
        self._run_seq([
            ([py, "-m", "src.evaluate",
              "--apply-conll", self._v_conll.get()], "Apply CoNLL"),
            ([py, "-m", "src.evaluate",
              "--gold", str(DATA / "processed" / "gold.jsonl"),
              "--model", self._v_eval_model.get()], "Evaluate"),
            ([py, "-m", "src.compare",
              "--source", "bert",
              "--model", self._v_eval_model.get()], "Compare (BERT)"),
        ])

    def _launch_bio_editor(self):
        """Start the Flask BIO editor in a background thread."""
        self._clog("\n[BIO Editor] Starting Flask at http://localhost:5050 …\n", "info")
        env = {**os.environ, "PYTHONPATH": str(ROOT), **self._hf_env()}
        try:
            proc = subprocess.Popen(
                [self._python(), str(TOOLS / "bio_editor.py")],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(ROOT), env=env, bufsize=1,
            )

            def stream():
                for line in proc.stdout:
                    self.output_q.put((line, None))
                    if "Running on" in line or "5050" in line:
                        self.after(600, lambda: webbrowser.open("http://localhost:5050"))

            threading.Thread(target=stream, daemon=True).start()
        except Exception as exc:
            self._clog(f"Error: {exc}\n", "err")

    # ── Environment helpers ───────────────────────────────────────────────────

    def _run_check_env(self):
        """Run environment check via the detected Python (CONDA_PREFIX-aware, cross-platform)."""
        root_escaped = str(ROOT).replace("\\", "\\\\")
        check_code = f"""
import sys, importlib.util, importlib, os, json
from pathlib import Path

ROOT = Path(r"{root_escaped}")
OK   = "[ok]  "; FAIL = "[FAIL]"; WARN = "[warn]"

print("=" * 66)
print(" SkillPostBERT -- environment check")
print("=" * 66)
print()
print("[python]")
print(f"  {{OK}} Python {{sys.version.split()[0]}}")
print(f"  {{OK}} Executable: {{sys.executable}}")

packages = [
    ("torch",        "torch",        True),
    ("transformers", "transformers", True),
    ("accelerate",   "accelerate",   True),
    ("datasets",     "datasets",     True),
    ("evaluate",     "evaluate",     True),
    ("seqeval",      "seqeval",      True),
    ("pandas",       "pandas",       False),
    ("scikit-learn", "sklearn",      False),
    ("matplotlib",   "matplotlib",   False),
    ("kaggle",       "kaggle",       True),
    ("pyyaml",       "yaml",         False),
    ("flask",        "flask",        False),
    ("spacy",        "spacy",        False),
]

fails = 0
for pkg_name, import_name, required in packages:
    if importlib.util.find_spec(import_name) is not None:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            print(f"  {{OK}} {{pkg_name}}: {{ver}}")
        except Exception:
            print(f"  {{OK}} {{pkg_name}}")
    else:
        tag = FAIL if required else WARN
        print(f"  {{tag}} {{pkg_name}}  <-- pip install -r requirements.txt")
        if required: fails += 1

print()
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  {{OK}} CUDA GPU: {{name}}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"  {{OK}} Apple MPS available")
    else:
        print(f"  {{WARN}} No GPU -- CPU mode")
except ImportError:
    print(f"  {{WARN}} GPU check skipped: torch not installed")

print()
print("[kaggle]")
kj = Path.home() / ".kaggle" / "kaggle.json"
if kj.exists():
    try:
        u = json.loads(kj.read_text(encoding="utf-8")).get("username", "?")
        print(f"  {{OK}} kaggle.json present (user: {{u}})")
    except Exception:
        print(f"  {{OK}} kaggle.json present")
else:
    print(f"  {{FAIL}} kaggle.json missing: {{kj}}")
    fails += 1

print()
print("[directories]")
raw = ROOT / "data" / "raw"
if raw.exists():
    csvs = list(raw.glob("*.csv"))
    print(f"  {{OK}} data/raw/ -- {{len(csvs)}} CSV file(s)")
else:
    print(f"  {{WARN}} data/raw/ missing -- run Download")

for label, rel in [
    ("corpus.jsonl", "data/processed/corpus.jsonl"),
    ("gold.conll",   "data/processed/gold.conll"),
]:
    p = ROOT / rel
    if p.exists():
        print(f"  {{OK}} {{label}} ({{p.stat().st_size // 1024}} KB)")
    else:
        print(f"  {{WARN}} {{label}} not yet generated")

mdir = ROOT / "models"
if mdir.exists():
    ms = [d.name for d in mdir.iterdir() if d.is_dir()]
    print(f"  {{OK}} models/ -- {{len(ms)}} model(s): {{', '.join(ms) or '(none)'}}")
else:
    print(f"  {{WARN}} models/ not yet created")

rdir = ROOT / "results"
if rdir.exists():
    print(f"  {{OK}} results/ -- {{len(list(rdir.iterdir()))}} file(s)")
else:
    print(f"  {{WARN}} results/ not yet created")

print()
print("=" * 66)
if fails == 0:
    print(" Environment OK -- ready to run the pipeline")
else:
    print(f" {{fails}} issue(s) found.")
    print(" Run 'Install Requirements' or: pip install -r requirements.txt")
print("=" * 66)
sys.exit(1 if fails else 0)
"""
        self._run([self._python(), "-c", check_code], "Environment Check")

    def _run_setup_env(self):
        if not messagebox.askyesno("Setup Environment",
                                    "Create conda env 'SSE691NLP' and install all dependencies?"):
            return
        if IS_WIN:
            self._run(["powershell", "-ExecutionPolicy", "Bypass",
                       "-File", str(SCRIPTS / "setup_env.ps1")], "Setup Environment")
        else:
            self._run(["bash", str(SCRIPTS / "setup_env.sh")], "Setup Environment")

    def _run_install_req(self):
        """Install requirements.txt into the active Python (self._python())."""
        py = self._python()
        req = ROOT / "requirements.txt"
        if not req.exists():
            messagebox.showerror("Not Found", f"requirements.txt not found at:\n{req}")
            return
        self._run(
            [py, "-m", "pip", "install", "-r", str(req)],
            "Install Requirements",
        )

    def _clear_cache(self):
        if messagebox.askyesno("Clear Cache", "Delete data/raw/?  Re-download will be needed."):
            self._shell_script("clear_cache", "Clear Cache")

    def _clear_training(self):
        if messagebox.askyesno("Clear Training", "Delete models/?  Re-training will be needed."):
            self._shell_script("clear_training", "Clear Training")

    def _clear_env(self):
        if messagebox.askyesno("Clear Environment", "Remove conda env 'SSE691NLP'?"):
            self._shell_script("clear_env", "Clear Environment")

    def _full_reset(self):
        if messagebox.askyesno(
            "Full Reset",
            "DELETE data/raw/, models/, results/, and the conda env?\n\n"
            "This CANNOT be undone.",
            icon="warning",
        ):
            self._shell_script("reset", "Full Reset")

    def _shell_script(self, name: str, label: str):
        if IS_WIN:
            self._run(["powershell", "-ExecutionPolicy", "Bypass",
                       "-File", str(SCRIPTS / f"{name}.ps1")], label)
        else:
            self._run(["bash", str(SCRIPTS / f"{name}.sh")], label)

    def _refresh_hardware(self):
        code = (
            "from src.utils import get_hardware_profile; import json; "
            "print(json.dumps(get_hardware_profile(), indent=2))"
        )
        self._hw_text.configure(state="normal")
        self._hw_text.delete("1.0", "end")
        try:
            r = subprocess.run(
                [self._python(), "-c", code],
                capture_output=True, text=True, cwd=str(ROOT),
                env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=20,
            )
            self._hw_text.insert("end", r.stdout if r.returncode == 0 else r.stderr)
        except Exception as exc:
            self._hw_text.insert("end", f"Error: {exc}")
        self._hw_text.configure(state="disabled")

    def _update_py_info(self):
        self._py_text.configure(state="normal")
        self._py_text.delete("1.0", "end")
        self._py_text.insert("end", "\n".join([
            f"Python    {sys.version.split()[0]}",
            f"Exec      {sys.executable}",
            f"Platform  {platform.system()} {platform.machine()}",
            f"CONDA     {os.environ.get('CONDA_PREFIX', '(none)')}",
        ]))
        self._py_text.configure(state="disabled")

    # ── Results helpers ───────────────────────────────────────────────────────

    def _refresh_metrics(self):
        comp = RESULTS / "comparison.json"
        self._metrics_text.configure(state="normal")
        self._metrics_text.delete("1.0", "end")
        if comp.exists():
            try:
                self._metrics_text.insert("end", json.dumps(json.loads(comp.read_text(encoding="utf-8")), indent=2))
            except Exception as exc:
                self._metrics_text.insert("end", f"Parse error: {exc}")
        else:
            self._metrics_text.insert("end",
                f"No results yet.\nExpected: {comp}\n\nRun Phase 2 to generate evaluation results.")
        self._metrics_text.configure(state="disabled")

    def _refresh_charts(self):
        pngs = sorted(RESULTS.glob("*.png")) if RESULTS.exists() else []
        self._chart_map = {p.name: p for p in pngs}
        self._chart_cb.configure(values=[p.name for p in pngs])
        if pngs:
            self._v_chart.set(pngs[0].name)
            self._show_chart()
        else:
            self._chart_lbl.configure(text="No charts found.\nRun Phase 2 to generate visualizations.")

    def _show_chart(self):
        name = self._v_chart.get()
        path = getattr(self, "_chart_map", {}).get(name)
        if not path or not path.exists():
            return
        try:
            from PIL import Image, ImageTk
            self._chart_canvas.update_idletasks()
            w = max(self._chart_canvas.winfo_width() - 20, 500)
            h = max(self._chart_canvas.winfo_height() - 20, 380)
            img = Image.open(path)
            img.thumbnail((w, h), Image.LANCZOS)
            self._chart_photo = ImageTk.PhotoImage(img)
            self._chart_lbl.configure(image=self._chart_photo, text="")
        except ImportError:
            self._chart_lbl.configure(
                image="",
                text=(f"Pillow not installed — cannot display image.\n"
                      f"Install with: pip install Pillow\n\nChart saved at:\n{path}"),
                fg=C["peach"],
            )
        except Exception as exc:
            self._chart_lbl.configure(image="", text=f"Error: {exc}", fg=C["red"])

    def _refresh_csv_list(self):
        csvs = sorted(RESULTS.glob("*.csv")) if RESULTS.exists() else []
        self._csv_map = {p.name: p for p in csvs}
        self._csv_cb.configure(values=[p.name for p in csvs])
        if csvs:
            self._v_csv.set(csvs[0].name)
            self._show_csv()

    def _show_csv(self):
        name = self._v_csv.get()
        path = getattr(self, "_csv_map", {}).get(name)
        self._csv_text.configure(state="normal")
        self._csv_text.delete("1.0", "end")
        if path and path.exists():
            try:
                self._csv_text.insert("end", path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._csv_text.insert("end", f"Error: {exc}")
        else:
            self._csv_text.insert("end", "File not found.")
        self._csv_text.configure(state="disabled")

    # ── Status helpers ────────────────────────────────────────────────────────

    def _refresh_status(self):
        checks = [
            ("data/raw/postings.csv",          DATA / "raw" / "postings.csv"),
            ("data/processed/corpus.jsonl",    DATA / "processed" / "corpus.jsonl"),
            ("data/processed/gold.conll",      DATA / "processed" / "gold.conll"),
            ("data/processed/gold.jsonl",      DATA / "processed" / "gold.jsonl"),
            ("models/bert-skills-ner/",        MODELS / "bert-skills-ner"),
            ("results/comparison.json",        RESULTS / "comparison.json"),
        ]

        try:
            self._file_text.configure(state="normal")
            self._file_text.delete("1.0", "end")
            self._file_text.insert("end", "Core Files\n", "hdr")
            self._file_text.insert("end", "─" * 54 + "\n")

            data_ok = model_ok = results_ok = True

            for name, path in checks:
                if path.exists():
                    if path.is_file():
                        sz = path.stat().st_size
                        sz_s = f"{sz/1024:.1f} KB" if sz < 1_048_576 else f"{sz/1_048_576:.1f} MB"
                        status, tag = f"✓  {sz_s}", "ok"
                    else:
                        count = sum(1 for _ in path.iterdir())
                        status, tag = f"✓  ({count} items)", "ok"
                    if "bert-skills-ner" in name:
                        model_ok = True
                    if "comparison.json" in name:
                        results_ok = True
                else:
                    status, tag = "✗  Missing", "miss"
                    if "postings.csv" in name or "corpus.jsonl" in name:
                        data_ok = False
                    if "bert-skills-ner" in name:
                        model_ok = False
                    if "comparison.json" in name:
                        results_ok = False

                self._file_text.insert("end", f"  {name:<38}  {status}\n", tag)

            # PNG charts
            pngs = sorted(RESULTS.glob("*.png")) if RESULTS.exists() else []
            self._file_text.insert("end", "\nResults Charts\n", "hdr")
            self._file_text.insert("end", "─" * 54 + "\n")
            if pngs:
                for p in pngs:
                    sz = p.stat().st_size
                    self._file_text.insert("end",
                        f"  {p.name:<38}  ✓  {sz/1024:.1f} KB\n", "ok")
            else:
                self._file_text.insert("end", "  (none generated yet)\n", "miss")

            self._file_text.configure(state="disabled")

            # Header indicators
            self._lbl_data.configure( fg=C["green"] if data_ok  else C["red"])
            self._lbl_model.configure(fg=C["green"] if model_ok else C["yellow"])

            # Check conda env (non-blocking best-effort)
            try:
                r = subprocess.run(["conda", "env", "list"], capture_output=True,
                                    text=True, timeout=4)
                env_ok = ENV_NAME in r.stdout
            except Exception:
                env_ok = False
            self._lbl_env.configure(fg=C["green"] if env_ok else C["yellow"])

        except Exception:
            pass

    def _count_corpus(self):
        corpus = DATA / "processed" / "corpus.jsonl"
        self._corpus_text.configure(state="normal")
        self._corpus_text.delete("1.0", "end")
        if not corpus.exists():
            self._corpus_text.insert("end", "corpus.jsonl not found.\nRun Preprocess first.")
        else:
            try:
                counts: dict[str, int] = {}
                total = 0
                with corpus.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        total += 1
                        try:
                            rec = json.loads(line)
                            disc = rec.get("discipline", "unknown")
                            counts[disc] = counts.get(disc, 0) + 1
                        except json.JSONDecodeError:
                            pass
                self._corpus_text.insert("end", f"Total records: {total}\n\n")
                self._corpus_text.insert("end", "By discipline:\n")
                for disc, cnt in sorted(counts.items()):
                    self._corpus_text.insert("end", f"  {disc:<12}  {cnt:>5}\n")
            except Exception as exc:
                self._corpus_text.insert("end", f"Error: {exc}")
        self._corpus_text.configure(state="disabled")

    def _check_model_info(self):
        mdir = MODELS / "bert-skills-ner"
        self._model_info_text.configure(state="normal")
        self._model_info_text.delete("1.0", "end")
        if not mdir.exists():
            self._model_info_text.insert("end", "No model directory found.\nRun training first.")
        else:
            try:
                state_f = mdir / "trainer_state.json"
                if state_f.exists():
                    state = json.loads(state_f.read_text(encoding="utf-8"))
                    self._model_info_text.insert("end",
                        f"Best F1:      {state.get('best_metric', 'N/A')}\n"
                        f"Checkpoint:   {Path(state.get('best_model_checkpoint', '') or '').name or 'N/A'}\n"
                        f"Global step:  {state.get('global_step', 'N/A')}\n\n"
                    )
                cps = sorted(mdir.glob("checkpoint-*"))
                self._model_info_text.insert("end", f"Checkpoints found: {len(cps)}\n")
                for cp in cps[-5:]:
                    self._model_info_text.insert("end", f"  {cp.name}\n")
            except Exception as exc:
                self._model_info_text.insert("end", f"Error: {exc}")
        self._model_info_text.configure(state="disabled")

    # ── Config helpers ────────────────────────────────────────────────────────

    # ── HuggingFace token ─────────────────────────────────────────────────────

    _ENV_FILE = ROOT / ".env"

    def _load_hf_token(self) -> str:
        """Read HF_TOKEN from env var or .env file."""
        if tok := os.environ.get("HF_TOKEN", ""):
            return tok
        if self._ENV_FILE.exists():
            for line in self._ENV_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    return line[len("HF_TOKEN="):].strip().strip('"').strip("'")
        return ""

    def _save_hf_token(self, token: str):
        """Persist HF_TOKEN to .env (creates or updates the file)."""
        token = token.strip()
        lines: list[str] = []
        if self._ENV_FILE.exists():
            lines = self._ENV_FILE.read_text(encoding="utf-8").splitlines()
        new_lines = [l for l in lines if not l.startswith("HF_TOKEN=")]
        if token:
            new_lines.append(f"HF_TOKEN={token}")
        self._ENV_FILE.write_text("\n".join(new_lines) + ("\n" if new_lines else ""),
                                   encoding="utf-8")
        self._hf_token = token

    def _hf_env(self) -> dict:
        """Return env-var dict to inject into subprocesses."""
        tok = self._hf_token or os.environ.get("HF_TOKEN", "")
        return {"HF_TOKEN": tok} if tok else {}

    # ── Config ────────────────────────────────────────────────────────────────

    def _load_config(self):
        self._config_text.delete("1.0", "end")
        if CONFIG.exists():
            self._config_text.insert("end", CONFIG.read_text(encoding="utf-8"))
        else:
            self._config_text.insert("end", f"# Config not found: {CONFIG}\n")

    def _save_config(self):
        content = self._config_text.get("1.0", "end-1c")
        try:
            import yaml
            yaml.safe_load(content)
        except Exception as exc:
            messagebox.showerror("Invalid YAML", f"Cannot save — YAML parse error:\n{exc}")
            return
        try:
            CONFIG.parent.mkdir(parents=True, exist_ok=True)
            CONFIG.write_text(content, encoding="utf-8")
            messagebox.showinfo("Saved", f"Config saved to:\n{CONFIG}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    # ── Console ───────────────────────────────────────────────────────────────

    def _clog(self, text: str, tag: str | None = None):
        self.output_q.put((text, tag))

    def _poll_output(self):
        """Drain the output queue into the console widget."""
        try:
            while True:
                text, tag = self.output_q.get_nowait()
                self.console.configure(state="normal")
                self.console.insert("end", text, tag or "")
                self.console.see("end")
                self.console.configure(state="disabled")
        except queue.Empty:
            pass
        finally:
            self.after(80, self._poll_output)

    def _clear_console(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def _stop_proc(self):
        if self.current_proc and self.current_proc.poll() is None:
            self.current_proc.terminate()
            self._clog("\n[Stopped by user]\n", "warn")
            self._proc_lbl.configure(text="Stopped", fg=C["peach"])

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _btn(self, parent, text, command, bg=None, fg=None, bold=False, small=False):
        kw = dict(
            text=text, command=command, relief="flat",
            bg=bg or C["surface0"], fg=fg or C["text"],
            font=("Helvetica", 8 if small else 10, "bold" if bold else "normal"),
            padx=4 if small else 8, pady=2 if small else 4,
            cursor="hand2",
        )
        b = tk.Button(parent, **kw)
        orig_bg = bg or C["surface0"]
        b.bind("<Enter>", lambda _: b.configure(bg=C["surface1"]))
        b.bind("<Leave>", lambda _: b.configure(bg=orig_bg))
        return b

    def _ro_text(self, parent, height=None, mono=False, font_size=10):
        """Return a read-only styled Text widget (not packed)."""
        font = ("Courier New" if IS_WIN else "Monaco", font_size) if mono else ("Helvetica", font_size)
        kw = dict(bg=C["surface0"], fg=C["text"], font=font,
                  state="disabled", relief="flat", wrap="word",
                  insertbackground=C["text"])
        if height:
            kw["height"] = height
        return tk.Text(parent, **kw)

    # ── File / folder dialogs ─────────────────────────────────────────────────

    def _browse_file(self, var: tk.StringVar, filetypes: list):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _browse_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    @staticmethod
    def _open_folder(path: Path):
        if IS_WIN:
            os.startfile(str(path))
        elif IS_MAC:
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])


# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = SkillPostBERTHub()
    app.mainloop()


if __name__ == "__main__":
    main()
