# 🧠💸 Budget Allocation

Budget allocation experiments for PETS, including offline evaluation and online streaming allocation.

## 🌟 Teaser

![Budget Allocation Teaser](assets/teaser2.png)

📄 Full PDF: [`assets/teaser2.pdf`](assets/teaser2.pdf)

## 🗂️ Structure

```text
PETS/budget_allocation/
├── README.md
├── MultiChoice_offline.py         # Offline OKG for multiple-choice predictions
├── FillintheBlank_offline.py      # Offline OKG for fill-in / numeric predictions
├── MultiChoice_online.py          # Online/streaming allocation (multiple-choice)
├── FillintheBlank_online.py       # Online/streaming allocation (fill-in)
├── multi_run_export.py            # Shared multi-run JSONL export (offline)
├── oracle_kmeans_common.py        # Shared KMeans/oracle allocation helpers (online)
├── assets/
│   ├── teaser2.png
│   └── teaser2.pdf
└── plots/
    ├── __init__.py
    ├── common.py                  # Shared plotting style + matplotlib setup
    ├── offline_curves.py          # Offline multi-run curve plotting + CSV export
    └── online_sweep.py            # Online sweep plotting + CSV export
```

## ✨ What Was Refactored

- 🎨 Plotting code moved out of experiment scripts into `plots/`.
- 🔁 Offline scripts now share multi-run export logic via `multi_run_export.py`.
- 🧩 Online scripts now share KMeans oracle helpers via `oracle_kmeans_common.py`.
- 🧼 Core allocation logic remains in main scripts, while shared infra is isolated.

## 🚀 Quick Start (Offline)

Run from `PETS/budget_allocation`:

```bash
# Multi-choice offline
python MultiChoice_offline.py \
  --preds /path/to/gpqa_preds.jsonl \
  --B 64 \
  --multi_runs 10 \
  --with_baseline

# Fill-in/offline
python FillintheBlank_offline.py \
  --preds /path/to/aime_preds.jsonl \
  --B 64 \
  --multi_runs 10 \
  --with_baseline
```

Both scripts can output:
- 📈 aggregated consistency/accuracy plots (`--consistency_plot`, `--accuracy_plot`)
- 🧾 corresponding CSV summaries (`--consistency_csv`, `--accuracy_csv`)
- 🗃️ optional multi-run JSONL stats (`--multi_run_jsonl`)

## 📥 Input Format (Offline)

Input is prediction JSONL where each line contains (at minimum):

- `id`: question id
- `answers`: sampled answers list
- label fields:
  - multiple-choice: typically `answer`
  - fill-in: typically `correct_answer` (fallback fields are handled in script)
- optional confidence traces:
  - `trace_confidence` for confidence-weighted variants

## ⚠️ Notes

- `MultiChoice_online.py` depends on `gpqa_streaming.py`.
- `FillintheBlank_online.py` depends on `mmlu_streaming.py`.
- If these base modules are not in your `PYTHONPATH`, online scripts will fail at import time.
