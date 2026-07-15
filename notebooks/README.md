# Notebooks — piano e stato

Piano derivato dalla review esterna del 2026-07-15
(`docs/external_review_2026-07-15.md`, §6) e adattato allo stato reale della
pipeline. **Regola**: i notebook visualizzano e analizzano; tutta la logica
riutilizzabile vive in `src/`. Nessun notebook addestra modelli.

Esecuzione: `uv run jupyter lab` (o aprire i `.ipynb` in VS Code usando il
kernel `.venv`). I notebook leggono solo artefatti già presenti
(`data/processed/`, `results/`); se un artefatto manca, la cella lo dichiara
e prosegue.

## Stato

| # | Notebook | Stato | Dipende da |
|---|---|---|---|
| 00 | `00_project_status.ipynb` | ✅ eseguibile ora | git, manifest, results/ |
| 01 | `01_dataset_audit.ipynb` | ✅ eseguibile ora | `data/processed/` v2 |
| 02 | tokenization & layout | ⏳ pianificato | tokenizer in cache |
| 03 | `03_attention_bottleneck_visualizer.ipynb` | ✅ eseguibile ora | `src/bottleneck.py` (celle Qwen opzionali) |
| 04 | exp0 baselines | 🔒 dopo Exp 0 v2 | prediction records per-esempio |
| 05 | training diagnostics | 🔒 dopo Exp 1b | TensorBoard events v2 |
| 06 | distance & relay ablation | 🔒 dopo Exp 2 | risultati Exp 2 + mask anchor-only |
| 07 | layerwise representation | 🔒 dopo Exp 3 | probe artifacts v2 |
| 08 | causal intervention | 🔒 dopo Exp 5 | intervention records |
| 09 | efficiency & rate-distortion | 🔒 dopo fast-path KV | decoder KV-pruned validato |
| 10 | error atlas | 🔒 dopo Exp 2 | predizioni etichettate per failure mode |
| 11 | paper figures | 🔒 ultimo | tutti gli artefatti congelati |

I notebook 🔒 hanno una ragione precisa per non esistere ancora: costruirli
prima degli artefatti che devono leggere produrrebbe codice morto da riscrivere
(stessa logica dei gate in CLAUDE.md). Il contenuto previsto per ciascuno è
specificato nella review esterna §6.

## Convenzioni

- Figure salvate in `results/figures/` (PNG, 150 dpi), tabelle in
  `results/tables/` (CSV), report machine-readable in `results/reports/`.
- Un solo hue per grafici a serie singola (`#2a78d6`); mask heatmap con
  due toni etichettati (blocked/allowed) — mai informazione affidata al solo
  colore.
- Ogni notebook stampa in testa: commit git, dirty flag, hash del manifest
  dati, così ogni figura è tracciabile allo stato che l'ha prodotta.
