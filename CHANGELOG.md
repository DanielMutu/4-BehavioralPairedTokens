# Changelog — Behavioral Paired Tokens

Il progetto non ha release: le voci seguono le milestone sperimentali e i branch.
Le decisioni con motivazione completa stanno in `experiments/decisions.md`;
questo file traccia **cosa è cambiato, file per file, e cosa è verificato**.

## [Unreleased] — branch `feat/true-compress-bottleneck-v2` (avviato 2026-07-15)

> ⚠️ **Stato: work-in-progress interrotto a metà.** La sessione autonoma che
> eseguiva il piano v2 è morta per crediti OpenRouter esauriti subito dopo un
> fix a `tests/test_bottleneck.py` mai ri-verificato. **La suite di test NON è
> garantita verde.** Parte del codice è stata generata col proxy in modalità
> OpenRouter (backend potenzialmente DeepSeek) e richiede review prima di
> essere considerata affidabile. Dettagli: `experiments/decisions.md`
> (2026-07-15) e sezione "Pipeline v2" in `CLAUDE.md`.

### Added

- `pyproject.toml` — progetto hatchling, dependency group `dev`, pin torch
  CPU via indice PyTorch dedicato (`tool.uv.sources` / `tool.uv.index`),
  config pytest con marker `integration`.
- `uv.lock` — ambiente bloccato realmente testato (`torch 2.12.0+cpu`).
- `Dockerfile` (riscritto) — lockfile-first con `uv sync --frozen`, base image
  pinnata per digest, `CMD` di default = `pytest` (il training è un comando
  esplicito, non il default).
- `.github/workflows/ci.yml` — CI CPU: pytest senza download di modelli grandi.
- `tests/` — `conftest.py` (fixture contratto + fake tokenizer),
  `test_data_contract.py`, `test_bottleneck.py` (truth-table della mask,
  layout validation, gradient flow, invarianza). **Ultima esecuzione:
  parzialmente rossa; fix applicato ma mai rieseguito.**
- `src/data_contract.py` — schema v2, `example_id`/`content_id` canonici,
  I/O JSONL atomico UTF-8, errori `path:line`, `assert_disjoint`, manifest
  con hash e provenance. **Mai eseguito.**
- `src/bottleneck.py` — mask additiva 4D `(B,1,T,T)` con invariante
  post-`[COMPRESS]`, layout validation centralizzata, decoder greedy
  reference full-recomputation (`use_cache=False`), boundary detection per
  `option_loglik`. **Mai eseguito oltre l'unica run pytest parziale.**
- `CHANGELOG.md` — questo file.

### Changed

- `data/generation/prepare_dataset.py` — riscritto: split disgiunti
  `[eval | test_in_style | train]` sul pool sintetico, dedupe per
  `content_id` (prima: primi 300 caratteri), held-out-style solo nel test,
  probe derivato solo da eval+test, manifest + modalità `--check`.
  Corregge il **leakage v0**: 148 contesti del train finivano in test e probe.
- `requirements.txt` — da range aperti a pin esatti.
  ⚠️ Incoerenza nota: pinna `torch==2.12.0` (PyPI → wheel CUDA) mentre il
  lock usa `+cpu`; da riallineare (preferire `uv sync --dev`).
- `.git/info/exclude` — esclusa `.claude/` (worktree e settings di sessione).
- `experiments/decisions.md` — voci 2026-07-15 (avvio v2, interruzione
  sessione, piano di ripresa).
- `CLAUDE.md` — sezione "Pipeline v2" (motivazioni, architettura, stato
  verificato/non verificato, prossimi passi vincolanti), checklist aggiornata.
- `experiments/exp1_stability/README.md` — tabella risultati con segni dei
  delta espliciti + nota sulla potenza statistica del gate downstream.
- `experiments/exp1_stability/qualitative_playground.md` — riferimento rotto
  corretto + caveat: i ✅ T1/T2 non dimostrano compressione sotto la v0
  (attention non mascherata).

### Fixed (2026-07-15, dopo il primo push)

- **Gate 0 chiuso: suite verde (31 passed) + ruff pulito.** Il primo run CI
  falliva al lint (4 errori banali, corretti). Il test rosso
  `test_gradient_reaches_context_only_via_anchor` era mal posto:
  `TinyAttention` a 1 layer non offre *alcuna* rotta contesto→post-anchor
  (le K/V dell'anchor sono il suo solo embedding) → portato a 2 layer, il
  minimo perché la rotta legittima esista. Dettagli: decisions.md 2026-07-15.

- **Review completata di `data_contract.py` e `bottleneck.py`** (2026-07-15):
  5 difetti corretti (inferenza `label_kind` su upgrade legacy — avrebbe
  rotto la rebuild; `assert_disjoint` con `pairs` → eval∩test=∅ imposto;
  `answer_idx` bool respinto; I/O cohort UTF-8; chiave legacy `distance`
  sempre rimossa) + guardia NaN in `bottleneck.py` (left padding rifiutato).
  Aggiunta `tests/test_qwen_integration.py` (6 test, marker `integration`,
  esclusi dalla CI): **la mask 4D è onorata da transformers 5.10.2 su Qwen2
  reale, zero leak attorno al bottleneck**. Suite: 40 passed.
  Dettagli: decisions.md 2026-07-15.

- **Build dati v2 eseguita** (2026-07-15): fixture gate (overlap zero,
  determinismo byte-identico su due run) e build reale —
  train=1197 / eval=149 / test=541 / probe=304, zero overlap
  train↔{eval,test,probe} e eval↔test, test 541/541 MCQ-annotato.
  Le 386 annotazioni CNN (che vivevano solo nel test v0) sono state
  backfillate nei raw via `content_id`; v0 preservato in
  `data/processed_v0/` (SHA-256 versionati in `hashes.json`); manifest
  versionato (`data/processed/manifest.json`, eccezione .gitignore).
  Decisione: TUTTI i CNN restano nel test (stile held-out), incluso
  `public_cnndm_train.jsonl`. Dettagli: decisions.md 2026-07-15.

### Known issues / debito aperto

1. ~~pytest mai rilanciato dopo l'ultimo fix → stato suite ignoto (gate 0)~~
   **risolto 2026-07-15**: 31 passed, lint pulito (v. Fixed).
2. ~~`data_contract.py` e `prepare_dataset.py` v2 mai eseguiti nemmeno su
   fixture.~~ **risolto 2026-07-15**: review + 40 test + build fixture/reale
   eseguite e verificate (v. Fixed).
3. `requirements.txt` incoerente con `uv.lock` su torch.
4. Codice generato via backend OpenRouter → review integrale richiesta di
   `data_contract.py` e `bottleneck.py` (un errore di sintassi già trovato).
5. Criterio gating Exp 2 da ri-pinnare (154 → 540 MCQ, split v2) **prima**
   di Exp 0 v2.
6. Proxy DeepClaude: passthrough dei modelli non mappati (`claude-fable-*`)
   instradato a OpenRouter invece che ad Anthropic — bug in
   `~/Work/2-DeepClaude/proxy/model-proxy.js`, fuori repo ma ha ucciso la
   sessione. Mitigazione: `proxy-an` prima di sessioni lunghe.

## v0 — `main` (2026-06-09 → 2026-06-14) — storico, preservato

Artefatti e numeri v0 restano validi come storia e **non vengono
sovrascritti** (SHA-256 rilevati il 2026-07-15). Limiti v0 scoperti a
posteriori: nessun vero bottleneck di attention; split con 148 contesti del
train in test/probe.

- **Exp 0** (gating): `mcq_from_summary` 0.82, `summary_fact_retrieval` 0.742,
  full-context 0.74 su n=50 — baseline forte, si procede
  (`results/exp0_results.json`).
- **Dataset**: A/B/C bilanciati, 3 famiglie via OpenRouter
  (train=1334, eval=148, test=540, probe=302); annotazione MCQ dei 386
  CNN/DailyMail; ri-annotazione delle 109 righe Llama 4 Scout.
- **Fix**: caricamento WikiText → `Salesforce/wikitext` (`src/eval.py`);
  filtro MCQ full-file in `run_exp0.py`.
- **Exp 1** (2026-06-14): training sano (nessun collasso della varianza
  hidden di `[COMPRESS]`) ma **gate FAIL** — WikiText ppl +24.7%,
  MMLU −4 pt → catastrophic forgetting; Exp 2 bloccato
  (`results/exp1_stability.json`, `experiments/exp1_stability/README.md`).
- **Playground qualitativo** (2026-07-14): pattern appreso in-distribution,
  failure mode su negazioni/argomentativo (inversione di polarità, T8).
