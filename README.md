# Behavioral Paired Tokens — `[COMPRESS]` / `[RECALL]`

Ricerca sperimentale su token comportamentali accoppiati in small LLMs
(Qwen2.5-0.5B → 1.5B): il modello viene addestrato a **comprimere** il
contesto nell'hidden state di `[COMPRESS]` e a **richiamarlo** su `[RECALL]`,
con verifica tramite probing rigoroso e intervento causale.

Contesto completo, ipotesi, letteratura e regole: **[CLAUDE.md](CLAUDE.md)**.
Log decisioni: **[experiments/decisions.md](experiments/decisions.md)**.
Stato file per file: **[CHANGELOG.md](CHANGELOG.md)**.

> **Stato verificato (2026-07-15, sera)** — pipeline v2 su `main`:
> - Test: **40 passed** (34 unit + 6 integration Qwen), ruff pulito, CI verde
> - Moduli v2 (`data_contract`, `bottleneck`): review completata, mask 4D
>   **verificata su Qwen reale** (zero leak)
> - Dati v2: build con manifest, split disgiunti (train=1197/eval=149/
>   test=541/probe=304, test tutto MCQ-annotato)
> - Toy gate code-recall: tent. 1 FAIL formale ma controlli causali PASS;
>   tent. 2 in corso
> - **P0 completato**: un solo percorso bottleneck per
>   train/eval/probe/intervention (`attention_mode` in config/checkpoint),
>   test anti-regressione con mask-spy su ogni entry point — 51 test verdi
> - **Gate bloccanti attuali**: toy gate PASS + ri-pin criterio Exp 2,
>   poi Exp 1b
> - Dettagli: `CLAUDE.md` → "Pipeline v2", `CHANGELOG.md`, `experiments/decisions.md`

## Setup

```bash
cd ~/Work/4-BehavioralPairedTokens
uv sync --dev            # ambiente bloccato da uv.lock (torch CPU)
uv run pytest -q         # unit; integration: uv run pytest -m integration
```

`requirements.txt` esiste come riferimento ma pinna `torch` senza suffisso
`+cpu` (incoerente col lock): preferire sempre `uv sync`.

Oppure via Docker: `docker build -t behavioral-tokens .` — il `CMD` di default
ora esegue **pytest** (validazione), non il training; il training è esplicito:
`docker run --rm behavioral-tokens python -m src.train --debug`

## Workflow (ordine obbligato)

```bash
# 1. Dataset — generatori multipli + dati pubblici + handwritten (anti-leakage)
python -m data.generation.generate_examples --type A --n 500
python -m data.generation.generate_examples --type B --n 500
python -m data.generation.generate_examples --type C --n 500
python -m data.generation.import_public --n 300
python -m data.generation.prepare_dataset          # -> data/processed/{train,eval,test,probe}.jsonl

# 2. Exp 0 — baseline prompt engineering (GATING: decide se procedere)
python experiments/exp0_prompt_baseline/run_exp0.py --data data/processed/test.jsonl

# 3. Debug run (sempre prima di un run serio: 100 esempi, 2 epoch, CPU)
python -m src.train --debug

# 4. Training vero
python -m src.train --run-name typeA-v1
tensorboard --logdir results/runs

# 5. Exp 1 — stabilità (perplexity prima/dopo)
python -m src.eval --task wikitext                                  # base
python -m src.eval --task wikitext --checkpoint results/checkpoints/typeA-v1/best

# 6. Exp 2 — ablazione distanza (recall quality per distanza)
python -m src.eval --task recall --data data/processed/test.jsonl \
    --checkpoint results/checkpoints/typeA-v1/best --out results/exp2.json
python -m src.eval --task mcq --data data/processed/test.jsonl \
    --checkpoint results/checkpoints/typeA-v1/best

# 7. Exp 3 — probing con tutti i controlli (usa un layer intermedio, es. 12)
python -m src.probe --data data/processed/probe.jsonl \
    --checkpoint results/checkpoints/typeA-v1/best --layer 12

# 8. Exp 5 — intervento causale (se Exp 3 positivo)
python -m src.intervention --data data/processed/probe.jsonl \
    --checkpoint results/checkpoints/typeA-v1/best \
    --probe results/exp3_probing/probe_results.npz \
    --target-class positive --alpha 4.0
```

Per la generazione sintetica servono: Ollama attivo su `localhost:11434`
e/o `OPENROUTER_API_KEY` nell'ambiente (config: `data/generation/generators.json`).

## Provare il modello addestrato (playground interattivo)

`src/try_model.py` carica il modello, incolla un tuo testo nel formato di
training (`contesto + [COMPRESS] + filler + [RECALL]`) e mostra cosa genera.

```bash
cd ~/Work/4-BehavioralPairedTokens
source .venv/bin/activate

# Modello base (nessun adapter) — utile come confronto
python -m src.try_model

# Modello addestrato: base Qwen + adapter LoRA di un run
python -m src.try_model --checkpoint results/checkpoints/exp1-stability/best

# Test distanza: inserisce testo tra [COMPRESS] e [RECALL]
python -m src.try_model --checkpoint results/checkpoints/exp1-stability/best \
    --filler "Testo intermedio che separa i due token..."

# Output più lungo (default: 200 token)
python -m src.try_model --checkpoint results/checkpoints/exp1-stability/best \
    --max-new-tokens 400
```

Uso: incolla il testo al prompt, **riga vuota** per inviarlo, **Ctrl+C** per uscire.

**Checkpoint disponibili** (`results/checkpoints/<run>/{best,last}`):

| Run | Note |
|---|---|
| `exp1-stability` | Run vero di Exp 1 — impara il pattern, ma la ricetta degrada il modello base (vedi `experiments/exp1_stability/README.md`) |
| `exp1-lora-debug` | Debug run (100 esempi, 2 epoch) — solo verifica pipeline |
| `smoke-debug`, `smoke2-debug` | Smoke test iniziali, non significativi |

Ogni run salva anche la sua `config.json` completa nella stessa directory.

> ⚠️ Il checkpoint non è un modello standalone: è un **adapter LoRA** che viene
> applicato sopra `Qwen/Qwen2.5-0.5B` (scaricato/cachato da HuggingFace al primo
> avvio). Il device viene risolto automaticamente (`cuda` se disponibile, altrimenti
> `cpu` — sulla GTX 970 il caricamento in VRAM da 4GB funziona per lo 0.5B).

## Struttura

```
src/            model, dataset, train, eval, probe, intervention
data/           raw (generati), processed (split), handwritten, generation (script)
experiments/    exp0..exp5 + decisions.md (log decisioni, da tenere aggiornato)
results/        checkpoints, run tensorboard, metriche json
```

## Regole rapide

- `seed=42` ovunque; ogni run salva la sua config completa
- `lambda_c=0` finché la base non funziona; cap 0.1 (rischio collasso)
- Exp 0 è gating: se il prompt baseline basta, il progetto si ferma
- Probing senza i 4 controlli non vale nulla
