# Behavioral Paired Tokens — `[COMPRESS]` / `[RECALL]`

Ricerca sperimentale su token comportamentali accoppiati in small LLMs
(Qwen2.5-0.5B → 1.5B): il modello viene addestrato a **comprimere** il
contesto nell'hidden state di `[COMPRESS]` e a **richiamarlo** su `[RECALL]`,
con verifica tramite probing rigoroso e intervento causale.

Contesto completo, ipotesi, letteratura e regole: **[CLAUDE.md](CLAUDE.md)**.
Log decisioni: **[experiments/decisions.md](experiments/decisions.md)**.

## Setup

```bash
cd ~/Work/4-BehavioralPairedTokens
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```

Oppure via Docker (CPU/debug): `docker build -t behavioral-tokens . && docker run --rm behavioral-tokens`

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
