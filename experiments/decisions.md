# Log delle decisioni — Behavioral Paired Tokens

Ogni scelta non banale va registrata qui (regola CLAUDE.md).

## 2026-06-09 — Setup iniziale

- **Embedding dei nuovi token**: training mirato delle sole righe nuove via
  `trainable_token_indices` (peft ≥ 0.15) invece di `modules_to_save=["embed_tokens"]`.
  Motivo: la matrice embedding completa di Qwen2.5-0.5B è ~136M parametri,
  insostenibile sulla GTX 970 4GB; le sole 3 righe nuove sono ~2.7K parametri.
  Fallback automatico a `modules_to_save` su peft vecchi (in `src/model.py`).
- **Resize embedding condizionale**: Qwen2.5 ha la matrice embedding già
  paddata oltre il vocab del tokenizer, quindi i 3 nuovi id rientrano senza
  resize. Si fa resize solo se `len(tokenizer)` supera le righe esistenti, e
  si inizializzano le righe nuove con la media degli embedding (più stabile
  del random init).
- **Loss solo sul target**: la cross-entropy è calcolata di default solo sui
  token dopo `[RECALL]` (`loss_on_target_only=True`). Motivo: il modello deve
  imparare il comportamento di recall, non a riprodurre contesto e filler.
  Disattivabile da config per ablazione.
- **Formato esempio**: `{context}\n[COMPRESS]\n{filler?}\n[RECALL]\n{target}<eos>`.
  Tipo C sostituisce `[COMPRESS]` con la sequenza di composizione.
- **Consistency loss**: `lambda_c = 0` al primo giro (regola CLAUDE.md), cap
  hard a 0.1 in `config.py`, warning runtime se la varianza degli hidden state
  di `[COMPRESS]` scende sotto 1e-4 (detection collasso).
- **Filler Tipo B**: generato da un pool di frasi neutre deterministico
  (seed 42) invece che da LLM. Motivo: il filler deve essere semanticamente
  irrilevante; generarlo con LLM aggiunge stile del generatore senza valore.
- **Controllo "posizioni shuffled" (Exp 3)**: implementato permutando gli
  hidden state di `[COMPRESS]` tra esempi diversi (rompe l'accoppiamento
  stato↔contenuto mantenendo la distribuzione marginale).
- **Intervento causale (Exp 5)**: l'hook agisce su un layer intermedio del
  decoder, non sull'ultimo hidden state (che è post-norm e non si propaga via
  attention alle posizioni successive). Il probe di Exp 3 va quindi addestrato
  sullo stesso layer intermedio usato per l'intervento (`--layer`).
- **Generatori dataset**: round-robin su Ollama locale + 2 modelli OpenRouter
  (config in `data/generation/generators.json`); esempi handwritten in
  `data/handwritten/` e CNN/DailyMail via `import_public.py` instradati
  preferenzialmente nel test set held-out (stile diverso dal training).

## 2026-06-09 — Generazione dataset

- **Ollama escluso dalla generazione bulk**: una singola richiesta a
  `qwen3.5:4b` sulla GTX 970 impiega ~5 minuti — inutilizzabile per 1500
  esempi. La diversità dei generatori (regola anti-leakage) è garantita da
  **3 famiglie diverse via OpenRouter**: DeepSeek V4 Flash, Llama 4 Scout,
  Mistral Small 2603 (round-robin, ~33% ciascuno per tipologia).
- **Generazione parallela**: `generate_examples.py` usa un ThreadPoolExecutor
  (`--workers`, default 6) con rng deterministico per task (`Random(10000+i)`);
  over-provisioning 1.5× per assorbire risposte non valide. 1500 esempi in
  ~15 minuti contro ~4 ore sequenziali.
- **Probe set held-out**: `probe.jsonl` contiene solo esempi etichettati NON
  presenti nel train (da eval+test). Probing su testi visti in training
  gonfierebbe l'accuratezza del probe (Exp 3).
- **Composizione split** (run del 2026-06-09): train=1334 (solo sintetico,
  A/B/C bilanciati), eval=148, test=540 (386 CNN/DailyMail + 6 handwritten +
  148 sintetici per confronto), probe=302.

## Template per nuove decisioni

```
## YYYY-MM-DD — Titolo
- **Decisione**: ...
- **Motivo**: ...
- **Alternative scartate**: ...
```
