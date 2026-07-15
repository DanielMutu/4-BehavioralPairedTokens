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

## 2026-06-09 — Exp 0: baseline prompt engineering (gating)

- **Run**: Qwen2.5-0.5B base (nessun token speciale), 50 esempi MCQ da
  `test.jsonl`, seed 42. Risultati (`results/exp0_results.json`):
  - `summary_fact_retrieval` = **0.742** (fatti conservati nel riassunto)
  - `mcq_from_summary` = **0.82** (41/50)
  - `mcq_full_context_upper_bound` = **0.74** (37/50)
  - chance level = 0.25
- **Decisione**: si **procede** con il training (Exp 1+), ma la baseline è
  forte e fissa l'asticella: i token addestrati devono battere chiaramente
  0.82 di MCQ accuracy e 0.74 di fact retrieval, a parità di protocollo.
  Il verdetto gating definitivo si dà al confronto con Exp 2 (come da
  guidance dello script): se i token non superano questi numeri, STOP.
- **Osservazione notevole**: `mcq_from_summary` (0.82) > full context (0.74).
  Il riassunto agisce da denoising del contesto per il modello 0.5B — la
  compressione *aiuta* anche senza training. Questo non invalida il progetto
  (che misura il *meccanismo*: probing, persistenza alla distanza, causalità —
  cose che il prompt baseline non può offrire), ma rende la barra
  comportamentale più alta del previsto.
- **Caveat**: i riassunti generati dal base model contengono artefatti
  (continuazioni spurie tipo "You are an AI assistant…" dopo i bullet);
  il fact retrieval medio 0.742 ha varianza alta (min 0.0 su alcuni esempi).
  Da tenere a mente nel confronto: anche i token addestrati verranno valutati
  con le stesse metriche rumorose.
- **Fix incluso nel run**: `run_exp0.py` ora carica l'intero file prima di
  filtrare gli esempi MCQ — in `test.jsonl` solo 154/540 righe hanno le
  opzioni e non sono distribuite uniformemente; il vecchio limite
  `max_samples * 3` ne scartava la maggior parte.

## 2026-06-09 — Criterio gating per Exp 2 (fissato PRIMA di vedere i risultati)

- **Decisione**: il verdetto gating al confronto Exp 2 vs baseline Exp 0 usa
  due criteri, decisi ora a risultati non visti:
  1. **Primario**: i token addestrati battono `mcq_from_summary = 0.82`
     (e `summary_fact_retrieval = 0.742`) a parità di protocollo.
  2. **Secondario (legittimo)**: pareggiano la baseline con compressione
     ~100× più densa — il riassunto del prompt baseline costa ~150 token,
     `[COMPRESS]` è un singolo hidden state. A parità di accuracy, vince
     l'efficienza; il valore del progetto si sposta su Exp 3/5 (meccanismo).
- **Soglia di pareggio**: dato il rumore con n=50, è "pareggio" tutto ciò
  che sta entro ±5 punti percentuali. Il verdetto finale si dà su **tutti
  i 154 esempi MCQ del test set**, non su 50.
- **Motivo**: fissare i criteri prima dei risultati evita di razionalizzare
  a posteriori un esito marginale (rischio principale del progetto: un Exp 2
  né vittoria né sconfitta). Un esito "i token non battono il baseline" con
  controlli rigorosi resta pubblicabile come negative result.
- **Prerequisito per Exp 2**: generare le opzioni MCQ per i 386 esempi
  CNN/DailyMail del test set (oggi quasi solo i sintetici le hanno):
  senza, la parte out-of-style — quella che distingue comportamento da
  stile — non viene misurata.

## 2026-06-09 — Annotazione MCQ del test set held-out (CNN/DailyMail)

- **Fatto**: i 386 esempi CNN/DailyMail di `test.jsonl` ora hanno
  question/options/answer_idx + facts (`data/generation/annotate_mcq.py`).
  Run: 386/386 annotati, 0 falliti, round-robin sulle 3 famiglie
  (mistral-small 149, deepseek-flash 128, llama4-scout 109).
  Ora tutti i 540 esempi di test sono valutabili con MCQ e fact retrieval —
  prima la metrica MCQ copriva quasi solo lo stile sintetico (154/540).
- **Shuffle posizione risposta corretta**: deterministico per riga
  (`Random(20000+i)`). Motivo: i generatori LLM tendono a mettere l'opzione
  giusta per prima; un bias posizionale corromperebbe la metrica MCQ via
  log-likelihood. Verifica post-run: answer_idx distribuiti 102/101/86/97.
- **Niente rischio leakage**: le annotazioni sono eval-only (mai usate in
  training); l'annotatore è tracciato in `meta.mcq_annotator` per eventuali
  analisi di robustezza per-famiglia.
- **Nota**: `data/processed/` è fuori da git (regenerable); versionato lo
  script, non i dati. Backup pre-annotazione in `test.jsonl.bak`.

## 2026-06-09 — Ri-annotazione delle 109 righe Llama 4 Scout

- **Problema**: ispezione manuale (10 esempi/famiglia) ha mostrato che le
  annotazioni di Llama 4 Scout erano sistematicamente più deboli: facts
  poveri (sole entità, ~57 chars medi vs ~250 delle altre famiglie) e
  almeno una annotazione errata (risposta "2013" a una domanda sull'età).
- **Decisione**: ri-annotate tutte le 109 righe Scout con le sole altre due
  famiglie (`--redo-annotator` / `--exclude-generator` aggiunti allo
  script). Esito: 109/109 ok, ora il test set CNN/DailyMail è annotato
  solo da Mistral Small (205) e DeepSeek Flash (181); answer_idx ben
  distribuiti (103/92/93/98); facts medi passati da 57 a 249 chars.
- **Fix incluso**: su redo lo script ora sovrascrive anche i `facts`
  (prima li riempiva solo se vuoti, lasciando quelli vecchi di Scout).
- **Nota**: Scout resta tra i generatori del *training* set (lì la
  diversità di famiglie è la mitigazione anti-leakage e lo stile debole
  di un generatore è rumore accettabile); è escluso solo dall'annotazione
  *eval*, dove la qualità delle domande determina la validità della metrica.

## 2026-06-13 — Exp 1: setup, gate pre-registrato e vincolo CPU

- **Vincolo hardware scoperto**: il torch installato è CPU-only
  (`2.12.0+cpu`, `cuda_avail=False`); la GTX 970 (Maxwell, 4GB) non è
  utilizzabile. Exp 1 gira interamente su CPU (i7-6700, 4 thread).
  Throughput misurato: ~8.9 s/batch (bs=4, len≤384) → 3 epoch ≈ 2.5 h.
  Coerente con "training overnight su config leggere".
- **Config run reale** (`experiments/exp1_stability/train_config.json`):
  `max_length=384` invece di 1024 — la distribuzione reale del train set ha
  max 342 token (p99=290), quindi nessun troncamento e CPU più veloce.
  `epochs=3, batch_size=4, grad_accum=4` (effettivo 16), `lambda_c=0`
  (consistency OFF al primo giro, regola CLAUDE.md), `seed=42`.
- **Gate pre-registrato (a risultati non visti)**: PASS sse
  (1) perplexity WikiText-2 aumenta di ≤ 5% (relativo), E
  (2) ogni accuracy downstream cala di ≤ 2.0 punti (assoluti).
  Motivo della soglia: Exp 1 verifica *stabilità*, non miglioramento — il
  fine-tuning LoRA + i nuovi token non devono rompere le capacità generali.
  Implementato in `experiments/exp1_stability/run_exp1.py`.
- **Benchmark downstream scelti: HellaSwag + MMLU** (multiple-choice,
  scored via log-likelihood normalizzata per lunghezza, stesso protocollo di
  `option_loglik` già usato per gli MCQ). **GSM8K escluso volutamente**:
  è generativo, un modello 0.5B base sta vicino al chance level (nessun
  headroom per rilevare una degradazione), e la generazione è lenta su CPU.
  Un test di stabilità ha bisogno di metriche con margine per *scendere*;
  GSM8K già a terra non può mostrare degrado. Numeri cloze-style: conta solo
  il DELTA base-vs-trained, non il valore assoluto vs leaderboard.
- **Confronto equo**: "base" = modello originale + token aggiunti
  (mean-init, mai usati nei testi dei benchmark) senza adapter; "trained" =
  + adapter LoRA. L'adapter agisce su q_proj/v_proj in ogni forward, quindi
  il delta misura proprio l'effetto del fine-tuning sulle capacità generali.

## 2026-06-14 — Exp 1: esito del training run (checkpoint `exp1-stability`)

- **Run completato**: 2026-06-13 21:51 → 2026-06-14 00:16 (~2h25), CPU,
  3 epoch (~250 optimizer step). Parametri trainabili 1.084.032 (0.22%).
- **Salute (TensorBoard `results/runs/exp1-stability`)**:
  - `train/loss_ce` 1.97 → 0.84 (min 0.72); `eval/loss_ce` 0.939 → **0.907**
    (best, step 249).
  - `train/perplexity` 7.15 → 2.31; `eval/perplexity` 2.60 → 2.52.
  - `loss_consistency` = 0.0 costante (atteso, `lambda_c=0`).
  - **Varianza hidden `[COMPRESS]`**: train 18.8 → 16.0 (range 12–39),
    eval 23.3 → 20.3. **Nessun collasso** — il controllo anti-degenerazione
    centrale del progetto è verde.
- **Decisione**: training accettato come sano; si procede allo step di
  *valutazione* di stabilità (separato dal training). Il verdetto PASS/FAIL
  di Exp 1 NON è ancora dato: dipende dal confronto base-vs-trained su
  WikiText-2/HellaSwag/MMLU prodotto da `run_exp1.py`.
- **Nota di processo**: il training e la valutazione sono due script distinti
  (`src/train.py` addestra, `experiments/exp1_stability/run_exp1.py` valuta).
  Aver lanciato solo il primo non chiude Exp 1.

## 2026-06-14 — Fix caricamento dataset WikiText in `src/eval.py`

- **Problema**: `run_exp1.py` crashava all'avvio della valutazione in
  `wikitext_perplexity`: `load_dataset("wikitext", "wikitext-2-raw-v1")` usa
  il nome canonico legacy, non più risolvibile con `datasets 5.0.0` /
  `huggingface_hub 1.18.0` (`HfUriError: Repository id must be
  'namespace/name'`).
- **Decisione**: usare il repo id namespaced `Salesforce/wikitext` (mirror
  ufficiale, stesso contenuto). HellaSwag (`Rowan/hellaswag`) e MMLU
  (`cais/mmlu`) erano già namespaced → nessuna modifica. Fix in `src/eval.py:48`.
- **Impatto**: sblocca Exp 1 e ogni futura chiamata a `wikitext_perplexity`
  (riusata negli esperimenti successivi). Valutazione rilanciata su CPU.

## 2026-06-14 — Exp 1: VERDETTO = FAIL (la ricetta degrada il modello)

- **Risultati** (`results/exp1_stability.json`, 200 campioni downstream,
  100 blocchi WikiText):
  - WikiText-2 ppl: 14.399 → 17.956 (**+24.7%**) → FAIL (gate ≤ +5%).
  - HellaSwag: 0.460 → 0.475 (−1.5 pt, migliora) → ok.
  - MMLU: 0.275 → 0.235 (**+4.0 pt di calo**) → FAIL (gate ≤ 2.0).
  - `PASS = false`.
- **Lettura**: over-specializzazione sul task. La `eval/loss_ce` del task era
  ottima (ppl 2.52), ma le capacità generali sono peggiorate → catastrophic
  forgetting da training troppo aggressivo su distribuzione stretta. Coerente
  col fatto che il *training* era "sano" (nessun collasso): collasso e
  forgetting sono problemi diversi — il primo è degenerazione dell'hidden
  state, il secondo è perdita di capacità generali. Exp 1 cattura il secondo.
- **Decisione**: Exp 2 **resta bloccato**. Va corretta la ricetta e ri-passato
  questo stesso gate (soglie invariate) prima di misurare il recall — altrimenti
  un eventuale Exp 2 debole sarebbe inattribuibile (token deboli o modello rotto?).
- **Fix candidati** (scelta da fare): (1) `lr` più basso (es. 5e-5) e/o 1 epoca;
  (2) **replay** di language modeling generico (~10–20% del train); (3) adapter
  meno capiente (`r=8`) o `lora_dropout` più alto; (4) selezione checkpoint con
  mini-gate di stabilità (ppl generica) durante il training, non solo sulla loss
  del task. Dettagli e tabella in `experiments/exp1_stability/README.md`.

## Template per nuove decisioni

```
## YYYY-MM-DD — Titolo
- **Decisione**: ...
- **Motivo**: ...
- **Alternative scartate**: ...
```
