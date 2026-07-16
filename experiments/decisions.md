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

## 2026-07-15 — Avvio pipeline v2: true bottleneck e salvaguardia v0

- **Branch**: creata `feat/true-compress-bottleneck-v2` dalla `main` corrente,
  preservando senza stash/reset le modifiche Exp 1 già staged/unstaged.
- **Perché serve una v2**: la pipeline v0 usa causal attention ordinaria, quindi
  i token dopo `[COMPRESS]` possono ancora leggere direttamente il contesto;
  inoltre lo split v0 include 148 contesti del train nel test e nel probe.
  Gli artefatti v0 restano storici, ma non supportano claim definitivi di true
  compression o probing held-out.
- **Politica artefatti**: nessun risultato v0 viene sovrascritto. Prima delle
  modifiche sono stati rilevati gli SHA-256 degli split e risultati correnti;
  saranno registrati nel manifest v0/v2 prodotto dal nuovo data contract.
- **Ordine dei gate**: test/contratto dati → split disgiunti → attention
  bottleneck toy gate → Exp 0 v2 → Exp 1b stabilità → Exp 2. Un FAIL blocca lo
  step successivo e viene documentato, non aggirato.
- **Decisione architetturale**: prima implementazione corretta tramite mask
  additiva 4D e decoder greedy full-recomputation (`use_cache=False`). La
  potatura KV è un'ottimizzazione successiva e sarà accettata solo se i logits
  coincidono con il decoder reference mantenendo le posizioni RoPE assolute.

## 2026-07-15 — Sessione v2 interrotta: stato verificato/non verificato e piano di ripresa

- **Cosa è successo**: l'esecuzione autonoma del piano v2 si è interrotta per
  esaurimento crediti OpenRouter (proxy DeepClaude in modalità `proxy-or`).
  Ultimo atto della sessione: fix a `tests/test_bottleneck.py` applicato ma
  **mai ri-verificato** — pytest non è stato rilanciato dopo il fix.
- **Stato VERIFICATO** (output visto a schermo prima del crash):
  - branch `feat/true-compress-bottleneck-v2` creata, modifiche v0 preservate;
  - SHA-256 di split e risultati v0 rilevati (per il manifest v0/v2);
  - `uv lock` + `uv sync --dev` completati; `torch 2.12.0+cpu` importa,
    `cuda_available=False` come atteso;
  - digest `python:3.11-slim` risolto e pinnato nel Dockerfile.
- **Stato NON VERIFICATO** (scritto ma mai eseguito con successo):
  - suite pytest: unica esecuzione parzialmente rossa, poi fix e crash;
  - `src/data_contract.py` (~390 righe): mai eseguito, nemmeno import;
  - `data/generation/prepare_dataset.py` v2: mai eseguito, nemmeno su fixture;
  - CI workflow: mai girata.
- **Caveat qualità del codice**: parte del codice v2 è stata generata con il
  proxy in modalità OpenRouter — il backend reale può essere stato DeepSeek
  anche quando la UI mostrava Opus/Fable. In `src/bottleneck.py` c'era già un
  errore di sintassi (`n most`) corretto a mano. Decisione: `data_contract.py`
  e `bottleneck.py` vanno **riletti integralmente** prima di fidarsi dei test
  che li coprono (i test stessi provengono dalla stessa sessione).
- **Incoerenza ambiente da sanare**: `requirements.txt` pinna `torch==2.12.0`
  (PyPI → wheel CUDA), mentre `uv.lock`/venv usano `2.12.0+cpu` dall'indice
  PyTorch dedicato dichiarato in `pyproject.toml`. Chi installa da
  requirements ottiene un ambiente diverso da quello testato.
- **Decisione di perimetro per la ripresa**: eseguire il piano fino a
  **toy gate bottleneck + Exp 0 v2** e fermarsi a rapporto prima di Exp 1b
  (~2.5 h CPU). I blocchi del piano su Exp 3/4/5 si implementano solo quando
  i gate precedenti passano: codice scritto ora verrebbe riscritto comunque.
- **Prerequisito scientifico prima di Exp 0 v2**: ri-pinnare il criterio
  gating di Exp 2 a risultati non visti. La voce 2026-06-09 fissa il verdetto
  su 154 MCQ, ma oggi il test set ne ha 540 annotati e gli split v2 lo
  cambieranno di nuovo. La nuova voce dovrà fissare: n esatto sul test v2,
  baseline Exp 0 ricalcolata sull'intero set (0.82 era su n=50, quasi solo
  sintetici), e un numero separato per la parte out-of-style CNN/DailyMail,
  che oggi non ha alcuna baseline.
- **Nota statistica per il gate di Exp 1b** (da applicare, soglie invariate):
  con 200 campioni downstream la sd binomiale di una accuracy è ~3 pt, quindi
  la soglia di gate (2.0 pt) è più stretta della precisione della misura — il
  FAIL v0 resta valido perché la ppl +24.7% è inequivocabile, ma il prossimo
  verdetto downstream va dato con più campioni o con McNemar appaiato sugli
  stessi item.
- **Bug infrastrutturale (fuori progetto, da verificare)**: `/compact` è
  fallito con `anthropic/claude-fable-5 ... may not exist` mentre il proxy era
  in modalità OpenRouter → il passthrough dei modelli non mappati in
  `~/Work/2-DeepClaude/proxy/model-proxy.js` non copre `claude-fable-*`,
  contraddicendo il CLAUDE.md homelab ("i modelli senza voce in MODEL_REMAP
  vanno sempre diretti ad Anthropic"). Mitigazione operativa: `proxy-an` prima
  di sessioni lunghe.
- **Fix documentali contestuali** (2026-07-15): il riferimento in
  `qualitative_playground.md` a una voce "2026-07-14" di questo log era rotto
  (la voce non è mai stata scritta; l'analisi vive nel README di Exp 1 e nel
  playground stesso) → corretto. Aggiunto al playground il caveat che i ✅
  T1/T2 non dimostrano compressione sotto la pipeline v0 (attention non
  mascherata → copying attentivo possibile). Le voci storiche di questo log
  restano intoccate.

## 2026-07-15 — Gate 0 chiuso: suite test verde (31 passed) + fix lint CI

- **Contesto**: al primo push su GitHub la CI è fallita allo step lint (4
  errori ruff banali: variabile `l`, import inutilizzato, `zip` senza
  `strict`, `warnings.warn` senza `stacklevel`). Corretti tutti.
- **Il test rosso era mal posto, non la mask**: il fallimento di
  `test_gradient_reaches_context_only_via_anchor` era causato dal modellino
  di test `TinyAttention` a **1 solo layer**: lì le K/V dell'anchor viste
  dalle query successive sono solo `emb[anchor]`, quindi il contesto non ha
  *nessuna* rotta verso le posizioni post-anchor — nemmeno quella legittima —
  e il gradiente zero era il comportamento corretto. Portato a 2 layer
  (minimo perché la rotta esista: layer 1 → l'anchor aggrega il contesto nel
  proprio hidden state; layer 2 → le query post-anchor leggono le K/V
  dell'anchor).
- **Nota concettuale non banale**: l'informazione attraversa il bottleneck
  solo tramite gli hidden state dell'anchor dei layer ≥ 1, mai al layer di
  embedding. Coerente con la decisione 2026-06-09 di fare probing e
  intervento causale su un layer intermedio, non sull'ultimo.
- **Esito**: `pytest` 31 passed, `ruff` pulito → gate 0 PASS. Restano dovute:
  la review severa di `data_contract.py`/`bottleneck.py` (i test provengono
  dalla stessa sessione non fidata) e la prima esecuzione reale di
  `prepare_dataset.py` su fixture (i gate successivi del piano).

## 2026-07-15 — Review di data_contract.py e bottleneck.py: esito e fix

Review integrale dei due moduli ereditati dalla sessione col backend
OpenRouter (obbligo registrato il 2026-07-15). Esito: **impianto corretto,
5 difetti reali corretti, 1 claim critica ora VERIFICATA su Qwen reale**.

- **Claim verificata (era solo asserita in docstring)**: transformers 5.10.2
  onora davvero la mask additiva 4D `(B,1,T,T)` passata come `attention_mask`
  a Qwen2 con SDPA. Nuova suite `tests/test_qwen_integration.py` (marker
  `integration`, esclusa dalla CI, gira sul checkpoint locale in ~9 s):
  1. mask causale 4D esplicita ≡ path causale di libreria (convenzione additiva ok);
  2. la mask bottleneck produce logits diversi dalla causale (non viene ignorata);
  3. controllo positivo: editando il contesto (a livello di ID) i logits
     post-anchor cambiano → la rotta legittima via anchor esiste;
  4. controllo negativo: bloccando anche la chiave dell'anchor, i logits
     post-anchor sono ESATTAMENTE invarianti all'edit del contesto → **zero
     leak attorno al bottleneck**;
  5. batch right-padded senza NaN; 6. smoke del reference decoder.
  Nota: verificata l'implementazione attention di default (SDPA); il path
  eager non è testato — se si cambia `attn_implementation`, ritestare.
- **Fix in `data_contract.py`**:
  1. `upgrade_legacy_example` ora inferisce `label_kind` dal valore di `label`
     quando assente (vocabolari disgiunti) — le righe v0 con label senza kind
     avrebbero fatto crashare l'intera rebuild;
  2. `assert_disjoint` accetta `pairs` extra: ora `prepare_dataset` impone
     anche eval∩test=∅ (prima solo train-vs-tutti; probe resta escluso perché
     vista derivata di eval+test);
  3. `answer_idx=True` non passa più la validazione MCQ (bool è int in Python);
  4. I/O `CohortSelection` con encoding UTF-8 esplicito;
  5. la chiave legacy `distance` viene sempre rimossa (prima poteva
     sopravvivere accanto a `distance_target_tokens`).
- **Fix in `bottleneck.py`**: guardia contro righe di query completamente
  mascherate (softmax tutta −inf → NaN che avvelena anche le posizioni reali
  via 0·NaN=NaN nei layer successivi). Succede col LEFT padding → ora
  rifiutato con errore esplicito; la pipeline richiede right padding.
- **Non-difetti verificati**: la strategia boundary di `option_loglik`
  (prefisso comune tra tokenizzazioni) è corretta; l'esclusione delle
  annotazioni MCQ da `example_id` è intenzionale (ri-annotare non cambia
  l'identità); l'overlap probe/eval/test nel manifest è by design.
- **Nota minore aperta**: `_git_state()` dipende dalla CWD (manifest costruiti
  fuori dalla root registrerebbero il git sbagliato o None) — accettato per ora.
- **Esito suite**: 40 passed (34 unit + 6 integration), ruff pulito.
  Prossimo gate: build dati v2 su fixture (`prepare_dataset.py`).

## 2026-07-15 — Build dati v2 eseguita: split disgiunti, manifest, annotazioni salvate

Gate "build dati v2" del piano: PASS. Sequenza eseguita e verificata:

- **Fixture gate**: build end-to-end su campione di righe raw reali (60
  sintetiche + 12 CNN + 6 handwritten) → conteggi conservati, overlap
  train/eval/test tutti a zero, `--check` verde, e **due build consecutive
  producono file byte-identici** (determinismo).
- **Salvaguardia v0**: split v0 copiati in `data/processed_v0/` con SHA-256
  registrati in `hashes.json` (versionato; i dati restano locali come da
  policy "script versionati, non dati"). Hash train v0:
  `0210448d…`, test v0: `9fe2cf4f…` (completi nel file).
- **Backfill annotazioni MCQ nei raw** (con backup `.pre_backfill.bak`):
  le 386 annotazioni CNN/DailyMail vivevano SOLO nel `test.jsonl` v0 — una
  rebuild ingenua dai raw le avrebbe perse. Riportate nei due file raw via
  `content_id` (che per design non cambia con le annotazioni): 400/400
  righe agganciate (386 uniche + 14 duplicati poi dedupe). Ora i raw sono
  autosufficienti per qualsiasi rebuild futura.
- **Build reale** (`disjoint-segments-v2`, seed 42):
  train=1197 (solo sintetico), eval=149, test=541 (386 CNN + 6 handwritten
  + 149 sintetici in-style), probe=304. Overlap: train↔{eval,test,probe}=0,
  eval↔test=0; probe↔{eval,test}>0 by design (vista derivata).
  **MCQ: 541/541 righe di test annotate.** Scarti: 5 dup sintetici,
  14 dup CNN, 0 incroci. Manifest: `data/processed/manifest.json`
  (versionato via eccezione .gitignore).
- **Decisione — tutti i CNN nel test, incluso `public_cnndm_train.jsonl`**:
  il file era stato importato con ruolo "train replay" (mai usato), ma
  instradare 300 righe CNN nel training distruggerebbe la proprietà
  "stile held-out" del test che distingue comportamento da stile. Finché il
  protocollo usa CNN come stile held-out, tutto CNN va nel test (stesso
  comportamento v0, ora esplicito). Alternativa scartata: rispettare il
  ruolo d'importazione — riconsiderabile solo se si introdurrà replay di
  dominio, che comunque dovrà usare testo generico (WikiText/C4), non CNN.
- **Nota**: il manifest registra `git.commit=9caee45, dirty=true` — è lo
  stato al momento della build (il commit che versiona il manifest è per
  forza successivo). L'integrità dei dati è garantita dagli SHA-256 degli
  split, non dal flag git.
- **Conseguenza per il gating Exp 2** (già previsto): il test v2 ha 541
  righe tutte MCQ-annotate; i 149 sintetici in-style sono righe MAI viste
  in training (in v0 erano 148 righe riusate dal train). La baseline Exp 0
  va ricalcolata su questo set (prossima voce di pre-registrazione).

## 2026-07-15 — Toy gate, tentativo 1: FAIL formale, meccanismo validato

- **Setup**: 160/40 codici a 4 cifre, LoRA r=8 q_proj+v_proj, lr=1e-3,
  20 epoche, bottleneck ON, soglie pre-registrate in `run_toy.py`
  (`results/toy_bottleneck.json`, ~21 min CPU).
- **Esito**: FAIL sul criterio 1 (`bottleneck_acc` 0.375 < 0.90), PASS sui
  tre controlli causali: untrained 0.000 (≤0.05), anchor-removed 0.000
  (≤0.05), context_override_rate 0.95 (≥0.90); swap_rate 0.45 (info,
  target 0.5). Come da protocollo il FAIL si documenta e non si aggira:
  nessuna soglia viene toccata.
- **Lettura**: il fallimento è di quantità di training, non di meccanismo.
  La loss resta inchiodata a 1.842 = 4·ln(10)/5 (formato imparato, cifre a
  caso) fino all'epoca ~6, poi transizione di fase fino a 0.22 con eval in
  salita (picco 0.55 a ep.19): il run finisce a metà salita. Train subset
  0.70 vs eval 0.375 → memorizzazione parziale, servono più codici per
  forzare la generalizzazione del routing.
- **Evidenze causali già acquisite** (indipendenti dal FAIL):
  (a) col patching, il 95% delle generazioni ignora il contesto reale del
  prompt e il 45% produce esattamente il codice dell'anchor trapiantato —
  la memoria vive nell'hidden state dell'anchor;
  (b) `full_context_acc = 0`: il modello addestrato col bottleneck non sa
  più usare l'attention libera — si è specializzato sul canale-anchor.
- **Decisione**: tentativo 2 con 400 codici train (was 160) e 30 epoche
  (was 20), ricetta e soglie invariate. Motivo della scelta dati>epoche:
  l'overfitting parziale indica che il collo è la varietà, non il tempo.
- **Nota infrastruttura**: run 1 eseguito mentre lo script veniva
  parametrizzato (mode/targets/n-train CLI); il processo in memoria non è
  stato toccato. `run_toy.py` ora supporta anche il run di controllo
  full-context per la condizione "token, unmasked" di Exp 2.

## 2026-07-15 — Triage review esterna: claim verificate, P0 accettato, notebook

Ricevuta una review esterna statica del repo (salvata integralmente in
`docs/external_review_2026-07-15.md`). Triage punto per punto:

- **Claim tecniche VERIFICATE sul codice (tutte confermate)**:
  1. `src/train.py:79,134` chiama `model(...)` con attention ordinaria — il
     bottleneck NON è collegato al training;
  2. `src/eval.py:69` usa `model.generate` standard e `option_loglik` (r.125)
     non passa la mask; le versioni `*_bottlenecked` esistono ma non sono
     importate da nessun modulo di `src/`;
  3. `src/eval.py:105` legge `meta.get("distance", 0)` — chiave che il
     contratto v2 rimuove: con dati v2 tutti gli esempi finirebbero nel
     bucket distanza 0 (bug bloccante per Exp 2);
  4. `src/probe.py:68` estrae hidden state con forward ordinario;
  5. `src/dataset.py:63` tokenizza con `truncation=True` senza
     `validate_layout` → possibile troncamento silenzioso dei token paired;
  6. accumulo gradienti senza flush finale + `total_steps` con divisione
     intera (`src/train.py:115,140`).
- **Conclusione accettata**: la milestone P0 della review diventa vincolante
  e si inserisce PRIMA di Exp 1b nell'ordine dei gate: *un unico percorso
  bottleneck condiviso da train/eval/probe/intervention, con un test
  end-to-end che fallisca se un componente torna alla causal attention
  ordinaria*. Senza questo, un Exp 1b "riuscito" produrrebbe di nuovo un
  checkpoint v0-style.
- **Punti già coperti dal lavoro fatto** (la review fotografa main a un
  commit precedente): suite verde e review moduli (voci precedenti), build
  dati v2 con manifest, McNemar/CI già pianificati per Exp 0 v2, condizione
  "token unmasked" già nelle 7 condizioni di Exp 2, nota potenza statistica
  già registrata.
- **Osservazione scientifica di merito accolta (relay)**: la mask attuale
  permette ai filler di leggere l'anchor e ai token successivi di leggere i
  filler → l'informazione può essere replicata a catena. Il bottleneck
  causale resta valido, ma l'ablazione distanza misura la sopravvivenza
  dell'informazione nella catena, non la persistenza del singolo hidden
  state. Exp 2 dovrà includere la condizione **anchor-only recall** (filler
  ciechi sull'anchor; solo [RECALL] lo legge). Da pre-registrare nel ri-pin
  del criterio gating.
- **Da adottare più avanti** (registrato, non bloccante ora): multi-seed sui
  risultati principali; ablazione nomi semantici vs token neutri; dedupe
  semantico (MinHash/embedding) oltre a content_id; probing con group-split
  per content_id e nested CV; controlli causali estesi per Exp 5; niente
  claim "compressione ~100×" senza definire la metrica (posizioni vs byte KV
  vs FLOPs); LICENSE/CITATION.
- **Notebook**: creati `notebooks/README.md` (piano 00–11 con stato e
  dipendenze) e i tre eseguibili oggi: `00_project_status`,
  `01_dataset_audit`, `03_attention_bottleneck_visualizer` — tutti
  smoke-testati cella per cella (output verificati; figure in
  `results/figures/`, tabelle in `results/tables/`). I notebook 🔒 si
  costruiscono quando esistono gli artefatti che devono leggere.
- **Fix documentale**: README allineato allo stato verificato (la nota "test
  non garantiti verdi" era rimasta dal 2026-07-15 mattina, superata dai fatti).

## 2026-07-15 — P0 completato: bottleneck end-to-end su tutta la pipeline

Chiuso il gate P0 (review esterna, triage nella voce precedente). Ora esiste
UN solo percorso di attention, dichiarato e verificato.

- **`config.py`**: nuovo campo `attention_mode`
  (`compress_bottleneck` default | `full_context` controllo), validato in
  `__post_init__`, salvato in ogni checkpoint via `cfg.save`.
- **`dataset.py`**: dopo la tokenizzazione con truncation, `validate_layout`
  su ogni riga + check "almeno un token di target" — una riga che perde i
  token paired o il target ora è `LayoutError` con l'identità dell'esempio,
  mai un esempio silenziosamente degenere nel batch.
- **`train.py`**: nuovo `forward_batch` (unico forward per training e eval
  interna, via `forward_bottlenecked` con `cfg.attention_mode`);
  fix accumulo gradienti (`should_step` con flush dell'ultima finestra
  parziale per epoca) e `total_steps` con `ceil` per epoca; CLI
  `--attention-mode`.
- **`eval.py`**: `generate_recall` → decoder reference bottleneck;
  `eval_mcq` → `option_loglik_bottlenecked`; il vecchio `option_loglik` è
  rinominato **`option_loglik_full_context`** e riservato ai benchmark di
  capacità generale (WikiText/HellaSwag/MMLU, che per protocollo restano
  full-context); **fix bug distanza**: `example_distance()` legge
  `distance_target_tokens` (KeyError esplicito su riga B senza campo, mai
  bucket-0 silenzioso); il mode di default si legge dalla config del
  checkpoint (`checkpoint_attention_mode`); ogni risultato registra
  `attention_mode`, `decoder`, e sha256 del manifest dati.
- **`probe.py`**: estrazione hidden state via `forward_bottlenecked`;
  `--label-kind` obbligatorio (sentiment|topic, mai mescolati);
  provenance nel risultato.
- **`intervention.py` / `try_model.py`**: mode threading; il playground usa
  lo stesso decoder reference dell'eval. Nota di onestà: i checkpoint v0
  (senza `attention_mode` in config) di default girano `full_context` nel
  playground — è il regime sotto cui sono stati addestrati.
- **`run_exp0.py`**: aggiornato al nome esplicito `option_loglik_full_context`
  (Exp 0 è la baseline prompt: testo piano, nessun token comportamentale).
- **Test anti-regressione** (`tests/test_pipeline_integration.py`, CI):
  modello-spia che cattura l'`attention_mask` ricevuta — train forward,
  generate, MCQ scoring e probe extraction DEVONO passare la mask 4D
  identica a `build_bottleneck_mask`, o il test è rosso. Più: predicato
  grad-accum ([F,T,F,T,T] su 5 batch/accum 2), contratto distanza, guardie
  truncation. Livello integration (`test_qwen_integration.py`): train step
  reale + generate + MCQ su Qwen attraverso il percorso condiviso.
- **Esito**: 51 passed (44 unit + 7 integration), ruff e compileall puliti.
- **Conseguenza**: il P0 è chiuso; Exp 1b resta bloccato solo dal toy gate
  (tentativo 2 in corso) e dal ri-pin del criterio Exp 2.

## 2026-07-15 — PRE-REGISTRAZIONE: criterio gating Exp 2 su test set v2 (a risultati non visti)

Sostituisce la voce 2026-06-09 (che fissava il verdetto su 154 MCQ del test
v0, oggi superato). Fissata ORA, prima di Exp 0 v2, prima di Exp 1b e prima
di qualsiasi risultato Exp 2. **Queste soglie non si ritoccano.**

- **Coorte del verdetto**: TUTTI i 541 esempi del test set v2, tutti
  MCQ-annotati (manifest sha256
  `d531d521b270b5edbc6e52f4f478c5472377da2266e9efba11caf4fb60c702ea`;
  composizione: 386 CNN/DailyMail out-of-style + 6 handwritten + 149
  sintetici in-style, questi ultimi MAI visti in training — a differenza
  del test v0, che riusava 148 contesti del train). Selezione per
  `example_id` via `CohortSelection`, mai "primi N".
- **Baseline**: la baseline v0 (0.82 su n=50) è storica e NON è l'asticella.
  L'asticella è **Exp 0 v2**: stesso protocollo two-stage
  (riassunto→risposta), rieseguito sull'intera coorte di 541, con prediction
  record per esempio. Va eseguito PRIMA di Exp 2 e con lo stesso scoring
  (`option_loglik_full_context` per la baseline prompt, che è testo piano).
- **Criterio primario (vittoria)**: la condizione *true bottleneck* batte la
  baseline prompt-summary sugli stessi 541 item — differenza appaiata > 0
  con McNemar p < 0.05.
- **Criterio secondario (pareggio legittimo)**: differenza appaiata entro
  **±3 punti** (CI 95% bootstrap della differenza contenuto in [−3, +3]) —
  con n=541 la sd binomiale è ~2 pt, quindi ±3 è più stretto del ±5 usato
  con n=50, com'è giusto. In caso di pareggio vince l'efficienza del canale
  (1 hidden state vs ~150 token di riassunto), e il valore del progetto si
  sposta su Exp 3/5. La densità NON va chiamata "compressione ~100×" senza
  specificare la metrica (posizioni attese vs byte KV vs FLOPs).
- **Numero separato obbligatorio per l'out-of-style**: accuracy e delta
  appaiato riportati anche sulla sola partizione CNN/DailyMail (n=386,
  finora senza alcuna baseline) — è la parte che distingue comportamento
  appreso da stile del generatore. Un PASS complessivo con FAIL netto
  sull'out-of-style va dichiarato come tale, non mediato via.
- **Condizioni di Exp 2** (8, estese col controllo relay):
  1. full context (upper bound); 2. prompt summary (baseline Exp 0);
  3. token addestrati, attention libera ("unmasked" — quantifica il
  copying); 4. **true bottleneck**; 5. anchor azzerato; 6. anchor
  shuffled/swap tra esempi; 7. token non addestrato; 8. **anchor-only
  recall** (i filler non leggono l'anchor; solo [RECALL] può — separa
  "persistenza del singolo hidden state" da "relay attraverso la catena").
- **Statistica**: per ogni condizione bootstrap CI (10k resample) su
  accuracy; McNemar appaiato per i confronti col baseline; breakdown per
  source e, sui tipi B, per `distance_target_tokens` (banda 3–100; le
  distanze con n<10 si riportano ma non supportano claim).
- **Fact retrieval**: metrica secondaria (rumorosa, v. caveat Exp 0 v0),
  riportata ma non gating.
- **Motivo del ri-pin ora**: il toy gate (tentativo 2 in corso) non informa
  queste soglie; fissarle prima di vedere QUALSIASI numero nuovo chiude la
  porta alla razionalizzazione a posteriori — lo stesso principio della
  voce 2026-06-09, aggiornato ai dati v2.

## 2026-07-15 — Toy gate tentativo 2: PASS su tutte le soglie pre-registrate

- **Setup**: identico al tentativo 1 tranne 400 codici train (was 160) e 30
  epoche (was 20); soglie invariate. ~76 min CPU
  (`results/toy_bottleneck_try2.json`).
- **Esito**: PASS. bottleneck_acc **0.925** su 40 codici mai visti (soglia
  0.90, chance ~1e-4); anchor_removed 0.000; context_override_rate **1.00**;
  swap_rate **0.90** (dal 0.45 del tent. 1); untrained 0.000 (misurato al
  tent. 1, stesso modello base). Train subset 1.00.
- **Lettura causale**: il trapianto dell'anchor (tutte le profondità) ora
  produce il codice dell'esempio sorgente nel 90% dei casi ignorando sempre
  il contesto del prompt ospite → l'hidden state di `[COMPRESS]` non solo
  contiene l'informazione: la DETERMINA. `full_context_acc` resta 0 — il
  modello addestrato col bottleneck non usa l'attention libera; il canale
  appreso è quello vincolato.
- **Diagnosi tent. 1 confermata**: era un problema di quantità (varietà dei
  codici), non di meccanismo. La transizione di fase si è ripresentata con
  la stessa firma (plateau a ~4·ln10/5, poi discesa).
- **Decisione**: gate bottleneck CHIUSO. Lanciato Exp 1b con la ricetta
  conservativa pre-registrata (`train_config_1b.json`: lr 5e-5, 1 epoca,
  r=8, α=16, dropout 0.1, lambda_c 0, attention_mode compress_bottleneck,
  dati v2) — primo training reale attraverso il bottleneck. Pre-flight dati
  superato (1197+149 righe, max 335 token ≤ 384). Il gate di stabilità
  (`run_exp1.py`, soglie INVARIATE: ppl ≤ +5%, calo ≤ 2 pt) girerà con
  **500 campioni downstream** (was 200) per la nota di potenza statistica;
  output separato in `results/exp1b_stability.json` (il v0 resta intatto).

## 2026-07-16 — Triage secondo feedback esterno (ricevuto durante Exp 1b)

Testo salvato in `docs/external_feedback_2026-07-16.md`. Qualità inferiore
alla review del 2026-07-15: contiene osservazioni utili ma anche errori
fattuali sul nostro codice e imprecisioni tecniche. Triage:

- **ACCOLTO — controllo aggiuntivo "anchor medio"**: sostituire l'anchor con
  l'hidden state MEDIO del contesto (norm-matched) come condizione
  diagnostica extra di Exp 2, accanto a zeroed/shuffled. Distingue
  "informazione specifica" da "rumore strutturato". NON tocca il criterio
  gating pre-registrato (resta a 8 condizioni + questa come diagnostica non
  gating — aggiunta dichiarata ORA, prima di ogni risultato Exp 2).
- **ACCOLTO — docs/mechanism.md**: documento dedicato alla meccanica del
  compressore (mask, invariante, decoder reference, ricetta, controlli).
  In backlog, da scrivere dopo il verdetto Exp 1b.
- **ACCOLTO — contingenza Exp 1c**: se Exp 1b FAIL, un tentativo con 2-3
  epoche + warmup prima di dichiarare il meccanismo insufficiente
  (coerente con la dinamica "transizione lenta" vista nel toy gate).
- **GIÀ PIANIFICATO** (il feedback non lo sapeva): Exp 0 v2 + Exp 2 sul
  modello trained come vera verifica della compressione (ordine gate);
  curva multi-anchor K e compressione ricorsiva (roadmap P2 della review
  precedente); anchor shuffled (condizione 6 pre-registrata).
- **GIÀ FATTO** (errore fattuale del feedback): l'init mean-embedding dei
  token speciali esiste da sempre (`_mean_init_new_tokens`, decisions
  2026-06-09) — il "miglioramento 4" descrive un problema che non abbiamo.
- **RESPINTO — loss di ricostruzione esplicita (aux decoder)**: ci
  trasformerebbe in una variante ICAE/autoencoder (il posizionamento è
  differenziarsi da ICAE, non reimplementarlo), reintrodurrebbe rischio di
  leakage dello stile del generatore dei riassunti target, e cambierebbe il
  claim (comportamento appreso via LM loss → compressione supervisionata).
  Riconsiderabile solo come ablazione separata e dichiarata, mai nel run
  principale.
- **RESPINTO — multi-anchor ORA**: cambierebbe l'ipotesi in corso di
  esperimento (il claim è sul SINGOLO anchor come canale obbligato).
- **RESPINTO — "context dropout PAD" come procedura di training**: mal
  posta — con contesto PAD l'anchor calcolato nel forward non contiene
  nulla da preservare; la versione corretta è il patching che già usiamo
  come controllo in eval.
- **Correzioni tecniche al testo**: la capacità del vettore è 896·16 =
  **14336 bit**, non "2^14336"; il confronto "loss 4.85 vs loss WikiText
  ~3" è mal posto (CE target-only sotto bottleneck vs LM loss full-context
  — non comparabili; la verifica giusta è il gate in corso).

## 2026-07-16 — Triage terzo feedback esterno (con verifica diretta del repo)

Feedback ricevuto dopo fetch diretto di raw/API GitHub (stato osservato:
commit `52c0395`). Il migliore dei tre per metodo: ha verificato invece di
opinare. Triage:

- **ACCOLTA — correzione interpretativa sul toy gate**: tra tent. 1 e 2
  sono cambiate DUE variabili insieme (codici 160→400, epoche 20→30);
  l'attribuzione "mancavano dati" nella voce del PASS era più forte del
  lecito. Conclusione corretta: capacità di training insufficiente, con
  contributo relativo di varietà vs durata NON isolato; il PASS esclude il
  difetto strutturale del meccanismo. Corretti i documenti vivi
  (`experiments/toy_bottleneck/README.md`, `docs/mechanism.md`); la voce
  storica del PASS resta com'era, questa nota fa da errata.
- **ACCOLTA — cautela su `full_context_acc = 0`**: declassata da "conferma
  indiretta del meccanismo" a **sensibilità al regime attentivo**
  (compatibile con la specializzazione sul canale ma anche con un semplice
  shift di distribuzione). Corretti gli stessi documenti vivi.
- **GIÀ RISOLTO — disallineamento documentale**: il feedback osservava il
  repo a `52c0395`; il commit `896ce92` (riallineamento snapshot/cronologia)
  è successivo e copre tutti i punti elencati.
- **CONFERMA UTILE**: lettura del toy gate condivisa ("la meccanica
  fondamentale è realizzabile; non dimostra ancora compressione semantica
  su testo naturale") — è la stessa formulazione dei nostri limiti in
  `docs/mechanism.md` §10.
- **Nota**: `exp1b_stability.json` assente dal repo perché il gate è ancora
  in esecuzione; verrà pushato col verdetto.

## 2026-07-16 — Triage quarto feedback esterno + dichiarazione condizione "forced relay"

Quarto feedback: in gran parte sintesi fedele dei nostri documenti (buon
segno: il repo si spiega da solo), con un errore fattuale, letture
pre-riallineamento e due spunti nuovi accolti.

- **ACCOLTO — condizione diagnostica "forced relay" per Exp 2** (dichiarata
  ORA, a risultati non visti, non-gating): filler che vedono l'anchor ma
  `[RECALL]` cieco sull'anchor — l'inverso dell'anchor-only. Le due
  condizioni insieme chiudono la matrice del relay: anchor-only misura la
  persistenza del singolo stato, forced-relay misura l'informazione che
  sopravvive alla catena di hop. Totale condizioni Exp 2: 8 gating
  pre-registrate + 2 diagnostiche (anchor-medio, forced-relay).
- **ACCOLTO (backlog Exp 3) — probing su fatti specifici**: oltre a
  sentiment/topic, probe su contenuti puntuali del contesto ("quale
  città?") — più vicino a ciò che il recall deve trasferire davvero.
- **Errore fattuale a verbale**: "anchor = 896×16 = 14336 dimensioni,
  16 layer" confonde dimensioni (896 per layer), bit (896 dim × 16 bit
  fp16 = 14336 bit di capacità grezza) e numero di layer (Qwen2.5-0.5B ne
  ha 24, non 16).
- **Letture stantie**: ripete l'attribuzione "dati insufficienti" del toy
  tent. 1 già corretta dall'errata (voce precedente) e cita come stato
  corrente sezioni documentali già storicizzate in `896ce92` — il feedback
  ha osservato il repo pre-riallineamento.
- **Convergenze** (già nostre posizioni): toy = prova del canale, non della
  compressione semantica (mechanism.md §10); rischio esito marginale di
  Exp 2 (voce 2026-06-09); monitoraggio varianza hidden (7.5→9.6) da
  proseguire in un eventuale Exp 1c.

## 2026-07-16 — Exp 1b: PASS — il training col bottleneck non degrada il modello

- **Setup**: primo training reale attraverso il bottleneck
  (`train_config_1b.json`: lr 5e-5, 1 epoca, r=8, α=16, dropout 0.1,
  `lambda_c=0`, `attention_mode=compress_bottleneck`, dati v2). Training
  ~45 min CPU (300 step, best eval loss_ce 4.54, varianza hidden 7.5→9.6,
  nessun collasso). Gate: `run_exp1.py` con **500 campioni downstream**
  (was 200, nota potenza statistica) e 100 blocchi WikiText, **soglie
  INVARIATE** dal gate v0. Output: `results/exp1b_stability.json`.
- **Risultati** (base → trained):
  WikiText-2 ppl 14.399 → 14.435 (**+0.24%**, gate ≤ +5%);
  HellaSwag 0.476 → 0.472 (−0.4 pt, gate ≤ 2.0);
  MMLU 0.280 → 0.290 (+1.0 pt, migliora). **PASS su tutti e tre i criteri.**
- **Confronto col FAIL v0** (stesse soglie): ppl +24.7% → **+0.24%**
  (100× meglio), MMLU −4.0 → +1.0. La combinazione ricetta conservativa +
  bottleneck ha eliminato il catastrophic forgetting. Nota causale onesta:
  ricetta E architettura sono cambiate insieme rispetto al v0 — non è
  isolato quale delle due pesi di più (non serve isolarlo per il gate, che
  chiede solo stabilità).
- **Cautele da mantenere**: (1) con n=500 la sd binomiale è ~2.2 pt, i
  delta downstream (−0.4/+1.0) sono nel rumore — il dato forte è la ppl,
  inequivocabile; (2) il PASS dice che il modello NON si è rotto, NON che
  il recall funzioni su testo naturale — quella è la domanda di
  Exp 0 v2 → Exp 2 (v. quarto feedback e mechanism.md §10); (3) il task
  learning in 1 epoca è modesto (loss 4.54): se Exp 2 desse recall nullo,
  la contingenza Exp 1c (2-3 epoche + warmup) resta aperta PRIMA di
  conclusioni sul meccanismo.
- **Decisione**: **Exp 2 SBLOCCATO**. Prossimo passo obbligato: Exp 0 v2
  (baseline sull'intera coorte pre-registrata di 541 MCQ, prediction record
  per esempio), poi Exp 2 con 8 condizioni gating + 2 diagnostiche.

## 2026-07-16 — Exp 0 v2 completato: la baseline è misurata, l'effetto bigino è morto

- **Run**: intera coorte pre-registrata (541 MCQ, ~8h50 CPU), modello base,
  protocollo v0 invariato. Record per esempio in
  `results/exp0_v2_records.jsonl`, coorte congelata in
  `results/exp0_v2_cohort.json`, aggregati in `results/exp0_v2.json`.
- **Risultati aggregati** (bootstrap CI 95%):
  `mcq_from_summary` **0.656** [0.616, 0.697];
  `mcq_full_context` **0.808** [0.774, 0.841];
  `summary_fact_retrieval` 0.245 [0.215, 0.276].
  McNemar summary vs full: n01=106 vs n10=24, **p=1.7e-13** — il contesto
  pieno batte il riassunto in modo schiacciante.
- **Per partizione** (obbligo pre-registrato):
  CNN/DailyMail (n=386): summary **0.593**, full 0.808;
  sintetici (n=149): summary 0.812 = full 0.812;
  handwritten (n=6): 0.833/0.667 (n troppo piccolo per claim).
- **L'effetto bigino del v0 è morto.** Il v0 (n=50, quasi solo sintetici)
  aveva trovato summary 0.82 > full 0.74; sulla coorte vera il rapporto si
  ribalta (aggregato) o si annulla (sintetici). Era rumore da campione
  piccolo + composizione. Lezione registrata: mai fissare narrazioni su
  n=50.
- **Il riassunto è un compressore pessimo su testo reale**: sui CNN
  sopravvive in media il **5.9%** dei fatti verificabili (82% dei riassunti
  ne conserva ZERO); quando i fatti sopravvivono l'MCQ risale a ~0.80.
  Il collo di bottiglia è la sommarizzazione, non la lettura.
- **Asticella ufficiale di Exp 2** (dal criterio pre-registrato, ora con i
  numeri): battere/pareggiare summary **0.656** aggregato sui 541 appaiati
  (McNemar), con numero separato vs **0.593** sulla partizione CNN.
  Upper bound di riferimento: full context 0.808.
- **Decisione**: lanciato Exp 2 (8 condizioni gating + 2 diagnostiche,
  checkpoint `exp1b-bottleneck-v2/best`, scoring MCQ batched, resumable).
  Nessuna soglia toccata.

## Template per nuove decisioni

```
## YYYY-MM-DD — Titolo
- **Decisione**: ...
- **Motivo**: ...
- **Alternative scartate**: ...
```
