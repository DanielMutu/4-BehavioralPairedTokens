# Review esterna della repository (2026-07-15)

> Revisione statica del branch `main` su GitHub al 2026-07-15, fornita
> dall'utente. Non include esecuzione di training o test. Il triage punto
> per punto (claim verificate sul codice, cosa è già coperto, cosa è
> accettato come roadmap) è in `experiments/decisions.md` → voce
> 2026-07-15 "Triage review esterna". Il piano notebook derivato è in
> `notebooks/README.md`.

## Sintesi veloce

L'idea è interessante e potenzialmente forte:

> insegnare a un piccolo language model a condensare un contesto nella rappresentazione interna associata al token `[COMPRESS]`, per poi recuperare quell'informazione quando incontra `[RECALL]`.

La direzione scientificamente corretta è la nuova pipeline **v2**, che introduce una maschera di attenzione per impedire ai token successivi a `[COMPRESS]` di rileggere direttamente il contesto. La versione precedente non aveva questo vincolo e quindi poteva semplicemente copiare dal testo originale. Inoltre, gli split precedenti avevano leakage tra train, test e probe.

**Il principale problema attuale è che la v2 non è ancora integrata end-to-end**:

- `src/bottleneck.py` contiene la nuova logica;
- i relativi test sembrano maturi;
- ma `train.py`, `eval.py`, `probe.py` e `intervention.py` continuano in diversi punti a chiamare direttamente il modello con l'attenzione normale.

Quindi oggi la repo contiene **una buona infrastruttura v2**, ma non ancora una pipeline sperimentale v2 completa e scientificamente utilizzabile.

## Giudizio complessivo

| Area | Valutazione | Commento |
|---|---:|---|
| Idea scientifica | **Molto buona** | Domanda chiara, falsificabile e con possibili negative results |
| Onestà metodologica | **Ottima** | I difetti della v0 sono documentati apertamente |
| Data governance | **Buona, in miglioramento** | Manifest, hash e split disgiunti sono un grande passo avanti |
| Test del bottleneck | **Buoni** | Ci sono test unitari e integrazione Qwen reale |
| Pipeline end-to-end | **Incompleta** | Il bottleneck non è ancora usato da tutti i percorsi |
| Valutazione statistica | **Debole/iniziale** | Un solo seed, campioni piccoli, poche incertezze |
| Probing | **Concettualmente buono, implementazione da rifare** | Mancano controlli statistici e integrazione v2 |
| Intervento causale | **Promettente ma prematuro** | "Output cambiato" non dimostra ancora causalità semantica |
| Notebook | **Assenti** | La cartella contiene soltanto `.gitkeep` |
| Riproducibilità | **Buona base** | `uv.lock`, CI, config e manifest; documentazione però disallineata |

## 2. Criticità tecniche prioritarie

### 2.1 Il true bottleneck non è ancora collegato al training

`src/bottleneck.py` definisce `forward_bottlenecked()`, presentandolo come il percorso condiviso da train, eval e probing. Ma nel training attuale viene ancora fatto `model(input_ids=..., attention_mask=..., labels=...)`: normale attenzione causale, non la maschera bottleneck 4D.

**Conseguenza**: un nuovo checkpoint addestrato con `src.train` continuerebbe a essere un checkpoint v0-style, anche se `src/bottleneck.py` è presente.

**Fix consigliato**: un solo entry point obbligatorio (`forward_bottlenecked` con `mode=cfg.attention_mode`), con `attention_mode` in config e salvato sempre nel checkpoint.

### 2.2 Anche evaluation e MCQ usano ancora il percorso vecchio

`generate_recall()` usa `model.generate()` standard, `option_loglik()` chiama direttamente il modello senza mask. `generate_bottlenecked()` e `option_loglik_bottlenecked()` esistono ma non sono collegati a `src/eval.py`. Ogni risultato dovrebbe registrare `attention_mode`, `decoder`, manifest e checkpoint hash.

### 2.3 Bug concreto nell'ablazione sulla distanza

Il data contract v2 usa `distance_target_tokens` e rimuove la vecchia chiave `distance`, ma `eval_recall()` legge ancora `meta.get("distance", 0)`: con i dati v2 tutti gli esempi finiscono nel bucket distanza 0. Fix piccolo ma bloccante; meglio fallire rumorosamente che usare un default silenzioso.

### 2.4 Il filler può diventare una memoria secondaria

La mask attuale consente a ogni query dopo `[COMPRESS]` di leggere tutte le chiavi da `[COMPRESS]` in poi: l'informazione può essere replicata e distribuita nei filler (relay). Il bottleneck causale resta reale, ma l'ablazione distanza misura "quanto sopravvive l'informazione in una catena di stati successivi", non "quanto a lungo il singolo hidden state di `[COMPRESS]` conserva l'informazione". Servono almeno due maschere: **relay allowed** (attuale) e **anchor-only recall** (i filler non leggono `[COMPRESS]`; solo `[RECALL]` può). Utile una terza: finestra locale fissa senza anchor.

### 2.5 Non è ancora compressione computazionale

Il decoder reference ricomputa tutta la sequenza a ogni token (`use_cache=False`): corretto come oracolo, ma il contesto resta nell'input e non si risparmia KV/latenza. Evitare la frase "compressione ~100×" finché non si definisce cosa si misura (posizioni accessibili, byte KV, FLOPs, latenza, accuratezza per byte). Il fast path con KV pruning è un esperimento separato, confrontato logit-per-logit col reference.

### 2.6 Troncamento potenzialmente silenzioso nel dataset

`BehavioralTokenDataset` tokenizza con `truncation=True`; `validate_layout()` esiste proprio per rifiutare righe dove i token paired sono stati troncati, ma non è usato dal dataset/training. Per i run scientifici qualsiasi troncamento inatteso dovrebbe essere un errore, con logging di percentuali e posizioni.

### 2.7 Accumulo del gradiente incompleto

`optimizer.step()` scatta solo su `(i+1) % grad_accum == 0`: l'ultimo gruppo parziale dell'epoca non viene applicato; `total_steps` usa divisione intera. Fix: `should_step = ((i+1) % grad_accum == 0) or (i+1 == len(loader))` e `math.ceil`.

## 3. Criticità scientifiche e statistiche

- **3.1 Un solo seed non basta**: per i risultati principali servono ≥5 seed (13, 42, 73, 101, 137) con media ± sd e bootstrap CI; il seed fisso resta per debug/regression.
- **3.2 Gate downstream sotto-potenziato**: con 200 item l'incertezza (~3 pt) supera la soglia (2 pt). McNemar appaiato, bootstrap appaiato, CI, ≥1000 item quando possibile; distinguere "nessuna evidenza di degrado" da "equivalenza dimostrata" (serve un test di equivalenza).
- **3.3 Baseline da ampliare**: full context; full context + bottleneck non addestrato (effetto architetturale puro); riassunto generato/estrattivo/oracle; token random; `[COMPRESS]` senza bottleneck (copying); bottleneck senza nome semantico; k memory token (curva capacità). Confrontare accuracy E costo totale.
- **3.4 Nomi semantici come confondente**: ablazione `[COMPRESS]/[RECALL]` vs `[TOKEN_A]/[TOKEN_B]` vs `[XQZ1]/[XQZ2]`, nomi scambiati, embedding congelati vs addestrabili.
- **3.5 Leakage semantico oltre quello esatto**: `content_id` non cattura parafrasi/template/near-duplicates. Aggiungere MinHash/Jaccard su n-gram, similarità embedding, group split per template/generatore, report nearest-neighbor train-test.

## 4. Probing: cosa migliorare

Il probe attuale: logistic regression, split 70/30 per riga, senza scaling/CV/permutation/CI, e estrae hidden state con forward ordinario (non `forward_bottlenecked`). Problemi: (A) non usa il bottleneck v2 — criticità principale; (B) split per riga → leakage se esistono varianti dello stesso contenuto (usare GroupShuffleSplit/StratifiedGroupKFold per `content_id`); (C) controlli non fattoriali (`recall_untrained` cambia 4 cose insieme); (D) rischio cherry-picking del layer (serve correzione o validation separato); (E) accuracy da sola insufficiente (balanced acc, macro-F1, AUC, permutation test, bootstrap, selectivity, curve per layer e per n esempi).

Pipeline suggerita: feature congelate → split per content_id → scaling in Pipeline sklearn → C scelto su fold interno → nested CV → permutation test → CI.

## 5. Intervento causale: cosa manca

L'intervento attuale sposta il vettore lungo la direzione del probe e misura se l'output è cambiato — non basta. Controlli indispensabili: direzione casuale norm-matched; direzione ortogonale; token casuale; token pre-`[COMPRESS]`; modello non addestrato; dose-response con α ∈ {−4,−2,−1,0,1,2,4}; specificità (sentiment non deve alterare topic/entità/lunghezza); valutazione automatica target-aware; annotazione cieca su campione; generazione via decoder bottleneck. Metrica primaria: P(output classificato come target | α) vs random/sham; il risultato convincente è una curva monotona in α, specifica per il target, assente nei controlli.

## 6. Piano notebook proposto

(Recepito integralmente in `notebooks/README.md` — 00 status, 01 dataset audit, 02 tokenization/layout, 03 bottleneck visualizer, 04 baselines, 05 training diagnostics, 06 distance/relay ablation, 07 layerwise representation, 08 causal intervention, 09 efficiency/rate-distortion, 10 error atlas, 11 paper figures.)

## 7. Nuovi esperimenti a più alto valore

Capacità del bottleneck con k ∈ {1,2,4,8,16} memory token; task sintetici controllati (key-value retrieval, associative recall, liste, binding, conteggio, negazioni, codici); generalizzazione composizionale (solo dopo definizione operativa dei token); cross-domain; cross-model; compressione ricorsiva (solo dopo il successo della singola).

## 8. Struttura Git

Documentazione disallineata (README/CHANGELOG dicevano "test non garantiti verdi" mentre decisions.md dichiara 40 test passati e build v2 completata) → blocco unico "Current verified status" nel README. Separare protocollo scientifico da istruzioni operative (docs/research_protocol.md, docs/current_status.md, ...). File mancanti: LICENSE, CITATION.cff, CONTRIBUTING, data card, model card. CI: aggiungere workflow integration manuale (workflow_dispatch), coverage sui moduli critici, smoke end-to-end su fixture minuscola. Artefatti: directory immutabile per run con config/environment/git/manifest/metrics/predictions.

## Roadmap consigliata

**P0 — correttezza**: (1) `forward_bottlenecked` in train.py; (2) `generate_bottlenecked`/`option_loglik_bottlenecked` in eval.py; (3) v2 in probe.py e intervention.py; (4) fix `distance` → `distance_target_tokens`; (5) `validate_layout` dopo tokenizzazione; (6) test end-to-end train→gen→MCQ sotto bottleneck; (7) allineare README/CHANGELOG allo stato verificato.

**P1 — validità scientifica**: (8) relay-allowed vs anchor-only; (9) Exp 0 su tutti i 541; (10) pre-registrazione metriche/coorte/soglie; (11) Exp 1b con replay, LR basso, early stop su stabilità; (12) multi-seed + CI; (13) probing con split per content_id e nested CV; (14) controlli causali per gli interventi.

**P2 — valore**: (15) curva memory token; (16) task sintetici di capacità; (17) misure KV/latenza/memoria; (18) fast decoder logit-equivalente; (19) OOD/multilingua/cross-model; (20) composizione.

## Conclusione

La repo ha una buona idea, un approccio onesto e una base metodologica promettente. Il rischio maggiore non è che l'idea sia sbagliata: è eseguire nuovi esperimenti pensando di usare la v2 mentre gli script principali passano ancora dalla causal attention ordinaria. Milestone prima di qualsiasi training serio:

> **un unico percorso bottleneck condiviso e verificato da training, evaluation, probing e intervento, con un test end-to-end che fallisca se anche uno solo di questi componenti torna accidentalmente alla causal attention ordinaria.**
