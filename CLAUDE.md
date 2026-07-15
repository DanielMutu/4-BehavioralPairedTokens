# CLAUDE.md — Project: Behavioral Paired Tokens (COMPRESS/RECALL)

## Obiettivo del progetto

Ricerca sperimentale su **token comportamentali paired** (`[COMPRESS]` / `[RECALL]`) in small LLMs.

L'ipotesi centrale: è possibile addestrare un modello a sviluppare una **memoria semantica dinamica mid-generation** tramite token espliciti accoppiati, verificabile attraverso probing degli hidden states e intervento causale.

Questo si distingue dalla letteratura esistente (GIST tokens, AutoCompressors) perché:
- I token sono **segnali comportamentali espliciti** inseriti dall'utente, non attention mask implicite
- Si misura la **struttura semantica** dell'hidden state associato a `[COMPRESS]` con controlli rigorosi
- Si testa la **persistenza** del recall al variare della distanza tra i due token
- Si esplora la **composizione** con altri behavioral token (`[REASON]`, ecc.)
- Si tenta un **intervento causale** sugli hidden state per dimostrare causalità (non solo correlazione)

---

## Riferimenti letteratura chiave

| Paper | Anno | Rilevanza |
|---|---|---|
| Mu et al. — *Learning to Compress Prompts with Gist Tokens* | 2023 | Baseline più vicina, da citare e superare |
| Chevalier et al. — *AutoCompressors* | 2023 | Compressione ricorsiva, ratio 30:1 |
| Ge et al. — *ICAE (In-context AutoEncoder)* | 2023/2024 | Encoder→memory slot tokens |
| Tarasov et al. — *Sentence-Anchored Gist Compression* | 2024 | Compressione a livello frase |
| *Forget, Then Recall* (gist sparse attention) | 2025 | Naming simile, approccio diverso |

**Posizionamento**: costruiamo *sopra* GIST, non lo reimplementiamo. Differenziamoci su **probing rigoroso, composizione, intervento causale**.

---

## Stack tecnico

### Modello base
- **Primario**: `Qwen/Qwen2.5-0.5B` (sviluppo e debug)
- **Secondario**: `Qwen/Qwen2.5-1.5B` (esperimenti seri)
- **Futuro**: `Qwen/Qwen2.5-7B` (con RTX 5060 Ti)

### Hardware
- **Dev/debug**: MSI Aegis — i7-6700, GTX 970 4GB, 16GB RAM, Ubuntu/Docker
- **Run seri**: stesso homelab, training overnight su config leggere
- **Upgrade pianificato**: RTX 5060 Ti 16GB (per modelli 7B+)

### Librerie
```
transformers
peft (LoRA)
torch
datasets
wandb (o tensorboard per logging locale)
numpy / matplotlib (analisi e plotting)
scikit-learn (linear probing)
```

### Ambiente
- Python 3.11+
- Docker su Ubuntu
- Sviluppo via Claude Code (WSL o homelab diretto)

---

## Architettura dell'esperimento

### 1. Aggiunta token al vocabolario
```python
special_tokens = ["[COMPRESS]", "[RECALL]", "[REASON]"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))
```

**Nota sul naming**: usiamo nomi semantici espliciti, non abbreviazioni (`[C]`, `[R]`).
Ipotesi: nomi semantici riducono il rischio di pattern matching puramente lessicale.

### 2. Fine-tuning con LoRA (Qwen frozen, solo adapter + nuovi embedding)
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
```

### 3. Struttura dataset (3 tipologie obbligatorie)
- **Tipo A** — Compress → Recall immediato (distanza 0)
- **Tipo B** — Compress → N token intermedi → Recall (distanza variabile: 3, 10, 20, 50, 100)
- **Tipo C** — Composizione: `[COMPRESS][REASON]` o `[REASON][COMPRESS]`

Target: **500–1000 esempi per tipologia**.

#### Attenzione al dataset sintetico
Se il dataset è generato con Claude/GPT-4, il modello rischia di imparare lo **stile di compressione del generatore** invece di un comportamento di compressione generale.

**Mitigazioni obbligatorie**:
- Variare i generatori (più LLM diversi, non solo uno)
- Includere esempi **scritti a mano** o estratti da dataset pubblici (es. CNN/DailyMail summaries)
- **Held-out test set** con stile diverso dal training (es. testi tecnici se il training è narrativo)

### 4. Loss function

#### Base
- Standard cross-entropy (causal LM)

#### Consistency loss (opzionale, da introdurre solo dopo che la base funziona)
```python
loss_consistency = 1 - F.cosine_similarity(h_compress, h_recall, dim=-1).mean()
loss_total = loss_ce + lambda_c * loss_consistency
```

**Warning collasso**: con `lambda_c` troppo alto, tutti gli hidden state di `[COMPRESS]` possono collassare in un unico vettore degenerato. La loss scende ma il modello non impara nulla.

**Mitigazioni**:
- Partire con `lambda_c = 0` (solo CE), aggiungere consistency solo se serve
- Monitorare la **varianza** degli hidden state di `[COMPRESS]` su batch diversi — se crolla → segnale di collasso
- Cap su `lambda_c <= 0.1` come default

---

## Esperimenti pianificati

### Exp 0 — Baseline prompt engineering (PRIMA DI TUTTO)
**Critico**: prima di addestrare qualsiasi cosa, verificare cosa fa il modello base con prompt esplicito.

```
"Riassumi il testo sopra in bullet points molto densi.
Poi, basandoti solo sul riassunto, rispondi a: <domanda>"
```

Se il modello base raggiunge già performance comparabili → tutto il progetto perde valore.
**Questo esperimento decide se vale la pena procedere.**

### Exp 1 — Stabilità baseline
Verifica che il fine-tuning non degradi le capacità generali del modello.
- Metrica primaria: perplexity su WikiText-2 prima/dopo
- Metrica secondaria: **task strutturati** (MMLU subset, GSM8K leggero, HellaSwag)
- L'aggiunta dei token + LoRA non deve rovinare le performance su task normali

### Exp 2 — Ablazione distanza
Variare la distanza tra `[COMPRESS]` e `[RECALL]` e misurare qualità del recall.
- ROUGE / BERTScore (rumorosi, da considerare indicativi)
- **Fact retrieval accuracy**: lista predefinita di fatti nel testo, conteggio espliciti corretti
- **QA multipla scelta** generata sul contenuto compresso (più dura, più affidabile)

### Exp 3 — Probing hidden states con controlli rigorosi
Linear probe sul vettore hidden state di `[COMPRESS]` su task semantici noti (sentiment, topic).

**Controlli obbligatori** (senza questi il risultato non vale nulla):
- **Probe su token random** nel contesto → deve fallire
- **Probe su `[RECALL]` non-trainato** (modello base con token aggiunti ma non fine-tunato) → deve fallire
- **Probe su posizioni shuffled** → deve fallire
- **Probe su ultimo token del contesto pre-`[COMPRESS]`** → se funziona quanto `[COMPRESS]`, il token sta solo copiando, non comprimendo

Solo se i controlli falliscono e il probe su `[COMPRESS]` funziona → segnale di compressione semantica reale.

### Exp 4 — Composizione token (se Exp 1-3 positivi)
Testare `[COMPRESS][REASON]` su esempi out-of-distribution.
- Se funziona senza training esplicito sulla composizione → risultato pubblicabile

### Exp 5 — Intervento causale (ambizioso, killer experiment)
Usando il probe addestrato in Exp 3:
1. Prendi l'hidden state di `[COMPRESS]` per un testo X
2. **Modifica** l'hidden state in direzione di un concetto target (es. sentiment opposto)
3. Verifica se il `[RECALL]` produce output coerente con la modifica

Se sì → dimostrazione di **causalità** (non solo correlazione tra hidden state e contenuto).
Questo trasforma il progetto da "interessante" a paper su venue serie.

---

## Struttura directory progetto

```
behavioral-tokens/
├── CLAUDE.md               # questo file
├── data/
│   ├── raw/                # dataset sorgente
│   ├── processed/          # esempi formattati per training
│   ├── handwritten/        # esempi manuali per anti-leakage
│   └── generation/         # script per generare esempi con LLM
├── src/
│   ├── model.py            # setup modello + token + LoRA
│   ├── dataset.py          # DataLoader e formatting
│   ├── train.py            # training loop + consistency loss
│   ├── eval.py             # metriche e valutazione
│   ├── probe.py            # linear probing + controlli
│   └── intervention.py     # causal intervention (Exp 5)
├── experiments/
│   ├── exp0_prompt_baseline/
│   ├── exp1_stability/
│   ├── exp2_distance/
│   ├── exp3_probing/
│   ├── exp4_composition/
│   ├── exp5_intervention/
│   └── decisions.md        # log delle decisioni importanti
├── notebooks/              # analisi esplorative
├── results/                # output runs, plot, metriche
└── requirements.txt
```

---

## Convenzioni di codice

- **Lingua**: codice e commenti in inglese, documentazione può essere in italiano
- **Logging**: ogni run deve loggare `loss_ce`, `loss_consistency`, `loss_total`, `perplexity`, **varianza hidden state di `[COMPRESS]`** (per detection collasso)
- **Checkpoint**: salvare ogni 500 steps, mantenere best + last
- **Riproducibilità**: sempre `seed=42`, salvare config completa per ogni run
- **Debug mode**: dataset di 100 esempi, 2 epoch, CPU — prima di qualsiasi run serio

---

## Note e decisioni di design

- **Exp 0 è gating**: se il baseline prompt-engineering funziona quanto i token addestrati, non procedere
- Iniziare sempre con Qwen2.5-0.5B per validare il codice, poi scalare
- Il dataset è il collo di bottiglia reale — diversificare generatori e includere esempi manuali
- Non reinventare GIST: citarlo, differenziarsi su probing+composizione+intervento
- Consistency loss disattivata al primo giro (`lambda_c = 0`)
- Probing senza controlli non vale niente — Exp 3 ha 4+ controlli obbligatori
- Tenere `experiments/decisions.md` aggiornato con ogni scelta non banale

---

## Exp 0 — I risultati spiegati semplici (2026-06-09)

**Cosa è stato fatto in Exp 0.** Prima di addestrare qualsiasi cosa, è stato chiesto al modello base (senza alcuna modifica): "riassumi questo testo, poi rispondi a domande usando solo il riassunto". È il test del "serve davvero il progetto, o basta chiedere per favore?".

**Cosa è uscito.** Il modello risponde correttamente all'**82%** delle domande a scelta multipla usando solo il proprio riassunto. Tirando a caso farebbe il 25%, quindi è un risultato forte. La sorpresa: quando gli dai il testo completo invece del riassunto, scende al **74%**. Sembra assurdo — più informazione, peggior risultato.

**Perché succede (in parole povere).** Un modello da 0.5B è come uno studente con poca memoria di lavoro. Se gli dai un articolo intero, si perde tra i dettagli irrilevanti. Se gli dai un bigino fatto bene, si concentra e risponde meglio. Il riassunto funziona da filtro anti-rumore. È un'arma a doppio taglio:

- **Cattiva notizia**: la compressione "gratis" (chiedere un riassunto) funziona già benissimo. I token addestrati devono battere 0.82, ed è un'asticella alta.
- **Buona notizia**: conferma la premessa del progetto — i modelli piccoli beneficiano della compressione. E il prompt baseline non può offrire ciò che i token offrono: il riassunto sono ~150 token di testo, `[COMPRESS]` è un singolo vettore. È una compressione ~100 volte più densa. Inoltre il riassunto non si può ispezionare dall'interno né manipolare causalmente.

**Il debug run del training.** Nessuna misura scientifica — solo verifica che la macchina funzioni. La loss scende regolarmente (il modello impara il pattern), nessun errore numerico, e soprattutto la varianza degli hidden state resta sana. Tradotto: il rischio era che il modello imparasse a produrre sempre lo stesso identico vettore per `[COMPRESS]` indipendentemente dal testo (il "collasso" — come uno studente che risponde sempre "42" a tutto). Non sta succedendo. Il codice è pronto per il run vero.

**Cosa significa "hidden state" e "probing"** (verranno usati molto). Quando il modello legge il token `[COMPRESS]`, internamente produce un vettore di numeri (l'hidden state) — è il suo "pensiero" in quel momento. Il probing significa addestrare un piccolo classificatore esterno che guarda solo quel vettore e prova a indovinare, ad esempio, il sentiment del testo originale. Se ci riesce, vuol dire che il vettore contiene davvero l'informazione compressa. I controlli servono a escludere spiegazioni banali (es. che qualsiasi token a caso contenga già quell'informazione).

### Come procedere (prima di lanciare i prossimi run)

1. **Criterio gating fissato ORA in decisions.md** (a risultati non visti): criterio primario "batte 0.82", criterio secondario legittimo "pareggia 0.82 con compressione ~100×". Soglia esplicita: dato il rumore con n=50, è "pareggio" tutto ciò che sta entro ±5 punti; il verdetto finale usa tutti i 154 MCQ del test set.
2. **Generare le opzioni MCQ per i 386 esempi CNN/DailyMail** prima di Exp 2, non dopo. Costo basso, valore alto: senza, il test "out-of-style" non misura nulla, ed è la parte che dimostra che il modello ha imparato un comportamento e non uno stile.
3. **Lanciare Exp 1 overnight.** Pronto, rischio basso. Se perplexity WikiText-2 e task downstream non degradano → via libera per Exp 2.

**Nota onesta sul rischio principale.** Il punto di maggior rischio del progetto non è la tecnica: è che Exp 2 dia un risultato marginale — né vittoria né sconfitta chiara — lasciando il progetto in limbo. Per questo il criterio secondario (efficienza a parità di accuracy) deciso adesso è importante: dà una via d'uscita onorevole e scientificamente legittima che sposta il valore su Exp 3 e 5, dove il progetto è davvero originale. Anche un risultato "i token non battono il prompt baseline" è pubblicabile come negative result ben documentato, se i controlli sono rigorosi come impostati.

---

## Pipeline v2 — true bottleneck (avviata 2026-07-15, IN CORSO)

**Branch**: `feat/true-compress-bottleneck-v2`. Log dettagliato: `experiments/decisions.md`
(voci 2026-07-15) e **`CHANGELOG.md`** (stato file per file).

### Perché esiste una v2 (i due difetti fatali della v0)

1. **Nessun vero bottleneck.** La v0 usa causal attention ordinaria: i token dopo
   `[COMPRESS]` possono leggere direttamente il contesto originale. Ogni successo v0 —
   inclusi i ✅ del playground qualitativo (T1/T2) — è compatibile con puro *copying
   attentivo* e NON dimostra compressione attraverso il token. Anche un Exp 1 riuscito
   non avrebbe provato il meccanismo.
2. **Split leaked.** Il `prepare_dataset.py` v0 riusava 148 contesti del train dentro
   test e probe → metriche held-out e probing gonfiati.

### Architettura v2 (decisa)

- **Mask additiva 4D** `(B,1,T,T)`: query fino a `[COMPRESS]` → causalità ordinaria;
  query dopo → solo chiavi da `[COMPRESS]` in poi. Invariante scientifica: dopo
  `[COMPRESS]` l'unica via al contesto è l'hidden state del token (`src/bottleneck.py`).
- **Decoder greedy reference** full-recomputation (`use_cache=False`) come oracolo;
  potatura KV solo come ottimizzazione successiva, accettata solo se i logits
  coincidono col reference mantenendo posizioni RoPE assolute.
- **Contratto dati v2** (`src/data_contract.py`): `example_id`/`content_id` canonici,
  split disgiunti provati da assert, manifest con hash e provenance per ogni build.
- **Ordine dei gate** (un FAIL blocca lo step successivo, si documenta e non si aggira):
  test verdi → build dati su fixture → toy code-recall gate → dati reali + manifest →
  Exp 0 v2 → Exp 1b stabilità → Exp 2.

### Stato al 2026-07-15 — sessione interrotta, cosa NON è verificato

La sessione autonoma che eseguiva il piano è **morta a metà** per crediti OpenRouter
esauriti (proxy DeepClaude in modalità `proxy-or`). Conseguenze da conoscere prima di
toccare qualsiasi cosa:

**Verificato** (output visto prima del crash):
- branch creata senza stash/reset, modifiche v0 preservate;
- SHA-256 degli artefatti v0 (split, risultati) rilevati per il manifest;
- `uv lock` + `uv sync --dev` completati, `torch 2.12.0+cpu` confermato nel venv;
- digest `python:3.11-slim` pinnato nel Dockerfile.

**NON verificato** (nessun run dopo la scrittura):
- **`pytest`: ultima esecuzione parzialmente rossa**; fix a `tests/test_bottleneck.py`
  applicato ma mai rieseguito → lo stato della suite è ignoto;
- `src/data_contract.py` (~390 righe) e il nuovo `prepare_dataset.py`: **mai eseguiti**,
  nemmeno su fixture;
- parte del codice v2 è stata generata col proxy in modalità OpenRouter (backend
  potenzialmente DeepSeek anche quando la UI mostrava Opus/Fable) → serve una **review
  più severa del solito**, in particolare su `data_contract.py` e `bottleneck.py`
  (dove c'era già un errore di sintassi corretto a mano).

**Incoerenza nota**: `requirements.txt` pinna `torch==2.12.0` (da PyPI → wheel CUDA),
ma lock e ambiente reale usano `2.12.0+cpu` dall'indice PyTorch dedicato
(`pyproject.toml`). Chi installa da requirements ottiene un ambiente diverso dal lock.

### Prossimi passi (ordine vincolante)

1. **`uv run pytest -q` e chiudere i rossi** — gate 0, decide tutto il resto.
2. Review di `src/data_contract.py` e `src/bottleneck.py` (codice non fidato, v. sopra).
3. Riallineare `requirements.txt` a torch `+cpu` (o rimandare esplicitamente a `uv sync`).
4. Build v2 su **fixture temporanee** (`prepare_dataset.py`) → solo dopo, dati reali
   con manifest e verifica zero-overlap (`--check`).
5. **Toy code-recall gate** (100–200 esempi "the secret code is NNNN"), 6 controlli:
   (a) recall corretto col bottleneck; (b) anchor azzerato → chance; (c) anchor swap →
   codice dell'altro esempio o fail; (d) contesto alterato post-anchor → logits
   invariati; (e) contesto visibile = upper bound; (f) token non addestrato → fail.
6. **Ri-pinnare il criterio gating di Exp 2 in decisions.md** (a risultati non visti):
   la voce 2026-06-09 parla di 154 MCQ, ma ora il test set ne ha 540 e gli split v2 li
   cambieranno. Servono: n esatto, baseline Exp 0 ricalcolata sull'intero test v2,
   numero separato per la parte out-of-style CNN/DailyMail (oggi senza baseline).
7. **Exp 0 v2**: prediction record per esempio, bootstrap CI, McNemar paired,
   breakdown per source/distanza.
8. **Exp 1b conservativo** (lr=5e-5, 1 epoca, r=8, dropout 0.1, `lambda_c=0`) con
   mini-gate di stabilità durante il training e selezione checkpoint vincolata.
   Nota potenza statistica: con n=200 downstream la soglia di 2 pt è sotto il rumore
   binomiale (~3 pt) — aumentare i campioni o usare McNemar sugli stessi item.
9. Solo su PASS di Exp 1b: **Exp 2** con le 7 condizioni (full context, prompt summary,
   token unmasked, true bottleneck, anchor zeroed, anchor shuffled, token untrained).

### Nota operativa — proxy DeepClaude

La sessione è morta con 402 OpenRouter e `/compact` è fallito su
`anthropic/claude-fable-5`: il passthrough dei modelli **non mappati** in
`~/Work/2-DeepClaude/proxy/model-proxy.js` evidentemente non copre i modelli nuovi
(`claude-fable-*`), contraddicendo il CLAUDE.md homelab ("i modelli senza voce in
MODEL_REMAP vanno diretti ad Anthropic"). Prima di sessioni/run lunghi su questo
progetto: **`proxy-an`** oppure crediti OpenRouter sufficienti; e verificare quel bug.

---

## Stato progetto

- [x] Setup ambiente Docker su homelab
- [x] **Exp 0 — Baseline prompt engineering** (gating decision) — baseline forte: MCQ 0.82, fact 0.74; si procede, barra fissata (vedi decisions.md 2026-06-09)
- [x] Implementazione `src/model.py` con token speciali + LoRA
- [x] Generazione dataset (A/B/C bilanciati, 3 famiglie via OpenRouter) — train=1334, eval=148, test=540, probe=302 (vedi decisions.md 2026-06-09)
- [x] Training loop base + logging (incluso varianza hidden state)
- [ ] **Exp 1 — stabilità baseline → FAIL** (2026-06-14): la ricetta degrada il modello (WikiText ppl **+24.7%**, MMLU **−4 pt**; HellaSwag ok; nessun collasso ma catastrophic forgetting). La correzione della ricetta è assorbita dalla pipeline v2 (Exp 1b). Dettagli: `experiments/exp1_stability/README.md`
- [ ] **Pipeline v2 — true bottleneck** (branch `feat/true-compress-bottleneck-v2`, vedi sezione sopra e `CHANGELOG.md`):
  - [x] Branch + salvaguardia artefatti v0 (SHA-256 rilevati)
  - [x] Ambiente riproducibile: `pyproject.toml` + `uv.lock` (torch cpu) + Dockerfile + CI
  - [x] **Suite test verde** (2026-07-15: 31 passed + ruff pulito; il test rosso era mal posto — TinyAttention a 1 layer non ha alcuna rotta contesto→post-anchor, portato a 2; vedi decisions.md)
  - [ ] **Review del codice generato via proxy OpenRouter** (`data_contract.py`, `bottleneck.py`) ← PROSSIMO PASSO
  - [ ] Build dati v2: fixture → dati reali + manifest, zero overlap verificato
  - [ ] Toy gate bottleneck (code-recall, 6 controlli causali)
  - [ ] Ri-pin criterio gating Exp 2 (test set 540, split v2) in decisions.md
  - [ ] Exp 0 v2 (test completo, per-example records, bootstrap/McNemar)
  - [ ] Exp 1b conservativo (lr 5e-5, 1 epoca, r=8) + mini-gate stabilità
- [ ] Exp 2 — ablazione distanza (bloccato: richiede Exp 1b PASS)
- [ ] Exp 3 — probing hidden states **con tutti i controlli**
- [ ] Dataset Tipo C + Exp 4 — composizione
- [ ] Exp 5 — intervento causale (se Exp 3 dà risultati positivi)
