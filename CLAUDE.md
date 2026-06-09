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

## Stato progetto

- [x] Setup ambiente Docker su homelab
- [ ] **Exp 0 — Baseline prompt engineering** (gating decision)
- [x] Implementazione `src/model.py` con token speciali + LoRA
- [ ] Generazione dataset Tipo A (500 esempi, generatori multipli)
- [x] Training loop base + logging (incluso varianza hidden state)
- [ ] Exp 1 — stabilità baseline + eval su task downstream
- [ ] Dataset Tipo B + Exp 2 — ablazione distanza
- [ ] Exp 3 — probing hidden states **con tutti i controlli**
- [ ] Dataset Tipo C + Exp 4 — composizione
- [ ] Exp 5 — intervento causale (se Exp 3 dà risultati positivi)
