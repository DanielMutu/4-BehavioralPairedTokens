# Exp 1 — Stability baseline

> **Domanda dell'esperimento.** Aggiungere i token comportamentali
> (`[COMPRESS]` / `[RECALL]` / `[REASON]`) al vocabolario e addestrare un
> adapter LoRA **degrada le capacità generali** del modello base?
>
> Exp 1 non misura il recall (è compito di Exp 2): misura solo che la ricetta
> di fine-tuning non rompa il modello. È un *gate di sicurezza* prima di
> investire negli esperimenti veri.

---

## 1. Cosa confronta

Due modelli, stesso seed (42), stesso protocollo di scoring:

| Modello | Definizione |
|---|---|
| **base** | Qwen2.5-0.5B originale **+ 3 token aggiunti** (init = media degli embedding, mai visti nei testi dei benchmark), **senza** adapter |
| **trained** | come sopra **+ adapter LoRA** addestrato (`results/checkpoints/exp1-stability/best`) |

L'adapter agisce su `q_proj`/`v_proj` a ogni forward, quindi il **delta
base→trained** isola esattamente l'effetto del fine-tuning sulle capacità
generali. I valori assoluti sono cloze-style (log-likelihood normalizzata per
lunghezza): **conta solo il delta**, non il confronto con le leaderboard.

### Metriche

- **WikiText-2 perplexity** (primaria) — abilità di language modeling.
- **HellaSwag accuracy** e **MMLU accuracy** (secondarie) — task strutturati
  multiple-choice, scoring via `option_loglik` (stesso usato per gli MCQ).
- **GSM8K escluso volutamente**: generativo, un 0.5B base è già al chance
  level → nessun margine per *rilevare* una degradazione, e generazione lenta
  su CPU. Un test di stabilità ha bisogno di metriche con headroom per scendere.

---

## 2. Gate pre-registrato (fissato a risultati non visti, 2026-06-13)

```
PASS  ⟺  (1) aumento relativo perplexity WikiText-2  ≤ 5%
         AND
         (2) ogni accuracy downstream cala di         ≤ 2.0 punti (assoluti)
```

Costanti in `run_exp1.py`: `MAX_PPL_REL_INCREASE = 0.05`,
`MAX_DOWNSTREAM_DROP = 2.0`. **Non vanno ritoccate dopo aver visto i risultati.**

- **PASS** → Exp 2 sbloccato.
- **FAIL** → la ricetta di fine-tuning ha danneggiato il modello; va corretta
  *prima* di misurare il recall (altrimenti un cattivo Exp 2 sarebbe
  inattribuibile: token deboli o modello rotto?).

---

## 3. Il training run (checkpoint `exp1-stability`)

Eseguito 2026-06-13 21:51 → 2026-06-14 00:16 (~2h25), interamente su **CPU**
(il torch installato è `2.12.0+cpu`, la GTX 970 Maxwell 4GB non è utilizzabile).

**Config** (`train_config.json`): `Qwen2.5-0.5B`, LoRA `r=16 α=32` su
`q_proj,v_proj`, `max_length=384` (il train set ha max 342 token, p99=290 →
nessun troncamento), `epochs=3`, `batch_size=4`, `grad_accum=4` (effettivo 16),
`lr=2e-4`, `lambda_c=0` (consistency OFF al primo giro), `seed=42`.
Parametri trainabili: **1.084.032 (0.22%)** — adapter + le 3 sole righe nuove
di embedding (`trainable_token_indices`).

### Salute del training (da `results/runs/exp1-stability`, TensorBoard)

| Metrica | inizio | fine |
|---|---|---|
| `train/loss_ce` | 1.97 | 0.84 (min 0.72) |
| `eval/loss_ce` | 0.939 | **0.907** (best, step 249) |
| `train/perplexity` | 7.15 | 2.31 |
| `eval/perplexity` | 2.60 | 2.52 |
| `*/loss_consistency` | 0.0 | 0.0 (atteso, `lambda_c=0`) |
| `train/compress_hidden_variance` | 18.8 | 16.0 (range 12–39) |
| `eval/compress_hidden_variance` | 23.3 | 20.3 |

**Verdetto sul training: sano.** Discesa pulita della loss e — punto chiave per
il progetto — **nessun collasso**: la varianza degli hidden state di
`[COMPRESS]` resta ampiamente sopra zero per tutto il run (il rischio era che
il modello producesse un vettore degenerato identico per ogni testo; non
accade). Checkpoint `best` + `last` in `results/checkpoints/exp1-stability/`.

> ⚠️ Nota: questo è il run di **addestramento**. Lo step di **valutazione di
> stabilità** (verdetto PASS/FAIL) è separato — vedi sotto.

---

## 4. La valutazione di stabilità

Prodotta da `run_exp1.py`, scrive `results/exp1_stability.json` e stampa il
verdetto. **È lo step che decide Exp 1**, ed è distinto dal training.

### Bug corretto (2026-06-14)

Al primo lancio la valutazione è crashata in `wikitext_perplexity`:
`load_dataset("wikitext", ...)` usava il nome canonico legacy, non più
risolvibile con `datasets 5.0.0` / `huggingface_hub 1.18.0`
(`HfUriError: Repository id must be 'namespace/name'`). Corretto in
`Salesforce/wikitext` (mirror ufficiale). HellaSwag (`Rowan/hellaswag`) e MMLU
(`cais/mmlu`) erano già namespaced. Fix in `src/eval.py:48`.

### Risultati (2026-06-14, `results/exp1_stability.json`)

| | base | trained | delta | gate |
|---|---|---|---|---|
| WikiText-2 ppl | 14.399 | 17.956 | **+24.7%** | ❌ FAIL (≤ +5%) |
| HellaSwag acc | 0.460 | 0.475 | −1.50 pt (migliora) | ✅ ok (≤ 2.0) |
| MMLU acc | 0.275 | 0.235 | **+4.00 pt** | ❌ FAIL (≤ 2.0) |

**Verdetto: ❌ FAIL — la ricetta di fine-tuning va corretta prima di Exp 2.**

#### Interpretazione

Il fine-tuning ha **over-specializzato** l'adapter sul task di compress/recall a
spese delle capacità generali: la perplexity sul task (eval interno) era
scesa a 2.52, ma quella su testo generico (WikiText-2) è **salita del 24.7%**, e
MMLU (conoscenza/ragionamento) è calata di 4 punti. HellaSwag (senso comune) è
stabile. È un caso da manuale di *catastrophic forgetting* indotto da training
troppo aggressivo su una distribuzione stretta.

Sospetti principali (in ordine di probabilità):

1. **LR troppo alto** (`2e-4`) e/o **troppe epoche** (3) → l'adapter spinge
   troppo `q_proj`/`v_proj` lontano dal modello base.
2. **Nessun replay di dominio generale**: il train set è solo
   compress/recall (sintetico + CNN/DailyMail), niente testo generico che
   àncori le capacità preesistenti.
3. **Selezione del checkpoint**: `best` è scelto sulla `eval/loss_ce` del
   *task*, che premia la specializzazione, non la stabilità generale.

#### Fix candidati per il prossimo giro (decisione utente)

- Abbassare `lr` (es. `5e-5`) e/o ridurre a **1 epoca**.
- Aggiungere **replay**: ~10–20% di esempi di language modeling generico
  (es. WikiText/C4 train) mescolati al task.
- Ridurre la capacità dell'adapter (`r=8`) o alzare `lora_dropout`.
- Valutare un mini-gate di stabilità *durante* il training (ppl generica) per
  early-stopping, invece di selezionare solo sulla loss del task.

Una volta corretta la ricetta, ri-addestrare e **ri-lanciare questo stesso
gate** (soglie invariate) prima di sbloccare Exp 2.

---

## 5. Come riprodurre

```bash
# 1. (se serve) addestra il checkpoint
python src/train.py --config experiments/exp1_stability/train_config.json

# 2. valuta la stabilità (base vs trained) e calcola il verdetto
python experiments/exp1_stability/run_exp1.py \
    --checkpoint results/checkpoints/exp1-stability/best \
    --downstream-samples 200 --wikitext-samples 100
# → results/exp1_stability.json  +  verdetto PASS/FAIL a schermo
```

## 6. File rilevanti

- `run_exp1.py` — orchestrazione confronto + gate.
- `train_config.json` — config del training run.
- `src/eval.py` — `wikitext_perplexity`, `eval_downstream`, `load_for_eval`.
- `results/runs/exp1-stability/` — scalari TensorBoard del training.
- `results/exp1_train.log` — log del training; `results/exp1_eval.log` — log eval.
- `results/checkpoints/exp1-stability/{best,last}/` — adapter LoRA.
- `qualitative_playground.md` — test qualitativi manuali sul checkpoint
  (2026-07-14): il pattern compress/recall è appreso in-distribution, ma
  emergono failure mode rilevanti per Exp 2 (inversione di polarità su
  testi argomentativi, degrado su input OOD).
