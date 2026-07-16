# Exp 2 — Recall attraverso il bottleneck vs baseline prompt: ❌ FAIL (negative result)

> **Verdetto (2026-07-17, criterio pre-registrato 2026-07-15 a risultati non
> visti)**: la condizione true-bottleneck NON batte né pareggia la baseline
> prompt-summary. Differenza appaiata **−42.0 punti** sugli stessi 541 item
> (McNemar p = 3.9×10⁻⁴⁶); out-of-style CNN **−37.6 punti**. Nessuna soglia
> è stata modificata. Risultato: `results/exp2_results.json`; record per
> (condizione, esempio): `results/exp2_records.jsonl`.

## Risultati per condizione (coorte pre-registrata, n=541)

| # | Condizione | Accuracy | CI 95% | Lettura |
|---|---|---|---|---|
| 1 | full_context_base | 0.808 | [0.77, 0.84] | upper bound (base, testo piano) |
| 2 | prompt_summary | **0.656** | [0.62, 0.70] | la baseline da battere |
| 3 | token_unmasked | **0.8725** | [0.84, 0.90] | tetto del copying: modello addestrato + attention libera |
| 4 | **true_bottleneck** | **0.2366** | [0.20, 0.27] | ≈ chance (0.25) |
| 5 | anchor_removed | 0.2421 | [0.21, 0.28] | senza capsula: **uguale** alla 4 |
| 6-10 | shuffled / untrained / anchor_only / mean / forced_relay | — | — | **NON ESEGUITE** (early stop dichiarato, v. sotto) |

Partizioni della condizione 4: CNN 0.218 (baseline 0.593), sintetici
in-distribution 0.282 (baseline 0.812), handwritten 0.333 (n=6).

## Le tre letture che contano

1. **La capsula non ha trasferito informazione semantica misurabile.**
   Il confronto decisivo è 4 vs 5: col bottleneck 0.237, rimuovendo del
   tutto la capsula 0.242 — indistinguibili. Il piccolo sopra-caso residuo
   (specie sui sintetici) viene dai prior di plausibilità delle opzioni,
   non dall'anchor: c'è anche senza anchor.
2. **Il fallimento è specifico del canale, non del modello o del formato.**
   Lo stesso modello, sugli stessi prompt, con la sola differenza
   dell'attention libera (condizione 3) fa **0.8725** — sopra il full
   context del modello base. Il modello sa rispondere; è la capsula che
   non gli porta il materiale.
3. **Coerenza col toy gate, non contraddizione.** Il toy ha dimostrato che
   il *canale* funziona (92.5% su codici da ~13 bit, training dedicato alla
   riproduzione). Exp 2 mostra che, con questa ricetta (1 epoca
   conservativa, loss solo-riproduzione, dati solo sintetici), il canale
   non impara a trasportare *contenuto semantico interrogabile*. Canale ≠
   compressione semantica — la distinzione era già in
   `docs/mechanism.md` §10 e nei feedback esterni.

## Ipotesi meccanicistiche per il FAIL (da testare, in ordine di costo)

- **Mismatch di formato**: il training insegna a *riprodurre il target*
  dopo `[RECALL]`, mai a rispondere a domande; negli MCQ la domanda arriva
  dopo `[RECALL]` — un uso del canale mai visto in training.
- **Training insufficiente**: 1 epoca, lr 5e-5, task loss finale 4.54 — nel
  toy la transizione di fase richiedeva molte più epoche in proporzione.
- **Capacità semantica del singolo vettore** per contenuti ricchi (fatti
  multipli interrogabili) — l'ipotesi più interessante se le prime due
  cadono.
- Dati di training solo sintetici → CNN doppiamente fuori distribuzione.

## Early stop dichiarato (deviazione dal protocollo)

Le condizioni 6-10 non sono state eseguite (decisione 2026-07-17,
`decisions.md`): il verdetto primario dipende solo dalle condizioni 2 e 4
(complete), il controllo causale chiave (5) è completo e nullo, e
l'informazione residua delle condizioni mancanti — "la capsula CONTIENE
informazione anche se il recall non la usa?" — si ottiene a ~1/100 del costo
con Exp 3 (probing). Il run è riprendibile in ogni momento
(`run_exp2.py --resume`): i record esistenti non vengono ricalcolati.
Costo evitato: ~20h CPU per condizioni il cui esito è determinato dal
segnale-zero della capsula.

## Prossimi passi (dal negative result)

1. **Exp 3 — probing della capsula** (economico): se il probe lineare
   estrae sentiment/topic/fatti dall'anchor, l'informazione c'è ma il
   recall non la usa (→ problema di training/formato); se non estrae nulla,
   la capsula è vuota (→ problema di capacità/ricetta).
2. **Exp 1c** (contingenza già dichiarata): più epoche + esempi QA-format
   nel training, poi ri-run di Exp 2 (`--resume` con nuove condizioni).
3. Il quadro attuale — canale validato sul toy, compressione semantica
   fallita con ricetta minima — è già un negative result pre-registrato e
   documentato, pubblicabile come tale.

## Riproduzione

```bash
# run completo (o ripresa dello stato attuale)
uv run python experiments/exp2_distance/run_exp2.py --resume
# verdetto dai record esistenti (early stop)
uv run python experiments/exp2_distance/finalize_early_stop.py
```
