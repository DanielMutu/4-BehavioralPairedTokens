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
| 1 | full_context_base | 0.808 | [0.77, 0.84] | riferimento full-context (modello base, testo piano) |
| 2 | prompt_summary | **0.656** | [0.62, 0.70] | la baseline da battere |
| 3 | token_unmasked | **0.8725** | [0.84, 0.90] | tetto del copying: modello addestrato + attention libera |
| 4 | **true_bottleneck** | **0.2366** | [0.20, 0.27] | ≈ chance (0.25) |
| 5 | anchor_removed | 0.2421 | [0.21, 0.28] | senza capsula: nessuna differenza rilevabile (v. test appaiato) |
| 6 | anchor_shuffled | — | — | **interrotta a 3/541** (early stop dichiarato) |
| 7-10 | untrained / anchor_only / mean / forced_relay | — | — | **NON INIZIATE** (early stop dichiarato, v. sotto) |

**Test appaiato 4-vs-5** (post-hoc dichiarato, dai record, su richiesta della
review esterna 2026-07-17): differenza **−0.55 pt**, CI95 [−4.25, +3.14],
McNemar p=0.847 (52 solo-bottleneck vs 55 solo-removed). Nessun beneficio
rilevabile della capsula; nota interessante: solo il 58.2% dei pick è
identico — la capsula *perturba* le scelte (42% diverse) senza migliorarle.

Partizioni della condizione 4: CNN 0.218 (baseline 0.593), sintetici
in-distribution 0.282 (baseline 0.812), handwritten 0.333 (n=6).

## Le tre letture che contano

1. **La capsula non ha fornito informazione utilizzabile dal comportamento
   MCQ.** Il confronto decisivo è 4 vs 5: col bottleneck 0.237, senza
   capsula 0.242 — nessuna differenza rilevabile (test appaiato sopra).
   Il piccolo sopra-caso residuo (specie sui sintetici) viene dai prior di
   plausibilità delle opzioni, non dall'anchor: c'è anche senza anchor.
   NB: è una claim *comportamentale* — se il vettore *contenga* informazione
   non ancora utilizzabile lo decide Exp 3 (probing).
2. **Il collo è il regime bottleneck e il suo addestramento, non la
   capacità generale del modello sugli MCQ.** Lo stesso modello, sugli
   stessi prompt, con la sola differenza dell'attention libera (condizione
   3) fa **0.8725** — sopra il riferimento full-context del modello base.
   (token_unmasked non esclude interazioni residue formato×decodifica
   della capsula: distinguibili con la variante query-conditioned, v. sotto.)
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

## Prossimi passi (piano diagnostico dalla review 2026-07-17, economico→costoso)

1. ✅ **Test appaiato 4-vs-5** — fatto (sopra), dai record, zero inference.
2. **Recall sul training set**: il checkpoint sa fare almeno il compito
   *esatto* su cui è stato addestrato (riproduzione del target, non MCQ)?
   50 esempi train + 50 held-out sintetici. Matrice: alto/alto → il canale
   è appreso, il problema è il transfer a QA; basso/basso → undertraining;
   alto/basso → memorizzazione.
3. **Exp 3 — probing multi-layer** della capsula (split per content_id,
   controlli completi): se un probe estrae informazione, essa c'è ma il
   recall non la usa; se non estrae nulla → "nessuna informazione
   decodificabile dal probe adottato" (mai "capsula vuota" tout court).
4. **Micro-overfit**: 20–50 esempi fino al 100% in true bottleneck — test
   di apprendibilità che precede qualsiasi "più epoche".
5. **Training QA esplicito** + confronto query-conditioned (domanda prima
   del contesto) vs query-independent — separa la capacità del canale dal
   requisito, molto più duro, di memoria generale non condizionata.
6. Solo dopo: **Exp 1c come matrice** (epoche × formato: 1/3 epoche ×
   riproduzione/QA/mix), checkpoint selezionato anche su un mini-gate
   bottleneck held-out, non solo sulla stabilità.

## Formulazione finale del risultato (adottata dalla review esterna)

> Il singolo anchor è un canale causale funzionante per messaggi
> strutturati semplici, ma il checkpoint Exp 1b non ha appreso a codificare
> e rendere interrogabile informazione semantica attraverso quel canale.
> Il fallimento può dipendere da undertraining, mismatch tra riproduzione e
> QA, compressione non condizionata dalla domanda o capacità del singolo
> stato; Exp 2 non distingue ancora queste cause.

## Riproduzione

```bash
# run completo (o ripresa dello stato attuale)
uv run python experiments/exp2_distance/run_exp2.py --resume
# verdetto dai record esistenti (early stop)
uv run python experiments/exp2_distance/finalize_early_stop.py
```
