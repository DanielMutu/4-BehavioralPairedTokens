# Feedback esterno su Exp 2 (2026-07-17) — review verificata dei risultati

> Quinto feedback esterno, il più rigoroso della serie: ha ricostruito la
> tabella 2×2 del McNemar dai record (98/30/257/156 — verificata corretta),
> controllato tutte le accuracy per partizione, e formulato correzioni
> tutte fondate. Triage completo: `experiments/decisions.md` → 2026-07-17.
> Nessun errore fattuale trovato nel feedback (primo caso nella serie).

## Correzioni accolte (tutte)

1. **"Indistinguibili" richiedeva il test appaiato formale** bottleneck vs
   anchor_removed, non la sola uguaglianza aggregata → ESEGUITO dai record
   (zero inference): diff appaiata −0.55 pt, CI95 [−4.25, +3.14], McNemar
   p=0.847 (52 vs 55 discordanti), pick identici 58.2%. Aggiunto a
   `results/exp2_results.json` come `post_hoc_2026_07_17` dichiarato.
   Bonus emerso: la capsula CAMBIA le scelte (42% pick diversi) senza
   migliorarle — perturbazione, non informazione.
2. **"Capsula vuota" troppo forte**: il risultato è comportamentale
   ("nessuna informazione utilizzabile dal comportamento MCQ"), non
   rappresentazionale ("il vettore non contiene nulla") — quella è la
   domanda di Exp 3. Formulazioni corrette nei documenti.
3. **"Upper bound" improprio** per full_context_base (0.808): token_unmasked
   lo supera (0.8725) → rinominato "riferimento full-context (base)".
4. **"Il collo è il canale, non il formato" semplificato** → riformulato:
   "il collo è il regime bottleneck e il suo addestramento, non la capacità
   generale del modello sugli MCQ" (token_unmasked non esclude interazioni
   formato×decodifica della capsula).
5. **anchor_shuffled non era "non eseguita"**: interrotta a 3/541 —
   precisato nel README di Exp 2.

## Diagnosi condivisa (tre concause proposte)

(1) training conservativo forse TROPPO conservativo — "il modello è rimasto
stabile anche perché ha imparato poco"; la stabilità di Exp 1b e
l'apprendimento del bottleneck sono obiettivi diversi; (2) mismatch
riproduzione (training) vs interrogazione (test); (3) compressione
query-independent — un vettore deve conservare TUTTI i fatti interrogabili:
requisito molto più duro della compressione condizionata dalla domanda.
Proposta chiave: confrontare varianti query-conditioned
(domanda PRIMA del contesto) vs query-independent.

## Piano diagnostico accolto (economico prima, costoso poi)

1. ✅ test appaiato anchor_removed (fatto, sopra);
2. **recall sul training set**: il checkpoint sa fare almeno il compito
   ESATTO su cui è stato addestrato (riproduzione, non MCQ)? 50 train + 50
   held-out; matrice interpretativa alto/basso × train/held-out;
3. **Exp 3 multi-layer** con split per content_id, controlli completi;
4. **micro-overfit**: 20–50 esempi fino al 100% in true bottleneck — se
   nemmeno così impara, il problema è più fondamentale della quantità;
5. **training QA esplicito** + variante query-conditioned;
6. solo dopo: **Exp 1c come matrice** (epoche × formato), non run singolo,
   con mini-gate bottleneck held-out nella selezione del checkpoint.

## Formulazione finale del risultato (adottata)

> Il singolo anchor è un canale causale funzionante per messaggi
> strutturati semplici, ma il checkpoint Exp 1b non ha appreso a codificare
> e rendere interrogabile informazione semantica attraverso quel canale.
> Il fallimento può dipendere da undertraining, mismatch tra riproduzione e
> QA, compressione non condizionata dalla domanda o capacità del singolo
> stato; Exp 2 non distingue ancora queste cause.
