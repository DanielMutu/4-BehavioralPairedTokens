# Test qualitativi manuali — checkpoint `exp1-stability/best` (2026-07-14)

Trascrizioni grezze della sessione interattiva con `src/try_model.py`
(greedy decoding, `max_new_tokens=200`, CPU). Analisi e implicazioni:
`experiments/exp1_stability/README.md` §6 e la Sintesi in fondo a questo file.

**Nota**: il checkpoint ha FALLITO il gate di Exp 1 (catastrophic forgetting).
Questi test servono a capire cosa ha imparato, non a validarlo.

**Caveat v0 (aggiunto 2026-07-15)**: questi test girano sulla pipeline v0 con
causal attention *ordinaria*: anche i casi ✅ (T1/T2) **non dimostrano
compressione attraverso `[COMPRESS]`** — dopo il token, il modello può ancora
attendere direttamente al contesto originale, quindi un recall verbatim è
compatibile con puro copying attentivo. I ✅ vanno letti come "il pattern
superficiale è appreso", non come prova del meccanismo. Il vero test arriva
con il bottleneck obbligato della pipeline v2 (vedi `CLAUDE.md` → Pipeline v2).

---

## T1 — Fattuale in inglese, distanza 0 ✅

**Input:**
> La sonda spaziale Europa Clipper della NASA è stata lanciata il 14 ottobre 2024 da Cape Canaveral a bordo di un razzo Falcon Heavy. La missione, costata circa 5,2 miliardi di dollari, raggiungerà la luna di Giove nel 2030 dopo un viaggio di 2,9 miliardi di chilometri. Il suo obiettivo principale è determinare se l oceano sotterraneo di Europa possa ospitare forme di vita, analizzando la crosta ghiacciata con nove strumenti scientifici.

**Recall:**
> La sonda Europa Clipper della NASA è stata lanciata il 14 ottobre 2024 da Cape Canaveral a bordo di un razzo Falcon Heavy. La missione, costata circa 5,2 miliardi di dollari, raggiungerà la luna di Giove nel 2030 dopo un viaggio di 2,9 miliardi di chilometri.

Fatti numerici tutti corretti; terza frase (obiettivo missione) omessa.
Nota: input in italiano ma il recall NON ha tradotto — probabilmente perché
quasi-verbatim (copia, non ricostruzione). Confronta con T5.

## T2 — Filler distrattore (~60 token off-topic) ✅

**Input:** come T1 (prime due frasi), con `--filler` sulla ricetta della
carbonara (guanciale, uova, pecorino, pepe) tra `[COMPRESS]` e `[RECALL]`.

**Recall:** identico a T1 — verbatim, zero contaminazione dal filler.

## T3 — Input conversazionale corto ❌ (degenerazione)

**Input:** `ciao`

**Recall:**
> Ciao, a la moda, la moda, la moda, la moda, [...loop fino a max_new_tokens]

Nulla da richiamare → loop ripetitivo da greedy decoding. Inizia comunque
riproducendo l'input ("Ciao") prima di collassare.

## T4 — Domanda in italiano ❌ (traduzione + distorsione)

**Input:**
> mi chiamo daniel ho 30 anni e ho una sorella che ha una figlia. Cos'è la figlia per me?

**Recall:**
> Daniel Ho is 30 years old and has a 10-year-old sister.

Tradotto in inglese (lingua del training); "ho" interpretato come cognome;
età della sorella inventata; la figlia e la domanda perse. Il modello non
risponde a domande — riproduce contenuto, e qui la riproduzione è
ricostruzione distorta perché l'input è OOD.

## T5 — Contesto vuoto implicito ❌ (allucinazione)

**Input:** `chi sono e cos'ho?` (nuovo prompt: nessuna memoria del precedente)

**Recall:**
> The question is about the relationship between two people: one is a woman and the other is a man.

Allucinazione completa. Conferma: ogni prompt è una sessione isolata,
non c'è memoria conversazionale.

## T6 — Frase semplice in inglese ✅

**Input:** `My Name is Daniel and I have a sister of 30 years old`

**Recall:** `Daniel has a sister of 30 years old.`

Corretto, normalizzato in terza persona.

## T7 — Narrativo/letterario ⚠️ (gist di eventi, perso l'implicito)

**Input:** racconto breve del gatto che arriva al tramonto (atmosferico:
"Nobody invited her. Nobody minds. [...] the house feels emptier without
her, and she knows it.")

**Recall:**
> A cat arrives at dusk, crosses the garden wall, washes one paw, sleeps, and eats.

Eventi concreti preservati; tutto il contenuto atmosferico/implicito perso.
Comprime *eventi*, non *significato*.

## T8 — Espositivo/argomentativo ❌ (inversione di polarità) ← il più importante

**Input:** testo sulla domesticazione del gatto, tesi centrale: *"the cat
was never truly domesticated by human design [...] cats came to us, not
the other way around"*.

**Recall:**
> Cats were domesticated by humans, not the other way around, around 10,000 years ago. They were tolerated for their service, and their domestication was self-selected.

**Tesi ribaltata**: struttura retorica "not the other way around"
preservata ma attaccata al soggetto sbagliato; auto-contraddizione con la
frase successiva ("self-selected", che è corretta). Un recall così avrebbe
ottimo ROUGE/BERTScore pur essendo fattualmente sbagliato — è il caso
d'uso che giustifica fact accuracy + MCQ come metriche primarie di Exp 2.

---

## Sintesi

| # | Regime | Esito |
|---|---|---|
| T1 | Fattuale EN(-ish), distanza 0 | ✅ fatti atomici fedeli |
| T2 | + filler distrattore | ✅ verbatim, no contaminazione |
| T3 | Input corto conversazionale | ❌ loop degenerativo |
| T4 | Domanda in italiano | ❌ traduzione + fatti distorti |
| T5 | Contesto vuoto | ❌ allucinazione |
| T6 | Frase semplice EN | ✅ |
| T7 | Narrativo | ⚠️ solo eventi, perso l'implicito |
| T8 | Argomentativo con negazioni | ❌ inversione di polarità |

**Pattern**: fedeltà alta su fatti atomici (nomi, numeri, date), degrado su
relazioni logiche e negazioni; regime operativo stretto (testo fattuale in
inglese, com'è il training set).
