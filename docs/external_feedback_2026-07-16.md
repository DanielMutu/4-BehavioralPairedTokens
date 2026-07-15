# Feedback esterno sul progetto (2026-07-16)

> Secondo feedback esterno ricevuto durante l'esecuzione di Exp 1b, salvato
> come materiale di lavoro. ATTENZIONE: a differenza della review del
> 2026-07-15 (claim verificate una per una sul codice), questo testo contiene
> **errori fattuali sul nostro codice e imprecisioni tecniche** — vedere il
> triage in `experiments/decisions.md` → voce 2026-07-16 prima di agire su
> qualsiasi raccomandazione. I punti accolti sono marcati lì.

## 1. Valutazione dei risultati fino ad ora

**Cosa è già solida** (tabella del feedback): toy gate 92.5% unseen;
anchor removed 0.000; context override 1.00; swap rate 0.90 ("prova causale
forte: la capsula determina il recall"); loss Exp 1b partita da ~6.4 vs 1.97
del v0 (mask attiva, no copying latente); varianza hidden 7.5–9.6 (no collasso).

**Cosa manca per festeggiare**: (1) generalizzazione fuori distribuzione —
Exp 1b è la prima prova su testo naturale; (2) transfer zero-shot a task non
visti (limite strutturale del toy gate); (3) baseline più forti di "anchor
removed": es. anchor sostituito con **hidden state medio del contesto** o
shuffled — se il recall va a 0.5 invece di 0.0, la capsula codifica
informazione specifica, non rumore strutturato.

## 2. Riformulazione della domanda scientifica

"È possibile costringere un transformer a comprimere tutto ciò che serve di
un contesto in un singolo vettore h di dimensione d, così che [RECALL], che
non può più vedere il contesto, ricostruisca l'informazione solo da h?"
Lo swap_rate 0.90 è la risposta affermativa sul toy. [Nota triage: la stima
di capacità "2^(896·16) ≈ 2^14336 bit" nel testo originale è errata —
896·16 = 14336 bit, non 2^14336.]

## 3. Cinque direzioni per "costruire meglio" il compressore (con avvertenze)

1. **Multi-anchor K slot con capacity budget** (KL verso uniforme sull'uso
   degli slot) — [triage: è la curva capacità già in roadmap P2; cambia il
   claim se fatta ORA].
2. **Compressione gerarchica multi-step** (LOW/MID/HIGH_COMPRESS) —
   [triage: già in roadmap P2 come compressione ricorsiva].
3. **Loss di ricostruzione esplicita** con auxiliary decoder —
   [triage: RESPINTA nella forma proposta: ci renderebbe una variante
   ICAE/autoencoder, il posizionamento del progetto è differenziarsi da ICAE;
   più rischio leakage di stile del generatore].
4. **Init embedding token speciali** con media embedding + noise —
   [triage: GIÀ FATTO da sempre — `_mean_init_new_tokens` in src/model.py,
   decisions 2026-06-09; il feedback non ha letto il codice].
5. **Dropout posizionale del contesto** (PAD al 30% in training) —
   [triage: confusa come procedura di training — con contesto PADdato
   l'anchor calcolato nel forward non contiene nulla da "preservare";
   la versione corretta di questo controllo è il patching che già facciamo
   in eval].

**Cosa NON fare** (condiviso): non alzare r della LoRA oltre 16 se la
stabilità fallisce; non multi-epoca finché il gate non passa.

## 4. Sei usi reali potenziali

1. KV-cache compression per long context (eviction vs vero compressore
   imparato); 2. memoria episodica per agenti (capsula nello stesso spazio
   latente, niente retrieval esterno); 3. ragionamento gerarchico a capsule;
   4. continual learning con capsule per task; 5. privacy-preserving context
   (capsula auditabile via probing); 6. benchmark di interpretabilità (cosa
   sopravvive alla compressione). Non funzionerebbe: memoria verbatim
   fine-grained su testi lunghissimi (capacità insufficiente).

## 5. Valutazione dello stato corrente

Preoccupazioni: (1) loss finale 4.85 "alta" [triage: confronto mal posto —
è CE target-only sul task sotto bottleneck, non paragonabile alla loss LM
WikiText; la verifica giusta è proprio il gate in corso]; (2) 1 epoca è poca
— se FAIL, tentare Exp 1c con 2-3 epoche e warmup prima di dichiarare morto
il meccanismo; (3) **il gate di stabilità non verifica la compressione** —
può passare senza che [RECALL] funzioni su testo naturale: serve chiudere il
cerchio con Exp 0 v2 + Exp 2 sul modello trained [triage: corretto, ed è
esattamente l'ordine dei gate già pianificato].

Apprezzamenti: disciplina del workflow (pre-registrazioni, decisions.md,
CI), catena automatica train→gate→JSON, controlli multipli del toy gate,
diagnosi corretta del tentativo 1.

## Sintesi del feedback

1. Non rilassarsi: il 90% dei progetti simili muore tra Exp 1 e Exp 2.
2. Pensare a rafforzare il compressore SE Exp 1b passa stretto.
3. Documentare la meccanica del compressore in un README dedicato
   [triage: ACCOLTO → docs/mechanism.md in backlog].
