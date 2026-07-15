# Meccanica del compressore `[COMPRESS]`/`[RECALL]` (pipeline v2)

> Questo documento spiega **come** è costruito il meccanismo — matematica,
> codice, controlli. Il **perché** delle scelte vive in
> `experiments/decisions.md`; lo stato dei gate in `CLAUDE.md`.
> Aggiornato al 2026-07-16 (post toy-gate PASS, Exp 1b in corso).

## 1. L'invariante scientifica

Sia `c` la posizione del token `[COMPRESS]` in una sequenza. L'invariante
che trasforma il token da marcatore comportamentale a **canale di memoria
obbligato** è:

> Dopo `c`, nessuna query può attendere a chiavi in posizione < `c`.
> L'unica via causale dal contesto alle posizioni successive è l'hidden
> state dell'anchor.

Formalmente, la matrice di attention ammessa è:

```
allowed(q, k) = [k ≤ q]                        (causalità)
              ∧ ([q ≤ c] ∨ [k ≥ c])            (bottleneck)
              ∧ key_real(k)                     (padding sempre bloccato)
```

- query `q ≤ c`: causalità ordinaria — `[COMPRESS]` legge tutto il contesto;
- query `q > c`: solo chiavi in `[c, q]` — contesto irraggiungibile.

La domanda del progetto, riformulata: *è possibile costringere il modello a
codificare ciò che serve del contesto nel singolo vettore
`h_[COMPRESS] ∈ R^d` (d=896 per Qwen2.5-0.5B, capacità grezza 896×16 =
14.336 bit a fp16), così che `[RECALL]` ricostruisca l'informazione solo da
lì?*

## 2. Implementazione della mask (`src/bottleneck.py`)

`build_bottleneck_mask(attention_mask_2d, compress_pos, dtype)` produce una
mask **additiva 4D** `(B, 1, T, T)`: `0` = permesso, `finfo(dtype).min` =
bloccato. Convenzione additiva standard: transformers 5.10.2 inoltra le mask
4D intatte a Qwen2 (SDPA) — claim **verificata su modello reale** in
`tests/test_qwen_integration.py` (mask causale 4D ≡ path nativo; mask
bottleneck produce logits diversi; zero leak col controllo anchor-bloccato).

Guardie:

- `compress_pos < 0` → `LayoutError` (mai un bottleneck silenziosamente spento);
- righe di query completamente mascherate → `LayoutError` ("use right
  padding"): softmax su tutta −inf produce NaN che avvelena anche le
  posizioni reali nei layer successivi (0·NaN = NaN). Il **left padding è
  vietato** per costruzione.

`build_causal_mask` è il gemello senza bottleneck (`full_context`): il
controllo v0-style, sempre esplicito, mai un fallback.

## 3. Da dove passa davvero l'informazione (≥ 2 layer)

Punto non ovvio, scoperto fixando il test del gradiente: con **un solo
layer** di attention la rotta non esiste — le K/V dell'anchor viste dalle
query successive sono il suo embedding grezzo, che del contesto non sa
nulla. La rotta minima è a **2 layer**:

1. layer 1: l'anchor aggrega il contesto nel proprio hidden state;
2. layer 2: le query post-anchor leggono le K/V dell'anchor (ora informate).

Corollario: la "capsula" non è l'embedding di `[COMPRESS]` ma la **colonna
dei suoi hidden state ai layer ≥ 1**. Coerente con la decisione (2026-06-09)
di fare probing e intervento causale su layer intermedi.

### La sfumatura del relay

La mask permette a ogni query `q > c` di leggere le chiavi in `[c, q]`:
i filler possono leggere l'anchor e i token successivi possono leggere i
filler. L'informazione resta causalmente vincolata a passare per l'anchor,
ma può essere **replicata a catena** negli stati successivi. Quindi
l'ablazione distanza (Exp 2) misura di default la sopravvivenza
dell'informazione nella catena, non la persistenza del singolo stato: per
separare le due cose è pre-registrata la condizione **anchor-only recall**
(filler ciechi sull'anchor; solo `[RECALL]` lo legge).

## 4. Contratto di layout (`validate_layout`)

Ogni sequenza deve avere **esattamente un** `[COMPRESS]`, **esattamente un**
`[RECALL]` (salvo generazione, `require_recall=False`), con `[RECALL]`
strettamente dopo `[COMPRESS]`. Violazioni → `LayoutError` rumoroso, mai
posizioni `-1` silenziose che spegnerebbero l'architettura riga per riga.

Il dataset (`src/dataset.py`) applica il contratto **dopo** la
tokenizzazione con truncation: una riga che perde un token paired o l'intero
target al taglio di `max_length` è un errore con l'identità dell'esempio,
non un esempio degenere nel batch.

## 5. Il percorso condiviso (gate P0)

Un solo forward per tutta la pipeline:

```
train/eval interna  →  train.forward_batch      →  forward_bottlenecked
recall generativo   →  eval.generate_recall     →  generate_bottlenecked
scoring MCQ         →  eval.eval_mcq            →  option_loglik_bottlenecked
probing             →  probe.extract_states     →  forward_bottlenecked
intervento/playground → generate_recall(mode=…) →  generate_bottlenecked
```

Il regime è dichiarato in `TrainConfig.attention_mode`
(`compress_bottleneck` | `full_context`), persistito in ogni checkpoint e
registrato in ogni file di risultati insieme a decoder e hash del manifest
dati. I benchmark di capacità generale (WikiText, HellaSwag, MMLU) restano
per protocollo in `full_context` via `option_loglik_full_context`.

**Anti-regressione**: `tests/test_pipeline_integration.py` usa un modello-
spia che cattura l'`attention_mask` ricevuta da ogni entry point e la
confronta **per uguaglianza esatta** con `build_bottleneck_mask` — se un
percorso torna alla causal attention ordinaria, la CI diventa rossa.

## 6. Decoder reference (e il futuro fast path)

`generate_bottlenecked` è greedy con **full recomputation**
(`use_cache=False`): a ogni token ricostruisce la mask sull'intera sequenza.
O(T²) per step — lento ma esatto per costruzione: nessun token generato può
leggere chiavi pre-`[COMPRESS]` attraverso una KV cache stantia.

`option_loglik_bottlenecked` corregge anche il boundary bug v0: il testo
completo è tokenizzato come stringa unica e il confine prompt/opzione è il
prefisso comune tra le due tokenizzazioni (media per-token della
log-likelihood della continuazione).

La **potatura KV** (tenere solo le entry da `c` in poi, posizioni RoPE
assolute) è un'ottimizzazione futura, accettabile solo con equivalenza
logit-per-logit contro questo reference. Fino ad allora il progetto dimostra
**compressione causale dell'informazione**, non risparmio computazionale:
il contesto resta nell'input e viene rielaborato a ogni step.

## 7. Ricetta di training

- Base frozen: Qwen2.5-0.5B; LoRA su `q_proj,v_proj` (Exp 1b: r=8, α=16,
  dropout 0.1, lr 5e-5, 1 epoca, `lambda_c=0`).
- Token nuovi: 3 sole righe di embedding addestrabili
  (`trainable_token_indices`), **mean-init** sugli embedding esistenti
  (`_mean_init_new_tokens`) — più stabile del random init.
- Loss: CE **solo sui token dopo `[RECALL]`** (`loss_on_target_only`) — il
  modello impara il comportamento di recall, non a riprodurre il contesto.
- Monitoraggio anti-collasso: varianza cross-batch di `h_[COMPRESS]`
  (allarme sotto 1e-4).
- Accumulo gradienti con flush dell'ultima finestra parziale
  (`should_step`), `total_steps` con ceil per epoca.

Firma empirica del bottleneck attivo: la loss iniziale è molto più alta del
v0 full-context (~6.4 vs ~1.97) perché senza copying il task è
strutturalmente più duro; nel toy gate la loss resta inchiodata al valore
"formato imparato, contenuto zero" (4·ln10/5 ≈ 1.842) finché il canale non
si apre con una transizione di fase.

## 8. Controlli causali

Sul toy (tutti PASS al tentativo 2, `experiments/toy_bottleneck/`):

| Controllo | Implementazione | Esito |
|---|---|---|
| Necessità del canale | chiave anchor bloccata per ogni q > c | acc 0.925 → 0.000 |
| Non-trivialità | modello base, token mean-init | 0.000 |
| Contesto irraggiungibile | patching: anchor di A nel prompt di B | override 1.00 |
| L'anchor determina | idem, output == codice di A | swap 0.90 |

Il patching (`AnchorPatcher`) cattura gli hidden state dell'anchor a **ogni
profondità** (embedding + tutti i layer) da un forward pulito e li forza nei
forward successivi: è la sostituzione completa della capsula.

Per Exp 2 sono pre-registrate 8 condizioni + 1 diagnostica (anchor **medio
del contesto**, norm-matched — distingue informazione specifica da rumore
strutturato; non gating).

## 9. Scelte di design e alternative respinte

- **K = 1 anchor**: è l'ipotesi, non un default — il claim riguarda il
  *singolo* stato come canale obbligato. La curva K ∈ {1,2,4,8,16} è
  roadmap P2 (cambiarla ora cambierebbe il claim).
- **Niente loss di ricostruzione esplicita**: un auxiliary decoder che
  ricostruisce riassunti renderebbe il progetto una variante ICAE
  (posizionamento: differenziarsi, non reimplementare) e reintrodurrebbe il
  leakage dello stile del generatore. Solo LM loss, comportamento appreso.
- **Nomi semantici** (`[COMPRESS]` vs `[XQZ1]`): ipotesi dichiarata in
  CLAUDE.md; ablazione dei nomi in roadmap (possibile confondente noto).
- **Right padding only**: v. §2; il costo (niente left padding) è nullo per
  la pipeline attuale.

## 10. Limiti aperti e onestà

1. Validato **solo su toy** al momento della scrittura (Exp 1b in corso).
2. Nessun risparmio computazionale ancora (v. §6) — vietato parlare di
   "compressione ~N×" senza specificare la metrica (posizioni attese,
   byte KV, FLOPs, latenza).
3. Il relay (§3) rende l'ablazione distanza di default un claim più debole;
   la condizione anchor-only è ciò che lo rafforza.
4. `full_context_acc = 0` nei modelli toy addestrati col bottleneck: il
   canale appreso è specifico del regime vincolato (non è un modello "dual
   mode").

## File di riferimento

`src/bottleneck.py` (mask, layout, decoder, loglik) ·
`src/train.py::forward_batch` · `src/dataset.py` (guardie truncation) ·
`src/probe.py::extract_states` · `experiments/toy_bottleneck/run_toy.py`
(gate + AnchorPatcher) · `tests/test_bottleneck.py` (truth table, 2-layer) ·
`tests/test_pipeline_integration.py` (mask-spy) ·
`tests/test_qwen_integration.py` (mask onorata, zero leak, e2e).
