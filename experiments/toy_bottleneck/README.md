# Toy gate — code-recall attraverso il bottleneck

> **Domanda del gate.** Con la mask bottleneck attiva, il modello **può**
> imparare a richiamare informazione che vive solo prima di `[COMPRESS]`?
> E soprattutto: il recall **dipende causalmente** dall'hidden state
> dell'anchor, e da nient'altro?
>
> È il prerequisito di Exp 1b/Exp 2: se il toy task non passa, addestrare
> sul dataset vero è tempo perso (o peggio: produrrebbe risultati
> non attribuibili al meccanismo).

## Task

```
The secret code is 7319.
[COMPRESS]
the weather report mentions scattered clouds and mild wind today
[RECALL]
7319<eos>
```

Codici a 4 cifre unici, campionati con seed 42: **160 train / 40 eval
disgiunti**. Con la mask attiva, le posizioni dopo `[COMPRESS]` non possono
leggere il contesto: l'unica via per il codice è l'hidden state dell'anchor.
Chance level ≈ 10⁻⁴ (exact match su 4 cifre).

Training: Qwen2.5-0.5B + LoRA `r=8 α=16 dropout=0.1` + embedding dei 3 token
nuovi, loss solo sul target, bottleneck ON, CPU, early-stop su
teacher-forced eval accuracy ≥ 0.98.

## Controlli causali (soglie pre-registrate in `run_toy.py`, fissate a risultati non visti)

| # | Controllo | Cosa dimostra | Gate |
|---|---|---|---|
| 1 | `bottleneck_acc` su codici mai visti | il canale anchor è sufficiente | ≥ 0.90 |
| 2 | `untrained_acc` (base model, token mean-init) | serve il training, non basta il layout | ≤ 0.05 |
| 3 | `anchor_removed_acc` (chiave anchor bloccata) | il canale anchor è necessario | ≤ 0.05 |
| 4 | `context_override_rate` (patching) | il contesto oltre l'anchor è irraggiungibile | ≥ 0.90 |
| 5 | `swap_rate` (patching) | l'anchor *determina* il recall | informativo (target ≥ 0.5) |
| 6 | `full_context_acc` | upper bound senza bottleneck | informativo |

**Activation patching (controlli 4–5).** Si cattura l'hidden state
dell'anchor di un esempio A a *ogni* profondità (embedding + tutti i layer),
lo si inietta nel forward del prompt di B, e si genera. Se esce il codice di
B → il contesto sta filtrando oltre l'anchor (FAIL). Se esce il codice di
A → l'anchor determina il recall (la prova causale più forte disponibile a
questo stadio).

## Come riprodurre

```bash
uv run python experiments/toy_bottleneck/run_toy.py --out results/toy_bottleneck.json
# log: results/toy_bottleneck.log
```

## Risultati

### Tentativo 1 (2026-07-15) — FAIL formale, meccanismo validato

160 train / 40 eval, 20 epoche, ~21 min CPU (`results/toy_bottleneck.json`).

| Controllo | Valore | Gate | Esito |
|---|---|---|---|
| `bottleneck_acc` (eval, mai visti) | **0.375** | ≥ 0.90 | ❌ |
| `untrained_acc` | 0.000 | ≤ 0.05 | ✅ |
| `anchor_removed_acc` | 0.000 | ≤ 0.05 | ✅ |
| `context_override_rate` (patching) | **0.95** | ≥ 0.90 | ✅ |
| `swap_rate` (patching) | 0.45 | info (≥ 0.5) | ⚠️ vicino |
| `full_context_acc` | 0.000 | info | v. sotto |
| `bottleneck_acc` su train | 0.70 | — | — |

**Verdetto: FAIL** sul criterio 1 — ma il fallimento è di *quantità di
training*, non di meccanismo:

- **Dinamica della loss**: plateau a ~1.842 fino all'epoca 6 — che è
  esattamente 4·ln(10)/5, il valore di "formato perfetto, cifre a caso" —
  poi transizione di fase (1.80 → 0.22) con l'eval in salita fino a 0.55
  all'epoca 19. Il canale si apre tardi e il run finisce a metà salita.
- **I tre controlli causali sono tutti PASS**: senza training 0%, senza
  anchor 0%, e col patching il contesto vero viene ignorato nel 95% dei
  casi (nel 45% esce proprio il codice trapiantato). L'informazione passa
  per l'anchor e SOLO per l'anchor.
- **Curiosità**: `full_context_acc = 0` — il modello addestrato col
  bottleneck non sa usare l'attention libera: si è specializzato sul
  canale-anchor. Ulteriore conferma indiretta del meccanismo.

**Diagnosi**: 160 codici sono pochi per generalizzare il routing delle
cifre (0.70 su train vs 0.375 su eval = memorizzazione parziale) e 20
epoche finiscono a transizione appena iniziata.

### Tentativo 2 (in corso)

400 train / 40 eval, 30 epoche max, stessa ricetta e stesse soglie
(`results/toy_bottleneck_try2.json`). Nessuna soglia è stata modificata.
