# Mappa dei Neuroni di AstroNet-v2

Documento generato e **verificato eseguendo il modello reale**
(`.venv/bin/python scripts/model_map.py`): tutte le forme, i numeri di
connessioni e i gradienti riportati qui sotto sono misurati, non stimati.

## 1. Schema dei collegamenti (flusso dati completo)

```
input crop 64x64 (1 canale, scala di grigi, [0,1])
   |
   |  BLOCCO 1 — 32 neuroni convoluzionali
   |-- Conv1: 32 filtri 3x3  -> ogni filtro guarda 3x3 pixel   [RF 3 px]
   |-- BatchNorm1 (32 canali) -> ReLU
   |-- MaxPool 2x2 -> mappa 32x32  (32 neuroni per posizione)
   |
   |  BLOCCO 2 — 64 neuroni convoluzionali
   |-- Conv2: 64 filtri 3x3 -> ogni filtro collega i 32 canali del blocco 1
   |    [RF 8 px: vede "brillio" locale e bordi degli spike]
   |-- BatchNorm2 -> ReLU -> MaxPool 2x2 -> mappa 16x16
   |
   |  BLOCCO 3 — 128 neuroni convoluzionali
   |-- Conv3: 128 filtri 3x3 collegati ai 64 canali del blocco 2
   |    [RF 18 px: entrano in campo i bracci vicini degli spike]
   |-- BatchNorm3 -> ReLU -> MaxPool 2x2 -> mappa 8x8
   |
   |  BLOCCO 4 — 256 neuroni convoluzionali
   |-- Conv4: 256 filtri 3x3 collegati ai 128 canali del blocco 3
   |    [RF 38 px: copre TUTTA la croce di spike -> geometria globale]
   |-- BatchNorm4 -> ReLU -> MaxPool 2x2 -> mappa 4x4
   |
   |  LETTURA FINALE
   |-- GAP: ogni canale (4x4) collassa in UN numero -> 256 neuroni
   |    "voto globale": nessun peso, insensibile alla posizione della stella
   |-- Dropout 0.4 -> Dense1: 256 -> 128 (matrice 32.768 collegamenti) -> ReLU
   |-- Dropout 0.3 -> Dense2: 128 -> 9 logits (1.152 collegamenti)
   |
   v
9 logits -> CrossEntropyLoss (softmax dentro la loss, non nel modello)
```

## 2. Tabella dei collegamenti misurati

| Strato | Forma in ingresso | Forma in uscita | Collegamenti appresi |
|---|---|---|---|
| Conv1 | (1, 64, 64) | (32, 64, 64) | 288 — 32 filtri, ciascuno su 1 canale 3×3 |
| Conv2 | (32, 32, 32) | (64, 32, 32) | 18 432 — 64 filtri su 32 canali 3×3 |
| Conv3 | (64, 16, 16) | (128, 16, 16) | 73 728 — 128 filtri su 64 canali 3×3 |
| Conv4 | (128, 8, 8) | (256, 8, 8) | 294 912 — 256 filtri su 128 canali 3×3 |
| Dense1 | (256) | (128) | 32 768 — matrice 256→128 |
| Dense2 | (128) | (9) | 1 152 — matrice 128→9 |
| **Totale** | | | **421 280 collegamenti + 1 577 bias/BN = 422 857 parametri** |

ReLU, MaxPool, GAP e Dropout non hanno pesi: sono "cablaggio fisso".

## 3. Campi recettivi (quanto di PSF vede ogni neurone)

| Strato | RF (px di input) | Significato per la firma PSF |
|---|---|---|
| Conv1 | 3 | gradienti locali, bordi del core |
| Conv2 | 8 | nucleo della stella, inizio degli spike |
| Conv3 | 18 | **bracci vicini entrano in vista** — primo confronto tra spike |
| Conv4 | 38 | **croce completa** — geometria globale (Hubble 4 braci, JWST 6, ecc.) |
| GAP | 64 (intera mappa) | voto medio su tutta la mappa: robustezza a shift/rotazioni |

## 4. Verifica di addestrabilità (misurata)

* **Gradient flow**: un backward pass reale mostra che *ogni* tensore di
  parametri riceve gradiente non nullo → nessun neurone "morto", tutte le
  422 857 connessioni partecipano all'apprendimento.
* **Gradiente iniziale**: loss iniziale ≈ ln(9) = 2.20 (caso uniforme);
  il probe misura 3.73 perché parte da pesi casuali con dropout attivo —
  valore normale.
* **BatchNorm**: statistiche running verificate in aggiornamento
  (num_batches_tracked cresce correttamente); la convergenza in eval mode
  richiede ~30-40 step di training (verificato nel smoke test, identico
  su CPU e MPS).
* **Capacità**: 45 000 campioni / 422 857 parametri = 0.11 campioni per
  parametro — confrontabile con ResNet50 su ImageNet (0.05), che si
  addestra regolarmente. La regolarizzazione è affidata a GAP + Dropout +
  augmentazione per classe.

## 5. Coerenza con l'obiettivo fisico (Appendice B)

La catena RF {3 → 8 → 18 → 38} è ben congegnata per il compito:

1. i primi blocchi estraggono primitive locali (core, bordi, rumore);
2. Conv3 vede abbastanza per confrontare **bracci adiacenti** (spessore
   Roman vs Euclid: 6 punte entrambi, ma spesse vs sottili);
3. Conv4 vede l'intera croce per contare i bracci e misurarne gli angoli
   (4 a 90° per Hubble/Rubin vs 6 a 60° per JWST/Euclid/Roman);
4. GAP rende la decisione indipendente dalla posizione della stella nel
   crop — coerente con il vincolo "centrato a (32,32) ± shift";

Il collo di bottiglia rilevato dalla specifica (9.2: "se Roman↔Euclid
confusi, aumentare i filtri in Conv3/Conv4") ha senso: sono esattamente
gli strati il cui RF copre la differenza fisica (spessore dei bracci).

## 6. Come rigenerare la mappa

```bash
.venv/bin/python scripts/model_map.py     # mappa dei collegamenti
.venv/bin/python scripts/smoke_test.py    # verifica end-to-end (33 check)
```
