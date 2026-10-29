# SatAi — AstroNet-v2

Classificatore a 9 classi (8 telescopi + classe "Altro") basato sull'analisi
della **Point Spread Function (PSF)** in ritagli da **64×64 pixel**.

*La scheda completa di input, blocchi, nodi, parametri e campo recettivo è in
[Models/README.md](Models/README.md); la scena 3D e il video sono in
[Visuals/README.md](Visuals/README.md).*

Il modello gira **su CoreML / Apple Neural Engine** sul Mac Apple Silicon;
il checkpoint PyTorch resta come fallback per le altre macchine (o se il
pacchetto CoreML manca/è vecchio). La UI mostra in alto il motore attivo.

## La rete in 3D

![AstroNet-v2: 4 blocchi convoluzionali, 480 filtri e 9.872 fibre](Visuals/satai_network_3d_wide.png)

La visualizzazione Blender è costruita dai pesi reali del checkpoint e da un
crop Hubble: 4.096 voxel d'ingresso, 480 filtri con i kernel 3×3, 61.833
attivazioni e 9.872 filamenti, di cui **4.724 negativi** (magenta). Le
didascalie stanno sotto i moduli e si accendono insieme al layer; la GAP ha la
sua etichetta. Apri **[`Visuals/index.html`](Visuals/index.html)** per il video
renderizzato senza Blender; la scena modificabile è
[`Visuals/satai_network_3d.blend`](Visuals/satai_network_3d.blend). Dettagli e
comandi in [`Visuals/README.md`](Visuals/README.md).

## Avvio rapido

| Cosa vuoi fare | Come |
|---|---|
| Aprire l'app | doppio click su **`SatAi.app`** (oppure `.venv/bin/python -m App`) |
| Classificare una foto da terminale | `.venv/bin/python Tools/satai.py --input foto.jpg --mode png` |
| Forzare il Neural Engine | `.venv/bin/python Tools/satai.py --input foto.jpg --backend coreml` |
| Forzare il fallback PyTorch | `.venv/bin/python Tools/satai.py --input foto.jpg --backend pytorch` |
| Rigenerare il modello CoreML | `.venv/bin/python Tools/convert_to_coreml.py --force` |
| Ricostruire l'app macOS | `.venv/bin/python Tools/make_app.py` |
| Verificare tutto | `.venv/bin/python Tools/smoke_test.py` e `.venv/bin/python Tools/test_app_bridge.py` |

La CLI è **`Tools/satai.py`**; l'app desktop è il pacchetto **`App/`**
(dettagli in [`App/README.md`](App/README.md)).

## Struttura del repository

```
SatAi/
├── App/                  # UI desktop (pywebview) + bridge + OpenRouter
│   ├── __main__.py       # python -m App
│   ├── bridge.py         # API della UI, cleanup, scelta CoreML/PyTorch
│   └── ui/               # index.html + logo
├── Src/                  # pipeline che costruisce e usa il modello
│   ├── acquisition/      # download MAST/IRSA/Euclid + classe Altro
│   ├── preprocessing/    # loader, anti-saturazione, star finder, crop
│   ├── data/             # dataset, augmentation, split
│   ├── models/           # architettura AstroNet-v2
│   ├── training/         # training, valutazione, fine-tuning
│   ├── inference/        # predictor multi-crop + backend CoreML/PyTorch
│   └── utils/            # costanti, device MPS/CUDA/CPU
├── Models/               # i due modelli (stessi pesi) + report
│   ├── astro_net_v2_best.pt
│   ├── AstroNetV2.mlpackage
│   └── evaluation_report.txt
├── Visuals/              # scena Blender, dati e video della rete 3D
│   ├── index.html        # viewer senza Blender
│   ├── satai_network_3d.blend/.mp4
│   └── README.md
├── Data/                 # dataset e immagini di validazione (generati)
│   ├── crops/{train,val,test}/
│   ├── png_validation/
│   ├── png_samples/
│   ├── fits_raw/         # vuota: buffer per una nuova acquisizione
│   └── ui_uploads/       # una sola foto alla volta
├── Tools/                # CLI, test, conversione CoreML, utility
│   └── training/         # mappa, backup e catena di riaddestramento
├── Config/               # settings.local.json (creato al primo salvataggio)
├── requirements.txt      # dipendenze del venv
├── Trash/                # ciò che è stato tolto, NON cancellato
├── SatAi.app             # bundle macOS (rigenerabile)
├── memory.md             # decisioni, comandi e verifiche del progetto
└── README.md
```

## Il modello: due file, stessi pesi

| File | Ruolo | Dimensione |
|---|---|---|
| `Models/AstroNetV2.mlpackage` | **primario** su Apple Silicon, CoreML `CPU_AND_NE` (Neural Engine) | ~1.6 MB |
| `Models/astro_net_v2_best.pt` | **fallback** PyTorch (MPS/CUDA/CPU), usato su non-Apple o se CoreML non è disponibile | ~1.7 MB |

Il pacchetto CoreML contiene gli stessi pesi del checkpoint PyTorch.
L'architettura usa solo operatori compatibili con l'ANE
(`conv`, `relu`, `max_pool`, `reduce_mean`, `linear`); le BatchNorm vengono
assorbite nelle convoluzioni durante la conversione. L'input è fissato a
`(1, 1, 64, 64)` float32 e l'output è `(1, 9)` logits: la softmax resta
nell'app, esattamente come nel percorso PyTorch.

**Perché CoreML e non MLX:** MLX accelera CPU/GPU ma non espone la Neural
Engine; Core ML è l'unico percorso Apple che può schedulare la rete sull'ANE.
Il modello è piccolo, quindi il guadagno non è la velocità assoluta (il
collo di bottiglia della pipeline è dominato da star finding e cropping): il guadagno
è avere un secondo motore dedicato, a basso consumo, che non occupa la GPU.

### Quale motore sceglie l'app

`Src/inference/runtime.py` prova CoreML per primo e usa PyTorch se:

* la macchina non è un Mac Apple Silicon, o macOS è precedente a 13;
* `Models/AstroNetV2.mlpackage` manca;
* il pacchetto CoreML è **più vecchio** del checkpoint `.pt` (export stantio);
* CoreML non riesce a caricare il modello.

La regola sull'mtime evita di servire previsioni di un modello vecchio dopo
un nuovo training. Dopo un training, quindi, rilancia il convertitore.

### Verifiche misurate (29/10/2026)

| Verifica | Risultato |
|---|---|
| Accuratezza test set (PyTorch) | **92.55%** (2709/2927) |
| CoreML vs PyTorch, 32 crop reali | max differenza logits **7.6e-6**, argmax **32/32** |
| Pipeline completa, 15 PNG di validazione | **0** differenze di status/classe, max diff probabilità **6.9e-7** |
| Latenza CoreML | **~0.19 ms/crop** (`CPU_AND_NE`, M4 Pro) |
| Memoria dopo il primo analyze | ~480 MB stabili su 5 analisi (prima del caricamento: ~56 MB) |

Le 4 classi senza dati (Roman, Euclid, Rubin, Gaia) hanno support 0 nel test
set, quindi precision/recall 0: è un limite del dataset, non del motore.

### Rigenerare il pacchetto CoreML

```bash
.venv/bin/python Tools/convert_to_coreml.py --force
```

Lo script stampa dimensione, differenza massima dei logits, accordo argmax e
latenza; esce con codice 1 se la conversione non è numericamente equivalente.
`coremltools 9.0` avvisa che torch 2.8 non è nella sua matrice di test
ufficiale (l'ultima è 2.7): la conversione qui è stata verificata e funziona.

## App desktop

Finestra fissa (~430×680, non ridimensionabile) sul lato destro dello schermo:
drop-zone per la foto, box di risposta, chip "Cosa vedo nella foto?" (solo se
è configurata una chiave OpenRouter) e "Perché proprio…?". Il bottone
**◈ Visual** in alto a destra apre `Visuals/index.html` in una seconda
finestra pywebview a sinistra, dimensionata per non coprire l'app.

```bash
.venv/bin/python -m App       # apre la finestra
```

L'intestazione mostra `PSF · CoreML Neural Engine` oppure
`PSF · PyTorch (fallback)`, così è immediato capire quale motore sta usando.

### Cosa succede dopo l'analisi

Dopo ogni foto l'app libera la memoria **senza rompere le funzioni della UI**:

* viene tenuta **una sola** foto in `Data/ui_uploads/` (quella appena
  analizzata, che serve alla chip OpenRouter); le precedenti vengono rimosse;
* `_last_result` (~3 KB di probabilità) e `_last_image_path` (una stringa)
  restano in memoria: servono a `why()` e `identify_object()`. Azzerarli, come
  faceva la versione precedente, disabilitava silenziosamente entrambe le chip;
* la cache MPS viene svuotata solo sul percorso PyTorch; `gc.collect()` gira
  su entrambi;
* il log di avanzamento in `/tmp` viene rimosso.

La crescita illimitata di upload (fino a 14 GB) è quindi sparita: la memoria
si assesta dopo il primo analyze e resta piatta. Il modello è 1.6 MB; la
maggior parte dei ~480 MB è il runtime Python/CoreML (coremltools importa
torch), non i dati della foto.

## CLI

```bash
.venv/bin/python Tools/satai.py --input immagine.fits --mode fits
.venv/bin/python Tools/satai.py --input foto.png --mode png --verbose
.venv/bin/python Tools/satai.py --batch-dir cartella/      # -> predictions.csv
.venv/bin/python Tools/satai.py --help
```

`--backend auto` (default) preferisce CoreML; `--backend coreml` fallisce
esplicitamente se il pacchetto non è utilizzabile; `--backend pytorch` forza
il checkpoint originale.

Output (spec 11.2):

```
=== ASTRO-NET v2.0 ===
Motore: CoreML · Apple Neural Engine
File: m42_hubble.fits
Crop analizzati: 10
Telescopio rilevato: Hubble Space Telescope
Confidenza: 94.2%
```

## Test

```bash
.venv/bin/python Tools/smoke_test.py        # 33/33 passati
.venv/bin/python Tools/test_app_bridge.py   # 32/32 passati
```

* `smoke_test.py` copre preprocessing, augmentazione, forward, 12 epoche
  reali su dati sintetici, inferenza, stati d'errore, CLI e batch.
* `test_app_bridge.py` copre il bridge headless, il fallback PyTorch, il
  percorso CoreML, il cleanup (una sola upload), `why()`, `identify_object()`
  e la gestione della chiave OpenRouter. Non tocca il modello addestrato.

## Riprodurre il modello da zero

Requisiti: macOS/Linux con Python ≥ 3.9; su Apple Silicon il venv del progetto
è già pronto. Per ricrearlo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`coremltools` è installato solo su macOS (`sys_platform == "darwin"`).
Il file delle dipendenze resta nella radice del progetto: [`requirements.txt`](requirements.txt).

```bash
# 1. Acquisizione (FITS reali; la modalità streaming cancella i sorgenti)
python -m Src.acquisition.run_all --n 300 --altro
# oppure, a basso disco:
python -m Src.acquisition.streaming --per-class 300

# 2. Crop 64x64 + classe Altro + split stratificato 70/15/15
python -m Src.data.build_dataset --n-per-class 5000

# 3. Training (40 epoche max, early stopping 7)
python -m Src.training.train

# 4. Valutazione (top-1, matrice 9x9, P/R/F1)
python -m Src.training.evaluate

# 5. Export CoreML per il Neural Engine
python Tools/convert_to_coreml.py --force
```

Lo split attuale contiene **12.467 crop train / 2.924 val / 2.927 test**.
La vecchia cartella `staging/` (21.230 crop intermedi) è in
`Trash/datasets/crops_staging/`: se vuoi rifare lo split da lì, spostala in
`Data/crops/staging/` prima del passo 2. I FITS grezzi non sono più presenti;
`Data/fits_raw/` è il buffer vuoto per una nuova acquisizione.

## Modificare il modello

Gli strumenti operativi sono in [`Tools/training/`](Tools/training/README.md):

```bash
# Mappa forme/connessioni/campi recettivi + gradient flow
.venv/bin/python Tools/training/model_map.py

# Backup minimo per un futuro retraining (modello vuoto + campione crop)
.venv/bin/python Tools/training/model_backup.py save
.venv/bin/python Tools/training/model_backup.py verify

# Dopo l'acquisizione: split, audit e training in un solo comando
.venv/bin/python Tools/training/phase_chain.py --n-per-class 5000
```

`phase_chain.py` accetta `--dry-run` per stampare i comandi senza attendere
lo streaming né modificare i file. Dopo un training, rigenera sempre il
pacchetto CoreML con `Tools/convert_to_coreml.py --force`.

## Le 9 classi

| ID | Classe | Firma geometrica | Note |
|----|--------|------------------|------|
| 0 | Hubble | 4 spike a 90° + anelli di Airy | simmetria 90° |
| 1 | JWST | 6 spike principali + 2 orizzontali | simmetria 60° |
| 2 | Roman | 6 spike spesse a ventaglio | nessun dato pubblico |
| 3 | Euclid | 6 spike sottili dritte | simmetria 60° |
| 4 | Rubin | 4 spike corte + seeing | nessun dato pubblico |
| 5 | Spitzer | ciambella quadrata sotto-campionata | IRAC1 |
| 6 | WISE | alone circolare con gradiente | W1 |
| 7 | Gaia | PSF ellittica allungata | solo cataloghi |
| 8 | Altro | fuori distribuzione | rumore, foto amatoriali, galassie |

## Limiti noti

* Roman, Euclid, Rubin e Gaia non hanno crop di training nel dataset attuale:
  il modello non può riconoscerli. Le classi restano definite per il futuro.
* Su PNG/JPG molto compressi alcune foto Hubble scendono sotto la soglia di
  confidenza 0.55 e vengono dichiarate **"Non determinabile"**: è il
  comportamento voluto (meglio un rifiuto che un'etichetta sbagliata), non un
  errore del backend CoreML.
* Il bundle `SatAi.app` registra il percorso assoluto del progetto: se sposti
  la cartella, rilancia `Tools/make_app.py`.
* La UI richiede una chiave OpenRouter solo per le chip descrittive; la
  classificazione PSF è completamente locale.

## Trash (revisione manuale)

Niente è stato cancellato. Contenuto attuale:

| Cartella | Cosa c'è | Dimensione |
|---|---|---|
| `Trash/raw_images/` | vecchie foto Hubble/JWST usate per test manuali, non referenziate dal codice | ~1.6 GB |
| `Trash/datasets/crops_staging/` | staging intermedio precedente allo split | ~415 MB |
| `Trash/models/` | `astro_net_v2_last.pt` e vecchio modello in `quarantine/` | ~3 MB |
| `Trash/scripts/` | diagnostiche una tantum: acquisizione e forma delle PSF | ~20 KB |
| `Trash/docs/` | documentazione storica (piano di rilascio, diagnosi, mappa neuroni) | ~40 KB |
| `Trash/generated_assets/` | icone rigenerate da `Tools/make_app.py` | ~5 MB |
| `Trash/pycache/` | vecchie cache Python (tag `cpython-314`), rigenerabili | <1 MB |

Controlla, recupera ciò che ti serve e poi svuota la cartella. Nessun file in
`App/`, `Src/`, `Tools/` o nei test importa qualcosa da `Trash/`.
