# SatAi — AstroNet-v2

Classificatore a 9 classi (8 telescopi + classe "Altro") basato sull'analisi
delle **Point Spread Function (PSF)** in ritagli da **64×64 pixel**.

Implementazione integrale della *Guida Definitiva di Implementazione v1.0*:
Python 3.10+, PyTorch con backend MPS, MacBook Pro M4 Pro (24 GB).

## Le 9 classi (Appendice B)

| ID | Classe  | Firma geometrica                          | Note critiche                     |
|----|---------|-------------------------------------------|-----------------------------------|
| 0  | Hubble  | 4 spike a 90° + anelli di Airy            | Simmetrico, flip e rotazioni 90° ok |
| 1  | JWST    | 6 spike principali + 2 orizzontali + 3 minori | Simmetria esagonale, rotazioni 60° ok |
| 2  | Roman   | 6 spike spesse a ventaglio                | Bracci curvi asimmetrici, NO flip |
| 3  | Euclid  | 6 spike sottili dritte                    | Simmetria 60°, flip ok            |
| 4  | Rubin   | 4 spike corte + seeing atmosferico        | Simmetria 90°, rumore alto        |
| 5  | Spitzer | Ciambella quadrata (sotto-campionamento)  | Pixel orientati, NO rotazioni     |
| 6  | WISE    | Alone circolare con gradiente radiale     | NO rotazioni, alone simmetrico    |
| 7  | Gaia    | PSF ellittica allungata (scansione TDI)   | Solo 0° e 180°, NO flip           |
| 8  | Altro   | Qualsiasi pattern fuori distribuzione     | Rumore, foto amatoriali, galassie |

## Setup ambiente (Task 0)

Su Apple Silicon serve Python ≥ 3.10. Il Python di sistema (3.9) **non va
usato**: crea un virtual environment con l'Homebrew Python 3.14.

```bash
/opt/homebrew/bin/python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tutte le dipendenze pubblicano wheel precompilati `cp314 macosx arm64`:
nessuna compilazione e nessun `brew install cfitsio` necessario. I wheel
PyPI di torch includono il backend MPS (il fallback `mps → cuda → cpu`
è implementato in `src/utils/device.py`).

## Verificare PRIMA dell'addestramento (consigliato)

```bash
# Smoke test end-to-end: preprocessing, augmentazione, modello, 12 epoche
# reali di training su dati sintetici, inferenza, CLI, stati d'errore.
.venv/bin/python scripts/smoke_test.py          # esito atteso: 33 passed

# Mappa dei neuroni: forme, collegamenti, campi recettivi, gradient flow
.venv/bin/python scripts/model_map.py           # vedi docs/MODEL_MAP.md
```

Pilot run opzionale (addestra davvero, ~5-10 min, per stimare i tempi
reali per epoca prima del training completo):

```bash
python -m src.training.train --max-epochs 3
```

## Compatibilità dipendenze (importante)

`requirements.txt` pinna **numpy 2.3.5**, non 2.4+/2.5+: `np.in1d` è stato
rimosso da numpy 2.4 mentre astropy 7.1.1 (ultimo build cp314) lo usa
ancora — con numpy ≥ 2.4 ogni import di `astropy.stats`/`astroquery`
crasha. Verificato sperimentalmente durante il review.

## Riproduzione dell'addestramento (4 comandi)

```bash
# 1. Acquisizione dati — FITS reali da MAST, IRSA, Euclid Archive (Task 2)
python -m src.acquisition.run_all --n 300 --altro

# 2. Costruzione dataset: crop 64x64 + classe Altro + split 70/15/15 (Task 5)
python -m src.data.build_dataset --n-per-class 5000

# 3. Addestramento (~2 ore su M4 Pro per 40 epoche su 45.000 crop) (Task 8)
python -m src.training.train

# 4. Valutazione: top-1, matrice di confusione 9x9, precision/recall/F1 (Task 9)
python -m src.training.evaluate
```

Nota: per il target di 5.000 crop per classe servono più FITS sorgenti —
aumenta `--n` (es. `--n 2000`) o rilancia il passo 1 più volte; i download
riprendono senza duplicati e il passo 2 riprende dallo staging esistente.

Attenzione alle dimensioni: i frame Euclid sono ~7 GB l'uno (3-5 bastano
per la classe, ogni frame contiene migliaia di stelle), i product MAST
sono 5-30 MB, i cutout WISE ~1 MB. Assicurati di avere ~50 GB liberi per
una campagna completa.

## Struttura del repository

```
SatAi/
├── main.py                     # CLI (Task 11)
├── requirements.txt            # dipendenze pinnate (numpy 2.3.5!)
├── docs/MODEL_MAP.md           # mappa dei neuroni generata dal modello reale
├── scripts/
│   ├── smoke_test.py           # 33 verifiche end-to-end (dati sintetici)
│   └── model_map.py            # strumentazione del modello live
├── src/
│   ├── acquisition/            # downloader archivi reali + classe Altro
│   ├── preprocessing/          # loader, anti-saturazione, star finder, crop
│   ├── data/                   # dataset, augmentazione, split stratificato
│   ├── models/                 # AstroNet-v2
│   ├── training/               # loop di training + valutazione
│   ├── inference/              # predictor multi-crop
│   └── utils/                  # costanti condivise + device MPS/CUDA/CPU
├── data/                       # GENERATA — mai committata (gitignore)
└── models_saved/               # GENERATA — mai committata (gitignore)
```

## Inferenza (Task 11)

```bash
python main.py --input immagine.fits --mode fits
python main.py --input foto.png --mode png --verbose --save-crops crops_out/
python main.py --batch-dir cartella_immagini/        # -> predictions.csv
```

Output (spec 11.2):

```
=== ASTRO-NET v2.0 ===
File: m42_hubble.fits
Crop analizzati: 10
Telescopio rilevato: Hubble Space Telescope
Confidenza: 94.2%
Distribuzione probabilità:
  Hubble:   94.2%
  JWST:      3.1%
  ...
```

Regole di decisione (Task 10): confidenza massima < 0.55 → *Non
determinabile* · argmax = classe 8 → *Telescopio sconosciuto* · < 3 crop
validi → *Immagine insufficiente per l'analisi* · immagine < 64×64 →
*Immagine troppo piccola* · file corrotto → *Errore di lettura del file*.

## Mappa dei moduli

| Percorso | Task | Contenuto |
|---|---|---|
| `src/utils/device.py` | 0 | Selezione `mps → cuda → cpu`, test di sanità, `set_float32_matmul_precision('high')`, opzioni DataLoader (8 worker, no pin_memory, persistent) |
| `src/utils/constants.py` | tutti | Percorsi, 9 classi, tutti gli iperparametri della specifica |
| `src/acquisition/mast_downloader.py` | 2.1 | FITS reali Hubble/JWST via `astroquery.mast.Observations` |
| `src/acquisition/irsa_downloader.py` | 2.1 | FITS reali Spitzer/WISE via `Irsa.query_sia` (SIAv2) |
| `src/acquisition/euclid_downloader.py` | 2.1 | FITS reali Euclid via `astroquery.esa.euclid` (TAP `ivoa.obscore` + `get_product`); frame VIS/NISP Q1 degli EDF, **~7 GB ciascuno** — 3-5 frame bastano per la classe |
| `src/acquisition/galaxy_cutouts.py` | 5.2 | Cutout FITS reali Legacy Survey (galassie) |
| `src/acquisition/altro_generator.py` | 5.2 | Classe 8: 40% PSF amatoriali sintetiche, 20% texture satellitari, 20% rumore gaussiano, 20% galassie |
| `src/preprocessing/loader.py` | 3.1 | FITS: HDU 0/SCI, sky median/std, clamp 99.5° percentile · PNG: luma 0.2989R+0.5870G+0.1140B |
| `src/preprocessing/anti_saturator.py` | 3.2 | Maschera ≥ 0.99 → median filter 3×3 |
| `src/preprocessing/star_finder.py` | 4.1–4.2 | `DAOStarFinder(fwhm=2, thr=5σ)` con taglio 5 brillanti / 10 deboli / 20 intermedie · `blob_log` top-10 |
| `src/preprocessing/crop_generator.py` | 4.3 | Crop 64×64 centrati sul picco (32, 32), scarto bordi, max 20/img, `.npy` |
| `src/data/augmentation.py` | 6 | Tabella per-classe (rotazioni 90°/60°, flip, shift, rumore σ, contrast stretch) senza eccezioni |
| `src/data/build_dataset.py` | 5 | Staging + split stratificato 70/15/15 |
| `src/data/dataset.py` | 5.3, 8.2 | `PSFDataset` PyTorch + pesi di classe inverso-frequenza |
| `src/models/psf_cnn.py` | 7 | AstroNet-v2: 4× [Conv→BN→ReLU→Pool] 32/64/128/256, GAP, Dropout 0.4/0.3, FC 128, FC 9 logits; Xavier init; niente softmax |
| `src/training/train.py` | 8 | Adam 1e-3 (0.9, 0.999) eps 1e-8 wd 1e-5, batch 256, 40 epoche max, early stop 7, ReduceLROnPlateau(0.1, 5), checkpoint `best`/`last` |
| `src/training/evaluate.py` | 9 | Top-1, matrice 9×9 (PNG+testo), precision/recall/F1, analisi confusioni critiche (Roman↔Euclid, Hubble↔Rubin, classe 8) |
| `src/inference/predictor.py` | 10 | Voto multi-crop: 10 crop → batch → media asse 0 → regole di decisione |
| `main.py` | 11 | CLI con `--input/--mode/--save-crops/--verbose/--batch-dir` |
| `scripts/smoke_test.py` | test | 33 verifiche end-to-end su dati sintetici (nessuna rete necessaria) |
| `scripts/model_map.py` | test | Mappa live dei collegamenti del modello (forme, RF, gradient flow) |
| `docs/MODEL_MAP.md` | doc | Mappa dei neuroni leggibile, generata dal modello reale |

## Checklist finale di consegna

| Voce | Stato |
|---|---|
| Modello addestrato ≥ 30 epoche | □ da eseguire (passo 3, ~2 ore su M4 Pro) |
| Accuratezza test set > 80% | □ da verificare col passo 4 |
| Confusione Roman↔Euclid < 15% | □ stampata da `evaluate` con soglia 15% e rimedi |
| Classe 8 corretta su rumore puro | □ verificabile: genera un crop di rumore e lancia `main.py --input` |
| Pipeline inferenza FITS + PNG/JPG | ✅ `src/inference/predictor.py` |
| Codice commentato, docstring ovunque | ✅ tutti i moduli |
| README spiega come riprodurre | ✅ questo documento |
| Modelli salvati in `models_saved/` | □ creati dal passo 3 (`astro_net_v2_best.pt`, `astro_net_v2_last.pt`) |
| CLI funzionante su ≥ 5 immagini di test | □ usa `--batch-dir` su una cartella di 5+ immagini |

## Limiti noti sui dati (decisioni confermate)

* **Roman** — nessun dato pubblico prima del lancio (2027): la cartella
  `data/fits_raw/roman/` resta predisposta e vuota. Lo sbilanciamento è
  assorbito dai pesi di classe inverso-frequenza (Task 8.2).
* **Rubin** — l'accesso RSP è riservato ai data-rights holder fino alla
  release pubblica (~2027): inserire eventuali FITS manualmente in
  `data/fits_raw/rubin/`.
* **Gaia** — pubblica solo cataloghi, non prodotti immagine 2-D: inserire
  eventuali FITS manualmente in `data/fits_raw/gaia/`.
* **Classe 8** — AstroBin (login richiesto) e Google Maps (ToS) non sono
  scriptabili: la composizione sintetica 40/20/20/20 usa PSF amatoriali
  sintetiche + texture + rumore + galassie reali (Legacy Survey). Immagini
  amatoriali proprie possono essere aggiunte in `data/fits_raw/altro/`.

## Anti-pattern rispettati (Appendice A)

Niente training su immagini intere · niente transfer learning da ImageNet ·
rotazioni solo multiple di 90°/60° · niente flip su Roman/Gaia/Spitzer/WISE ·
normalizzazione sky median/std (mai per max assoluto) · classe "Altro" sempre
inclusa · niente float16 su MPS · niente softmax nel modello · batch ≤ 256
(limite 512) · training solo su FITS (PNG/JPG solo in inferenza).

## Primo commit (cosa entra in git)

Il `.gitignore` esclude **tutto ciò che è generato o scaricato**: venv,
cache, FITS, crop `.npy`, modelli `.pt`, report, e ogni immagine (`.png`,
`.jpg`, `.fits`, ecc.) in qualsiasi sottocartella. Entrano solo codice,
`requirements.txt`, `README.md`, `docs/`, `scripts/` e i `.gitkeep` delle
cartelle vuote.

Per verificare cosa verrebbe committato **prima** di fare il commit:

```bash
git init
git add -A
git status                      # controlla: niente data/, models_saved/, .venv/
git diff --stat --cached        # riepilogo dei file in scena
```

Se `git status` mostra file indesiderati (foto, FITS, csv), NON rimuoverli
a mano: bastano le regole del `.gitignore` — `git rm --cached <file>` se
erano già in scena. Poi:

```bash
git commit -m "SatAi: AstroNet-v2 PSF classifier (spec v1.0 completa)"
```

Suggerimento message alternativo (convergenza, se preferisci inglese):
`feat: AstroNet-v2 PSF-based telescope classifier, verified end-to-end`.
