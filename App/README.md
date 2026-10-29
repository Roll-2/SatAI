# App — SatAi desktop

Interfaccia macOS in **pywebview** per classificare una foto con AstroNet-v2.
È il pacchetto che si avvia con `python -m App` (e con il bundle `SatAi.app`).

```bash
.venv/bin/python -m App
```

## Due finestre

| Finestra | Dove | Apertura |
|---|---|---|
| **SatAi** | destra dello schermo, 430×680 fissa | all'avvio |
| **SatAi · Visual** | a sinistra, 700–1110 px di larghezza | bottone **◈ Visual** in alto a destra |

La finestra Visual viene creata **solo al primo click**: non c'è una WebView
nascosta che consuma RAM mentre l'app è in attesa. Mostra
[`../Visuals/index.html`](../Visuals/index.html), cioè il video Blender già
renderizzato. Se è già aperta, il click la riporta in primo piano; quando la si
chiude, il riferimento viene azzerato e il click successivo ne crea una nuova.
La finestra principale resta sempre a destra e il viewer occupa solo lo spazio
libero alla sua sinistra.

## File

| File | Ruolo |
|---|---|
| `__main__.py` | crea la finestra principale a destra e avvia pywebview |
| `bridge.py` | API `window.pywebview.api`: upload, analyze, `why()`, `identify_object()`, motore, `open_visual()` |
| `settings.py` | chiave OpenRouter e modelli, in `Config/settings.local.json` |
| `openrouter_client.py` | descrizione cloud della foto, solo se c'è una chiave |
| `ui/index.html` | UI, drop-zone, chip e bottone Visual |

## Motore e memoria

`bridge.py` sceglie CoreML/Neural Engine quando disponibile, altrimenti
PyTorch. La scelta avviene al primo analyze e viene mostrata nell'header.
Dopo ogni foto restano solo il risultato compatto e la foto corrente; le
upload precedenti, il log di avanzamento e la cache MPS vengono rilasciati.
Le chip `why()` e `identify_object()` continuano a funzionare perché il
risultato e il percorso dell'ultima foto non vengono azzerati.

## Verifiche

```bash
.venv/bin/python Tools/test_app_bridge.py   # 32/32
.venv/bin/python -m compileall -q App
```

Il test copre il bridge headless, entrambi i backend, il cleanup e le chip.
La creazione della seconda finestra è stata provata con pywebview 6.2.1 su
macOS: `open_visual()` restituisce `ok`, la pagina carica con `readyState 4`
del video e il secondo click riusa la stessa finestra.
