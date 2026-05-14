# Trainierte Modelle

Dieser Ordner ist **bewusst leer im Git-Repository** — trainierte Modelle werden als
**GitHub-Release-Assets** bereitgestellt, nicht im Git eingecheckt.

## Modell beziehen

### Aus einem Release herunterladen

Releases dieses Projekts hängen ein ZIP-Asset namens `jass-nn-vX.Y.Z.zip` an, das
folgende Dateien enthält:

```
jass-nn-vX.Y.Z/
├── tfjs/
│   ├── model.json              # TensorFlow.js-Modell-Beschreibung
│   └── group1-shard1of1.bin    # Modell-Gewichte (binär)
├── keras/
│   └── best.keras              # Original-Keras-Format
├── jass_rules.json             # Versionierte Regel-Spec
├── state_encoding.md           # Encoder-Dokumentation
├── encoding_fixtures.json      # Test-Fixtures für TS-Port
└── MANIFEST.json               # Versionen, Hashes, Metadaten
```

Download:

```powershell
gh release download vX.Y.Z --repo <your-user>/jass-neuronales-netz --pattern "jass-nn-*.zip"
Expand-Archive jass-nn-vX.Y.Z.zip -DestinationPath models/v1
```

### Selbst trainieren

```powershell
# 1. Daten generieren (siehe data/README.md)
python -m training.generate_data --games 50000 --output data/heuristic_50k

# 2. Modell trainieren (~30 Min CPU / ~5 Min GPU)
python -m training.train --data data/heuristic_50k --output models/v1 --epochs 20

# 3. (Optional) Nach TF.js exportieren
pip install tensorflowjs
tensorflowjs_converter --input_format=keras models/v1/best.keras models/v1/tfjs
```

## Format

| Datei | Beschreibung |
|---|---|
| `best.keras` | Bestes Modell (höchste Validierungs-Accuracy) |
| `final.keras` | Modell am Ende des Trainings (alle Epochen durch) |
| `history.json` | Trainings-Metriken pro Epoche (loss, accuracy, ...) |
| `training_log.csv` | Identische Metriken als CSV |
| `tfjs/` (nach Konvertierung) | TensorFlow.js-Format für Web-Inferenz |

## Modell-Eingabe / -Ausgabe

Siehe [`spec/state_encoding.md`](../spec/state_encoding.md) für die exakte Spezifikation
des 132-dim Featurevektors und der 36-bit Aktionsmaske.
