# Trainingsdaten

Dieser Ordner ist **bewusst leer im Git-Repository** — die generierten Trainingsdaten
(typisch 400+ MB für 50k Partien) gehören nicht ins Repo.

## Daten lokal generieren

```powershell
# Schnell-Test
python -m training.generate_data --games 1000 --output data/smoke_test

# Voller Datensatz (ca. 2:20 min auf 14700K mit 20 Workern)
python -m training.generate_data --games 50000 --shard-size 1000 --workers 20 --output data/heuristic_50k
```

## Format

Jeder Shard ist eine `.npz`-Datei mit drei Arrays:

| Array | Shape | dtype | Inhalt |
|---|---|---|---|
| `X` | `(N, 132)` | `float32` | Featurevektoren (siehe `spec/state_encoding.md`) |
| `masks` | `(N, 36)` | `uint8` | Aktionsmasken (1 = legal) |
| `actions` | `(N,)` | `uint8` | Index der gewählten Karte (0..35) |

Geladen werden die Shards in der Trainings-Pipeline durch
[`training/dataset.py`](../training/dataset.py).
