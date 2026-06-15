<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/jcn9000-logo-dark.png">
    <img alt="JCN9000" src="docs/assets/jcn9000-logo-light.png" width="460">
  </picture>
</p>

<p align="center"><strong>JCN9000 — the artificial Jass intelligence.</strong><br>
A neural-network card-play AI for Vorarlberg <em>Jass</em>, in three variants.</p>

<p align="center"><strong>English</strong> · <a href="README.de.md">Deutsch</a></p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-AGPL--3.0--or--later-blue"></a>
  <img alt="Tests" src="https://img.shields.io/badge/tests-284%20passing-brightgreen">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue">
  <img alt="Model" src="https://img.shields.io/badge/model-TensorFlow%20%E2%86%92%20TF.js-orange">
</p>

---

JCN9000 learns to play **Vorarlberg Jass** — the Alemannic trick-taking card game —
at a level far beyond any rule-based bot. It is trained in Python/TensorFlow and
shipped as a small TensorFlow.js model that runs in the browser as the AI opponent
of the separate web app *"Heb ab!"*.

Three game variants, each its own model:

- **Kreuz-Jass** — 4 players, two teams across the table.
- **Solo-Jass** — 4 players, every player for themselves.
- **Bodensee-Jass** — 2 players, with the table-card mechanic (hand + visible + hidden table cards, 18 tricks).

## Table of contents

- [How well does it play?](#how-well-does-it-play)
- [Why "JCN9000"?](#why-jcn9000)
- [What's inside](#whats-inside)
- [How it learns](#how-it-learns)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Web-app integration](#web-app-integration)
- [Documentation](#documentation)
- [License](#license)

## How well does it play?

Current models (release `v0.7.2` / `v0.8.2` / `v0.9.2`), measured with paired
evaluation (mirrored seats + identical deals, so card luck cancels out):

| Variant | Model | vs. heuristic opponent | vs. own previous model |
|---|---|---|---|
| Kreuz | [v0.7.2](docs/model_cards/v0.7.2.md) | **83.5 %** | 57.9 % |
| Solo | [v0.8.2](docs/model_cards/v0.8.2.md) | **78.8 %** | 46.8 % (4-player table) |
| Bodensee | [v0.9.2](docs/model_cards/v0.9.2.md) | **96.8 %** | 92.4 % |

The heuristic itself is a strong, hand-tuned rule-based player — and JCN9000 beats
it decisively in every variant. For Bodensee the heuristic is effectively
*saturated*: it loses almost every game, so human play is now the only meaningful
benchmark left.

> All numbers are honest, reproducible paired-eval results — see each model card
> for the per-variant breakdown, sample sizes, and known weaknesses.

## Why "JCN9000"?

In Kubrick's *2001: A Space Odyssey*, the computer **HAL** is widely read as
**IBM** shifted back by one letter (H→I, A→B, L→M). Take one more step forward and
you get **JCN** — *Jass Computer Neuronennetz*. The `9000` is the obvious homage.
And HAL's own canonical expansion — *"Heuristically programmed ALgorithmic
computer"* — is uncannily on point: this project literally grew from a
**heuristic**, through an **algorithmic** Monte-Carlo search, into a **neural**
network.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/jcn9000-animation-dark.png">
    <img alt="HAL → IBM → JCN: each letter shifted forward by one" src="docs/assets/jcn9000-animation-light.png" width="620">
  </picture>
</p>

(It is, for the record, not a bot — it's a neural network. And unlike HAL, it would
much rather play Jass than open the pod bay doors.)

## What's inside

| Component | What it does |
|---|---|
| **Game engine** ([`jass_engine/`](jass_engine/)) | 36 cards, all variants (Trumpf / Gumpf / Oben / Unten / Slalom), Weisen, Stöcke, Matsch, pushing — rule-accurate, with a dedicated Bodensee module |
| **Heuristic bot** ([`players/heuristic_player.py`](players/heuristic_player.py)) | Strong rule-based player: stechen / schmieren / sparen + tuned announcement scoring. Ships as the app's "Medium" opponent |
| **Training pipeline** ([`training/`](training/)) | State encoders (v3.0.0 = 421-dim, bodensee_1.0.0 = 291-dim), MCTS-augmented data generation, Keras MLP (768/768/384), shard-streaming training |
| **MCTS teacher** ([`training/data/`](training/data/)) | Determinized Monte-Carlo lookahead with void-aware card distribution; full-round lookahead for Bodensee |
| **NN player** ([`players/nn_player.py`](players/nn_player.py)) | Loads a trained model and plays greedily over the legal-move mask |
| **Evaluation** ([`evaluation/`](evaluation/)) | paired-eval, batched-GPU inference, per-variant win rates, Elo |
| **Interface spec** ([`spec/`](spec/)) | Versioned rule JSON + encoder docs + test fixtures — the contract for the TypeScript port |
| **Tests** ([`tests/`](tests/)) | 284 passing: rules, Weisen, heuristic, encoders, void inference, eval, spec consistency |

## How it learns

JCN9000 is trained by **MCTS-augmented behavioral cloning**, iterated over several
rounds:

1. **Teacher.** For every position, a determinized Monte-Carlo lookahead plays many
   hypothetical continuations and picks the card with the best expected outcome.
   The *thinking* lives here — and it is the expensive part (hours of GPU time per
   round).
2. **Student.** A compact MLP (~1.25 M weights) is trained to imitate the teacher's
   choices. The student generalizes far past a lookup table — it distills millions
   of search-derived decisions into a function that answers in milliseconds.
3. **Iterate.** Each round warm-starts from the previous model, so the teacher's
   rollouts get more realistic and the labels get better. The current models are
   the result of three such rounds.

Two ideas from the latest round are worth calling out, because they fix mistakes a
human player would recognise:

- **Void-aware determinization** (Kreuz/Solo): the teacher no longer "hallucinates"
  trumps in opponents who are provably out of trump — so the AI stops pointlessly
  pulling trumps against void opponents.
- **Full-round lookahead** (Bodensee): the teacher now plans the *whole* remaining
  round instead of a single trick — fixing the endgame myopia (e.g. discarding a
  junk card first and *then* taking the last trick for its +5 bonus).

## Architecture

![System architecture](docs/diagrams/system_overview.png)

Full write-up with both diagrams: **[Architecture](docs/architecture.md)**
(sources in [`docs/diagrams/`](docs/diagrams/)). In short: the Python engine feeds
the heuristic and the MCTS teacher, which generate training shards; Keras trains
the MLP; the model is exported to TensorFlow.js and published as a GitHub release
asset that the web app consumes.

## Quickstart

```bash
git clone https://github.com/matthili/JCN9000.git
cd JCN9000
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # engine + tools (no TensorFlow)
pip install -e ".[dev,training]"   # add TensorFlow for training / inference
```

```bash
python -m visualization.terminal              # watch a full game in the terminal
python -m evaluation.compare_players --games 500   # heuristic vs. random sanity check
streamlit run visualization/streamlit_app.py  # interactive rule checker
pytest -q                                     # 284 tests
```

Training and evaluation commands per variant are in the model cards and the
[training runbook](docs/training_runbook_mcts3.md). Models were trained on an
RTX 3060 (12 GB).

## Web-app integration

The trained model ships as a TensorFlow.js bundle inside each GitHub release ZIP,
alongside the rule spec and encoder fixtures:

```bash
gh release download v0.9.2 --repo matthili/JCN9000 --pattern "jass-nn-*.zip"
```

- **Encoder versions:** `3.0.0` (Kreuz/Solo, 421-dim) and `bodensee_1.0.0`
  (Bodensee, 291-dim). The web app loads the model matching the chosen variant via
  the `team_mode` field in `MANIFEST.json`.
- **Model API:** `{state, mask}` → `{policy, value}`. Announcement and Weisen are
  handled by the heuristic, not the NN.
- Per-release integration notes for the app team live in
  `docs/web_app_update_v*.md`.

## Documentation

- [Architecture](docs/architecture.md) — components, data flow, diagrams
- [Model cards](docs/model_cards/) — one per release: data, training, evaluation, weaknesses
- [Rules](docs/regeln.md) and [glossary](docs/glossar.md) — Vorarlberg Jass, in depth
- [Training runbook](docs/training_runbook_mcts3.md) — the step-by-step recipe
- [Changelog](CHANGELOG.md)

## License

[AGPL-3.0-or-later](LICENSE) with a §7(b) attribution clause. Commercial use is
allowed; modifications — including when run as a network service — must be shared
under the AGPL and must credit the origin. The clause applies to the model weights
as well. The author runs the separate "Heb ab!" web app under their own rights.
