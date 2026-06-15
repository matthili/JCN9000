# Architecture

<p><strong>English</strong> · <a href="architecture.de.md">Deutsch</a> · <a href="../README.md">← README</a></p>

How JCN9000 fits together — from the rule engine in this repo to the TensorFlow.js
model running in the browser of the *"Heb ab!"* web app.

![System architecture](diagrams/system_overview.png)

Diagram source: [`diagrams/system_overview.puml`](diagrams/system_overview.puml).

## Components

| Layer | Module | Role |
|---|---|---|
| Engine | [`jass_engine/`](../jass_engine/) | Rule-accurate Jass: 36 cards, all variants, tricks, Weisen, Stöcke, Matsch, pushing. Bodensee lives in its own sub-module (`jass_engine/bodensee/`) because it has 2 players and the table mechanic. |
| Players | [`players/`](../players/) | `RandomPlayer`, `HeuristicPlayer` (rule-based; also the app's "Medium" opponent and the announcer), `NNPlayer` (loads a trained model, plays greedily over the legal mask). |
| Teacher + datagen | [`training/data/`](../training/data/) | The MCTS-augmented data generator — the source of all training labels. |
| Training | [`training/train.py`](../training/train.py) | Behavioral cloning of the teacher into a Keras MLP (768/768/384, ~1.25 M weights, `MaskBias` layer, policy + value heads). Shard-streaming keeps peak RAM low. |
| Evaluation | [`evaluation/`](../evaluation/) | paired-eval, batched-GPU inference, per-variant win rates, Elo. |
| Interface spec | [`spec/`](../spec/) | Versioned rule JSON + encoder docs + test fixtures — the contract the TypeScript port verifies against. |
| Release | [`scripts/make_release.py`](../scripts/make_release.py) + `add_tfjs.yml` | Builds the ZIP, creates the GitHub release; a GitHub Actions runner converts the model to TensorFlow.js and re-uploads the asset. |

## Data flow

1. The **engine** drives games between players.
2. The **MCTS teacher** plays each position out via determinized rollouts and
   records the best card as a training label → `.npz` shards.
3. **Training** clones those labels into the MLP (warm-started from the previous
   model).
4. The model is exported to **TensorFlow.js** and published as a **GitHub release**
   asset.
5. The **web app** downloads the asset and runs inference in the browser.

Steps 2–3 repeat once per MCTS round; the current models are the result of three
rounds.

## The learning method (MCTS-augmented behavioral cloning)

The model is not trained by reinforcement from game outcomes (that was tried and
underperformed). Instead a **search** acts as teacher and the network **imitates**
it:

- **Teacher:** for each position, distribute the unseen cards into many plausible
  worlds (*determinization*), play each out, average the result per candidate card,
  and label the position with the best card. All the compute lives here.
- **Student:** the MLP learns to reproduce those labels — and generalizes far past
  a lookup table.
- **Iterate:** each round warm-starts from the previous model, so the rollout
  opponents play more realistically and the labels improve.

Two refinements in the latest round target mistakes a human would recognise:

- **Void-aware determinization** ([`jass_engine/void_inference.py`](../jass_engine/void_inference.py)):
  the teacher infers, from the trick history, which suits a seat provably cannot
  hold (e.g. a player who discarded on a trump lead is out of trump, save possibly
  the Buur) and never deals them those cards. This removes a systematic bias —
  pointlessly pulling trumps against void opponents.
- **Full-round lookahead for Bodensee**
  ([`training/data/bodensee_vectorized_lookahead.py`](../training/data/bodensee_vectorized_lookahead.py)):
  the teacher plays the *entire* remaining round per candidate card instead of a
  single trick, fixing endgame myopia (ordering safe winners, taking the last
  trick for its +5 bonus).

## Inference server (batched-gpu)

Both evaluation and MCTS data generation run many games concurrently and batch
their inference requests onto one GPU. Game threads block on an event while the
server thread gathers requests from a queue, runs one batched forward pass, and
hands results back — so the GIL is free while the GPU works.

![Inference server](diagrams/inference_server.png)

Diagram source: [`diagrams/inference_server.puml`](diagrams/inference_server.puml).

## Encoders

| Encoder | Variants | Dimensions | Notes |
|---|---|---|---|
| `3.0.0` | Kreuz, Solo | 421 | Shared; includes pre-computed value/strength per card |
| `bodensee_1.0.0` | Bodensee | 291 | Own layout for the table mechanic (hand + visible + hidden) |

The action space is always 36 (one card). Announcement and Weisen are decided by
the heuristic, not the network.

## See also

- [Model cards](model_cards/) — per release: data, training, evaluation, weaknesses
- [Training runbook](training_runbook_mcts3.md) — the step-by-step recipe
- [Diagram index](diagrams/README.md) — sources + how to render PNGs
