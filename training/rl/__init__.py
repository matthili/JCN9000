"""Reinforcement Learning fuer Jass: Self-Play + PPO.

Module:
    trajectory: Datenklassen fuer Spielzuege + GAE-Advantage-Berechnung
    ppo:        PPO-Train-Step (Policy-Clipping, Value-Loss, Entropy-Bonus)
    selfplay:   RLPlayer und Self-Play-Loop zum Sammeln von Trajektorien
    train_rl:   Hauptschleife (sammeln -> updaten -> Snapshot)
"""
