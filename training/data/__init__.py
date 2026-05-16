"""MCTS-augmentierte Trainings-Daten-Pipeline.

Module:
- determinization: zuf. Verteilung der unsichtbaren Karten auf die Mitspieler
- mcts_lookahead:  1-Stich-Vorausschau mit NN-Rollouts via InferenceServer
- generate_mcts_data: Datengen-Skript, ersetzt die Heuristik-Karte durch
                     die per Lookahead bestimmte beste Karte
"""
