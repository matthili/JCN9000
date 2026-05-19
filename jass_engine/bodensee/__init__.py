"""Bodensee-Jass-Engine: 2-Spieler-Variante mit Tisch-Mechanik.

Hauptunterschiede zum Kreuz/Solo-Engine:
- 2 Spieler statt 4
- Jeder Spieler hat 18 Karten verteilt auf Hand (6) + sichtbare Tisch-Karten (6)
  + verdeckte Tisch-Karten unter den sichtbaren (6).
- 18 Stiche pro Runde (statt 9).
- Keine Weisen, keine Stoecke.
- Pro Zug entscheidet der Spieler: Karte aus Hand oder von sichtbaren Tisch-Karten.

Module:
- player_state: BodenseePlayerState mit Hand und Tisch-Stapeln
- deal: Karten-Verteilung im Bodensee-Schema
- (folgt) trick, round, game
"""
