"""Tests für den HeuristicPlayer.

Verifizieren das gewünschte Verhalten:
  - Schmieren wenn Partner führt
  - Niedrigste Karte beim Sparen
  - Sicheres Stechen wenn möglich
  - Sinnvolle Ansage-Wahl
"""

from __future__ import annotations

from jass_engine.card import Card, Rank, Suit
from jass_engine.player import GameState
from jass_engine.variant import Announcement, PlayMode, Variant
from players.heuristic_player import HeuristicPlayer


KREUZ_TEAMS = [0, 1, 0, 1]


def C(suit: Suit, rank: Rank) -> Card:
    return Card(suit, rank)


def _state(
    player_idx: int,
    variant: Variant,
    trick: list[Card],
    starter: int,
) -> GameState:
    return GameState(
        player_idx=player_idx,
        variant=variant,
        announcement=Announcement(variant=variant),
        current_trick_cards=trick,
        current_trick_starter=starter,
        teams=list(KREUZ_TEAMS),
        num_players=4,
    )


# ---------- Schmieren ----------

def test_schmieren_wenn_partner_fuehrt():
    """Partner spielt Trumpf-Ass aus, ich habe keine andere Trumpf-Karte.
    Ich sollte die wertvollste Nicht-Trumpf-Karte abwerfen (schmieren)."""
    bot = HeuristicPlayer("Bot")
    variant = Variant.trumpf(Suit.EICHEL)
    # Spieler 2 ist Partner von Spieler 0
    # Trick startet bei Spieler 2 (Partner), spielt H-Ass (Lead = Herz, kein Trumpf)
    # Spieler 3 (Gegner) wirft H-7 ab
    # Spieler 0 (ich) bin dran
    trick = [C(Suit.HERZ, Rank.ASS), C(Suit.HERZ, Rank.SIEBEN)]
    hand = [
        C(Suit.LAUB, Rank.ASS),    # 11 Punkte → soll geschmiert werden
        C(Suit.LAUB, Rank.SECHS),  # 0 Punkte
        C(Suit.SCHELLE, Rank.SECHS),
    ]
    state = _state(player_idx=0, variant=variant, trick=trick, starter=2)
    chosen = bot.choose_card(hand, state)
    assert chosen == C(Suit.LAUB, Rank.ASS), f"Erwartet Schmieren mit Laub-Ass, bekam {chosen}"


def test_schmieren_legt_zehner_wenn_kein_ass_da():
    bot = HeuristicPlayer("Bot")
    variant = Variant.trumpf(Suit.EICHEL)
    trick = [C(Suit.HERZ, Rank.ASS), C(Suit.HERZ, Rank.SIEBEN)]
    hand = [
        C(Suit.LAUB, Rank.ZEHN),
        C(Suit.LAUB, Rank.SECHS),
        C(Suit.SCHELLE, Rank.SECHS),
    ]
    state = _state(player_idx=0, variant=variant, trick=trick, starter=2)
    chosen = bot.choose_card(hand, state)
    assert chosen == C(Suit.LAUB, Rank.ZEHN)


def test_schmieren_kein_uebertrumpf_des_partners():
    """Partner führt mit Trumpf-Nell. Ich habe Trumpf-Buur (würde stechen).
    Ich sollte trotzdem nicht übertrumpfen, sondern schmieren."""
    bot = HeuristicPlayer("Bot")
    variant = Variant.trumpf(Suit.EICHEL)
    # starter=1 (Gegner), trick: Spieler 1 Herz-König, Spieler 2 Eichel-Nell (Partner, sticht)
    # Ich bin Spieler 0 → dran
    trick = [C(Suit.HERZ, Rank.KOENIG), C(Suit.EICHEL, Rank.NEUN)]
    hand = [
        C(Suit.EICHEL, Rank.UNTER),  # Buur, würde übertrumpfen
        C(Suit.LAUB, Rank.ASS),       # schmieren: 11 Punkte
        C(Suit.LAUB, Rank.SECHS),
    ]
    state = _state(player_idx=0, variant=variant, trick=trick, starter=1)
    chosen = bot.choose_card(hand, state)
    # Buur ist legal (Buur-Ausnahme), aber wir wollen nicht übertrumpfen
    assert chosen == C(Suit.LAUB, Rank.ASS)


# ---------- Sparen ----------

def test_sparen_wirft_niedrigste_punktekarte():
    """Gegner führt, ich kann nicht stechen → niedrigste Punktkarte werfen."""
    bot = HeuristicPlayer("Bot")
    variant = Variant.trumpf(Suit.EICHEL)
    # starter=2 (Partner), trick: Partner Herz-7, Gegner Eichel-Buur (sticht)
    # Ich bin Spieler 0 → dran, kann nicht stechen
    trick = [C(Suit.HERZ, Rank.SIEBEN), C(Suit.EICHEL, Rank.UNTER)]
    hand = [
        C(Suit.LAUB, Rank.ASS),     # 11 Pkt, sollte gespart bleiben
        C(Suit.LAUB, Rank.SECHS),   # 0 Pkt, niedrigste → werfen
        C(Suit.SCHELLE, Rank.OBER), # 3 Pkt
    ]
    state = _state(player_idx=0, variant=variant, trick=trick, starter=2)
    chosen = bot.choose_card(hand, state)
    assert chosen == C(Suit.LAUB, Rank.SECHS)


# ---------- Stechen ----------

def test_stechen_als_letzter_minimal():
    """Als letzter Spieler im Stich, der Gegner führt → übernehme so knapp wie möglich."""
    bot = HeuristicPlayer("Bot")
    variant = Variant.trumpf(Suit.EICHEL)
    # Lead Herz-König (Gegner), Partner H-7, Gegner H-9. Ich kann mit H-Ass stechen.
    trick = [
        C(Suit.HERZ, Rank.KOENIG),
        C(Suit.HERZ, Rank.SIEBEN),
        C(Suit.HERZ, Rank.NEUN),
    ]
    hand = [
        C(Suit.HERZ, Rank.ASS),     # sticht
        C(Suit.HERZ, Rank.SECHS),   # sticht nicht
    ]
    state = _state(player_idx=0, variant=variant, trick=trick, starter=1)
    chosen = bot.choose_card(hand, state)
    assert chosen == C(Suit.HERZ, Rank.ASS)


def test_stechen_mit_hoher_karte_wenn_noch_gegner_kommen():
    """Wenn noch Spieler nach mir kommen, wähle hohe Stich-Karte (sicherer)."""
    bot = HeuristicPlayer("Bot")
    variant = Variant.trumpf(Suit.EICHEL)
    # Lead Herz-7 (Gegner). Ich bin zweiter im Stich. Habe H-Ass und H-9.
    trick = [C(Suit.HERZ, Rank.SIEBEN)]
    hand = [
        C(Suit.HERZ, Rank.ASS),
        C(Suit.HERZ, Rank.NEUN),
    ]
    state = _state(player_idx=2, variant=variant, trick=trick, starter=1)
    chosen = bot.choose_card(hand, state)
    # Ass ist die höchste sichere Wahl
    assert chosen == C(Suit.HERZ, Rank.ASS)


# ---------- Anspielen ----------

def test_opening_trumpf_buur_zuerst():
    bot = HeuristicPlayer("Bot")
    variant = Variant.trumpf(Suit.EICHEL)
    hand = [
        C(Suit.EICHEL, Rank.UNTER),  # Buur
        C(Suit.EICHEL, Rank.ASS),
        C(Suit.HERZ, Rank.SIEBEN),
    ]
    state = _state(player_idx=0, variant=variant, trick=[], starter=0)
    chosen = bot.choose_card(hand, state)
    assert chosen == C(Suit.EICHEL, Rank.UNTER)


def test_opening_oben_ass():
    bot = HeuristicPlayer("Bot")
    variant = Variant.oben()
    hand = [
        C(Suit.HERZ, Rank.ASS),
        C(Suit.LAUB, Rank.SECHS),
        C(Suit.EICHEL, Rank.SIEBEN),
    ]
    state = _state(player_idx=0, variant=variant, trick=[], starter=0)
    chosen = bot.choose_card(hand, state)
    assert chosen == C(Suit.HERZ, Rank.ASS)


def test_opening_unten_sechs():
    bot = HeuristicPlayer("Bot")
    variant = Variant.unten()
    hand = [
        C(Suit.HERZ, Rank.ASS),
        C(Suit.LAUB, Rank.SECHS),   # niedrigste = stärkste bei Geiss
        C(Suit.EICHEL, Rank.SIEBEN),
    ]
    state = _state(player_idx=0, variant=variant, trick=[], starter=0)
    chosen = bot.choose_card(hand, state)
    assert chosen == C(Suit.LAUB, Rank.SECHS)


# ---------- Ansage ----------

def test_ansage_buur_und_nell_mit_niedrigem_rest_macht_gumpf_attraktiv():
    """Lehrbuch-Gumpf: Buur+Nell+Ass+Koenig in einer Farbe als sichere Top-
    Truempfe, plus durchgehend niedrige Karten (6/7) in den Nicht-Trumpf-Farben.
    Da die niedrigen Karten in Gumpf nach Geiss-Logik bewertet werden (6 sticht
    in der Lead-Farbe alles), ist Gumpf hier signifikant besser als reines
    Trumpf-Spiel.
    """
    bot = HeuristicPlayer("Bot")
    hand = [
        C(Suit.EICHEL, Rank.UNTER),  # Buur
        C(Suit.EICHEL, Rank.NEUN),    # Nell
        C(Suit.EICHEL, Rank.ASS),
        C(Suit.EICHEL, Rank.KOENIG),
        C(Suit.HERZ, Rank.SECHS),
        C(Suit.HERZ, Rank.SIEBEN),
        C(Suit.LAUB, Rank.SECHS),
        C(Suit.LAUB, Rank.SIEBEN),
        C(Suit.SCHELLE, Rank.SECHS),
    ]
    ann = bot.choose_announcement(hand, round_idx=0, can_push=False)
    assert ann is not None
    assert ann.variant.mode == PlayMode.GUMPF
    assert ann.variant.trump_suit == Suit.EICHEL


def test_ansage_buur_und_nell_mit_hohem_rest_macht_trumpf_attraktiv():
    """Gegenstueck: Buur+Nell+Ass+Koenig in Eichel, aber **hohe** Karten in
    den Nicht-Trumpf-Farben (Asse, Zehner). Da hohe Karten in Gumpf nach
    Geiss-Logik wertlos sind, gewinnt hier Trumpf-Eichel.
    """
    bot = HeuristicPlayer("Bot")
    hand = [
        C(Suit.EICHEL, Rank.UNTER),  # Buur
        C(Suit.EICHEL, Rank.NEUN),    # Nell
        C(Suit.EICHEL, Rank.ASS),
        C(Suit.EICHEL, Rank.KOENIG),
        C(Suit.HERZ, Rank.ASS),
        C(Suit.HERZ, Rank.ZEHN),
        C(Suit.LAUB, Rank.ASS),
        C(Suit.LAUB, Rank.ZEHN),
        C(Suit.SCHELLE, Rank.ASS),
    ]
    ann = bot.choose_announcement(hand, round_idx=0, can_push=False)
    assert ann is not None
    assert ann.variant.mode == PlayMode.TRUMPF
    assert ann.variant.trump_suit == Suit.EICHEL


def test_ansage_gumpf_verboten_faellt_auf_trumpf_zurueck():
    """Hausregel 'kein Gumpf': bei einer Lehrbuch-Gumpf-Hand muss die Heuristik
    sauber auf die zweitbeste Wahl (Trumpf-Eichel) zurueckfallen.
    """
    bot = HeuristicPlayer(
        "Bot",
        allowed_modes={PlayMode.TRUMPF, PlayMode.OBEN, PlayMode.UNTEN},
    )
    hand = [
        C(Suit.EICHEL, Rank.UNTER),
        C(Suit.EICHEL, Rank.NEUN),
        C(Suit.EICHEL, Rank.ASS),
        C(Suit.EICHEL, Rank.KOENIG),
        C(Suit.HERZ, Rank.SECHS),
        C(Suit.HERZ, Rank.SIEBEN),
        C(Suit.LAUB, Rank.SECHS),
        C(Suit.LAUB, Rank.SIEBEN),
        C(Suit.SCHELLE, Rank.SECHS),
    ]
    ann = bot.choose_announcement(hand, round_idx=0, can_push=False)
    assert ann is not None
    assert ann.variant.mode == PlayMode.TRUMPF
    assert ann.variant.trump_suit == Suit.EICHEL


def test_ansage_slalom_verboten():
    """Wenn allow_slalom=False, darf die Heuristik niemals Slalom waehlen, auch
    nicht bei einer Hand, die sonst klar Slalom favorisiert."""
    bot = HeuristicPlayer("Bot", allow_slalom=False)
    hand = [
        # Eichel: 6, 7, 8, 9 (4 unten-starke Karten)
        C(Suit.EICHEL, Rank.SECHS),
        C(Suit.EICHEL, Rank.SIEBEN),
        C(Suit.EICHEL, Rank.ACHT),
        C(Suit.EICHEL, Rank.NEUN),
        # Laub: Ober, Koenig, Ass (3 oben-starke Karten)
        C(Suit.LAUB, Rank.OBER),
        C(Suit.LAUB, Rank.KOENIG),
        C(Suit.LAUB, Rank.ASS),
        C(Suit.HERZ, Rank.ASS),
        C(Suit.SCHELLE, Rank.SECHS),
    ]
    ann = bot.choose_announcement(hand, round_idx=0, can_push=False)
    assert ann is not None
    assert ann.slalom is False


def test_ansage_viele_asse_macht_bock_attraktiv():
    bot = HeuristicPlayer("Bot")
    hand = [
        C(Suit.EICHEL, Rank.ASS),
        C(Suit.HERZ, Rank.ASS),
        C(Suit.LAUB, Rank.ASS),
        C(Suit.SCHELLE, Rank.ASS),
        C(Suit.EICHEL, Rank.ZEHN),
        C(Suit.HERZ, Rank.ZEHN),
        C(Suit.LAUB, Rank.SECHS),
        C(Suit.SCHELLE, Rank.SECHS),
        C(Suit.EICHEL, Rank.SIEBEN),
    ]
    ann = bot.choose_announcement(hand, round_idx=0, can_push=False)
    assert ann is not None
    assert ann.variant.mode == PlayMode.OBEN


def test_ansage_viele_sechser_macht_geiss_attraktiv():
    bot = HeuristicPlayer("Bot")
    hand = [
        C(Suit.EICHEL, Rank.SECHS),
        C(Suit.HERZ, Rank.SECHS),
        C(Suit.LAUB, Rank.SECHS),
        C(Suit.SCHELLE, Rank.SECHS),
        C(Suit.EICHEL, Rank.SIEBEN),
        C(Suit.HERZ, Rank.SIEBEN),
        C(Suit.LAUB, Rank.SIEBEN),
        C(Suit.SCHELLE, Rank.ACHT),
        C(Suit.EICHEL, Rank.ACHT),
    ]
    ann = bot.choose_announcement(hand, round_idx=0, can_push=False)
    assert ann is not None
    assert ann.variant.mode == PlayMode.UNTEN


def test_ansage_schwache_hand_schiebt():
    """Sehr schwache Hand → Bot schiebt, wenn er darf."""
    bot = HeuristicPlayer("Bot", push_threshold=200)  # sehr hoher Threshold zum Erzwingen
    hand = [
        C(Suit.EICHEL, Rank.SIEBEN),
        C(Suit.HERZ, Rank.ACHT),
        C(Suit.LAUB, Rank.NEUN),
        C(Suit.SCHELLE, Rank.SIEBEN),
        C(Suit.EICHEL, Rank.ACHT),
        C(Suit.HERZ, Rank.NEUN),
        C(Suit.LAUB, Rank.SECHS),
        C(Suit.SCHELLE, Rank.NEUN),
        C(Suit.EICHEL, Rank.NEUN),
    ]
    ann = bot.choose_announcement(hand, round_idx=1, can_push=True)
    assert ann is None  # geschoben


def test_ansage_kann_nicht_schieben_in_runde_1():
    bot = HeuristicPlayer("Bot", push_threshold=99999)
    hand = [C(Suit.EICHEL, Rank.SIEBEN)] * 9  # dummy
    ann = bot.choose_announcement(hand[:1] + [C(Suit.HERZ, Rank.ACHT)] * 8, round_idx=0, can_push=False)
    assert ann is not None


def test_ansage_slalom_bei_komplementaerer_hand():
    """Hand mit niedrigen Karten (6,7,8) in einer Farbe und hohen (O,K,A) in einer
    anderen sollte Slalom auslösen, weil beide Modi unterstützt werden."""
    bot = HeuristicPlayer("Bot")
    hand = [
        # F1 = Eichel: 6, 7, 8, 9 (4 Karten, alle unten-stark)
        C(Suit.EICHEL, Rank.SECHS),
        C(Suit.EICHEL, Rank.SIEBEN),
        C(Suit.EICHEL, Rank.ACHT),
        C(Suit.EICHEL, Rank.NEUN),
        # F2 = Laub: Ober, König, Ass (3 Karten, alle oben-stark)
        C(Suit.LAUB, Rank.OBER),
        C(Suit.LAUB, Rank.KOENIG),
        C(Suit.LAUB, Rank.ASS),
        # 2 weitere komplementäre Karten
        C(Suit.HERZ, Rank.ASS),
        C(Suit.SCHELLE, Rank.SECHS),
    ]
    ann = bot.choose_announcement(hand, round_idx=0, can_push=False)
    assert ann is not None
    assert ann.slalom is True, f"Erwartete Slalom-Ansage, bekam {ann}"


def test_ansage_kein_slalom_bei_einseitiger_hand():
    """Hand mit nur hohen Karten → Slalom-Bonus = 0 → Oben gewinnt."""
    bot = HeuristicPlayer("Bot")
    hand = [
        C(Suit.EICHEL, Rank.ASS),
        C(Suit.HERZ, Rank.ASS),
        C(Suit.LAUB, Rank.ASS),
        C(Suit.SCHELLE, Rank.ASS),
        C(Suit.EICHEL, Rank.KOENIG),
        C(Suit.HERZ, Rank.KOENIG),
        C(Suit.LAUB, Rank.KOENIG),
        C(Suit.SCHELLE, Rank.KOENIG),
        C(Suit.EICHEL, Rank.OBER),
    ]
    ann = bot.choose_announcement(hand, round_idx=0, can_push=False)
    assert ann is not None
    # Mit dieser Hand sollte OBEN gewählt werden, nicht Slalom
    assert ann.slalom is False
    assert ann.variant.mode == PlayMode.OBEN


def test_konzentrierte_hand_bekommt_hoeheren_slalom_score_als_verstreute():
    """Eine Hand mit 3 oben-/3 unten-Karten in jeweils gleicher Farbe sollte einen
    deutlich höheren Slalom-Score haben als eine Hand mit gleich vielen Karten,
    die aber über alle 4 Farben verstreut sind. Das prüft die Konzentrations-Logik."""
    bot = HeuristicPlayer("Bot")

    # Konzentriert: Eichel = O,K,A + Laub = 6,7,8 + 3 Filler
    konzentriert = [
        C(Suit.EICHEL, Rank.OBER),
        C(Suit.EICHEL, Rank.KOENIG),
        C(Suit.EICHEL, Rank.ASS),
        C(Suit.LAUB, Rank.SECHS),
        C(Suit.LAUB, Rank.SIEBEN),
        C(Suit.LAUB, Rank.ACHT),
        C(Suit.HERZ, Rank.NEUN),
        C(Suit.HERZ, Rank.ZEHN),
        C(Suit.SCHELLE, Rank.UNTER),
    ]
    # Verstreut: 3 oben- und 3 unten-Karten, aber über alle Farben verteilt
    verstreut = [
        C(Suit.EICHEL, Rank.ASS),
        C(Suit.HERZ, Rank.KOENIG),
        C(Suit.LAUB, Rank.OBER),
        C(Suit.SCHELLE, Rank.SECHS),
        C(Suit.EICHEL, Rank.SIEBEN),
        C(Suit.HERZ, Rank.ACHT),
        C(Suit.LAUB, Rank.NEUN),
        C(Suit.SCHELLE, Rank.ZEHN),
        C(Suit.EICHEL, Rank.UNTER),
    ]

    # Direkter Vergleich: gleiche Anzahl A/K/O und 6/7/8, aber Verteilung anders.
    # Wir prüfen es über die gewählte Ansage:
    ann_konz = bot.choose_announcement(konzentriert, round_idx=0, can_push=False)
    ann_verstreut = bot.choose_announcement(verstreut, round_idx=0, can_push=False)
    # Konzentrierte Hand → Slalom; verstreute Hand → eher nicht Slalom
    assert ann_konz is not None and ann_konz.slalom is True
    # Die verstreute Hand sollte Slalom NICHT als Top-Ansage haben
    assert ann_verstreut is not None
    # (Die Trumpf-Score bei verstreuter Hand mit 4 Assen mag hoch sein → andere Ansage)
    assert ann_verstreut.slalom is False, (
        f"Verstreute Hand sollte Slalom NICHT triggern, bekam {ann_verstreut}"
    )
