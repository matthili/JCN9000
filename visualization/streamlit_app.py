"""Streamlit-Web-UI zur Regelverifikation.

Aufruf:
    streamlit run visualization/streamlit_app.py

Vier Seiten:
  1. Regelwerk — statische Anzeige aller Karten/Werte je Variante
  2. Regelprüfer — interaktive Prüfung von legal_moves() je Variante
  3. Weis-Prüfer — interaktive Weis-Erkennung
  4. Demo-Partie — Random-vs-Random Stich für Stich
"""

from __future__ import annotations

import random

import pandas as pd
import streamlit as st

from jass_engine.card import ALL_RANKS, ALL_SUITS, Card, Rank, Suit
from jass_engine.deck import deal, find_weli_holder
from jass_engine.player import GameState
from jass_engine.rules import (
    LAST_TRICK_BONUS,
    MATCH_BONUS,
    POINT_VALUES_NORMAL,
    POINT_VALUES_OBEN_UNTEN,
    POINT_VALUES_TRUMP,
    TOTAL_POINTS_PER_ROUND,
    TRUMP_RANK_ORDER,
    card_value,
    legal_moves,
)
from jass_engine.trick import Trick
from jass_engine.variant import Announcement, PlayMode, Variant
from jass_engine.variants.kreuz_jass import KREUZ_JASS_TEAMS
from jass_engine.weis import find_weise, has_stoecke, stoecke_weis
from players.random_player import RandomPlayer


st.set_page_config(page_title="Jass-Engine Regelprüfer", layout="wide")


def card_label(c: Card) -> str:
    return f"{c.suit.german_name}-{c.rank.german_name}"


def select_cards(label: str, key: str, max_count: int = 9) -> list[Card]:
    options = [Card(s, r) for s in ALL_SUITS for r in ALL_RANKS]
    labels = [card_label(c) for c in options]
    chosen_labels = st.multiselect(label, labels, key=key, max_selections=max_count)
    return [options[labels.index(lbl)] for lbl in chosen_labels]


def select_suit(label: str, key: str) -> Suit:
    names = {s.german_name: s for s in ALL_SUITS}
    sel = st.selectbox(label, list(names.keys()), key=key)
    return names[sel]


def select_variant(label_prefix: str, key_prefix: str) -> Variant:
    """UI-Steuerelement zur Auswahl einer effektiven Variante (Trumpf/Oben/Unten)."""
    mode = st.selectbox(
        f"{label_prefix}: Modus",
        ["Trumpf", "Bock (oben)", "Geiss (unten)"],
        key=f"{key_prefix}_mode",
    )
    if mode == "Trumpf":
        suit = select_suit(f"{label_prefix}: Trumpf-Farbe", key=f"{key_prefix}_suit")
        return Variant.trumpf(suit)
    if mode == "Bock (oben)":
        return Variant.oben()
    return Variant.unten()


# ---------- Seite: Regelwerk ----------

def page_regelwerk():
    st.header("Regelwerk")

    st.subheader("Kartenwerte je Variante")
    trumpf_choice = select_suit("Trumpf-Farbe für die Werte-Tabelle", key="rw_trumpf")
    rows = []
    for s in ALL_SUITS:
        for r in ALL_RANKS:
            c = Card(s, r)
            wert_normal_trumpf_modus = str(POINT_VALUES_NORMAL[r])
            wert_trumpf = (
                str(POINT_VALUES_TRUMP[r]) if s == trumpf_choice else "-"
            )
            wert_oben_unten = str(POINT_VALUES_OBEN_UNTEN[r])
            rows.append(
                {
                    "Karte": card_label(c),
                    "Wert (im Trumpf-Modus, Nicht-Trumpf-Farbe)": wert_normal_trumpf_modus,
                    "Wert (im Trumpf-Modus, Trumpf-Farbe)": wert_trumpf,
                    "Wert (Bock/Geiss)": wert_oben_unten,
                    "Weli?": "ja" if c.is_weli else "",
                }
            )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Reihenfolge je Variante")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Trumpf** ({trumpf_choice.german_name}, hoch → niedrig)")
        sorted_trump = sorted(ALL_RANKS, key=lambda r: TRUMP_RANK_ORDER[r], reverse=True)
        st.text(" > ".join(r.full_name for r in sorted_trump))
        st.markdown("**Nicht-Trumpf-Farben** (hoch → niedrig)")
        st.text("Ass > König > Ober > Unter > 10 > 9 > 8 > 7 > 6")
    with col2:
        st.markdown("**Bock (oben), Lead-Farbe**")
        st.text("Ass > König > Ober > Unter > 10 > 9 > 8 > 7 > 6")
        st.caption("Andere Farben können nicht stechen.")
    with col3:
        st.markdown("**Geiss (unten), Lead-Farbe**")
        st.text("6 > 7 > 8 > 9 > 10 > Unter > Ober > König > Ass")
        st.caption("Andere Farben können nicht stechen.")

    st.subheader("Punktebilanz")
    st.markdown(
        f"""
- Stichpunkte gesamt: **152** (in allen Varianten — bei Bock/Geiss gleichen 4×8er den Buur+Nell-Wegfall aus)
- Bonus letzter Stich: **+{LAST_TRICK_BONUS}**
- Summe regulär: **{TOTAL_POINTS_PER_ROUND}**
- Matsch-Bonus (alle 9 Stiche): **+{MATCH_BONUS}** → **{TOTAL_POINTS_PER_ROUND + MATCH_BONUS}**
"""
    )

    st.subheader("Slalom")
    st.markdown(
        """
**Slalom** ist eine eigenständige Ansage. Der Ansager wählt, ob mit *oben* (Bock) oder *unten*
(Geiss) begonnen wird; der Modus wechselt dann **nach jedem Stich**. Die Kartenwerte folgen
dem Bock/Geiss-Schema (8er=8 Punkte, kein Buur/Nell-Bonus). Stöcke gibt es nicht, da kein
Trumpf existiert.
"""
    )

    st.subheader("Weis-Tabelle")
    st.markdown(
        """
| Kombination | Punkte |
|---|---|
| 3-Blatt-Sequenz | 20 |
| 4-Blatt-Sequenz | 50 |
| 5-Blatt-Sequenz | 100 |
| 6-Blatt-Sequenz | 120 |
| 7-Blatt-Sequenz | 140 |
| 8-Blatt-Sequenz | 160 |
| 9-Blatt-Sequenz | 180 |
| 4× Zehner/Ober/König/Ass | 100 |
| 4× Neuner | 150 |
| 4× Unter | 200 |
| Stöcke (Trumpf-Ober + Trumpf-König) | 20 (nur Trumpf-Variante) |
"""
    )

    st.subheader("Regelzwänge")
    st.markdown(
        """
- **Farbzwang** (alle Varianten): Lead-Farbe bedienen wenn möglich
- **Buur-Ausnahme** (nur Trumpf): Trumpf-Unter darf immer gespielt werden
- **Kein Untertrumpfen** (nur Trumpf): niedriger Trumpf nur als letzte Möglichkeit
- **Kein Stichzwang** (alle Varianten): wer nicht bedienen kann, darf frei abwerfen
- **Schieben**: ab Runde 2 erlaubt, der ursprüngliche Ansager spielt aber aus
"""
    )


# ---------- Seite: Regelprüfer ----------

def page_regelpruefer():
    st.header("Regelprüfer — welche Karte darf ich spielen?")
    st.write(
        "Wähle Variante, deine Hand, und die bisher im Stich gespielten Karten. "
        "Die App zeigt, welche deiner Handkarten erlaubt sind und warum."
    )

    col1, col2 = st.columns(2)
    with col1:
        variant = select_variant("Variante", key_prefix="rp")
        hand = select_cards("Deine Hand (max. 9 Karten)", key="rp_hand", max_count=9)
    with col2:
        current_trick = select_cards(
            "Karten im aktuellen Stich (in Spielreihenfolge; leer = du fängst an)",
            key="rp_trick",
            max_count=3,
        )

    if not hand:
        st.info("Bitte mindestens eine Karte für die Hand wählen.")
        return

    legal = set(legal_moves(hand, current_trick, variant))

    rows = []
    for c in hand:
        is_legal = c in legal
        reason = _explain_legality(c, hand, current_trick, variant, is_legal)
        rows.append(
            {
                "Karte": card_label(c),
                "Erlaubt?": "ja" if is_legal else "nein",
                "Begründung": reason,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _explain_legality(
    card: Card,
    hand: list[Card],
    current_trick: list[Card],
    variant: Variant,
    is_legal: bool,
) -> str:
    if not current_trick:
        return "Erste Karte des Stichs — alles ist erlaubt."
    lead_suit = current_trick[0].suit

    if variant.mode != PlayMode.TRUMPF:
        # Bock/Geiss: reiner Farbzwang
        same_suit = [c for c in hand if c.suit == lead_suit]
        if same_suit:
            if card.suit == lead_suit:
                return f"Bedienst die Lead-Farbe ({lead_suit.german_name})."
            return f"Lead-Farbe ({lead_suit.german_name}) bedienen Pflicht; diese Karte verstößt gegen den Farbzwang."
        return "Lead-Farbe nicht bedienbar — beliebige Karte erlaubt."

    # Trumpf-Modus
    assert variant.trump_suit is not None
    trumpf = variant.trump_suit

    if lead_suit == trumpf:
        trumps_in_hand = [c for c in hand if c.suit == trumpf]
        non_buur_trumps = [c for c in trumps_in_hand if c.rank != Rank.UNTER]
        if non_buur_trumps:
            if card.suit == trumpf:
                return "Trumpf angespielt, du musst Trumpf bedienen — diese Karte ist Trumpf."
            return "Trumpf angespielt; du hast andere Trümpfe als nur den Buur → musst Trumpf bedienen."
        if trumps_in_hand:
            return "Einziger Trumpf ist der Buur — Buur-Ausnahme, beliebige Karte erlaubt."
        return "Kein Trumpf in der Hand — beliebige Karte erlaubt."

    same_suit = [c for c in hand if c.suit == lead_suit]
    if same_suit:
        if card.suit == lead_suit:
            return f"Bedienst die Lead-Farbe ({lead_suit.german_name})."
        if card.suit == trumpf and card.rank == Rank.UNTER:
            return "Buur-Ausnahme: Trumpf-Unter darf immer gespielt werden."
        return f"Lead-Farbe ({lead_suit.german_name}) bedienbar; Farbzwang verletzt."

    trumps_in_trick = [c for c in current_trick if c.suit == trumpf]
    if not trumps_in_trick:
        return "Nicht bedienbar, kein Trumpf im Stich → frei abwerfen."
    highest_in_trick = max(trumps_in_trick, key=lambda c: TRUMP_RANK_ORDER[c.rank])
    if card.suit != trumpf:
        return "Nicht bedienbar → abwerfen (kein Stichzwang)."
    if TRUMP_RANK_ORDER[card.rank] > TRUMP_RANK_ORDER[highest_in_trick.rank]:
        return "Übertrumpfen erlaubt — höher als der höchste Trumpf im Stich."
    if is_legal:
        return "Nur niedrigere Trümpfe verfügbar → Untertrumpfen erzwungen."
    return "Untertrumpfen verboten — Alternativen vorhanden."


# ---------- Seite: Weis-Prüfer ----------

def page_weispruefer():
    st.header("Weis-Prüfer")
    st.write("Wähle 9 Karten und einen Trumpf — die App erkennt alle Weisen und Stöcke.")

    trumpf = select_suit("Trumpf (für Stöcke-Erkennung; bei Bock/Geiss irrelevant)", key="wp_trumpf")
    hand = select_cards("Hand (9 Karten)", key="wp_hand", max_count=9)

    if len(hand) != 9:
        st.info(f"Wähle genau 9 Karten ({len(hand)}/9).")
        return

    weise = find_weise(hand)
    rows = [
        {
            "Typ": w.kind.value,
            "Karten": ", ".join(card_label(c) for c in w.cards),
            "Punkte": w.points,
        }
        for w in weise
    ]
    if has_stoecke(hand, trumpf):
        sw = stoecke_weis(trumpf)
        rows.append(
            {
                "Typ": sw.kind.value + " (nur bei Trumpf-Variante)",
                "Karten": ", ".join(card_label(c) for c in sw.cards),
                "Punkte": sw.points,
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.success(f"Summe potenzieller Weis-Punkte: {sum(r['Punkte'] for r in rows)}")
    else:
        st.write("Keine Weisen in dieser Hand.")


# ---------- Seite: Demo-Partie ----------

def page_demo_partie():
    st.header("Demo-Partie (Random-vs-Random)")
    st.write(
        "Klick 'Neue Runde' für eine frisch gemischte Runde, dann 'Nächster Zug' "
        "für jeden Spielzug. Hände sind offen sichtbar zur Verifikation."
    )

    if "demo_state" not in st.session_state or st.button("Neue Runde", key="new_round"):
        _init_demo()

    state = st.session_state["demo_state"]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Ansage**: {state['announcement']}")
        st.markdown(f"**Ansager**: {state['player_names'][state['announcer_idx']]}")
        current_variant = state["announcement"].variant_for_trick(len(state["completed_tricks"]))
        st.markdown(f"**Aktueller Stich-Modus**: {current_variant}")
    with col2:
        st.markdown("**Team-Stichpunkte**")
        for tid, pts in state["team_points"].items():
            st.text(f"Team {tid}: {pts}")

    st.subheader("Hände (offen)")
    cols = st.columns(4)
    for idx, p_hand in enumerate(state["hands"]):
        with cols[idx]:
            st.markdown(f"**{state['player_names'][idx]}** (Team {state['teams'][idx]})")
            st.text("\n".join(
                card_label(c) for c in sorted(p_hand, key=lambda c: (int(c.suit), int(c.rank)))
            ))

    st.subheader(f"Aktueller Stich (Modus: {current_variant})")
    if state["current_trick"].cards:
        rows = [
            {
                "Pos": i,
                "Spieler": state["player_names"][state["current_trick"].player_idx_for_card(i)],
                "Karte": card_label(c),
                "Wert": card_value(c, current_variant),
            }
            for i, c in enumerate(state["current_trick"].cards)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.text("(noch leer)")

    if state["completed_tricks"]:
        st.subheader("Vergangene Stiche")
        for i, (t, w) in enumerate(zip(state["completed_tricks"], state["trick_winners"])):
            v_i = state["announcement"].variant_for_trick(i)
            cards_str = ", ".join(card_label(c) for c in t.cards)
            st.text(f"Stich {i + 1} ({v_i}): {cards_str} — Gewinner: {state['player_names'][w]}")

    if not state["round_done"]:
        if st.button("Nächster Zug", key="next_move"):
            _advance_demo()
    else:
        st.success("Runde abgeschlossen.")


def _init_demo():
    rng = random.Random()
    players = [
        RandomPlayer(name=f"P{i}", rng=random.Random(rng.randint(0, 10**9)))
        for i in range(4)
    ]
    hands = deal(num_players=4, rng=rng)
    weli_holder = find_weli_holder(hands)
    announcement = players[weli_holder].choose_announcement(hands[weli_holder], 0, can_push=False)
    assert announcement is not None
    teams = list(KREUZ_JASS_TEAMS)
    st.session_state["demo_state"] = {
        "players": players,
        "player_names": [p.name for p in players],
        "teams": teams,
        "hands": hands,
        "announcement": announcement,
        "announcer_idx": weli_holder,
        "current_trick": Trick(starting_player_idx=weli_holder, num_players=4),
        "completed_tricks": [],
        "trick_winners": [],
        "team_points": {tid: 0 for tid in set(teams)},
        "round_done": False,
    }


def _advance_demo():
    state = st.session_state["demo_state"]
    trick = state["current_trick"]
    trick_idx = len(state["completed_tricks"])
    variant_this = state["announcement"].variant_for_trick(trick_idx)

    if trick.is_complete():
        winner = trick.winner_idx(variant_this)
        is_last = trick_idx == 8
        pts = trick.points(variant_this, is_last=is_last)
        state["team_points"][state["teams"][winner]] += pts
        state["completed_tricks"].append(trick)
        state["trick_winners"].append(winner)
        if len(state["completed_tricks"]) == 9:
            state["round_done"] = True
            return
        state["current_trick"] = Trick(starting_player_idx=winner, num_players=4)
        return

    cur = trick.next_player_idx()
    state_obj = GameState(
        player_idx=cur,
        variant=variant_this,
        announcement=state["announcement"],
        current_trick_cards=list(trick.cards),
        current_trick_starter=trick.starting_player_idx,
        teams=list(state["teams"]),
        completed_tricks=[list(t.cards) for t in state["completed_tricks"]],
        trick_idx=trick_idx,
        num_players=4,
    )
    chosen = state["players"][cur].choose_card(state["hands"][cur], state_obj)
    state["hands"][cur].remove(chosen)
    trick.add_card(chosen)


# ---------- Navigation ----------

PAGES = {
    "Regelwerk": page_regelwerk,
    "Regelprüfer": page_regelpruefer,
    "Weis-Prüfer": page_weispruefer,
    "Demo-Partie": page_demo_partie,
}

st.sidebar.title("Jass-Engine")
choice = st.sidebar.radio("Seite", list(PAGES.keys()))
PAGES[choice]()
