"""Generator: erzeugt `spec/jass_rules.json` aus den Python-Konstanten.

Garantiert, dass die JSON-Spezifikation immer synchron mit dem Engine-Code ist.
Wird in CI ausgeführt und der Diff geprüft.

Aufruf:
    python -m scripts.generate_jass_rules_json
"""

from __future__ import annotations

import json
from pathlib import Path

from jass_engine.card import ALL_RANKS, ALL_SUITS, Rank, Suit
from jass_engine.rules import (
    LAST_TRICK_BONUS,
    MATCH_BONUS,
    POINT_VALUES_NORMAL,
    POINT_VALUES_OBEN_UNTEN,
    POINT_VALUES_TRUMP,
    TOTAL_POINTS_PER_ROUND,
    TRUMP_RANK_ORDER,
)
from jass_engine.weis import (
    FOUR_OF_KIND_POINTS,
    SEQUENCE_POINTS,
    STOECKE_POINTS,
)


SPEC_VERSION = "1.2.0"  # +score_composition, +play_order_anchor (additiv, kein Bruch)

GAME_NAME = "Vorarlberger Kreuz-Jass"
GAME_DESCRIPTION = (
    "Vorarlberger Variante des Schweizer Jass mit 4 Spielern, einfachem deutschen "
    "Blatt (36 Karten), Trumpf-/Gumpf-/Bock-/Geiss-/Slalom-Varianten, Schieben, Weisen, "
    "Stöcken und Matsch-Bonus."
)


def _ranks() -> list[dict]:
    short_names = {
        Rank.SECHS: "6",
        Rank.SIEBEN: "7",
        Rank.ACHT: "8",
        Rank.NEUN: "9",
        Rank.ZEHN: "10",
        Rank.UNTER: "U",
        Rank.OBER: "O",
        Rank.KOENIG: "K",
        Rank.ASS: "A",
    }
    return [
        {
            "id": int(r),
            "name": r.name,
            "german": r.full_name,
            "short": short_names[r],
        }
        for r in ALL_RANKS
    ]


def _suits() -> list[dict]:
    return [
        {
            "id": int(s),
            "name": s.name,
            "german": s.german_name,
        }
        for s in ALL_SUITS
    ]


def _trumpf_points() -> dict[str, dict[str, int]]:
    return {
        "non_trump": {r.name: POINT_VALUES_NORMAL[r] for r in ALL_RANKS},
        "trump": {r.name: POINT_VALUES_TRUMP[r] for r in ALL_RANKS},
    }


def _gumpf_points() -> dict[str, dict[str, int]]:
    """Gumpf-Wertpunkte: identisch mit Trumpf-Variante. Nur Stärke ist invertiert."""
    return {
        "non_trump": {r.name: POINT_VALUES_NORMAL[r] for r in ALL_RANKS},
        "trump": {r.name: POINT_VALUES_TRUMP[r] for r in ALL_RANKS},
    }


def _non_trump_rank_order_desc_inverted() -> list[str]:
    """Gumpf-Nicht-Trumpf: 6 stärkste, Ass schwächste (Geiss-mäßig)."""
    return [r.name for r in sorted(ALL_RANKS, key=lambda r: int(r))]


def _oben_unten_points() -> dict[str, int]:
    return {r.name: POINT_VALUES_OBEN_UNTEN[r] for r in ALL_RANKS}


def _trump_rank_order_desc() -> list[str]:
    return sorted(
        (r.name for r in ALL_RANKS),
        key=lambda name: -TRUMP_RANK_ORDER[Rank[name]],
    )


def _non_trump_rank_order_desc() -> list[str]:
    return [r.name for r in sorted(ALL_RANKS, key=lambda r: -int(r))]


def _unten_rank_order_desc() -> list[str]:
    # Bei Unten ist die niedrigste Karte die stärkste
    return [r.name for r in sorted(ALL_RANKS, key=lambda r: int(r))]


def _sequences() -> dict[str, int]:
    return {f"{length}_blatt": pts for length, pts in SEQUENCE_POINTS.items()}


def _four_of_kind() -> dict[str, int]:
    return {r.name: pts for r, pts in FOUR_OF_KIND_POINTS.items()}


def build_spec() -> dict:
    return {
        "spec_version": SPEC_VERSION,
        "name": GAME_NAME,
        "description": GAME_DESCRIPTION,
        "sources": [
            "https://jassa.at/regeln/",
            "https://www.mohrenbrauerei.at/biererlebniswelt/community/haeufig-gestellte-fragen-faq/",
            "https://www.jasskarten.at/jassregeln",
        ],

        "deck": {
            "total_cards": 36,
            "suits": _suits(),
            "ranks": _ranks(),
            "card_index_formula": "suit_id * 9 + rank_id",
            "index_range": [0, 35],
        },

        "special_cards": {
            "weli": {
                "suit": "SCHELLE",
                "rank": "SECHS",
                "card_index": int(Suit.SCHELLE) * 9 + int(Rank.SECHS),
                "role": "trump_announcer_in_round_1",
                "note": (
                    "Im laufenden Spiel zählt der Weli wie jede andere 6 (0 Punkte, keine "
                    "Sonderstärke). Er bestimmt nur, wer in Runde 1 Trumpf ansagt."
                ),
            }
        },

        "variants": {
            "trumpf": {
                "id": "trumpf",
                "german": "Trumpf",
                "has_trump_suit": True,
                "trump_suit_options": [s.name for s in ALL_SUITS],
                "card_points": _trumpf_points(),
                "rank_order_desc": {
                    "non_trump": _non_trump_rank_order_desc(),
                    "trump": _trump_rank_order_desc(),
                },
                "rules": {
                    "follow_lead_suit_if_possible": True,
                    "buur_exception": {
                        "active": True,
                        "description": (
                            "Trumpf-Unter (Buur) darf immer gespielt werden, auch wenn "
                            "der Spieler die Lead-Farbe bedienen könnte. Bei Trumpf-Lead "
                            "darf der Buur als einziger Trumpf zurückgehalten werden."
                        ),
                    },
                    "no_undertrumping": {
                        "active": True,
                        "description": (
                            "Liegt schon ein Trumpf im Stich, darf nur höher getrumpft "
                            "werden; Untertrumpfen ist nur erlaubt, wenn keine andere Karte "
                            "spielbar ist."
                        ),
                    },
                    "stichzwang": {
                        "active": False,
                        "description": (
                            "Wer die Lead-Farbe nicht bedienen kann, darf frei abwerfen — "
                            "es besteht keine Pflicht zu trumpfen."
                        ),
                    },
                },
                "stoecke": {
                    "active": True,
                    "cards": ["trump_OBER", "trump_KOENIG"],
                    "points": STOECKE_POINTS,
                    "announce_timing": "after_second_stock_card_played",
                    "competes_with_other_weisen": False,
                },
            },

            "gumpf": {
                "id": "gumpf",
                "german": "Gumpf",
                "has_trump_suit": True,
                "trump_suit_options": [s.name for s in ALL_SUITS],
                "card_points": _gumpf_points(),
                "rank_order_desc": {
                    # Nicht-Trumpf-Reihenfolge ist invertiert (6 stärkste, Ass schwächste)
                    "non_trump": _non_trump_rank_order_desc_inverted(),
                    "trump": _trump_rank_order_desc(),
                },
                "rules": {
                    "follow_lead_suit_if_possible": True,
                    "buur_exception": {
                        "active": True,
                        "description": (
                            "Trumpf-Unter (Buur) darf auch im Gumpf jederzeit gespielt "
                            "werden — die Trumpf-Farbe verhält sich genauso wie bei der "
                            "normalen Trumpf-Variante."
                        ),
                    },
                    "no_undertrumping": {
                        "active": True,
                        "description": (
                            "In der Trumpf-Farbe gilt im Gumpf das gleiche Untertrumpf-"
                            "Verbot wie in der normalen Trumpf-Variante."
                        ),
                    },
                    "stichzwang": {
                        "active": False,
                        "description": (
                            "Wer die Lead-Farbe nicht bedienen kann, darf frei abwerfen — "
                            "es besteht keine Pflicht zu trumpfen."
                        ),
                    },
                    "note": (
                        "Gumpf = 'Geiss + Trumpf': In der Trumpf-Farbe gelten Trumpf-Werte "
                        "(Buur=20, Nell=14) und Trumpf-Reihenfolge. In allen anderen Farben "
                        "ist die Stärke invertiert: die 6 in der Lead-Farbe sticht alles "
                        "Nicht-Trumpf. Wertpunkte in Nicht-Trumpf bleiben wie bei der "
                        "Trumpf-Variante (8er=0, kein Geiss-8er-Bonus)."
                    ),
                },
                "stoecke": {
                    "active": True,
                    "cards": ["trump_OBER", "trump_KOENIG"],
                    "points": STOECKE_POINTS,
                    "announce_timing": "after_second_stock_card_played",
                    "competes_with_other_weisen": False,
                },
            },

            "oben": {
                "id": "oben",
                "german": "Bock (oben)",
                "has_trump_suit": False,
                "card_points": _oben_unten_points(),
                "rank_order_desc": {
                    "lead_suit": _non_trump_rank_order_desc(),
                },
                "rules": {
                    "follow_lead_suit_if_possible": True,
                    "buur_exception": {"active": False},
                    "no_undertrumping": {"active": False},
                    "stichzwang": {"active": False},
                    "note": "Nur Lead-Farbe sticht — Karten anderer Farben können nie gewinnen.",
                },
                "stoecke": {"active": False},
            },

            "unten": {
                "id": "unten",
                "german": "Geiss (unten)",
                "has_trump_suit": False,
                "card_points": _oben_unten_points(),
                "rank_order_desc": {
                    "lead_suit": _unten_rank_order_desc(),
                },
                "rules": {
                    "follow_lead_suit_if_possible": True,
                    "buur_exception": {"active": False},
                    "no_undertrumping": {"active": False},
                    "stichzwang": {"active": False},
                    "note": (
                        "Umgekehrte Reihenfolge: die 6 sticht alles in der Lead-Farbe. "
                        "Karten anderer Farben können nie gewinnen."
                    ),
                },
                "stoecke": {"active": False},
            },

            "slalom": {
                "id": "slalom",
                "german": "Slalom",
                "alternating_modes": ["oben", "unten"],
                "starter_chooses_first_mode": True,
                "card_points_inherit_from": ["oben", "unten"],
                "rules": {
                    "mode_switches_per_trick": True,
                    "description": (
                        "Der Ansager wählt, ob mit Bock (oben) oder Geiss (unten) "
                        "begonnen wird; der Modus wechselt dann nach jedem Stich."
                    ),
                },
                "stoecke": {"active": False},
            },
        },

        "scoring": {
            "card_points_per_round": 152,
            "last_trick_bonus": LAST_TRICK_BONUS,
            "total_points_per_round": TOTAL_POINTS_PER_ROUND,
            "match_bonus": MATCH_BONUS,
            "match_definition": "Ein Team gewinnt alle 9 Stiche einer Runde.",
            "team_scoring_convention": (
                "Üblicherweise zählt nur das Team mit weniger Stichen seine Punkte; "
                "das andere Team bekommt (TOTAL_POINTS_PER_ROUND − x). Beide Werte "
                "müssen sich zu 157 (bzw. 257 bei Matsch) summieren — wichtige "
                "Konsistenzprüfung in jeder Engine-Implementierung."
            ),
            "score_composition": {
                "description": (
                    "Wie die Rundenwertung pro Team aus den einzelnen Punkt-Quellen "
                    "zusammengesetzt wird. Eine kompatible Engine-Implementierung muss "
                    "diese Formel exakt einhalten, sonst weichen Statistiken und "
                    "Konsistenz-Prüfungen zwischen Engines voneinander ab."
                ),
                "formula": (
                    "team_total_points[team] = team_card_points[team] "
                    "+ team_weise_points[team] + team_stoecke_points[team]"
                ),
                "team_card_points_formula": (
                    "team_card_points[team] = "
                    "Σ trick_points(t) für t in tricks von team "
                    "+ (MATCH_BONUS wenn matsch_team == team else 0)"
                ),
                "trick_points_formula": (
                    "trick_points(t) = Σ card_value(c, variant) für c in t.cards "
                    "+ (LAST_TRICK_BONUS wenn t letzter Stich der Runde else 0)"
                ),
                "match_bonus_handling": "added_to_team_card_points",
                "match_bonus_handling_note": (
                    "Der MATCH_BONUS (100) wird DIREKT zu team_card_points[team] "
                    "addiert, nicht in einem separaten Feld geführt. Konsistenz-"
                    "Prüfung: sum(team_card_points.values()) == "
                    "TOTAL_POINTS_PER_ROUND + (MATCH_BONUS wenn Matsch vorliegt). "
                    "Konkret: 157 ohne Matsch, 257 mit Matsch."
                ),
                "last_trick_bonus_handling": (
                    "Der LAST_TRICK_BONUS (5) ist bereits in trick_points des "
                    "letzten Stichs enthalten, somit transitiv in team_card_points. "
                    "Kein separates Feld."
                ),
                "fields_outside_team_card_points": [
                    "team_weise_points",
                    "team_stoecke_points",
                ],
            },
        },

        "weise": {
            "sequences": _sequences(),
            "four_of_kind": _four_of_kind(),
            "stoecke_points": STOECKE_POINTS,
            "comparison_rules": {
                "winning_team_takes_all_their_weisen": True,
                "loser_team_gets_zero": True,
                "primary_sort_key": "weis_points_descending",
                "secondary_sort_key": "top_rank_descending",
                "tie_breaker": "earlier_in_play_order_wins",
                "play_order_anchor": "original_announcer",
                "play_order_definition": (
                    "play_order = [(announcer_idx + i) mod num_players "
                    "für i in 0..num_players-1]. Der Anker ist der ursprüngliche "
                    "Ansager (announcer_idx), auch wenn er an seinen Partner "
                    "geschoben hat. Konsistent zur Konvention, dass der erste "
                    "Stich vom ursprünglichen Ansager ausgespielt wird "
                    "(siehe round_flow.trick_start.first_trick)."
                ),
                "stoecke_independent": True,
            },
            "announce_timing": "before_first_trick",
            "stoecke_announce_timing": "after_second_stock_card_played",
        },

        "round_flow": {
            "num_players": 4,
            "cards_per_player": 9,
            "teams": [[0, 2], [1, 3]],
            "team_count": 2,
            "trump_announcement": {
                "round_1": {
                    "announcer": "player_holding_weli",
                    "push_allowed": False,
                },
                "round_n": {
                    "announcer": "rotates_clockwise_from_previous_announcer",
                    "push_allowed": True,
                },
            },
            "push": {
                "to": "partner",
                "communication_allowed": False,
                "first_trick_played_by": "original_announcer",
                "partner_may_not_push_back": True,
            },
            "tricks_per_round": 9,
            "trick_start": {
                "first_trick": "announcer (or original_announcer if pushed)",
                "subsequent_tricks": "winner_of_previous_trick",
            },
        },

        "game_flow": {
            "default_target_score": 1000,
            "target_score_configurable": True,
            "winner": "first_team_to_reach_target_score_at_end_of_round",
        },

        "action_space": {
            "size": 36,
            "encoding": "card_index = suit_id * 9 + rank_id",
            "mask_semantics": "1 = legal action, 0 = illegal action",
        },
    }


def main():
    spec = build_spec()
    out_path = Path("spec/jass_rules.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": auf Windows wuerde write_text() sonst \n -> \r\n umsetzen,
    # was den Spec-Drift-Check der Release-Pipeline (.gitattributes: *.json eol=lf)
    # nach jedem Aufruf scheitern laesst.
    out_path.write_text(
        json.dumps(spec, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    print(f"Geschrieben: {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
