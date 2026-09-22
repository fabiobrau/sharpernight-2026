"""Every user-facing string in the demo, in Italian and English.

Nothing that appears on screen may be written inline in the drawing code -- the
science fair is most likely Italian-speaking, and the operator must be able to
flip the whole interface with `--lang`.

Keep every line short enough to read from 3 m and simple enough for a 7-year-old.
"""
from __future__ import annotations

LANGUAGES = ("it", "en")
DEFAULT_LANG = "it"

_STRINGS: dict[str, dict[str, str]] = {
    "it": {
        # --- panel headings -------------------------------------------------
        "panel_human": "QUELLO CHE VEDI TU",
        "panel_robot": "QUELLO CHE VEDE IL ROBOT",
        # --- robot speech ---------------------------------------------------
        "robot_sleeping": "Portami un umano!",
        "robot_detected": "TI VEDO!",
        "robot_vanished": "DOVE SEI FINITO?!",
        "robot_found": "TI HO TROVATO!",
        # --- detection banner ----------------------------------------------
        "label_human": "UMANO",
        "banner_detected": "UMANO TROVATO",
        "banner_lost": "NESSUN UMANO",
        # --- the big timer --------------------------------------------------
        "invisible_for": "INVISIBILE DA",
        "seconds_short": "s",
        "top_board": "I 5 PIU' INVISIBILI",
        "new_record": "NUOVO RECORD!",
        "board_empty": "Nessuno ancora!",
        # --- posters --------------------------------------------------------
        "poster_title": "CARTELLONE",
        "poster_magic": "1 - Il cane magico",
        "poster_dog": "2 - Un cane normale",
        "poster_none": "0 - Niente",
        "hint_try_posters": "Prova i cartelloni 1 e 2",
        # --- confidence -----------------------------------------------------
        "confidence": "QUANTO E' SICURO",
        "threshold": "soglia",
        # --- modes ----------------------------------------------------------
        "mode_fallback": "VIDEO REGISTRATO",
        "board_not_found": "Mostra i 4 quadratini!",
        # --- expert mode ----------------------------------------------------
        "expert_on": "MODO ESPERTO",
        "expert_explain": "Questo robot e' piu' nuovo: il trucco non funziona con lui",
        "expert_model": "robot n.2",
        # --- footer ---------------------------------------------------------
        "privacy": "Non salviamo niente: nessuna foto, nessun video, nessun volto.",
        "keys_help": "ESC esci   F schermo intero   R azzera   E esperto",
        # --- trouble --------------------------------------------------------
        "camera_lost": "La telecamera fa i capricci... un attimo!",
        "camera_retry": "Riprovo a collegarmi...",
        "starting": "Accendo il robot...",
    },
    "en": {
        "panel_human": "WHAT YOU SEE",
        "panel_robot": "WHAT THE ROBOT SEES",
        "robot_sleeping": "Bring me a human!",
        "robot_detected": "I SEE YOU!",
        "robot_vanished": "WHERE DID YOU GO?!",
        "robot_found": "FOUND YOU!",
        "label_human": "HUMAN",
        "banner_detected": "HUMAN DETECTED",
        "banner_lost": "NO HUMAN",
        "invisible_for": "INVISIBLE FOR",
        "seconds_short": "s",
        "top_board": "TOP 5 INVISIBLE",
        "new_record": "NEW RECORD!",
        "board_empty": "Nobody yet!",
        "poster_title": "POSTER",
        "poster_magic": "1 - The magic dog",
        "poster_dog": "2 - An ordinary dog",
        "poster_none": "0 - Nothing",
        "hint_try_posters": "Try posters 1 and 2",
        "confidence": "HOW SURE IT IS",
        "threshold": "threshold",
        "mode_fallback": "RECORDED VIDEO",
        "board_not_found": "Show me the 4 little squares!",
        "expert_on": "EXPERT MODE",
        "expert_explain": "This robot is newer: the trick does not work on it",
        "expert_model": "robot no.2",
        "privacy": "We save nothing: no photos, no video, no faces.",
        "keys_help": "ESC quit   F fullscreen   R reset   E expert",
        "camera_lost": "The camera is being silly... one moment!",
        "camera_retry": "Trying to reconnect...",
        "starting": "Waking the robot up...",
    },
}


class Strings:
    """Tiny lookup object so drawing code reads `S.get('robot_found')`."""

    def __init__(self, lang: str = DEFAULT_LANG):
        if lang not in _STRINGS:
            lang = DEFAULT_LANG
        self.lang = lang
        self._d = _STRINGS[lang]

    def get(self, key: str) -> str:
        # A missing key must never crash the demo mid-show.
        return self._d.get(key, _STRINGS[DEFAULT_LANG].get(key, key))

    __call__ = get


def poster_label(strings: Strings, poster_key: str) -> str:
    return strings.get({
        "adversarial": "poster_magic",
        "dog": "poster_dog",
        "none": "poster_none",
    }.get(poster_key, "poster_none"))
