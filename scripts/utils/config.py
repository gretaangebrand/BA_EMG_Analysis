"""
config.py
=========
Zentrale Konfiguration für die EMG-Pipeline.

Steuert:
  - BASELINE_MODE: Welche Übung als Baseline-Normalisierung dient ('SQ' oder 'DJ')
  - EXERCISE_CONFIGS: Welche Übungen in der Auswertung enthalten sind
  - EXERCISE_MAP: Mapping von Label → (Ordner, Seite) für Feature-Extraktion und Plots

Regel:
  Die Baseline-Übung wird automatisch aus der Auswertung ausgeschlossen.
  Bei DJ-Baseline wird der bilaterale Squat als Auswertungsübung ergänzt.

Verwendung in allen Skripten:
  from scripts.utils.config import BASELINE_MODE, EXERCISE_CONFIGS, EXERCISE_MAP, BASELINE_FOLDERS
"""

# ============================================================
# FARBEN: Zyklusphasen (barrierefrei, Okabe-Ito / seaborn colorblind)
# ============================================================
PHASE_COLORS = {
    "PER": "#D55E00",   # Vermillion (Rot)
    "OVU": "#0173B2",   # Blau
    "LUT": "#CC78BC",   # Pink/Lila
}

PHASE_LINESTYLES = {
    "PER": "-",         # durchgezogen
    "OVU": "--",        # gestrichelt
    "LUT": "-.",        # Strich-Punkt
}

# Schraffuren für Balkendiagramme (SW-Druck)
PHASE_HATCHES = {
    "PER": "",          # keine Schraffur (voll gefüllt)
    "OVU": "//",        # diagonal schraffiert
    "LUT": "\\",        # diagonal schraffiert
}


# ============================================================
# MUSKELN: Reihenfolge und Farben
# ============================================================
# Reihenfolge: von proximal nach distal (Körper von oben nach unten)
#   1. Gluteus Medius        – Hüfte
#   2. Vastus Lateralis      – Oberschenkel vorne
#   3. Biceps Femoris        – Oberschenkel hinten
#   4. Semitendinosus        – Oberschenkel hinten (medial)
#   5. Gastrocnemius medial  – Unterschenkel

MUSCLE_NAMES = [
    "Gluteus Medius",
    "Vastus Lateralis",
    "Biceps Femoris",
    "Semitendinosus",
    "Gastrocnemius medial",
]

# Farben: barrierefrei, keine Überschneidung mit PHASE_COLORS
MUSCLE_COLORS = {
    "Gluteus Medius":       "#029E73",   # Grün
    "Vastus Lateralis":     "#DE8F05",   # Orange
    "Biceps Femoris":       "#CA9161",   # Braun
    "Semitendinosus":       "#ECE133",   # Gelb
    "Gastrocnemius medial": "#56B4E9",   # Hellblau
}

MUSCLE_LINESTYLES = {
    "Gluteus Medius":       "-",         # durchgezogen
    "Vastus Lateralis":     "--",        # gestrichelt
    "Biceps Femoris":       "-.",        # Strich-Punkt
    "Semitendinosus":       ":",         # gepunktet
    "Gastrocnemius medial": (0, (5, 1)), # eng gestrichelt
}


# ============================================================
# BASELINE-MODUS
# ============================================================
# Hier umschalten: 'SQ' oder 'DJ'
BASELINE_MODE = 'SQ'

# Soll die Baseline-Übung auch als eigenständige Übung ausgewertet werden?
#   True  = Baseline-Übung wird NICHT ausgewertet (Standard)
#   False = Baseline-Übung wird trotzdem mit ausgewertet
EXCLUDE_BASELINE_FROM_ANALYSIS = True

# Zuordnung: Mode → (Übungsordner, Seitenordner)
BASELINE_FOLDERS = {
    'SQ': ('SQ', 'BILATERAL'),
    'DJ': ('DJ', 'BILATERAL'),
}


# ============================================================
# ÜBUNGEN: TRIAL-SELEKTION (Skript 05)
# ============================================================
# Alle möglichen Übungen für die Trial-Selektion (bester Trial).
# Die Baseline-Übung wird automatisch ausgeschlossen.

_ALL_TRIAL_SELECTION_CONFIGS = [
    {"exercise": "CMJ", "side": "BILATERAL", "method": "jumpheight",
     "label": "CMJ bilateral"},
    {"exercise": "CMJ", "side": "RIGHT",     "method": "jumpheight",
     "label": "CMJ einbeinig R"},
    {"exercise": "DJ",  "side": "BILATERAL", "method": "jumpheight",
     "label": "DJ bilateral"},
    {"exercise": "SQ",  "side": "BILATERAL", "method": "knee_angle",
     "label": "SQ bilateral"},
    {"exercise": "SQ",  "side": "RIGHT",     "method": "knee_angle",
     "label": "SQ einbeinig R"},
]

_baseline_ex, _baseline_side = BASELINE_FOLDERS[BASELINE_MODE]

def _is_baseline(exercise: str, side: str) -> bool:
    """Prüft ob eine Übung die aktuelle Baseline-Übung ist."""
    return exercise == _baseline_ex and side == _baseline_side

TRIAL_SELECTION_CONFIGS = [
    cfg for cfg in _ALL_TRIAL_SELECTION_CONFIGS
    if not (EXCLUDE_BASELINE_FROM_ANALYSIS and _is_baseline(cfg["exercise"], cfg["side"]))
]


# ============================================================
# ÜBUNGEN: PIPELINE-VERARBEITUNG (Notebook 03)
# ============================================================
# Welche Übungen durch die EMG-Pipeline verarbeitet werden.

_ALL_PIPELINE_CONFIGS = [
    {'exercise': 'CMJ', 'side_folder': 'BILATERAL', 'label': 'CMJ bilateral'},
    {'exercise': 'CMJ', 'side_folder': 'RIGHT',     'label': 'CMJ einbeinig'},
    {'exercise': 'DJ',  'side_folder': 'BILATERAL', 'label': 'DJ bilateral'},
    {'exercise': 'SQ',  'side_folder': 'BILATERAL', 'label': 'SQ bilateral'},
    {'exercise': 'SQ',  'side_folder': 'RIGHT',     'label': 'SQ einbeinig'},
]

EXERCISE_CONFIGS = [
    cfg for cfg in _ALL_PIPELINE_CONFIGS
    if not (EXCLUDE_BASELINE_FROM_ANALYSIS and _is_baseline(cfg['exercise'], cfg['side_folder']))
]


# ============================================================
# EXERCISE_MAP: Label → (Ordner, Seite)
# ============================================================
# Verwendet in Skripten 06 und 07 für Feature-Extraktion und Plots.

_ALL_EXERCISE_MAP = {
    "CMJ bilateral":   ("CMJ", "BILATERAL"),
    "CMJ einbeinig R": ("CMJ", "RIGHT"),
    "DJ bilateral":    ("DJ",  "BILATERAL"),
    "SQ bilateral":    ("SQ",  "BILATERAL"),
    "SQ einbeinig R":  ("SQ",  "RIGHT"),
}

EXERCISE_MAP = {
    label: folders for label, folders in _ALL_EXERCISE_MAP.items()
    if not (EXCLUDE_BASELINE_FROM_ANALYSIS and _is_baseline(folders[0], folders[1]))
}


# ============================================================
# INFO-AUSGABE
# ============================================================

def print_config():
    """Gibt die aktuelle Konfiguration aus."""
    excl = "ja" if EXCLUDE_BASELINE_FROM_ANALYSIS else "nein (wird mit ausgewertet)"
    print(f"Baseline-Modus       : {BASELINE_MODE} ({_baseline_ex}/{_baseline_side})")
    print(f"Baseline ausschließen: {excl}")
    print(f"Pipeline-Übungen     : {[c['label'] for c in EXERCISE_CONFIGS]}")
    print(f"Trial-Selektion      : {[c['label'] for c in TRIAL_SELECTION_CONFIGS]}")
    print(f"Exercise-Map         : {list(EXERCISE_MAP.keys())}")


if __name__ == "__main__":
    print_config()