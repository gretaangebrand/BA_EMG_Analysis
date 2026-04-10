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
# BASELINE-MODUS
# ============================================================
# Hier umschalten: 'SQ' oder 'DJ'
BASELINE_MODE = 'DJ'

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

TRIAL_SELECTION_CONFIGS = [
    cfg for cfg in _ALL_TRIAL_SELECTION_CONFIGS
    if not (cfg["exercise"] == _baseline_ex and cfg["side"] == _baseline_side)
]


# ============================================================
# ÜBUNGEN: PIPELINE-VERARBEITUNG (Notebook 03)
# ============================================================
# Welche Übungen durch die EMG-Pipeline verarbeitet werden.
# Die Baseline-Übung wird automatisch ausgeschlossen.

_ALL_PIPELINE_CONFIGS = [
    {'exercise': 'CMJ', 'side_folder': 'BILATERAL', 'label': 'CMJ bilateral'},
    {'exercise': 'CMJ', 'side_folder': 'RIGHT',     'label': 'CMJ einbeinig'},
    {'exercise': 'DJ',  'side_folder': 'BILATERAL', 'label': 'DJ bilateral'},
    {'exercise': 'SQ',  'side_folder': 'BILATERAL', 'label': 'SQ bilateral'},
    {'exercise': 'SQ',  'side_folder': 'RIGHT',     'label': 'SQ einbeinig'},
]

EXERCISE_CONFIGS = [
    cfg for cfg in _ALL_PIPELINE_CONFIGS
    if not (cfg['exercise'] == _baseline_ex and cfg['side_folder'] == _baseline_side)
]


# ============================================================
# EXERCISE_MAP: Label → (Ordner, Seite)
# ============================================================
# Verwendet in Skripten 06 und 07 für Feature-Extraktion und Plots.
# Die Baseline-Übung wird automatisch ausgeschlossen.

_ALL_EXERCISE_MAP = {
    "CMJ bilateral":   ("CMJ", "BILATERAL"),
    "CMJ einbeinig R": ("CMJ", "RIGHT"),
    "DJ bilateral":    ("DJ",  "BILATERAL"),
    "SQ bilateral":    ("SQ",  "BILATERAL"),
    "SQ einbeinig R":  ("SQ",  "RIGHT"),
}

EXERCISE_MAP = {
    label: folders for label, folders in _ALL_EXERCISE_MAP.items()
    if not (folders[0] == _baseline_ex and folders[1] == _baseline_side)
}


# ============================================================
# INFO-AUSGABE
# ============================================================

def print_config():
    """Gibt die aktuelle Konfiguration aus."""
    print(f"Baseline-Modus       : {BASELINE_MODE} ({_baseline_ex}/{_baseline_side})")
    print(f"Baseline ausgeschl.  : {_baseline_ex}/{_baseline_side}")
    print(f"Pipeline-Übungen     : {[c['label'] for c in EXERCISE_CONFIGS]}")
    print(f"Trial-Selektion      : {[c['label'] for c in TRIAL_SELECTION_CONFIGS]}")
    print(f"Exercise-Map         : {list(EXERCISE_MAP.keys())}")


if __name__ == "__main__":
    print_config()