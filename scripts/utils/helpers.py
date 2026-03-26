from pathlib import Path
import re

# =============================================================================
# helpers.py
#
# Kleine, universelle Hilfsfunktionen fuer die EMG-Pipeline.
# Wird von 01_convert_raw_data_into_csv.py und 02_preprocess_emg.py verwendet.
# =============================================================================


# -----------------------------------------------------------------------------
# Konstante Zuordnungen
# -----------------------------------------------------------------------------

SUBJECT_MAP = {
    # Rohdaten-Dateinamen -> anonymisierte Subject-ID
    "P01_K":       "S01",
    "P02_A":       "S02",
    "P05_A":       "S03",
    "P06_D":       "S04",
    "P07_P":       "S05",
    "P09_B":       "S06",
    "P10_P":       "S07",
    "P01_Batzner": "S08",
    "P02_Lorenz":  "S09",
    "P03_Feik":    "S10",
    "P04_Platzer": "S11",
    # Bereits anonymisierte Dateien (Passthrough)
    "S01": "S01", "S02": "S02", "S03": "S03", "S04": "S04",
    "S05": "S05", "S06": "S06", "S07": "S07", "S08": "S08",
    "S09": "S09", "S10": "S10", "S11": "S11",
}

PHASE_MAP = {
    # Ordnernamen -> Phasenkuerzel
    "01_period":    "01_PER",
    "02_ovulation": "02_OVU",
    "03_luteal":    "03_LUT",
    # Bereits anonymisierte Dateien (Passthrough)
    "01_PER": "01_PER",
    "02_OVU": "02_OVU",
    "03_LUT": "03_LUT",
}

# Subjects mit abweichender Sampling-Rate
_HIGH_FS_SUBJECTS = {"S01", "S02", "S03", "S04"}


# -----------------------------------------------------------------------------
# Sampling-Rate
# -----------------------------------------------------------------------------

def get_sampling_rate_for_subject(subject_id: str) -> int:
    """
    Gibt die Sampling-Rate in Hz fuer eine Subject-ID zurueck.
    Standard: 2000 Hz. Ausnahmen S01-S04: 2100 Hz.
    """
    return 2100 if subject_id in _HIGH_FS_SUBJECTS else 2000


# -----------------------------------------------------------------------------
# Parsing aus Dateinamen / Pfaden
# (wird in 01_convert_raw_data_into_csv.py verwendet)
# -----------------------------------------------------------------------------

def get_subject_id_from_filename(file_path: str | Path) -> str:
    """
    Extrahiert die Subject-ID aus dem Dateinamen.
    Erwartet Format wie: S01_01_PER_CMJ.csv
    """
    return Path(file_path).stem.split("_")[0]


def detect_phase(file_path: Path) -> str | None:
    """
    Erkennt die Zyklusphase anhand der Ordnernamen im Pfad.
    Erwartet Ordner wie '01_period', '02_ovulation', '03_luteal'
    oder bereits anonymisierte Kuerzel '01_PER' usw.
    """
    for part in file_path.parts:
        if part in PHASE_MAP:
            return PHASE_MAP[part]
    return None


def detect_subject(filename: str) -> str | None:
    """
    Sucht einen bekannten Participant-Key im Dateinamen und gibt die
    anonymisierte Subject-ID zurueck. Prueft laengere Keys zuerst,
    damit z.B. 'P01_Batzner' vor 'P01' matched.
    """
    for pid in sorted(SUBJECT_MAP.keys(), key=len, reverse=True):
        if pid.lower() in filename.lower():
            return SUBJECT_MAP[pid]
    return None


def detect_movement(file_path: Path) -> str | None:
    """
    Erkennt den Bewegungstyp aus Ordner- oder Dateinamen.
    Wird nur in 01_convert_raw_data_into_csv.py benoetigt,
    um den Output-Ordner zu bestimmen.

    Gibt einen der folgenden Strings zurueck (oder 'UNKNOWN_MOVEMENT'):
      'Counter-Movement Jump Bilateral'
      'Counter-Movement Jump Left'
      'Counter-Movement Jump Right'
      'Drop Jump Bilateral'
      'Squatting Bilateral'
      'Squatting Left'
      'Squatting Right'
    """
    s = str(file_path).lower()

    if "drop jump" in s or "_dj" in s:
        return "DJ"

    if "counter-movement jump" in s or "_cmj" in s:
        if "left" in s:
            return "Counter-Movement Jump Left"
        if "right" in s:
            return "Counter-Movement Jump Right"
        return "CMJ"

    if "squatting" in s or "squat" in s or "_sq" in s:
        if "left" in s:
            return "Squatting Left"
        if "right" in s:
            return "Squatting Right"
        return "SQ"

    print(f"[WARNUNG] Bewegung nicht erkannt in: {file_path.name}")
    return "UNKNOWN_MOVEMENT"



# -----------------------------------------------------------------------------
# Trial-Kurzname
# (wird in 02_preprocess_emg.py verwendet)
# -----------------------------------------------------------------------------

def make_short_trial_name(trial_name: str) -> str:
    """
    Wandelt einen langen Trial-Namen in ein kompaktes Kuerzel um.

    Eingabe: der Dateiname (mit oder ohne .c3d), z.B.:
      'Counter-Movement Jump 1.c3d'       -> 'CMJ_01'
      'Counter-Movement Jump Left 2.c3d'  -> 'CMJ_L_02'
      'Counter-Movement Jump Right 3.c3d' -> 'CMJ_R_03'
      'Drop Jump Bilateral 1.c3d'         -> 'DJ_01'
      'Squatting 3.c3d'                   -> 'SQ_03'
      'Squatting Left 1.c3d'              -> 'SQ_L_01'
    """
    # Nur den letzten Pfadteil / Dateinamen behalten und .c3d entfernen
    name = re.sub(r"\.c3d$", "", Path(str(trial_name)).name, flags=re.IGNORECASE).strip()

    def _num(s: str) -> int:
        m = re.search(r"(\d+)$", s)
        return int(m.group(1)) if m else 0

    # CMJ, hier gibt es Rechts und Links schon
    if re.search(r"Counter-Movement Jump Left", name, re.IGNORECASE):
        return f"CMJ_L_{_num(name):02d}"
    if re.search(r"Counter-Movement Jump Right", name, re.IGNORECASE):
        return f"CMJ_R_{_num(name):02d}"
    if re.search(r"Counter-Movement Jump", name, re.IGNORECASE):
        return f"CMJ_{_num(name):02d}"

    # DJ, gibt hier nur DJ Bilateral, RE und LI könnt eig raus, aber mal noch drinne gelassen
    if re.search(r"Drop Jump Left", name, re.IGNORECASE):
        return f"DJ_L_{_num(name):02d}"
    if re.search(r"Drop Jump Right", name, re.IGNORECASE):
        return f"DJ_R_{_num(name):02d}"
    if re.search(r"Drop Jump Bilateral", name, re.IGNORECASE):
        return f"DJ_{_num(name):02d}"

    # Squat, hier gibt es Rechts und Links schon
    if re.search(r"Squat(?:ting)? Left", name, re.IGNORECASE):
        return f"SQ_L_{_num(name):02d}"
    if re.search(r"Squat(?:ting)? Right", name, re.IGNORECASE):
        return f"SQ_R_{_num(name):02d}"
    if re.search(r"Squat(?:ting)?", name, re.IGNORECASE):
        return f"SQ_{_num(name):02d}"

    # Fallback: bereinigten Namen zurueckgeben
    return re.sub(r"\s+", "_", name)
