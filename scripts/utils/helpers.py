from pathlib import Path
import re
# import numpy as np
# import pandas as pd

# kleine, universelle Tools (z. B. Sampling‑Rate‑Lookup, Scaling, Parsing-Funktionen)
#---------------------------------------------------------------------------
# Konstante Zuordnungen

SUBJECT_MAP = {
    "P01_K": "S01",
    "P02_A": "S02",
    "P05_A": "S03",
    "P06_D": "S04",
    "P07_P": "S05",
    "P09_B": "S06",
    "P10_P": "S07",
    "P01_Batzner": "S08",
    "P02_Lorenz": "S09",
    "P03_Feik": "S10",
    "P04_Platzer": "S11",
}

PHASE_MAP = {
    "01_period": "01_PER",
    "02_ovulation": "02_OVU",
    "03_luteal": "03_LUT",
}


# Sampling-Frequenz per Subject-ID mappen
def get_sampling_rate_for_subject(subject_id: str) -> int:
    """
    Gibt die Sampling-Rate in Hz für jede Versuchsperson zurück.
    Standard: 2000 Hz
    Ausnahmen: P01_K, P02_A, P05_A, P06_D -> 2100 Hz
    """
    high_fs_subjects = {"S01", "S02", "S03", "S04"}  # deine anonymisierten IDs
    return 2100 if subject_id in high_fs_subjects else 2000


def get_subject_id_from_filename(file_path: str | Path) -> str:
    """
    Extrahiert die Subject-ID aus einem Dateinamen zb aus:
    S01_01_PER_CMJ.csv
    """
    file_name = Path(file_path).stem
    return file_name.split("_")[0]

# ============================================================
# Erkennung von Phase, Subject und Bewegung
# ============================================================

def detect_phase(file_path: Path) -> str | None:
    """
    Erkennt die Phase über einen der übergeordneten Ordner.
    Erwartet, dass Ordnernamen wie '01_period', '02_ovulation', '03_luteal' enthalten sind.
    """

    for part in file_path.parts:
        if part in PHASE_MAP:
            return PHASE_MAP[part]
    return None


def detect_subject(filename: str) -> str | None:
    """
    Sucht participant_id im Dateinamen und übersetzt zu subject_id.
    Prüft längsten Key zuerst (z. B. P01_Batzner vor P01).
    """
    for pid in sorted(SUBJECT_MAP.keys(), key=len, reverse=True):
        if pid.lower() in filename.lower():
            return SUBJECT_MAP[pid]
    return None


def detect_movement(file_path: Path) -> str | None:
    """
    Erkennt Bewegung anhand des Ordnernamens.
    """
    folder_map = {
        "cmj": "CMJ",
        "dj": "DJ",
        "squatting": "SQ",
    }
    for part in file_path.parts:
        low = part.lower()
        if low in folder_map:
            return folder_map[low]
    return None


# Bilaterale Übungen, nur Übung selbst und Trail Nummer im Namen
def get_bilateral_trials(trial_names):
    return [
        t for t in trial_names
        if "Left" not in str(t) and "Right" not in str(t)
    ]

# Einbeinige Übungen, Übunge selbst, Trail Nummer und Seite im Namen
def get_left_trials(trial_names):
    return [t for t in trial_names if "Left" in str(t)]

# Einbeinige Übungen, Übunge selbst, Trail Nummer und Seite im Namen
def get_right_trials(trial_names):
    return [t for t in trial_names if "Right" in str(t)]


def apply_s08_scaling(df, subject_id: str):
    """
    Skaliert nur für S08 die EMG-Daten.
    """
    if subject_id == "S08":
        return df * 1_000_000
    return df

def make_short_trial_name(trial_name: str) -> str:
    """
    Wandelt lange Trialnamen/Pfade in kurze, saubere Namen um.

    Beispiele:
    Counter-Movement Jump 1.c3d       -> CMJ_01
    Counter-Movement Jump Left 2.c3d  -> CMJ_L_02
    Counter-Movement Jump Right 3.c3d -> CMJ_R_03
    Drop Jump 1.c3d                   -> DJ_01
    Drop Jump Left 2.c3d              -> DJ_L_02
    Squat 3.c3d                       -> SQ_03
    """
    name = str(trial_name)

    # Nur den Dateinamen bzw. letzten Teil behalten
    last_part = Path(name).name

    # .c3d entfernen
    last_part = re.sub(r"\.c3d$", "", last_part, flags=re.IGNORECASE)

    # CMJ
    if "Counter-Movement Jump Left" in last_part:
        match = re.search(r"(\d+)$", last_part)
        number = int(match.group(1)) if match else 0
        return f"CMJ_L_{number:02d}"

    if "Counter-Movement Jump Right" in last_part:
        match = re.search(r"(\d+)$", last_part)
        number = int(match.group(1)) if match else 0
        return f"CMJ_R_{number:02d}"

    if "Counter-Movement Jump" in last_part:
        match = re.search(r"(\d+)$", last_part)
        number = int(match.group(1)) if match else 0
        return f"CMJ_{number:02d}"

    # DJ
    if "Drop Jump Left" in last_part:
        match = re.search(r"(\d+)$", last_part)
        number = int(match.group(1)) if match else 0
        return f"DJ_L_{number:02d}"

    if "Drop Jump Right" in last_part:
        match = re.search(r"(\d+)$", last_part)
        number = int(match.group(1)) if match else 0
        return f"DJ_R_{number:02d}"

    if "Drop Jump" in last_part:
        match = re.search(r"(\d+)$", last_part)
        number = int(match.group(1)) if match else 0
        return f"DJ_{number:02d}"

    # Squat
    if "Squat Left" in last_part:
        match = re.search(r"(\d+)$", last_part)
        number = int(match.group(1)) if match else 0
        return f"SQ_L_{number:02d}"

    if "Squat Right" in last_part:
        match = re.search(r"(\d+)$", last_part)
        number = int(match.group(1)) if match else 0
        return f"SQ_R_{number:02d}"

    if "Squat" in last_part:
        match = re.search(r"(\d+)$", last_part)
        number = int(match.group(1)) if match else 0
        return f"SQ_{number:02d}"

    # Fallback
    return last_part