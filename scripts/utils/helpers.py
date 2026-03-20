from pathlib import Path
import re


def get_subject_id_from_filename(file_path: str | Path) -> str:
    """
    Extrahiert die Subject-ID aus einem Dateinamen zb aus:
    S01_01_PER_CMJ.csv
    """
    file_name = Path(file_path).stem
    return file_name.split("_")[0]

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