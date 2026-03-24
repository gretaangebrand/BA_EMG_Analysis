from pathlib import Path
import re
import shutil

# Skript durchsucht zwei Quellordner mit Rohdaten (_EMG_ORIGINAL und _EMG_RAW), erkennt automatisch, zu welcher Versuchsperson (Subject), 
# Phase und Bewegungsart jede Textdatei gehört, benennt die Datei anonymisiert um und speichert sie als 
# CSV‑Datei in einer neuen, strukturierten Ordnerhierarchie.

# ============================================================
# 1) PFADE
# ============================================================
# Hauptordner
SOURCE_FOLDER = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\raw_data")
# Unterordner
emg_original_path = SOURCE_FOLDER / "_EMG_ORIGINAL"
emg_raw_path = SOURCE_FOLDER / "_EMG_RAW"
SOURCE_FOLDERS = [emg_original_path, emg_raw_path]

# Outputordner
OUTPUT_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\anonymized_csv_data")


# ============================================================
# 2) MAPPING: participant_id -> subject_id
# ============================================================

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

# ============================================================
# 3) PHASEN-MAPPING
# ============================================================

PHASE_MAP = {
    "01_period": "01_PER",
    "02_ovulation": "02_OVU",
    "03_luteal": "03_LUT",
}


# ============================================================
# 4) HILFSFUNKTIONEN
# ============================================================

def detect_phase(file_path: Path) -> str | None:
    """
    Erkennt die Phase über einen der übergeordneten Ordner.
    """
    for part in file_path.parts:
        if part in PHASE_MAP:
            return PHASE_MAP[part]
    return None


def detect_subject(filename: str) -> str | None:
    """
    Sucht participant_id im Dateinamen und übersetzt zu subject_id.
    Es wird der längste passende Key zuerst geprüft, damit z.B.
    P01_Batzner vor P01 erkannt wird.
    """
    for participant_id in sorted(SUBJECT_MAP.keys(), key=len, reverse=True):
        if participant_id.lower() in filename.lower():
            return SUBJECT_MAP[participant_id]
    return None


def detect_movement(file_path: Path) -> str | None:
    """
    Erkennt Bewegung über den Ordnernamen.
    """
    folder_map = {
        "cmj": "CMJ",
        "dj": "DJ",
        "squatting": "SQ",
    }

    for part in file_path.parts:
        part_lower = part.lower()
        if part_lower in folder_map:
            return folder_map[part_lower]

    return None


def ensure_output_folder(base_output: Path, phase: str, movement: str) -> Path:
    """
    Erstellt Zielordner, falls nicht vorhanden.
    """
    target_dir = base_output / phase / movement
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def convert_txt_to_csv(source_file: Path, target_file: Path):
    """
    Einfache Umwandlung: Inhalt wird 1:1 übernommen, nur Dateiendung wird .csv.
    Das ist sinnvoll, wenn die .txt-Datei bereits tabellarisch strukturiert ist
    (z.B. tab-separiert oder komma-separiert).

    Falls deine Dateien ein spezielles Format haben, kann man das später anpassen.
    """
    text = source_file.read_text(encoding="utf-8", errors="ignore")
    target_file.write_text(text, encoding="utf-8")


# ============================================================
# 5) HAUPTSKRIPT
# ============================================================

def main():
    txt_files = []

    for folder in SOURCE_FOLDERS:
        if folder.exists():
            txt_files.extend(folder.rglob("*.txt"))
        else:
            print(f"[WARNUNG] Ordner nicht gefunden: {folder}")

    if not txt_files:
        print("[INFO] Keine .txt-Dateien gefunden.")
        return

    print(f"[INFO] Gefundene .txt-Dateien: {len(txt_files)}")


    # Protokoll für problematische Dateien
    created_count = 0
    skipped_files = []

    for txt_file in sorted(txt_files):
        filename = txt_file.name

        phase = detect_phase(txt_file)
        subject = detect_subject(filename)
        movement = detect_movement(txt_file)

        missing = []
        if phase is None:
            missing.append("Phase")
        if subject is None:
            missing.append("Subject")
        if movement is None:
            missing.append("Movement")

        if missing:
            skipped_files.append((txt_file, missing))
            print(f"[ÜBERSPRUNGEN] {txt_file}")
            print(f"   -> Nicht erkannt: {', '.join(missing)}")
            continue

        new_filename = f"{subject}_{phase}_{movement}.csv"

        target_dir = ensure_output_folder(OUTPUT_DIR, phase, movement)
        target_file = target_dir / new_filename

        convert_txt_to_csv(txt_file, target_file)
        created_count += 1

        print(f"[OK] {txt_file.name} -> {target_file}")

    print("\n========================")
    print("FERTIG")
    print("========================")
    print(f"Erzeugte Dateien: {created_count}")
    print(f"Übersprungene Dateien: {len(skipped_files)}")

    if skipped_files:
        print("\nProblematische Dateien:")
        for file_path, reasons in skipped_files:
            print(f"- {file_path}")
            print(f"  Fehlend: {', '.join(reasons)}")


if __name__ == "__main__":
    main()