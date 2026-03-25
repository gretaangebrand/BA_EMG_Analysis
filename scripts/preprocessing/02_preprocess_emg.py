from pathlib import Path
import pandas as pd
import numpy as np

from scripts.utils.helpers import (
    get_sampling_rate_for_subject,
    get_subject_id_from_filename,
    detect_phase,
    apply_s08_scaling,
    make_short_trial_name,
)

# =============================================================================
# 02_preprocess_emg.py
#
# Liest anonymisierte CSV-Dateien aus dem Preprocessing-Schritt 01 ein,
# erkennt automatisch alle enthaltenen Trials und Muskeln direkt aus dem
# Header (Zeile 0 enthaelt den c3d-Dateipfad je Spalte), und speichert
# pro Trial eine saubere CSV mit time_s + EMG-Kanaelen.
#
# Ordnerstruktur des Outputs:
#   <OUTPUT_DIR>/<subject_id>/<phase>/<bewegung>/<seite>/<trial_kuerzel>.csv
#
# Beispiel:
#   preprocessed/S01/01_PER/CMJ/BILATERAL/CMJ_01.csv
#   preprocessed/S01/01_PER/CMJ/LEFT/CMJ_L_01.csv
# =============================================================================

# -----------------------------------------------------------------------------
# Pfade
# -----------------------------------------------------------------------------
SOURCE_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\anonymized_csv_data")
OUTPUT_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\preprocessed_emg_data")

# Anzahl Header-Zeilen je Subject-Gruppe
_HEADER_ROWS_DEFAULT  = 5   # S01-S07
_HEADER_ROWS_SPECIAL  = 4   # S08 (und ggf. S09-S11, hier erweiterbar)
_SPECIAL_SUBJECTS     = {"S08"}


# -----------------------------------------------------------------------------
# Schritt 1: Header analysieren
# -----------------------------------------------------------------------------

def _n_header_rows(subject_id: str) -> int:
    return _HEADER_ROWS_SPECIAL if subject_id in _SPECIAL_SUBJECTS else _HEADER_ROWS_DEFAULT


def read_header(file_path: Path, n_rows: int) -> pd.DataFrame:
    """Liest die ersten n_rows Zeilen als rohen String-DataFrame."""
    return pd.read_csv(file_path, header=None, nrows=n_rows, low_memory=False).fillna("")


def extract_trial_column_map(header: pd.DataFrame) -> dict[str, list[int]]:
    """
    Liest Zeile 0 (Trial-Pfade) und ermittelt pro einzigartigem Trial,
    welche Spalten-Indizes die EMG-Daten enthalten.

    Kriterien fuer EMG-Spalten:
      - Zeile 0 enthaelt den Trial-Pfad (*.c3d)
      - Zeile mit 'Typ' enthaelt 'ANALOG'
      - Zeile mit 'Subtyp' enthaelt 'EMG_RAW' (falls im File vorhanden)
        oder wird weggelassen (S08+), dann gilt: alle ANALOG-Spalten

    Gibt zurueck:
      { trial_stem_string: [spalten_index, ...] }
    z.B. { "Counter-Movement Jump 1": [66, 67, 68, ...] }
    """
    n_rows = len(header)

    row0 = header.iloc[0].astype(str)          # Trial-Pfade
    row_type = header.iloc[2].astype(str).str.upper()   # METRIC / ANALOG / ...

    # Subtyp-Zeile nur auswerten wenn sie existiert (5-Zeilen-Format)
    has_subtype_row = n_rows >= 4
    row_sub = header.iloc[3].astype(str).str.upper() if has_subtype_row else None

    # Pruefe ob EMG_RAW als Subtyp vorkommt (S01-S07)
    has_emg_raw = has_subtype_row and (row_sub == "EMG_RAW").any()

    # Trials in Reihenfolge ihres ersten Auftretens sammeln
    trial_order: list[str] = []
    seen_paths: set[str] = set()
    for val in row0:
        if ".c3d" in val.lower() and val not in seen_paths:
            seen_paths.add(val)
            trial_order.append(val)

    trial_column_map: dict[str, list[int]] = {}

    for trial_path in trial_order:
        trial_stem = Path(trial_path).stem  # z.B. "Counter-Movement Jump 1"

        if has_emg_raw:
            # S01-S07: nur Spalten die ANALOG + EMG_RAW sind
            col_indices = [
                i for i, v in enumerate(row0)
                if v == trial_path
                and row_type.iloc[i] == "ANALOG"
                and row_sub.iloc[i] == "EMG_RAW"
            ]
        else:
            # S08+: alle ANALOG-Spalten dieses Trials
            col_indices = [
                i for i, v in enumerate(row0)
                if v == trial_path and row_type.iloc[i] == "ANALOG"
            ]

        if not col_indices:
            print(f"  [WARNUNG] Keine EMG-Spalten fuer Trial '{trial_stem}' gefunden.")
            continue

        trial_column_map[trial_stem] = col_indices

    return trial_column_map


def extract_muscle_names(header: pd.DataFrame, col_indices: list[int]) -> list[str]:
    """Liest die Muskelnamen (Zeile 1) fuer die gegebenen Spalten-Indizes."""
    return header.iloc[1].astype(str).iloc[col_indices].tolist()


# -----------------------------------------------------------------------------
# Schritt 2: Daten laden und pro Trial extrahieren
# -----------------------------------------------------------------------------

def load_data(file_path: Path, n_header_rows: int) -> pd.DataFrame:
    """
    Laedt den Datenteil der CSV (alles nach den Header-Zeilen).
    Spalte 0 enthaelt den ITEM-Zeitschritt (wird nicht benoetigt).
    """
    return pd.read_csv(file_path, header=None, skiprows=n_header_rows, low_memory=False)


def extract_trial_df(
    data: pd.DataFrame,
    col_indices: list[int],
    muscle_names: list[str],
    subject_id: str,
    fs: float,
) -> pd.DataFrame:
    """
    Extrahiert die EMG-Spalten eines Trials aus dem Daten-DataFrame,
    benennt die Spalten nach den Muskeln und fuegt eine Zeitspalte hinzu.
    Wendet S08-Skalierung an falls noetig.
    """
    df = data.iloc[:, col_indices].copy()
    df.columns = muscle_names
    df = df.apply(pd.to_numeric, errors="coerce")
    df = apply_s08_scaling(df, subject_id)
    df.insert(0, "time_s", np.arange(len(df)) / fs)
    return df


# -----------------------------------------------------------------------------
# Schritt 3: Ordnerstruktur und Dateinamen ableiten
# -----------------------------------------------------------------------------

def trial_stem_to_folder_and_side(trial_stem: str) -> tuple[str, str]:
    """
    Leitet aus dem langen Trial-Namen (Stem des c3d-Dateipfads) den
    Bewegungsordner (CMJ / DJ / SQ / OTHER) und die Seite (BILATERAL /
    LEFT / RIGHT) ab.

    Eingabe:  'Counter-Movement Jump Left 2'
    Ausgabe:  ('CMJ', 'LEFT')
    """
    s = trial_stem.lower()

    # Seite
    if "left" in s:
        side = "LEFT"
    elif "right" in s:
        side = "RIGHT"
    else:
        side = "BILATERAL"

    # Bewegung
    if "counter-movement jump" in s or "counter movement jump" in s:
        movement = "CMJ"
    elif "drop jump" in s:
        movement = "DJ"
    elif "squat" in s:
        movement = "SQ"
    else:
        movement = "OTHER"

    return movement, side


# -----------------------------------------------------------------------------
# Hauptfunktion pro Datei
# -----------------------------------------------------------------------------

def preprocess_emg_file(file_path: Path) -> int:
    """
    Verarbeitet eine einzelne CSV-Datei vollstaendig:
      1. Metadaten aus Dateinamen lesen
      2. Header analysieren -> Trial-Spalten-Map aufbauen
      3. Daten laden
      4. Pro Trial EMG-Daten extrahieren und als CSV speichern

    Gibt die Anzahl der gespeicherten Trial-Dateien zurueck.
    """
    subject_id = get_subject_id_from_filename(file_path)
    phase      = detect_phase(file_path)
    fs         = get_sampling_rate_for_subject(subject_id)
    n_hdr      = _n_header_rows(subject_id)

    if phase is None:
        print(f"[FEHLER] Phase nicht erkannt: {file_path} – Datei wird uebersprungen.")
        return 0

    print(f"\n=== {file_path.name} | {subject_id} | {phase} | fs={fs} Hz ===")

    # Header und Daten einlesen
    header = read_header(file_path, n_hdr)
    trial_col_map = extract_trial_column_map(header)

    if not trial_col_map:
        print(f"  [FEHLER] Keine Trials gefunden – Datei wird uebersprungen.")
        return 0

    data = load_data(file_path, n_hdr)
    saved = 0

    for trial_stem, col_indices in trial_col_map.items():
        muscle_names = extract_muscle_names(header, col_indices)
        short_name   = make_short_trial_name(trial_stem)   # z.B. "CMJ_L_02"
        movement, side = trial_stem_to_folder_and_side(trial_stem)

        trial_df = extract_trial_df(data, col_indices, muscle_names, subject_id, fs)

        # NaN-Zeilen zaehlen (informativ, kein Abbruch)
        n_nan = trial_df.iloc[:, 1:].isna().all(axis=1).sum()
        if n_nan > 0:
            print(f"  [INFO] {short_name}: {n_nan} vollstaendige NaN-Zeilen (normal bei unterschiedlichen Trial-Laengen)")

        # Ausgabepfad
        out_dir = OUTPUT_DIR / subject_id / phase / movement / side
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{short_name}.csv"

        trial_df.to_csv(out_path, index=False)
        print(f"  [OK] {out_path.relative_to(OUTPUT_DIR)}")
        saved += 1

    return saved


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    all_files = sorted(SOURCE_DIR.rglob("*.csv"))
    print(f"Gefundene CSV-Dateien: {len(all_files)}")

    total_saved  = 0
    total_errors = 0

    for file_path in all_files:
        try:
            n = preprocess_emg_file(file_path)
            total_saved += n
        except Exception as e:
            print(f"[FEHLER] {file_path.name}: {e}")
            total_errors += 1

    print(f"\n{'='*50}")
    print(f"FERTIG")
    print(f"  Gespeicherte Trial-Dateien : {total_saved}")
    print(f"  Dateien mit Fehler         : {total_errors}")
    print(f"  Output-Ordner              : {OUTPUT_DIR}")
    print(f"{'='*50}")
