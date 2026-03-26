import pandas as pd
from pathlib import Path, PureWindowsPath
import re
from collections import defaultdict




"""
04_check_missing_files.py
=========================
Vergleicht die Vicon-Metadaten (c3d_metadata_export.xlsx) mit den tatsaechlich
vorhandenen CSV-Dateien in anonymized_csv_data und erstellt einen Excel-Bericht
ueber fehlende Dateien.

Kernlogik:
  - Der Phase-Key (01_PER / 02_OVU / 03_LUT) kommt aus dem Dateinamen der CSV
    (z.B. S01_02_OVU_CMJ.csv) – NICHT aus der chronologischen Reihenfolge der
    Sitzungsdaten. Die Sitzungsdaten in der Metadata werden stattdessen verwendet
    um herauszufinden, welche c3d-Trials pro Phase erwartet werden.
  - Der Abgleich erfolgt auf Trial-Ebene: jede CSV wird geoeffnet, die c3d-Stems
    aus Zeile 0 extrahiert, und mit der Metadata verglichen.
  - Da in einer CSV alle Trials einer Session + Exercise stecken, gilt:
    Metadata-Datum -> Phase wird durch den Dateinamen der CSV beantwortet
    (weil du die CSVs korrekt mit der Phase benannt hast).
"""

# ============================================================
# PFADE 
# ============================================================
RAW_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\anonymized_csv_data")
METADATA_XLS = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\c3d_metadata_export.xlsx")
REPORT_PATH  = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\anonymized_csv_data\preprocessing_summary.xlsx")

# ============================================================
# KONSTANTEN
# ============================================================

SUBJECT_MAP = {
    "K_P01": "S01",  "A_P02": "S02",  "A_P05": "S03",  "D_P06": "S04",
    "P_P07": "S05",  "B_P09": "S06",  "P_P10": "S07",
    "Batzner_P01": "S08",  "Batzner_P01_JH": "S08",
    "Lorenz_P02":  "S09",  "Feik_P03": "S10",  "Platzer_P04": "S11",
}

TARGET_EXERCISES = [
    "Counter-movement jump session",
    "Counter-movement jump session_2",
    "Drop jump session",
    "Drop jump session_2",
    "Squatting session",
    "Squatting session_3",
]


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def exercise_to_short(exercise_name: str) -> str:
    """Vicon Exercise-Name -> Kurzkuerzel (CMJ / DJ / SQ)."""
    e = exercise_name.lower()
    if "counter" in e: return "CMJ"
    if "drop"    in e: return "DJ"
    if "squat"   in e: return "SQ"
    return "OTHER"


def stem_to_short(c3d_stem: str) -> str:
    """c3d-Dateiname-Stem -> Kurzkuerzel."""
    s = c3d_stem.upper()
    if "COUNTER-MOVEMENT JUMP" in s: return "CMJ"
    if "DROP JUMP"             in s: return "DJ"
    if "SQUAT"                 in s: return "SQ"
    return "OTHER"


def read_c3d_stems_from_csv(csv_path: Path) -> set[str]:
    """
    Oeffnet eine CSV, liest Zeile 0 (die Trial-Pfade) und gibt
    die Dateinamen-Stems (ohne .c3d) als Upper-Case-Set zurueck.
    Funktioniert auch wenn die Pfade Windows-Backslashes enthalten.
    """
    try:
        row0 = pd.read_csv(csv_path, header=None, nrows=1, low_memory=False).iloc[0].dropna()
        stems = set()
        for val in row0:
            s = str(val)
            if ".c3d" in s.lower():
                stems.add(PureWindowsPath(s).stem.upper())
        return stems
    except Exception as e:
        print(f"  [WARNUNG] Konnte {csv_path.name} nicht lesen: {e}")
        return set()


def parse_csv_filename(filename: str) -> tuple[str, str, str] | None:
    """
    Parst den Dateinamen einer anonymisierten CSV.
    Erwartet Format: <SubjectID>_<Phase>_<Exercise>.csv
    Beispiel: S01_02_OVU_CMJ.csv  ->  ('S01', '02_OVU', 'CMJ')
    Gibt None zurueck wenn das Format nicht passt.
    """
    stem = Path(filename).stem   # z.B. "S01_02_OVU_CMJ"
    parts = stem.split("_")
    # Mindest-Format: S01_02_OVU_CMJ -> ['S01','02','OVU','CMJ']
    if len(parts) < 4:
        return None
    subject_id  = parts[0]                    # S01
    phase       = f"{parts[1]}_{parts[2]}"    # 02_OVU
    exercise    = parts[3]                    # CMJ
    return subject_id, phase, exercise


# ============================================================
# SCHRITT 1: Metadaten laden und aufbereiten
# ============================================================

def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Laedt die Metadaten-Excel, filtert auf relevante Subjects + Exercises
    und gibt einen DataFrame zurueck mit Spalten:
      subject_id, SESSION_DATE, EXERCISE, exercise_short, c3d_stem_upper
    """
    df = pd.read_excel(metadata_path)

    df["subject_id"] = df["PARTICIPANT"].map(SUBJECT_MAP)
    df = df[df["subject_id"].notna() & df["EXERCISE"].isin(TARGET_EXERCISES)].copy()

    df["exercise_short"]  = df["EXERCISE"].apply(exercise_to_short)
    df["c3d_stem_upper"]  = df["C3D_FILENAME"].apply(lambda x: Path(str(x)).stem.upper())

    return df[["subject_id", "SESSION_DATE", "EXERCISE", "exercise_short",
               "c3d_stem_upper", "C3D_FILENAME"]].reset_index(drop=True)


# ============================================================
# SCHRITT 2: Phase-Datum-Mapping aus den CSV-Dateien ableiten
# ============================================================

def build_phase_date_map(csv_files: list[Path], df_meta: pd.DataFrame) -> dict:
    """
    Liest jede CSV-Datei aus, schaut welche c3d-Stems drin sind, und
    matcht diese gegen die Metadata um das Sitzungsdatum zu ermitteln.

    Gibt zurueck:
      { (subject_id, phase, exercise_short) : SESSION_DATE }

    Damit wissen wir: diese CSV gehoert zu diesem Metadata-Sitzungsdatum.
    """
    phase_date_map = {}

    for csv_path in csv_files:
        parsed = parse_csv_filename(csv_path.name)
        if parsed is None:
            continue
        subject_id, phase, exercise_short = parsed
        if subject_id not in {v for v in SUBJECT_MAP.values()}:
            continue

        stems_in_csv = read_c3d_stems_from_csv(csv_path)
        if not stems_in_csv:
            continue

        # Finde das Sitzungsdatum in der Metadata das zu diesen Stems passt
        candidates = df_meta[
            (df_meta["subject_id"]       == subject_id) &
            (df_meta["exercise_short"]   == exercise_short) &
            (df_meta["c3d_stem_upper"].isin(stems_in_csv))
        ]

        if candidates.empty:
            print(f"  [INFO] Kein Metadata-Match fuer {csv_path.name}")
            continue

        # Alle Stems sollten aus derselben Sitzung kommen
        dates = candidates["SESSION_DATE"].unique()
        if len(dates) > 1:
            print(f"  [WARNUNG] {csv_path.name} matcht mehrere Sitzungsdaten: {dates}")

        session_date = dates[0]
        key = (subject_id, phase, exercise_short)
        phase_date_map[key] = session_date

    return phase_date_map


# ============================================================
# SCHRITT 3: Vergleich Metadata vs. CSV-Inhalt
# ============================================================

def find_missing_trials(
    csv_files: list[Path],
    df_meta: pd.DataFrame,
    phase_date_map: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Vergleicht pro CSV-Datei die enthaltenen c3d-Stems mit den erwarteten
    Stems aus der Metadata.

    Gibt zurueck:
      missing_trials: Trials die laut Metadata existieren sollten, aber fehlen
      extra_trials:   Trials die in der CSV sind, aber nicht in der Metadata
    """
    missing_trials = []
    extra_trials   = []

    for csv_path in csv_files:
        parsed = parse_csv_filename(csv_path.name)
        if parsed is None:
            continue
        subject_id, phase, exercise_short = parsed

        key = (subject_id, phase, exercise_short)
        session_date = phase_date_map.get(key)

        if session_date is None:
            print(f"  [INFO] Kein Sitzungsdatum bekannt fuer {csv_path.name} – uebersprungen")
            continue

        # Erwartete Stems aus der Metadata fuer diese Sitzung
        expected = set(
            df_meta.loc[
                (df_meta["subject_id"]     == subject_id) &
                (df_meta["exercise_short"] == exercise_short) &
                (df_meta["SESSION_DATE"]   == session_date),
                "c3d_stem_upper"
            ]
        )

        # Tatsaechliche Stems in der CSV
        actual = read_c3d_stems_from_csv(csv_path)

        # Vergleich
        missing = expected - actual
        extra   = actual - expected

        for stem in sorted(missing):
            # Originalzeilendaten fuer den Bericht
            meta_row = df_meta[
                (df_meta["subject_id"]     == subject_id) &
                (df_meta["exercise_short"] == exercise_short) &
                (df_meta["SESSION_DATE"]   == session_date) &
                (df_meta["c3d_stem_upper"] == stem)
            ]
            c3d_filename = meta_row["C3D_FILENAME"].iloc[0] if not meta_row.empty else stem
            missing_trials.append({
                "CSV_Datei":      csv_path.name,
                "Subject_ID":     subject_id,
                "Phase":          phase,
                "Exercise":       exercise_short,
                "Sitzungsdatum":  str(session_date)[:10],
                "Fehlender_Trial": c3d_filename,
                "c3d_Stem":       stem,
            })

        for stem in sorted(extra):
            extra_trials.append({
                "CSV_Datei":   csv_path.name,
                "Subject_ID":  subject_id,
                "Phase":       phase,
                "Exercise":    exercise_short,
                "Extra_Trial": stem,
                "Hinweis":     "In CSV gefunden, aber nicht in Metadata",
            })

    return missing_trials, extra_trials


# ============================================================
# SCHRITT 4: Fehlende CSV-Dateien pruefen
# ============================================================

def find_missing_csv_files(
    csv_files: list[Path],
    df_meta: pd.DataFrame,
    phase_date_map: dict,
) -> list[dict]:
    """
    Prueft ob fuer alle Subject + Phase + Exercise Kombinationen
    eine CSV-Datei vorhanden ist.
    """
    existing_keys = set()
    for csv_path in csv_files:
        parsed = parse_csv_filename(csv_path.name)
        if parsed:
            existing_keys.add(parsed)

    # Alle Kombinationen die laut Metadata existieren sollten
    missing_csvs = []
    for (subject_id, phase, exercise_short), _ in phase_date_map.items():
        if (subject_id, phase, exercise_short) not in existing_keys:
            missing_csvs.append({
                "Subject_ID": subject_id,
                "Phase":      phase,
                "Exercise":   exercise_short,
                "Erwartete_CSV": f"{subject_id}_{phase}_{exercise_short}.csv",
                "Hinweis": "CSV-Datei fehlt komplett",
            })

    return missing_csvs


# ============================================================
# HAUPTFUNKTION
# ============================================================

def main():
    print("=" * 60)
    print("Starte Lueckenanalyse: Metadata vs. CSV-Dateien")
    print("=" * 60)

    # Dateien sammeln
    all_csvs = sorted(RAW_DIR.rglob("*.csv"))
    print(f"\nGefundene CSV-Dateien in {RAW_DIR.name}: {len(all_csvs)}")
    if not all_csvs:
        print("[FEHLER] Keine CSV-Dateien gefunden. Pfad pruefen.")
        return

    # Metadaten laden
    print(f"\nLade Metadaten: {METADATA_XLS.name} ...")
    df_meta = load_metadata(METADATA_XLS)
    print(f"Relevante Metadata-Zeilen: {len(df_meta)}")

    # Phase-Datum-Mapping aufbauen
    print("\nOrdne CSV-Dateien ihren Sitzungsdaten zu ...")
    phase_date_map = build_phase_date_map(all_csvs, df_meta)
    print(f"Gemappte CSV-Sitzungen: {len(phase_date_map)}")

    # Fehlende Trials innerhalb vorhandener CSVs
    print("\nPruefe Trial-Vollstaendigkeit in jeder CSV ...")
    missing_trials, extra_trials = find_missing_trials(all_csvs, df_meta, phase_date_map)

    # Fehlende CSV-Dateien insgesamt
    missing_csvs = find_missing_csv_files(all_csvs, df_meta, phase_date_map)

    # Ergebnisse ausgeben
    print("\n" + "=" * 60)
    print("ERGEBNIS")
    print("=" * 60)

    if not missing_csvs and not missing_trials:
        print("\n[OK] Alle Dateien und Trials vollstaendig vorhanden!")
    else:
        if missing_csvs:
            print(f"\n[!] {len(missing_csvs)} fehlende CSV-Dateien (komplett):")
            for r in missing_csvs:
                print(f"    {r['Erwartete_CSV']}")

        if missing_trials:
            print(f"\n[!] {len(missing_trials)} fehlende Trials in vorhandenen CSVs:")
            for r in missing_trials:
                print(f"    {r['CSV_Datei']}  ->  fehlt: {r['Fehlender_Trial']}")

    if extra_trials:
        print(f"\n[i] {len(extra_trials)} Extra-Trials (in CSV, nicht in Metadata):")
        for r in extra_trials:
            print(f"    {r['CSV_Datei']}  ->  extra: {r['Extra_Trial']}")

    # Excel-Bericht speichern
    print(f"\nSpeichere Bericht: {REPORT_PATH}")
    with pd.ExcelWriter(REPORT_PATH, engine="openpyxl") as writer:
        if missing_csvs:
            pd.DataFrame(missing_csvs).to_excel(writer, sheet_name="Fehlende_CSV_Dateien", index=False)
        if missing_trials:
            pd.DataFrame(missing_trials).to_excel(writer, sheet_name="Fehlende_Trials", index=False)
        if extra_trials:
            pd.DataFrame(extra_trials).to_excel(writer, sheet_name="Extra_Trials", index=False)

        # Zusammenfassung
        summary = pd.DataFrame([{
            "Gefundene_CSV_Dateien":    len(all_csvs),
            "Gemappte_Sitzungen":       len(phase_date_map),
            "Fehlende_CSV_Dateien":     len(missing_csvs),
            "Fehlende_Trials_in_CSVs":  len(missing_trials),
            "Extra_Trials":             len(extra_trials),
        }])
        summary.to_excel(writer, sheet_name="Zusammenfassung", index=False)

    print("\nFERTIG.")


if __name__ == "__main__":
    main()