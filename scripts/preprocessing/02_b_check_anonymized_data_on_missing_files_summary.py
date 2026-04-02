"""
02_b_check_missing_files_summary.py
====================================
Vergleicht die Vicon-Metadaten (c3d_metadata_export.xlsx) mit den tatsaechlich
vorhandenen preprocessed Dateien in 03_preprocessed_emg_data.

Kernlogik:
  1. Aus der c3d-Metadata wird ermittelt, wie viele Trials pro Subject, Phase,
     Uebung und Seite laut Vicon-Aufnahme existieren SOLLTEN.
  2. Im preprocessed-Ordner wird gezaehlt, wie viele _emg.csv Dateien
     pro Subject/Phase/Uebung/Seite tatsaechlich vorhanden sind.
  3. Die Differenz ergibt die wirklich fehlenden Trials.

Ergebnis:
  - Excel-Bericht mit fehlenden Trials und Zusammenfassung
  - Konsolenausgabe mit Uebersicht

Relevante Seiten fuer die Auswertung: BILATERAL und RIGHT
(LEFT wird mitgezaehlt, aber separat ausgewiesen).
"""

from pathlib import Path
import pandas as pd
import numpy as np
# from collections import defaultdict


# ============================================================
# PFADE
# ============================================================
PREPROCESSED_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\03_preprocessed_emg_data")
METADATA_XLS     = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\c3d_metadata_export.xlsx")
REPORT_PATH      = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\Pipeline_Reports.xlsx")


# ============================================================
# KONSTANTEN
# ============================================================

SUBJECT_MAP = {
    "K_P01": "S01",  "A_P02": "S02",  "A_P05": "S03",  "D_P06": "S04",
    "P_P07": "S05",  "B_P09": "S06",  "P_P10": "S07",
    "Batzner_P01": "S08",  "Batzner_P01_JH": "S08",
    "Lorenz_P02":  "S09",  "Feik_P03": "S10",  "Platzer_P04": "S11",
}

# Nur relevante Subjects (die auch in der Pipeline verarbeitet werden)
RELEVANT_SUBJECTS = set(SUBJECT_MAP.values())

# Phasen-Mapping: Sitzungsdaten werden chronologisch den Phasen zugeordnet.
# Die Zuordnung muss pro Subject erfolgen (1. Datum = PER, 2. = OVU, 3. = LUT).
# AUSNAHME: S08 hat 4 Termine, die nicht chronologisch den Phasen entsprechen.
PHASE_ORDER = ["01_PER", "02_OVU", "03_LUT"]
PHASE_LABELS = {"01_PER": "PER", "02_OVU": "OVU", "03_LUT": "LUT"}

# Manuelle Phase-Zuordnung fuer Subjects, bei denen die chronologische
# Reihenfolge nicht PER -> OVU -> LUT entspricht.
# Format: { subject_id: { datetime.date: phase_key } }
# Termine die hier nicht gelistet sind, werden ignoriert.
from datetime import date
MANUAL_PHASE_OVERRIDE = {
    "S01": {
        date(2024, 6,  6): "03_LUT",
        date(2024, 6, 13): "01_PER",
        date(2024, 7,  1): "02_OVU",
    },
    "S06": {
        date(2024, 12, 13): "02_OVU",
        date(2025,  1, 20): "01_PER",
        date(2025,  2,  4): "03_LUT",
    },
    "S08": {
        date(2025, 2, 19): "02_OVU",
        date(2025, 2, 27): "03_LUT",
        date(2025, 3,  6): "01_PER",
        # 2024-11-13 wird bewusst NICHT zugeordnet -> ignoriert
    },
}

# Exercises und deren Zuordnung
TARGET_EXERCISES = [
    "Counter-movement jump session",
    "Counter-movement jump session_2",
    "Drop jump session",
    "Drop jump session_2",
    "Squatting session",
    "Squatting session_3",
]

# Laut Studienprotokoll werden pro Subject × Phase × Uebung × Seite
# immer 3 gueltige Trials erwartet. Wenn die Metadata weniger als 3
# listet, wurden weniger aufgenommen -> trotzdem als fehlend zaehlen.
EXPECTED_TRIALS_PER_COMBINATION = 3


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def exercise_to_short(exercise_name: str) -> str:
    e = exercise_name.lower()
    if "counter" in e: return "CMJ"
    if "drop"    in e: return "DJ"
    if "squat"   in e: return "SQ"
    return "OTHER"


def c3d_filename_to_side(filename: str) -> str:
    """Bestimmt die Seite aus dem c3d-Dateinamen."""
    s = filename.upper()
    if " LEFT " in s or " LEFT" in s.rstrip(".C3D"):
        return "LEFT"
    if " RIGHT " in s or " RIGHT" in s.rstrip(".C3D"):
        return "RIGHT"
    return "BILATERAL"


# ============================================================
# SCHRITT 1: Erwartete Trials aus Metadata zaehlen
# ============================================================

def load_expected_trials(metadata_path: Path) -> pd.DataFrame:
    """
    Laedt die c3d-Metadaten und zaehlt die erwarteten Trials
    pro Subject, Phase, Uebung, Seite.

    Phase-Zuordnung:
      - Subjects in MANUAL_PHASE_OVERRIDE: explizite Datum->Phase Zuordnung
      - Alle anderen: chronologisch (1. Datum = PER, 2. = OVU, 3. = LUT)

    Deduplizierung:
      Gleicher c3d-Dateiname am gleichen Datum fuer dasselbe Subject wird
      nur einmal gezaehlt (behebt Duplikate durch mehrere Exercise-Sessions).
    """
    df = pd.read_excel(metadata_path)

    # Subject-ID zuordnen
    df["subject_id"] = df["PARTICIPANT"].map(SUBJECT_MAP)
    df = df[df["subject_id"].notna()].copy()
    df = df[df["subject_id"].isin(RELEVANT_SUBJECTS)].copy()

    # Nur relevante Exercises
    df = df[df["EXERCISE"].isin(TARGET_EXERCISES)].copy()

    # Kurzbezeichnungen
    df["exercise_short"] = df["EXERCISE"].apply(exercise_to_short)
    df["side"] = df["C3D_FILENAME"].apply(c3d_filename_to_side)

    df["SESSION_DATE"] = pd.to_datetime(df["SESSION_DATE"], errors="coerce")

    # Duplikate entfernen: gleicher Subject + Datum + c3d-Dateiname
    # (entsteht wenn derselbe Trial in mehreren Exercise-Sessions auftaucht)
    df = df.drop_duplicates(
        subset=["subject_id", "SESSION_DATE", "C3D_FILENAME"]
    ).copy()

    records = []
    for (subj, ex_short), grp in df.groupby(["subject_id", "exercise_short"]):

        # Phase-Zuordnung: manuell oder chronologisch
        if subj in MANUAL_PHASE_OVERRIDE:
            override = MANUAL_PHASE_OVERRIDE[subj]
            date_to_phase = {}
            for d in grp["SESSION_DATE"].unique():
                d_date = d.date() if hasattr(d, "date") else d
                if d_date in override:
                    date_to_phase[d] = override[d_date]
        else:
            dates = sorted(grp["SESSION_DATE"].unique())
            date_to_phase = {}
            for i, d in enumerate(dates):
                if i < len(PHASE_ORDER):
                    date_to_phase[d] = PHASE_ORDER[i]

        for _, row in grp.iterrows():
            phase = date_to_phase.get(row["SESSION_DATE"])
            if phase is None:
                continue
            records.append({
                "subject_id": subj,
                "phase": phase,
                "exercise": ex_short,
                "side": row["side"],
                "c3d_filename": row["C3D_FILENAME"],
            })

    df_expected = pd.DataFrame(records)
    return df_expected


# ============================================================
# SCHRITT 2: Vorhandene Trials im preprocessed-Ordner zaehlen
# ============================================================

def count_preprocessed_trials(preprocessed_dir: Path) -> pd.DataFrame:
    """
    Zaehlt die tatsaechlich vorhandenen _emg.csv Dateien im
    preprocessed-Ordner pro Subject/Phase/Uebung/Seite.

    Erwartet die Ordnerstruktur:
      <preprocessed_dir>/<subject_id>/<phase>/<exercise>/<side>/<trial>_emg.csv
    """
    records = []

    for emg_file in preprocessed_dir.rglob("*_emg.csv"):
        parts = emg_file.relative_to(preprocessed_dir).parts
        # Erwartet: subject_id / phase / exercise / side / filename
        if len(parts) < 5:
            continue

        subject_id, phase, exercise, side, filename = parts[:5]
        trial_name = filename.replace("_emg.csv", "")

        records.append({
            "subject_id": subject_id,
            "phase": phase,
            "exercise": exercise,
            "side": side,
            "trial_name": trial_name,
        })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["subject_id", "phase", "exercise", "side", "trial_name"]
    )


# ============================================================
# SCHRITT 3: Vergleich und fehlende Trials ermitteln
# ============================================================

def find_missing_trials(
    df_expected: pd.DataFrame,
    df_actual: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vergleicht erwartete vs. vorhandene Trials und gibt eine Tabelle
    der Differenzen zurueck (nur wo Trials fehlen).

    Die Erwartung ist mindestens EXPECTED_TRIALS_PER_COMBINATION (= 3 laut
    Studienprotokoll). Wenn die Metadata weniger listet, wird trotzdem
    3 als Soll verwendet — es wurden dann weniger aufgenommen als geplant.
    """
    # Erwartete Trials pro Gruppe zaehlen (laut Metadata)
    expected_counts = (
        df_expected
        .groupby(["subject_id", "phase", "exercise", "side"])
        .size()
        .reset_index(name="laut_metadata")
    )

    # Soll-Wert: Maximum aus Metadata und Studienprotokoll
    expected_counts["erwartet"] = expected_counts["laut_metadata"].clip(
        lower=EXPECTED_TRIALS_PER_COMBINATION
    )

    # Vorhandene Trials pro Gruppe zaehlen
    if df_actual.empty:
        actual_counts = pd.DataFrame(
            columns=["subject_id", "phase", "exercise", "side", "vorhanden"]
        )
    else:
        actual_counts = (
            df_actual
            .groupby(["subject_id", "phase", "exercise", "side"])
            .size()
            .reset_index(name="vorhanden")
        )

    # Merge: left join, damit auch Gruppen ohne vorhandene Trials erscheinen
    merged = expected_counts.merge(
        actual_counts,
        on=["subject_id", "phase", "exercise", "side"],
        how="left",
    )
    merged["vorhanden"] = merged["vorhanden"].fillna(0).astype(int)
    merged["fehlend"]   = merged["erwartet"] - merged["vorhanden"]

    # Grund fuer das Fehlen: nicht aufgenommen vs. nicht verarbeitet
    merged["grund"] = np.where(
        merged["laut_metadata"] < EXPECTED_TRIALS_PER_COMBINATION,
        "weniger aufgenommen",
        "nicht verarbeitet",
    )

    # Nur Zeilen mit fehlenden Trials
    missing = merged[merged["fehlend"] > 0].copy()

    # Phase-Label hinzufuegen
    missing["phase_label"] = missing["phase"].map(PHASE_LABELS)

    # Sortieren
    missing = missing.sort_values(
        ["subject_id", "exercise", "phase", "side"]
    ).reset_index(drop=True)

    return missing


# ============================================================
# HAUPTFUNKTION
# ============================================================

def main():
    print("=" * 70)
    print("02_b  –  Fehlende Trials: Metadata vs. Preprocessed-Ordner")
    print("=" * 70)

    # 1) Erwartete Trials aus Metadata
    print(f"\nLade Metadaten: {METADATA_XLS.name} ...")
    df_expected = load_expected_trials(METADATA_XLS)
    print(f"  Erwartete Trials gesamt (alle Seiten): {len(df_expected)}")

    # 2) Vorhandene Trials im preprocessed-Ordner
    print(f"\nScanne preprocessed-Ordner: {PREPROCESSED_DIR} ...")
    df_actual = count_preprocessed_trials(PREPROCESSED_DIR)
    print(f"  Vorhandene _emg.csv Dateien: {len(df_actual)}")

    # 3) Vergleich
    print("\nVergleiche erwartete vs. vorhandene Trials ...")
    df_missing = find_missing_trials(df_expected, df_actual)

    # Aufteilen: relevante Seiten (BILATERAL + RIGHT) vs. LEFT
    df_missing_relevant = df_missing[
        df_missing["side"].isin(["BILATERAL", "RIGHT"])
    ].copy()
    df_missing_left = df_missing[
        df_missing["side"] == "LEFT"
    ].copy()

    total_missing_relevant = df_missing_relevant["fehlend"].sum()
    total_missing_left     = df_missing_left["fehlend"].sum()

    # 4) Konsolenausgabe
    print(f"\n{'='*70}")
    print("ERGEBNIS")
    print(f"{'='*70}")

    if df_missing_relevant.empty:
        print("\n[OK] Keine fehlenden Trials bei BILATERAL/RIGHT.")
    else:
        print(f"\n[!] {total_missing_relevant} fehlende Trials bei BILATERAL/RIGHT:")
        print(f"    (Soll = {EXPECTED_TRIALS_PER_COMBINATION} Trials pro Kombination laut Studienprotokoll)\n")
        for _, row in df_missing_relevant.iterrows():
            print(f"    {row['subject_id']:5s} | {row['phase_label']:4s} | "
                  f"{row['exercise']:4s} | {row['side']:10s} | "
                  f"soll: {row['erwartet']}  metadata: {row['laut_metadata']}  "
                  f"vorhanden: {row['vorhanden']}  fehlend: {row['fehlend']}  "
                  f"({row['grund']})")

    if not df_missing_left.empty:
        print(f"\n[i] Zusaetzlich {total_missing_left} fehlende Trials bei LEFT "
              f"(nicht auswertungsrelevant)")

    # 5) Uebersichtstabelle: alle Subject × Phase × Uebung × Seite
    all_counts = (
        df_expected
        .groupby(["subject_id", "phase", "exercise", "side"])
        .size()
        .reset_index(name="laut_metadata")
    )
    all_counts["erwartet"] = all_counts["laut_metadata"].clip(
        lower=EXPECTED_TRIALS_PER_COMBINATION
    )
    if not df_actual.empty:
        act = (
            df_actual
            .groupby(["subject_id", "phase", "exercise", "side"])
            .size()
            .reset_index(name="vorhanden")
        )
        all_counts = all_counts.merge(act, on=["subject_id","phase","exercise","side"], how="left")
    else:
        all_counts["vorhanden"] = 0
    all_counts["vorhanden"] = all_counts["vorhanden"].fillna(0).astype(int)
    all_counts["fehlend"] = all_counts["erwartet"] - all_counts["vorhanden"]
    all_counts["phase_label"] = all_counts["phase"].map(PHASE_LABELS)
    all_counts["vollstaendig"] = np.where(all_counts["fehlend"] == 0, "✓", "✗")
    all_counts["grund"] = np.where(
        all_counts["laut_metadata"] < EXPECTED_TRIALS_PER_COMBINATION,
        "weniger aufgenommen",
        np.where(all_counts["fehlend"] > 0, "nicht verarbeitet", ""),
    )

    # 6) Excel-Bericht -> Pipeline_Reports.xlsx
    summary = pd.DataFrame([{
        "Erwartete_Trials_gesamt": len(df_expected),
        "Vorhandene_Trials_gesamt": len(df_actual),
        "Fehlende_Trials_BILATERAL_RIGHT": int(total_missing_relevant),
        "Fehlende_Trials_LEFT": int(total_missing_left),
        "Fehlende_Trials_gesamt": int(total_missing_relevant + total_missing_left),
    }])

    overview = all_counts[
        ["subject_id", "phase_label", "exercise", "side",
         "laut_metadata", "erwartet", "vorhanden", "fehlend", "vollstaendig", "grund"]
    ].rename(columns={"phase_label": "Phase", "subject_id": "Subject",
                      "exercise": "Uebung", "side": "Seite",
                      "laut_metadata": "in_Metadata", "grund": "Grund"})
    overview = overview.sort_values(["Subject","Uebung","Phase","Seite"])

    sheets = {
        "02b_Fehlende_Zusammenfassung": summary,
        "02b_Fehlende_Gesamtuebersicht": overview,
    }

    if not df_missing_relevant.empty:
        out = df_missing_relevant[
            ["subject_id", "phase_label", "exercise", "side",
             "laut_metadata", "erwartet", "vorhanden", "fehlend", "grund"]
        ].rename(columns={"phase_label": "Phase", "subject_id": "Subject",
                          "exercise": "Uebung", "side": "Seite",
                          "laut_metadata": "in_Metadata", "grund": "Grund"})
        sheets["02b_Fehlende_BIL_RIGHT"] = out

    if not df_missing_left.empty:
        out_left = df_missing_left[
            ["subject_id", "phase_label", "exercise", "side",
             "laut_metadata", "erwartet", "vorhanden", "fehlend", "grund"]
        ].rename(columns={"phase_label": "Phase", "subject_id": "Subject",
                          "exercise": "Uebung", "side": "Seite",
                          "laut_metadata": "in_Metadata", "grund": "Grund"})
        sheets["02b_Fehlende_LEFT"] = out_left

    _save_to_pipeline_report(sheets)

    print(f"\n{'='*70}")
    print("FERTIG")
    print(f"  Fehlende Trials (BILATERAL+RIGHT) : {total_missing_relevant}")
    print(f"  Fehlende Trials (LEFT)            : {total_missing_left}")
    print(f"  Bericht                           : {REPORT_PATH}")
    print(f"{'='*70}")


# ============================================================
# ZENTRALE REPORT-FUNKTION
# ============================================================

def _save_to_pipeline_report(sheets: dict[str, pd.DataFrame]):
    """
    Speichert mehrere DataFrames als Reiter in die zentrale
    Pipeline_Reports.xlsx. Bestehende Reiter anderer Skripte
    bleiben erhalten; eigene Reiter werden ueberschrieben.
    """
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if REPORT_PATH.exists():
        from openpyxl import load_workbook
        with pd.ExcelWriter(REPORT_PATH, engine="openpyxl", mode="a",
                            if_sheet_exists="replace") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
    else:
        with pd.ExcelWriter(REPORT_PATH, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)


if __name__ == "__main__":
    main()