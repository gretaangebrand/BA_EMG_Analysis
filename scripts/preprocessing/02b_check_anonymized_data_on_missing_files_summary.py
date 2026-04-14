"""
02_b_check_missing_files_summary.py
====================================
Vergleicht DREI Quellen, um fehlende Trials zuverlaessig zu klassifizieren:

  1. Metadata (c3d_metadata_export.xlsx) – was laut Vicon existieren SOLLTE
  2. Trial-Inventar (trial_inventar_rohdaten.csv) – was tatsaechlich in den
     Rohdateien gefunden wurde (erzeugt von 02_a_preprocess.py)
  3. Preprocessed-Ordner (03_preprocessed_emg_data) – was nach Verarbeitung
     als _emg.csv vorhanden ist

Damit wird der Grund fuer fehlende Trials praezise bestimmt:
  - "weniger aufgenommen":  Metadata listet Trial, aber Rohdatei enthaelt ihn nicht
  - "nicht verarbeitet":    Trial existiert in Rohdatei, wurde aber nicht gespeichert
  - "Pipeline-Fehler":      Trial hat EMG-Spalten, aber Speicherung schlug fehl
  - "kein EMG in Rohdatei": Trial existiert in Rohdatei, hat aber keine EMG-Spalten

Ergebnis:
  - Excel-Bericht mit fehlenden Trials und Zusammenfassung
  - Konsolenausgabe mit Uebersicht
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date


# ============================================================
# PFADE
# ============================================================
PREPROCESSED_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\03_preprocessed_emg_data")
METADATA_XLS     = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\c3d_metadata_export.xlsx")
INVENTORY_CSV    = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\03_preprocessed_emg_data\trial_inventar_rohdaten.csv")
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

RELEVANT_SUBJECTS = set(SUBJECT_MAP.values())

PHASE_ORDER = ["01_PER", "02_OVU", "03_LUT"]
PHASE_LABELS = {"01_PER": "PER", "02_OVU": "OVU", "03_LUT": "LUT"}

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
    },
}

TARGET_EXERCISES = [
    "Counter-movement jump session",
    "Counter-movement jump session_2",
    "Drop jump session",
    "Drop jump session_2",
    "Squatting session",
    "Squatting session_3",
]

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
    """
    df = pd.read_excel(metadata_path)

    df["subject_id"] = df["PARTICIPANT"].map(SUBJECT_MAP)
    df = df[df["subject_id"].notna()].copy()
    df = df[df["subject_id"].isin(RELEVANT_SUBJECTS)].copy()

    df = df[df["EXERCISE"].isin(TARGET_EXERCISES)].copy()

    df["exercise_short"] = df["EXERCISE"].apply(exercise_to_short)
    df["side"] = df["C3D_FILENAME"].apply(c3d_filename_to_side)

    df["SESSION_DATE"] = pd.to_datetime(df["SESSION_DATE"], errors="coerce")

    df = df.drop_duplicates(
        subset=["subject_id", "SESSION_DATE", "C3D_FILENAME"]
    ).copy()

    records = []
    for (subj, ex_short), grp in df.groupby(["subject_id", "exercise_short"]):

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
# SCHRITT 2: Trial-Inventar aus Rohdateien laden
# ============================================================

def load_trial_inventory(inventory_path: Path) -> pd.DataFrame:
    """
    Laedt das Trial-Inventar, das von 02_a erzeugt wurde.
    Enthaelt jeden Trial, der in den Rohdateien gefunden wurde,
    mit Info ob EMG vorhanden und ob erfolgreich gespeichert.
    """
    if not inventory_path.exists():
        print(f"[WARNUNG] Trial-Inventar nicht gefunden: {inventory_path}")
        print(f"          Bitte zuerst 02_a_preprocess.py ausfuehren.")
        return pd.DataFrame(
            columns=["subject_id", "phase", "exercise", "side",
                     "trial_name", "hat_emg", "emg_gespeichert"]
        )

    return pd.read_csv(inventory_path)


# ============================================================
# SCHRITT 3: Vorhandene Trials im preprocessed-Ordner zaehlen
# ============================================================

def count_preprocessed_trials(preprocessed_dir: Path) -> pd.DataFrame:
    """
    Zaehlt die tatsaechlich vorhandenen _emg.csv Dateien im
    preprocessed-Ordner pro Subject/Phase/Uebung/Seite.
    """
    records = []

    for emg_file in preprocessed_dir.rglob("*_emg.csv"):
        parts = emg_file.relative_to(preprocessed_dir).parts
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
# SCHRITT 4: Vergleich mit drei Quellen
# ============================================================

def find_missing_trials(
    df_expected: pd.DataFrame,
    df_actual: pd.DataFrame,
    df_inventory: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vergleicht erwartete vs. vorhandene Trials mit dem Inventar als
    dritte Quelle fuer die praezise Bestimmung des Grundes.

    Gruende:
      - "weniger aufgenommen":  Metadata erwartet Trial, aber Rohdatei hat ihn nicht
      - "nicht verarbeitet":    Trial in Rohdatei, aber nicht im preprocessed-Ordner
      - "kein EMG in Rohdatei": Trial in Rohdatei vorhanden, aber ohne EMG-Spalten
      - "Pipeline-Fehler":      EMG-Spalten vorhanden, aber Speicherung fehlgeschlagen
    """
    group_cols = ["subject_id", "phase", "exercise", "side"]

    # Erwartete Trials pro Gruppe (laut Metadata)
    expected_counts = (
        df_expected
        .groupby(group_cols)
        .size()
        .reset_index(name="laut_metadata")
    )
    expected_counts["erwartet"] = expected_counts["laut_metadata"].clip(
        lower=EXPECTED_TRIALS_PER_COMBINATION
    )

    # Trials in Rohdateien pro Gruppe (laut Inventar)
    if not df_inventory.empty and "subject_id" in df_inventory.columns:
        inv_counts = (
            df_inventory
            .groupby(group_cols)
            .size()
            .reset_index(name="in_rohdatei")
        )
        # Wie viele davon hatten EMG-Spalten?
        inv_emg_counts = (
            df_inventory[df_inventory["hat_emg"] == True]
            .groupby(group_cols)
            .size()
            .reset_index(name="mit_emg_in_rohdatei")
        )
    else:
        inv_counts = pd.DataFrame(columns=group_cols + ["in_rohdatei"])
        inv_emg_counts = pd.DataFrame(columns=group_cols + ["mit_emg_in_rohdatei"])

    # Vorhandene Trials im preprocessed-Ordner
    if df_actual.empty:
        actual_counts = pd.DataFrame(columns=group_cols + ["vorhanden"])
    else:
        actual_counts = (
            df_actual
            .groupby(group_cols)
            .size()
            .reset_index(name="vorhanden")
        )

    # Alles zusammenfuehren
    merged = expected_counts.merge(
        inv_counts, on=group_cols, how="left"
    ).merge(
        inv_emg_counts, on=group_cols, how="left"
    ).merge(
        actual_counts, on=group_cols, how="left"
    )

    merged["in_rohdatei"]         = merged["in_rohdatei"].fillna(0).astype(int)
    merged["mit_emg_in_rohdatei"] = merged["mit_emg_in_rohdatei"].fillna(0).astype(int)
    merged["vorhanden"]           = merged["vorhanden"].fillna(0).astype(int)
    merged["fehlend"]             = merged["erwartet"] - merged["vorhanden"]

    # Praezisen Grund bestimmen
    def _bestimme_grund(row):
        if row["in_rohdatei"] < row["laut_metadata"]:
            return "weniger aufgenommen"
        if row["mit_emg_in_rohdatei"] < row["in_rohdatei"]:
            return "kein EMG in Rohdatei"
        if row["vorhanden"] < row["mit_emg_in_rohdatei"]:
            return "Pipeline-Fehler"
        if row["laut_metadata"] < EXPECTED_TRIALS_PER_COMBINATION:
            return "weniger aufgenommen"
        return "nicht verarbeitet"

    merged["grund"] = merged.apply(
        lambda r: _bestimme_grund(r) if r["fehlend"] > 0 else "", axis=1
    )

    # Nur Zeilen mit fehlenden Trials
    missing = merged[merged["fehlend"] > 0].copy()
    missing["phase_label"] = missing["phase"].map(PHASE_LABELS)

    missing = missing.sort_values(
        ["subject_id", "exercise", "phase", "side"]
    ).reset_index(drop=True)

    return missing, merged


# ============================================================
# HAUPTFUNKTION
# ============================================================

def main():
    print("=" * 70)
    print("02_b  –  Fehlende Trials: Metadata vs. Rohdateien vs. Preprocessed")
    print("=" * 70)

    # 1) Erwartete Trials aus Metadata
    print(f"\nLade Metadaten: {METADATA_XLS.name} ...")
    df_expected = load_expected_trials(METADATA_XLS)
    print(f"  Erwartete Trials gesamt (alle Seiten): {len(df_expected)}")

    # 2) Trial-Inventar aus Rohdateien
    print(f"\nLade Trial-Inventar: {INVENTORY_CSV.name} ...")
    df_inventory = load_trial_inventory(INVENTORY_CSV)
    if not df_inventory.empty:
        print(f"  Trials in Rohdateien: {len(df_inventory)}")
        print(f"  Davon mit EMG:        {df_inventory['hat_emg'].sum()}")
        print(f"  Davon EMG gespeichert: {df_inventory['emg_gespeichert'].sum()}")
    else:
        print("  [!] Kein Inventar vorhanden – Grund-Bestimmung eingeschraenkt.")

    # 3) Vorhandene Trials im preprocessed-Ordner
    print(f"\nScanne preprocessed-Ordner: {PREPROCESSED_DIR} ...")
    df_actual = count_preprocessed_trials(PREPROCESSED_DIR)
    print(f"  Vorhandene _emg.csv Dateien: {len(df_actual)}")

    # 4) Vergleich
    print("\nVergleiche alle drei Quellen ...")
    df_missing, df_all_counts = find_missing_trials(df_expected, df_actual, df_inventory)

    # Aufteilen: relevante Seiten vs. LEFT
    df_missing_relevant = df_missing[
        df_missing["side"].isin(["BILATERAL", "RIGHT"])
    ].copy()
    df_missing_left = df_missing[
        df_missing["side"] == "LEFT"
    ].copy()

    total_missing_relevant = df_missing_relevant["fehlend"].sum()
    total_missing_left     = df_missing_left["fehlend"].sum()

    # 5) Konsolenausgabe
    print(f"\n{'='*70}")
    print("ERGEBNIS")
    print(f"{'='*70}")

    if df_missing_relevant.empty:
        print("\n[OK] Keine fehlenden Trials bei BILATERAL/RIGHT.")
    else:
        print(f"\n[!] {total_missing_relevant} fehlende Trials bei BILATERAL/RIGHT:")
        print(f"    (Soll = {EXPECTED_TRIALS_PER_COMBINATION} Trials pro Kombination)\n")

        # Nach Grund gruppiert ausgeben
        for grund, grp in df_missing_relevant.groupby("grund"):
            print(f"  --- {grund} ({len(grp)} Kombinationen) ---")
            for _, row in grp.iterrows():
                print(f"    {row['subject_id']:5s} | {row['phase_label']:4s} | "
                      f"{row['exercise']:4s} | {row['side']:10s} | "
                      f"metadata: {row['laut_metadata']}  "
                      f"rohdatei: {row['in_rohdatei']}  "
                      f"mit_emg: {row['mit_emg_in_rohdatei']}  "
                      f"vorhanden: {row['vorhanden']}  "
                      f"fehlend: {row['fehlend']}")
            print()

    if not df_missing_left.empty:
        print(f"[i] Zusaetzlich {total_missing_left} fehlende Trials bei LEFT "
              f"(nicht auswertungsrelevant)")

    # 6) Excel-Bericht
    summary = pd.DataFrame([{
        "Erwartete_Trials_gesamt": len(df_expected),
        "Trials_in_Rohdateien": len(df_inventory) if not df_inventory.empty else "n/a",
        "Vorhandene_Trials_gesamt": len(df_actual),
        "Fehlende_Trials_BILATERAL_RIGHT": int(total_missing_relevant),
        "Fehlende_Trials_LEFT": int(total_missing_left),
    }])

    # Gesamtuebersicht mit allen Spalten
    display_cols = [
        "subject_id", "phase", "exercise", "side",
        "laut_metadata", "in_rohdatei", "mit_emg_in_rohdatei",
        "erwartet", "vorhanden", "fehlend", "grund",
    ]
    overview = df_all_counts[display_cols].copy()
    overview["phase_label"] = overview["phase"].map(PHASE_LABELS)
    overview["vollstaendig"] = np.where(overview["fehlend"] <= 0, "✓", "✗")
    overview = overview.sort_values(
        ["subject_id", "exercise", "phase", "side"]
    ).reset_index(drop=True)

    sheets = {
        "02b_Fehlende_Zusammenfassung": summary,
        "02b_Fehlende_Gesamtuebersicht": overview,
    }

    if not df_missing_relevant.empty:
        out_cols = [
            "subject_id", "phase_label", "exercise", "side",
            "laut_metadata", "in_rohdatei", "mit_emg_in_rohdatei",
            "erwartet", "vorhanden", "fehlend", "grund",
        ]
        sheets["02b_Fehlende_BIL_RIGHT"] = df_missing_relevant[out_cols].copy()

    if not df_missing_left.empty:
        out_cols = [
            "subject_id", "phase_label", "exercise", "side",
            "laut_metadata", "in_rohdatei", "mit_emg_in_rohdatei",
            "erwartet", "vorhanden", "fehlend", "grund",
        ]
        sheets["02b_Fehlende_LEFT"] = df_missing_left[out_cols].copy()

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