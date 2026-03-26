"""
check_emg_amplitudes.py
=======================
Prueft alle anonymisierten CSV-Dateien auf ungewoehnliche EMG-Amplituden,
um zu erkennen ob weitere Subjects (neben S08) mit einer anderen
Verstaerkerstufe aufgezeichnet wurden.

Hintergrund:
  Normale EMG-Rohdaten in µV haben typischerweise:
    |Max|  ~  100 – 5000 µV
    |Mean| ~  5   – 300  µV

  Wenn eine Datei in Volt statt µV vorliegt (Faktor 1.000.000 zu klein):
    |Max|  ~  0.0001 – 0.005
    |Mean| ~  0.000005 – 0.0003

  S08 ist der bekannte Sonderfall: die Rohdaten liegen im Volt-Bereich
  und werden mit * 1.000.000 korrigiert.

Output:
  - Konsolenausgabe mit Ampitudenübersicht pro Subject + Datei
  - Excel-Bericht unter REPORT_PATH
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PFADE – anpassen!
# ============================================================
# Rohdaten prüfen
SOURCE_DIR  = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\anonymized_csv_data")
REPORT_PATH = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\emg_amplitude_check.xlsx")

# Bekannte Sonderfälle (bereits korrekt behandelt in 02_preprocess_emg.py)
KNOWN_SPECIAL = {"S08"}

# Schwellenwerte fuer die Klassifikation
THRESHOLD_VOLT_MAX  = 0.1    # Max-Wert unter diesem Grenzwert -> wahrscheinlich in Volt
THRESHOLD_VOLT_MEAN = 0.01   # Mean-Wert unter diesem Grenzwert -> wahrscheinlich in Volt
THRESHOLD_HUGE_MAX  = 50_000 # Max-Wert ueber diesem Grenzwert -> ungewoehnlich gross

# Anzahl Header-Zeilen je Subject
_SPECIAL_SUBJECTS    = {"S08"}
_HEADER_ROWS_DEFAULT = 5
_HEADER_ROWS_SPECIAL = 4


def get_subject_id(file_path: Path) -> str:
    return file_path.stem.split("_")[0]


def n_header_rows(subject_id: str) -> int:
    return _HEADER_ROWS_SPECIAL if subject_id in _SPECIAL_SUBJECTS else _HEADER_ROWS_DEFAULT


def get_emg_amplitudes(file_path: Path) -> dict | None:
    """
    Liest die EMG-Spalten des ersten Trials einer CSV und gibt
    Amplitude-Kennwerte zurueck.
    Gibt None zurueck wenn keine EMG-Spalten gefunden wurden.
    """
    subject_id = get_subject_id(file_path)
    n_hdr      = n_header_rows(subject_id)

    try:
        header = pd.read_csv(file_path, header=None, nrows=n_hdr,
                             low_memory=False).fillna("")
        data   = pd.read_csv(file_path, header=None, skiprows=n_hdr,
                             low_memory=False)
    except Exception as e:
        print(f"  [FEHLER] {file_path.name}: {e}")
        return None

    row0 = header.iloc[0].astype(str)
    row2 = header.iloc[2].astype(str).str.upper()
    row3 = header.iloc[3].astype(str).str.upper() if n_hdr >= 4 \
           else pd.Series([""] * len(header.columns))

    has_emg_raw = (row3 == "EMG_RAW").any()

    # Ersten Trial holen
    trial_path = next((v for v in row0 if ".c3d" in v.lower()), None)
    if trial_path is None:
        return None

    # EMG-Spaltenindizes
    if has_emg_raw:
        emg_idx = [i for i, v in enumerate(row0)
                   if v == trial_path
                   and row2.iloc[i] == "ANALOG"
                   and row3.iloc[i] == "EMG_RAW"]
    else:
        emg_idx = [i for i, v in enumerate(row0)
                   if v == trial_path and row2.iloc[i] == "ANALOG"]

    if not emg_idx:
        return None

    emg_data = data.iloc[:, emg_idx].apply(pd.to_numeric, errors="coerce")

    # Nur nicht-NaN Zeilen
    emg_data = emg_data.dropna(how="all")
    if emg_data.empty:
        return None

    abs_vals = emg_data.abs()

    return {
        "subject_id":   subject_id,
        "file":         file_path.name,
        "abs_max":      abs_vals.max().max(),
        "abs_mean":     abs_vals.mean().mean(),
        "abs_median":   abs_vals.median().median(),
        "n_emg_cols":   len(emg_idx),
        "has_emg_raw":  has_emg_raw,
        "n_rows":       len(emg_data),
    }


def classify_amplitude(row: dict) -> str:
    """Klassifiziert die Amplitudenlage einer Datei."""
    if row["abs_max"] < THRESHOLD_VOLT_MAX and row["abs_mean"] < THRESHOLD_VOLT_MEAN:
        return "VOLT (?)"        # Wahrscheinlich in Volt, nicht µV
    if row["abs_max"] > THRESHOLD_HUGE_MAX:
        return "UNGEWOEHNLICH GROSS"
    return "OK (µV)"


def main():
    print("=" * 65)
    print("EMG-Amplitudenprüfung – alle anonymisierten CSV-Dateien")
    print("=" * 65)

    all_csvs = sorted(SOURCE_DIR.rglob("*.csv"))
    print(f"\nGefundene CSV-Dateien: {len(all_csvs)}\n")

    if not all_csvs:
        print(f"[FEHLER] Keine CSV-Dateien in {SOURCE_DIR}")
        return

    results = []

    for csv_path in all_csvs:
        amp = get_emg_amplitudes(csv_path)
        if amp is None:
            print(f"  [SKIP] {csv_path.name} – keine EMG-Spalten")
            continue
        amp["status"] = classify_amplitude(amp)
        results.append(amp)

    if not results:
        print("Keine auswertbaren Dateien gefunden.")
        return

    df = pd.DataFrame(results)

    # -------------------------------------------------------
    # Konsolenausgabe: Übersicht pro Subject
    # -------------------------------------------------------
    print(f"\n{'Datei':<35} {'Sub':>4}  {'Max':>10}  {'Mean':>8}  {'Median':>8}  Status")
    print("-" * 85)

    for _, row in df.iterrows():
        flag = ""
        if row["subject_id"] in KNOWN_SPECIAL:
            flag = "  [bekannt]"
        elif row["status"] != "OK (µV)":
            flag = "  *** PRÜFEN ***"

        print(f"  {row['file']:<33} {row['subject_id']:>4}  "
              f"{row['abs_max']:>10.4f}  {row['abs_mean']:>8.4f}  "
              f"{row['abs_median']:>8.4f}  {row['status']}{flag}")

    # -------------------------------------------------------
    # Zusammenfassung
    # -------------------------------------------------------
    print("\n" + "=" * 65)
    print("ZUSAMMENFASSUNG")
    print("=" * 65)

    ok        = df[df["status"] == "OK (µV)"]
    volt      = df[df["status"] == "VOLT (?)"]
    huge      = df[df["status"] == "UNGEWOEHNLICH GROSS"]
    known_sp  = df[df["subject_id"].isin(KNOWN_SPECIAL)]

    print(f"\n  OK (µV-Bereich):          {len(ok)} Dateien")
    print(f"  Bekannte Sonderfälle:     {len(known_sp)} Dateien  "
          f"({', '.join(sorted(known_sp['subject_id'].unique()))})")

    if not volt.empty:
        subjects = sorted(volt["subject_id"].unique())
        print(f"\n  [!] Wahrscheinlich in VOLT: {len(volt)} Dateien")
        print(f"      Betroffene Subjects: {subjects}")
        print("      -> In helpers.py zu _SPECIAL_SUBJECTS hinzufügen und")
        print("         Skalierung mit * 1.000.000 in 02_preprocess_emg.py ergaenzen!")
        for _, row in volt.iterrows():
            print(f"      {row['file']}: Max={row['abs_max']:.6f}, "
                  f"Mean={row['abs_mean']:.6f}")
    else:
        print(f"\n  [OK] Keine weiteren Dateien mit Volt-Skalierung gefunden.")

    if not huge.empty:
        print(f"\n  [!] Ungewoehnlich grosse Amplituden: {len(huge)} Dateien")
        for _, row in huge.iterrows():
            print(f"      {row['file']}: Max={row['abs_max']:.1f}")

    # -------------------------------------------------------
    # Excel-Bericht
    # -------------------------------------------------------
    df["skalierung_korrekt"] = df.apply(
        lambda r: "bereits korrigiert" if r["subject_id"] in KNOWN_SPECIAL
                  else r["status"], axis=1
    )

    with pd.ExcelWriter(REPORT_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Alle_Dateien", index=False)

        # Nur auffällige Dateien (ohne bekannte Sonderfälle)
        auffaellig = df[
            (df["status"] != "OK (µV)") &
            (~df["subject_id"].isin(KNOWN_SPECIAL))
        ]
        if not auffaellig.empty:
            auffaellig.to_excel(writer, sheet_name="Auffaellig", index=False)

        # Übersicht pro Subject (Median der Medians)
        summary = df.groupby("subject_id").agg(
            n_dateien    = ("file",       "count"),
            max_median   = ("abs_max",    "median"),
            mean_median  = ("abs_mean",   "median"),
            status_mode  = ("status",     lambda x: x.mode()[0])
        ).reset_index()
        summary.to_excel(writer, sheet_name="Zusammenfassung_Subject", index=False)

    print(f"\nBericht gespeichert: {REPORT_PATH}")
    print("\nFERTIG.")


if __name__ == "__main__":
    main()