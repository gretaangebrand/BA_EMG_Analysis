"""
check_emg_amplitudes.py
=======================
Prueft alle preprocessed Trial-CSVs auf ungewoehnliche EMG-Amplituden.
Laeuft NACH 02_preprocess_emg.py auf dem 03_preprocessed_emg_data-Ordner.

Erkennt EMG-Spalten anhand des Namens-Praefixes: L_ oder R_
(z.B. L_Vastus Lateralis, R_Biceps Femoris).

Klassifikation:
  OK (µV)              Max > 0.1  oder  Mean > 0.01  -> normaler µV-Bereich
  VOLT (?)             Max < 0.1  und  Mean < 0.01   -> wahrscheinlich in Volt
  UNGEWOEHNLICH GROSS  Max > 50.000                  -> ungewoehnlich gross

Output:
  - Konsolenausgabe pro Subject und Datei
  - Excel-Bericht unter REPORT_PATH
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PFADE – anpassen!
# ============================================================
SOURCE_DIR  = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\03_preprocessed_emg_data")
REPORT_PATH = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\03_preprocessed_emg_data\emg_amplitude_check.xlsx")

# Bekannte Sonderfälle – werden separat markiert, nicht als Fehler gewertet
KNOWN_SPECIAL = {"S08", "S09", "S11"}

# Schwellenwerte
THRESHOLD_VOLT_MAX  = 0.1       # Max unter diesem Wert  -> wahrscheinlich Volt
THRESHOLD_VOLT_MEAN = 0.01      # Mean unter diesem Wert -> wahrscheinlich Volt
THRESHOLD_HUGE_MAX  = 50_000    # Max ueber diesem Wert  -> ungewoehnlich gross


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def get_subject_id(file_path: Path) -> str:
    """
    Liest die Subject-ID aus dem Ordnerpfad.
    Erwartet Struktur: .../<subject_id>/<phase>/<exercise>/<side>/<trial>.csv
    """
    # Gehe die Pfad-Teile durch und nimm das erste S01..S11-Segment
    for part in file_path.parts:
        if part.startswith("S") and part[1:].isdigit():
            return part
    # Fallback: ersten Teil des Dateinamens
    return file_path.stem.split("_")[0]


def get_emg_columns(df: pd.DataFrame) -> list[str]:
    """
    Gibt alle EMG-Spaltennamen zurueck.
    EMG-Spalten haben das Praefix L_ oder R_ (z.B. L_Vastus Lateralis).
    """
    return [c for c in df.columns if c.startswith("L_") or c.startswith("R_")]


def classify(abs_max: float, abs_mean: float) -> str:
    if abs_max < THRESHOLD_VOLT_MAX and abs_mean < THRESHOLD_VOLT_MEAN:
        return "VOLT (?)"
    if abs_max > THRESHOLD_HUGE_MAX:
        return "UNGEWOEHNLICH GROSS"
    return "OK (µV)"


def check_file(file_path: Path) -> dict | None:
    """
    Laedt eine preprocessed CSV, extrahiert EMG-Spalten und
    berechnet Amplitude-Kennwerte.
    Gibt None zurueck wenn keine EMG-Spalten gefunden wurden.
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"  [FEHLER] {file_path.name}: {e}")
        return None

    emg_cols = get_emg_columns(df)
    if not emg_cols:
        return None

    # Nur Zeilen wo mindestens eine EMG-Spalte nicht NaN ist
    emg_data = df[emg_cols].apply(pd.to_numeric, errors="coerce")
    emg_data = emg_data.dropna(how="all")

    if emg_data.empty:
        return None

    abs_vals   = emg_data.abs()
    abs_max    = float(abs_vals.max().max())
    abs_mean   = float(abs_vals.mean().mean())
    abs_median = float(abs_vals.median().median())

    subject_id = get_subject_id(file_path)

    # Relativen Pfad ab SOURCE_DIR fuer lesbare Ausgabe
    try:
        rel_path = file_path.relative_to(SOURCE_DIR)
    except ValueError:
        rel_path = file_path

    return {
        "subject_id":    subject_id,
        "phase":         rel_path.parts[1] if len(rel_path.parts) > 1 else "",
        "exercise":      rel_path.parts[2] if len(rel_path.parts) > 2 else "",
        "side":          rel_path.parts[3] if len(rel_path.parts) > 3 else "",
        "file":          file_path.name,
        "rel_path":      str(rel_path),
        "abs_max":       abs_max,
        "abs_mean":      abs_mean,
        "abs_median":    abs_median,
        "n_emg_cols":    len(emg_cols),
        "n_rows_valid":  len(emg_data),
        "status":        classify(abs_max, abs_mean),
    }


# ============================================================
# HAUPTFUNKTION
# ============================================================

def main():
    print("=" * 70)
    print("EMG-Amplitudenprüfung – preprocessed_emg_data")
    print("=" * 70)

    all_csvs = sorted(SOURCE_DIR.rglob("*.csv"))
    print(f"\nGefundene CSV-Dateien: {len(all_csvs)}\n")

    if not all_csvs:
        print(f"[FEHLER] Keine CSV-Dateien in {SOURCE_DIR}")
        return

    results   = []
    skipped   = []

    for csv_path in all_csvs:
        result = check_file(csv_path)
        if result is None:
            skipped.append(csv_path.name)
        else:
            results.append(result)

    if skipped:
        print(f"Übersprungen (keine EMG-Spalten): {len(skipped)} Dateien")
        for s in skipped[:5]:
            print(f"  {s}")
        if len(skipped) > 5:
            print(f"  ... und {len(skipped)-5} weitere")
        print()

    if not results:
        print("Keine auswertbaren Dateien gefunden.")
        return

    df = pd.DataFrame(results)

    # -------------------------------------------------------
    # Konsolenausgabe gruppiert nach Subject
    # -------------------------------------------------------
    for subject_id, grp in df.groupby("subject_id"):
        is_known = subject_id in KNOWN_SPECIAL
        n_ok     = (grp["status"] == "OK (µV)").sum()
        n_bad    = (grp["status"] != "OK (µV)").sum()

        status_line = f"[bekannt skaliert]" if is_known else \
                      f"[OK – {n_ok}/{len(grp)} Dateien]" if n_bad == 0 else \
                      f"[!!! {n_bad} AUFFÄLLIG !!!]"

        print(f"\n--- {subject_id}  {status_line} ---")

        # Nur auffällige Dateien im Detail zeigen (ausser bei bekannten)
        show = grp if (n_bad > 0 and not is_known) else \
               grp[grp["status"] != "OK (µV)"] if is_known else \
               grp.head(1)  # nur eine Beispielzeile wenn alles OK

        for _, row in show.iterrows():
            flag = ""
            if row["subject_id"] in KNOWN_SPECIAL:
                flag = "  [bekannt]"
            elif row["status"] != "OK (µV)":
                flag = "  *** PRÜFEN ***"

            print(f"  {row['file']:<20}  "
                  f"Max={row['abs_max']:>10.4f}  "
                  f"Mean={row['abs_mean']:>8.4f}  "
                  f"Median={row['abs_median']:>8.4f}  "
                  f"{row['status']}{flag}")

        if n_bad == 0 and not is_known and len(grp) > 1:
            print(f"  (alle {len(grp)} Dateien im OK-Bereich, "
                  f"Beispiel-Max: {grp['abs_max'].median():.2f} µV)")

    # -------------------------------------------------------
    # Gesamtzusammenfassung
    # -------------------------------------------------------
    print("\n" + "=" * 70)
    print("GESAMTZUSAMMENFASSUNG")
    print("=" * 70)

    ok_df     = df[df["status"] == "OK (µV)"]
    volt_df   = df[(df["status"] == "VOLT (?)") & (~df["subject_id"].isin(KNOWN_SPECIAL))]
    huge_df   = df[df["status"] == "UNGEWOEHNLICH GROSS"]
    known_df  = df[df["subject_id"].isin(KNOWN_SPECIAL)]

    print(f"\n  Gesamt geprüft:           {len(df)} Dateien")
    print(f"  OK (µV-Bereich):          {len(ok_df)} Dateien")
    print(f"  Bekannte Sonderfälle:     {len(known_df)} Dateien "
          f"({', '.join(sorted(KNOWN_SPECIAL))})")

    if not volt_df.empty:
        subjects = sorted(volt_df["subject_id"].unique())
        print(f"\n  [!] Wahrscheinlich in VOLT:  {len(volt_df)} Dateien")
        print(f"      Betroffene Subjects:  {subjects}")
        print()
        print("      Massnahme:")
        print("      1. In helpers.py: Subject zu _SPECIAL_SUBJECTS hinzufügen")
        print("      2. 02_preprocess_emg.py erneut ausfuehren")
        print("      3. Dieses Skript erneut ausfuehren zur Verifikation")
    else:
        print(f"\n  [OK] Keine weiteren Volt-skalierten Dateien gefunden.")

    if not huge_df.empty:
        print(f"\n  [!] Ungewoehnlich grosse Amplituden: {len(huge_df)} Dateien")
        for _, row in huge_df.iterrows():
            print(f"      {row['rel_path']}: Max={row['abs_max']:.1f}")

    # -------------------------------------------------------
    # Excel-Bericht
    # -------------------------------------------------------
    print(f"\nSpeichere Bericht: {REPORT_PATH}")

    with pd.ExcelWriter(REPORT_PATH, engine="openpyxl") as writer:

        # Sheet 1: Alle Dateien
        df.to_excel(writer, sheet_name="Alle_Dateien", index=False)

        # Sheet 2: Nur auffällige (ohne bekannte Sonderfälle)
        auffaellig = df[
            (df["status"] != "OK (µV)") &
            (~df["subject_id"].isin(KNOWN_SPECIAL))
        ]
        if not auffaellig.empty:
            auffaellig.to_excel(writer, sheet_name="Auffaellig", index=False)

        # Sheet 3: Zusammenfassung pro Subject
        summary = (
            df.groupby("subject_id")
            .agg(
                n_dateien       = ("file",       "count"),
                max_abs_max     = ("abs_max",    "max"),
                median_abs_max  = ("abs_max",    "median"),
                median_abs_mean = ("abs_mean",   "median"),
                n_ok            = ("status",     lambda x: (x == "OK (µV)").sum()),
                n_auffaellig    = ("status",     lambda x: (x != "OK (µV)").sum()),
            )
            .reset_index()
        )
        summary["bekannter_sonderfall"] = summary["subject_id"].isin(KNOWN_SPECIAL)
        summary.to_excel(writer, sheet_name="Zusammenfassung_Subject", index=False)

    print("FERTIG.")



if __name__ == "__main__":
    main()

