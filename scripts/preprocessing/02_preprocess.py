from pathlib import Path
import pandas as pd
import numpy as np

from scripts.utils.helpers import (
    get_sampling_rate_for_subject,
    get_subject_id_from_filename,
    detect_phase,
    make_short_trial_name,
)

# =============================================================================
# 02_preprocess_emg.py
#
# Liest anonymisierte CSV-Dateien ein und speichert pro Trial eine kombinierte
# CSV mit drei Abschnitten in dieser Spaltenreihenfolge:
#
#   time_s | Scalar-Metriken | Kinematik (GRF, Angles, Moments, Power) | EMG (roh)
#
# Vorteile der kombinierten Datei:
#   - EMG und Kinematik sind bereits ab Rohexport synchronisiert (selbe Zeitachse)
#   - Kein spaeteres Alignment noetig
#   - Ankerpunkte (Peak-GRF, minimaler Kniewinkel) koennen direkt aus derselben
#     Datei berechnet werden
#
# Spaltenkonvention Kinematik:
#   X-Achse (Hauptkomponente): nur Variablenname, z.B. "Left Knee Angles"
#   Y- und Z-Achse:            Variablenname + Suffix, z.B. "Left Knee Angles_Y"
#
# Ordnerstruktur Output:
#   <OUTPUT_DIR>/<subject_id>/<phase>/<bewegung>/<seite>/<trial_kuerzel>.csv
# =============================================================================

# -----------------------------------------------------------------------------
# Pfade
# -----------------------------------------------------------------------------
SOURCE_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\anonymized_csv_data")
OUTPUT_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\preprocessed_emg_data")

# Anzahl Header-Zeilen je Subject-Gruppe
_HEADER_ROWS_DEFAULT = 5   # S01-S07
_HEADER_ROWS_SPECIAL = 4   # S08 (und ggf. S09-S11)
_SPECIAL_SUBJECTS    = {} # Annahme, dass es ab S08 anders sei, stimmte nicht


# -----------------------------------------------------------------------------
# Hilfsfunktionen: Header
# -----------------------------------------------------------------------------

def _n_header_rows(subject_id: str) -> int:
    return _HEADER_ROWS_SPECIAL if subject_id in _SPECIAL_SUBJECTS else _HEADER_ROWS_DEFAULT


def read_header(file_path: Path, n_rows: int) -> pd.DataFrame:
    """Liest die ersten n_rows Zeilen als rohen String-DataFrame."""
    return pd.read_csv(file_path, header=None, nrows=n_rows, low_memory=False).fillna("")


def _make_col_name(variable: str, axis: str) -> str:
    """
    Baut den Spaltennamen aus Variablenname und Achse zusammen.
    X-Achse erhaelt kein Suffix (= Hauptkomponente).
    Y- und Z-Achse erhalten _Y / _Z als Suffix.
    """
    variable = variable.strip()
    axis     = axis.strip().upper()
    if axis in ("Y", "Z"):
        return f"{variable}_{axis}"
    return variable


# -----------------------------------------------------------------------------
# Hilfsfunktionen: Spaltenindizes bestimmen
# -----------------------------------------------------------------------------

def _get_trial_paths_in_order(row0: pd.Series) -> list[str]:
    """Gibt alle einzigartigen Trial-Pfade in Reihenfolge ihres ersten Auftretens zurueck."""
    seen, ordered = set(), []
    for v in row0:
        if ".c3d" in v.lower() and v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def _col_indices_by_type(
    row0: pd.Series,
    row2: pd.Series,
    row3: pd.Series,
    trial_path: str,
    has_emg_raw: bool,
) -> dict[str, list[int]]:
    """
    Gibt fuer einen Trial vier Listen von Spaltenindizes zurueck:
 
      'events' : EVENT_LABEL  (start, take_off, landing, end_jump etc.)
                 Jeder Event hat genau einen Zeitwert in Sekunden (Zeile 1),
                 alle weiteren Zeilen sind NaN.
      'scalar' : METRIC + DERIVED  (HEIGHT, MASS, Jumpheight, etc.)
      'kin'    : LINK_MODEL_BASED  (GRF, Angles, Moments, Power)
      'emg'    : ANALOG + EMG_RAW  (oder alle ANALOG wenn kein EMG_RAW vorhanden)
    """
    result = {"events": [], "scalar": [], "kin": [], "emg": []}

    for i, v in enumerate(row0):
        if v != trial_path:
            continue
        typ = row2.iloc[i]
        sub = row3.iloc[i]

        if typ == "EVENT_LABEL":
            result["events"].append(i)

        elif typ == "ANALOG":
            if has_emg_raw:
                if sub == "EMG_RAW":
                    result["emg"].append(i)
            else:
                result["emg"].append(i)

        elif typ == "LINK_MODEL_BASED":
            result["kin"].append(i)

        elif typ in ("METRIC", "DERIVED"):
            result["scalar"].append(i)

    return result


# -----------------------------------------------------------------------------
# Hilfsfunktionen: DataFrames aufbauen
# -----------------------------------------------------------------------------

def build_trial_dataframe(
    data:       pd.DataFrame,
    header:     pd.DataFrame,
    col_map:    dict[str, list[int]],
    subject_id: str,
    fs:         float,
    phase:      str = "",
    movement:   str = "",
) -> pd.DataFrame:
    """
    Baut den kombinierten DataFrame fuer einen Trial auf:

      time_s | scalar | kinematik | EMG

    Kinematik-Spaltennamen folgen der _make_col_name-Konvention.
    EMG-Spaltennamen kommen direkt aus Zeile 1 (Muskelname).
    Scalar-Spaltennamen kommen direkt aus Zeile 1.

    phase und movement werden fuer die Subject-spezifische Skalierung benoetigt.
    """
    row1 = header.iloc[1].astype(str)
    # Achsen-Zeile (Zeile 4) nur wenn 5-Zeilen-Format vorhanden
    row4 = header.iloc[4].astype(str) if len(header) >= 5 \
           else pd.Series([""] * len(header.columns))

    n = len(data)

    # --- Zeitspalte ---
    time_s = pd.Series(np.arange(n) / fs, name="time_s")

    # --- Event-Spalten ---
    # Jeder Event (start, take_off, landing, end_jump, landing1, landing2)
    # hat genau einen Zeitwert in Sekunden in der ersten Datenzeile.
    # Alle weiteren Zeilen sind NaN (Events sind Zeitpunkte, keine Zeitreihen).
    # Die Werte werden als Konstante in einer eigenen Spalte gespeichert,
    # damit sie direkt beim Plotten und bei der Segmentierung greifbar sind.
    # Spaltenname-Schema: "event_<name>_s"  (z.B. "event_take_off_s")
    event_cols = []
    for i in col_map["events"]:
        event_name = row1.iloc[i].strip()
        col_name   = f"event_{event_name}_s"
        # Zeitwert aus Zeile 0 der Datenspalte lesen
        t_val = pd.to_numeric(data.iloc[0, i], errors="coerce")
        # Als konstante Spalte speichern (nur Zeile 0 hat den Wert, Rest NaN - beides OK)
        series = pd.Series([t_val] + [np.nan] * (n - 1), name=col_name)
        event_cols.append(series)

    # --- Scalar-Spalten ---
    scalar_cols = []
    for i in col_map["scalar"]:
        col_name = row1.iloc[i].strip()
        series   = data.iloc[:, i].apply(pd.to_numeric, errors="coerce")
        series.name = col_name
        scalar_cols.append(series)

    # --- Kinematik-Spalten ---
    kin_cols = []
    for i in col_map["kin"]:
        col_name = _make_col_name(row1.iloc[i], row4.iloc[i])
        series   = data.iloc[:, i].apply(pd.to_numeric, errors="coerce")
        series.name = col_name
        kin_cols.append(series)

    # --- EMG-Spalten ---
    # Einige Subjects wurden mit anderer Verstaerkerstufe aufgezeichnet
    # und muessen mit Faktor 1.000.000 skaliert werden (Volt -> µV):
    #
    #   S08: alle Phasen, alle Uebungen
    #   S09: NUR Phase 02_OVU + Uebung SQ und CMJ
    #   S11: NUR Phase 03_LUT + Uebung DJ  (alle anderen S11-Dateien sind OK)
    #
    # WICHTIG: bei neuen Sonderfaellen hier ergaenzen.
    emg_cols = []
    for i in col_map["emg"]:
        col_name = row1.iloc[i].strip()
        series   = data.iloc[:, i].apply(pd.to_numeric, errors="coerce")
        series.name = col_name
        emg_cols.append(series)

    needs_scaling = (
        (subject_id == "S08" and not (phase == "02_OVU" and movement == "CMJ"))
        or (subject_id == "S11" and phase == "03_LUT" and movement == "DJ")
        or (subject_id == "S11" and phase == "03_LUT" and movement == "SQ")
        or (subject_id == "S09" and phase == "02_OVU" and movement == "SQ")
        or (subject_id == "S09" and phase == "02_OVU" and movement == "CMJ")
    )
    if needs_scaling and emg_cols:
        emg_cols = [s * 1_000_000 for s in emg_cols]
        print(f"  [SKALIERUNG] {subject_id} | {phase} | {movement} "
              f"-> * 1.000.000 angewendet")
        

    # Alles zusammenfuehren:
    #   time_s | events | scalar | kinematik | EMG
    #
    # Events stehen direkt nach time_s damit sie beim Plotten
    # sofort greifbar sind ohne die Spalten zu durchsuchen.
    all_parts = [time_s] + event_cols + scalar_cols + kin_cols + emg_cols
    return pd.concat(all_parts, axis=1)


# -----------------------------------------------------------------------------
# Hilfsfunktionen: Ordnerstruktur
# -----------------------------------------------------------------------------

def trial_stem_to_folder_and_side(trial_stem: str) -> tuple[str, str]:
    """
    Leitet Bewegungsordner (CMJ / DJ / SQ / OTHER) und
    Seite (BILATERAL / LEFT / RIGHT) aus dem Trial-Namen ab.
    """
    s = trial_stem.lower()
    side = (
        "LEFT"      if "left"  in s else
        "RIGHT"     if "right" in s else
        "BILATERAL"
    )
    movement = (
        "CMJ"   if "counter-movement jump" in s or "counter movement jump" in s else
        "DJ"    if "drop jump" in s else
        "SQ"    if "squat" in s else
        "OTHER"
    )
    return movement, side


# -----------------------------------------------------------------------------
# Hauptfunktion pro Datei
# -----------------------------------------------------------------------------

def preprocess_file(file_path: Path) -> int:
    """
    Verarbeitet eine einzelne CSV vollstaendig:
      1. Metadaten aus Dateinamen lesen
      2. Header analysieren -> Spalten-Map je Trial
      3. Daten laden
      4. Pro Trial kombinierten DataFrame speichern

    Gibt die Anzahl gespeicherter Trial-Dateien zurueck.
    """
    subject_id = get_subject_id_from_filename(file_path)
    phase      = detect_phase(file_path)
    fs         = get_sampling_rate_for_subject(subject_id)
    n_hdr      = _n_header_rows(subject_id)

    if phase is None:
        print(f"[FEHLER] Phase nicht erkannt: {file_path} – uebersprungen.")
        return 0

    print(f"\n=== {file_path.name} | {subject_id} | {phase} | fs={fs} Hz ===")

    # Header einlesen
    header      = read_header(file_path, n_hdr)
    row0        = header.iloc[0].astype(str)
    row2        = header.iloc[2].astype(str).str.upper()
    row3        = header.iloc[3].astype(str).str.upper() if n_hdr >= 4 \
                  else pd.Series([""] * len(header.columns))
    has_emg_raw = (row3 == "EMG_RAW").any()

    trial_paths = _get_trial_paths_in_order(row0)
    if not trial_paths:
        print("  [FEHLER] Keine Trials gefunden – uebersprungen.")
        return 0

    # Datenteil einlesen
    data  = pd.read_csv(file_path, header=None, skiprows=n_hdr, low_memory=False)
    saved = 0

    for trial_path in trial_paths:
        trial_stem = Path(trial_path).stem
        short_name = make_short_trial_name(trial_stem)
        movement, side = trial_stem_to_folder_and_side(trial_stem)

        col_map = _col_indices_by_type(row0, row2, row3, trial_path, has_emg_raw)

        if not col_map["emg"]:
            print(f"  [WARNUNG] {short_name}: keine EMG-Spalten – uebersprungen.")
            continue

        trial_df = build_trial_dataframe(
            data, header, col_map, subject_id, fs,
            phase=phase, movement=movement,
        )

        # NaN-Info (informativ, kein Abbruch)
        emg_names = [header.iloc[1].iloc[i].strip() for i in col_map["emg"]]
        n_nan = trial_df[emg_names].isna().all(axis=1).sum()
        if n_nan > 0:
            print(f"  [INFO] {short_name}: {n_nan} vollst. NaN-Zeilen "
                  f"(normal bei unterschiedl. Trial-Laengen)")

        # Ausgabepfad
        out_dir = OUTPUT_DIR / subject_id / phase / movement / side
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{short_name}.csv"
        trial_df.to_csv(out_path, index=False)

        n_sca = len(col_map["scalar"])
        n_kin = len(col_map["kin"])
        n_emg = len(col_map["emg"])
        print(f"  [OK] {out_path.relative_to(OUTPUT_DIR)}  "
              f"({n_sca} scalar | {n_kin} kin | {n_emg} EMG | {len(trial_df)} Zeilen)")
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
            n = preprocess_file(file_path)
            total_saved += n
        except Exception as e:
            print(f"[FEHLER] {file_path.name}: {e}")
            total_errors += 1

    print(f"\n{'='*55}")
    print("FERTIG")
    print(f"  Gespeicherte Trial-Dateien : {total_saved}")
    print(f"  Dateien mit Fehler         : {total_errors}")
    print(f"  Output-Ordner              : {OUTPUT_DIR}")
    print(f"{'='*55}")