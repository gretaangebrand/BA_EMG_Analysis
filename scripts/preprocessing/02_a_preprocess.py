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
# Liest anonymisierte CSV-Dateien ein und speichert pro Trial ZWEI Dateien:
#
#   <trial>_emg.csv  – EMG-Rohdaten bei voller Analog-Rate (2000/2100 Hz)
#   <trial>_kin.csv  – Kinematik (GRF, Angles, Moments, Power) bei Video-Rate
#
# Beide Dateien enthalten Event- und Scalar-Spalten als Metadaten
# (nur Zeile 0 hat Werte, Rest NaN).
#
# Hintergrund:
#   Die Quell-CSVs aus Vicon enthalten Zeilen bei der Analog-Rate
#   (= EMG-Abtastrate). Kinematik-Werte erscheinen nur jede N-te Zeile
#   (N = fs_emg / fs_video), der Rest ist NaN.
#   Die Event-Zeitpunkte sind in absoluter Vicon-Session-Zeit angegeben.
#   Die Daten jedes Trials beginnen beim fruehesten Event und enden
#   beim spaetesten -> time_s = event_start + row_index / fs_emg.
#
# Spaltenkonvention Kinematik:
#   X-Achse (Hauptkomponente): nur Variablenname, z.B. "Left Knee Angles"
#   Y- und Z-Achse:            Variablenname + Suffix, z.B. "Left Knee Angles_Y"
#
# Ordnerstruktur Output:
#   <OUTPUT_DIR>/<subject_id>/<phase>/<bewegung>/<seite>/<trial>_emg.csv
#   <OUTPUT_DIR>/<subject_id>/<phase>/<bewegung>/<seite>/<trial>_kin.csv
# =============================================================================

# -----------------------------------------------------------------------------
# Pfade
# -----------------------------------------------------------------------------
SOURCE_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\02_anonymized_csv_data")
OUTPUT_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\03_preprocessed_emg_data")

# Anzahl Header-Zeilen je Subject-Gruppe
_HEADER_ROWS_DEFAULT = 5   # S01-S07
_HEADER_ROWS_SPECIAL = 4   # S08 (und ggf. S09-S11)
_SPECIAL_SUBJECTS    = {}


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
# Hilfsfunktionen: Event-Offset und Sampling-Rate ermitteln
# -----------------------------------------------------------------------------

def _read_event_times(data: pd.DataFrame, header: pd.DataFrame,
                      col_map: dict) -> dict[str, float]:
    """
    Liest alle Event-Zeitpunkte (absolut, in Sekunden) aus Zeile 0.
    Gibt dict mit {event_name: zeit_s} zurueck.
    """
    row1 = header.iloc[1].astype(str)
    events = {}
    for i in col_map["events"]:
        name  = row1.iloc[i].strip()
        t_val = pd.to_numeric(data.iloc[0, i], errors="coerce")
        if not np.isnan(t_val):
            events[name] = float(t_val)
    return events


def _derive_fs_emg(data: pd.DataFrame, col_map: dict,
                   events: dict[str, float],
                   fs_fallback: float) -> float:
    """
    Leitet die tatsaechliche EMG-Abtastrate aus den Daten ab:
      fs = (n_valid_emg_samples - 1) / event_span

    Faellt auf fs_fallback zurueck wenn nicht ableitbar.
    """
    if len(events) < 2 or not col_map["emg"]:
        return fs_fallback

    # Anzahl gueltiger EMG-Samples (erster EMG-Kanal)
    emg_vals = pd.to_numeric(data.iloc[:, col_map["emg"][0]], errors="coerce")
    n_valid  = emg_vals.notna().sum()
    if n_valid < 2:
        return fs_fallback

    evt_span = max(events.values()) - min(events.values())
    if evt_span <= 0:
        return fs_fallback

    fs_derived = (n_valid - 1) / evt_span
    # Auf bekannte Raten runden (2000 oder 2100)
    fs_rounded = round(fs_derived / 100) * 100
    if fs_rounded in (2000, 2100):
        return float(fs_rounded)

    return fs_fallback


# -----------------------------------------------------------------------------
# Hilfsfunktionen: DataFrames aufbauen
# -----------------------------------------------------------------------------

def _build_meta_columns(data: pd.DataFrame, header: pd.DataFrame,
                        col_map: dict, n_rows: int) -> list[pd.Series]:
    """
    Baut Event- und Scalar-Spalten auf.
    Nur Zeile 0 hat den Wert, Rest NaN (Metadaten, keine Zeitreihen).
    """
    row1 = header.iloc[1].astype(str)
    cols = []

    # Events
    for i in col_map["events"]:
        event_name = row1.iloc[i].strip()
        col_name   = f"event_{event_name}_s"
        t_val = pd.to_numeric(data.iloc[0, i], errors="coerce")
        series = pd.Series([t_val] + [np.nan] * (n_rows - 1), name=col_name)
        cols.append(series)

    # Scalars
    for i in col_map["scalar"]:
        col_name = row1.iloc[i].strip()
        val = pd.to_numeric(data.iloc[0, i], errors="coerce")
        series = pd.Series([val] + [np.nan] * (n_rows - 1), name=col_name)
        cols.append(series)

    return cols


def build_emg_dataframe(
    data:       pd.DataFrame,
    header:     pd.DataFrame,
    col_map:    dict[str, list[int]],
    subject_id: str,
    fs_emg:     float,
    t_start:    float,
    phase:      str = "",
    movement:   str = "",
) -> pd.DataFrame:
    """
    Baut den EMG-DataFrame auf:
      time_s | events | scalars | EMG-Spalten

    Nur Zeilen mit gueltigen EMG-Daten werden behalten.
    time_s = t_start + row_index / fs_emg  (absolute Vicon-Zeit).
    """
    row1 = header.iloc[1].astype(str)

    # EMG-Spalten extrahieren
    emg_series = []
    for i in col_map["emg"]:
        col_name = row1.iloc[i].strip()
        s = pd.to_numeric(data.iloc[:, i], errors="coerce")
        s.name = col_name
        emg_series.append(s)

    # Maske: Zeilen wo mindestens ein EMG-Kanal gueltig ist
    emg_df_raw = pd.concat(emg_series, axis=1)
    valid_mask = emg_df_raw.notna().any(axis=1)
    n_valid    = valid_mask.sum()

    if n_valid == 0:
        return pd.DataFrame()

    # Gueltige Zeilen-Indizes (fuer Zeitachsen-Berechnung)
    valid_indices = np.where(valid_mask.values)[0]

    # Zeitachse: absolute Vicon-Zeit
    time_s = pd.Series(
        t_start + valid_indices / fs_emg,
        name="time_s",
    ).reset_index(drop=True)

    # EMG-Daten nur gueltige Zeilen
    emg_valid = emg_df_raw.loc[valid_mask].reset_index(drop=True)

    # EMG-Skalierung (Volt -> µV) fuer bestimmte Subjects
    needs_scaling = (
        (subject_id == "S08" and not (phase == "02_OVU" and movement == "CMJ"))
        or (subject_id == "S11" and phase == "03_LUT" and movement == "DJ")
        or (subject_id == "S11" and phase == "03_LUT" and movement == "SQ")
        or (subject_id == "S09" and phase == "02_OVU" and movement == "SQ")
        or (subject_id == "S09" and phase == "02_OVU" and movement == "CMJ")
    )
    if needs_scaling:
        emg_valid = emg_valid * 1_000_000
        print(f"  [SKALIERUNG] {subject_id} | {phase} | {movement} "
              f"-> * 1.000.000 angewendet")

    # Meta-Spalten (Events + Scalars)
    meta = _build_meta_columns(data, header, col_map, n_valid)

    # Zusammenfuegen: time_s | events | scalars | EMG
    all_parts = [time_s] + meta + [emg_valid]
    return pd.concat(all_parts, axis=1)


def build_kin_dataframe(
    data:       pd.DataFrame,
    header:     pd.DataFrame,
    col_map:    dict[str, list[int]],
    fs_emg:     float,
    t_start:    float,
) -> pd.DataFrame:
    """
    Baut den Kinematik-DataFrame auf:
      time_s | events | scalars | Kinematik-Spalten

    Nur Zeilen mit gueltigen Kinematik-Daten werden behalten.
    time_s = t_start + row_index / fs_emg  (absolute Vicon-Zeit).
    Hinweis: row_index bezieht sich auf die Analog-Zeilen der Quelldatei.
    """
    row1 = header.iloc[1].astype(str)
    row4 = header.iloc[4].astype(str) if len(header) >= 5 \
           else pd.Series([""] * len(header.columns))

    if not col_map["kin"]:
        return pd.DataFrame()

    # Kinematik-Spalten extrahieren
    kin_series = []
    for i in col_map["kin"]:
        col_name = _make_col_name(row1.iloc[i], row4.iloc[i])
        s = pd.to_numeric(data.iloc[:, i], errors="coerce")
        s.name = col_name
        kin_series.append(s)

    kin_df_raw = pd.concat(kin_series, axis=1)
    valid_mask = kin_df_raw.notna().any(axis=1)
    n_valid    = valid_mask.sum()

    if n_valid == 0:
        return pd.DataFrame()

    valid_indices = np.where(valid_mask.values)[0]

    # Zeitachse: gleiche Basis wie EMG (row_index / fs_emg)
    time_s = pd.Series(
        t_start + valid_indices / fs_emg,
        name="time_s",
    ).reset_index(drop=True)

    # Kinematik-Daten nur gueltige Zeilen
    kin_valid = kin_df_raw.loc[valid_mask].reset_index(drop=True)

    # Meta-Spalten (Events + Scalars)
    meta = _build_meta_columns(data, header, col_map, n_valid)

    # Zusammenfuegen: time_s | events | scalars | Kinematik
    all_parts = [time_s] + meta + [kin_valid]
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
      4. Pro Trial EMG- und Kinematik-DataFrame separat speichern

    Gibt die Anzahl gespeicherter Trial-Dateien zurueck (EMG + KIN).
    """
    subject_id = get_subject_id_from_filename(file_path)
    phase      = detect_phase(file_path)
    fs_hint    = get_sampling_rate_for_subject(subject_id)
    n_hdr      = _n_header_rows(subject_id)

    if phase is None:
        print(f"[FEHLER] Phase nicht erkannt: {file_path} – uebersprungen.")
        return 0

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

    # fs_emg wird aus dem ersten Trial abgeleitet und fuer alle Trials
    # in dieser Datei verwendet (selbe Aufnahme-Session).
    fs_emg   = None
    fs_video = None

    for trial_path in trial_paths:
        trial_stem = Path(trial_path).stem
        short_name = make_short_trial_name(trial_stem)
        movement, side = trial_stem_to_folder_and_side(trial_stem)

        col_map = _col_indices_by_type(row0, row2, row3, trial_path, has_emg_raw)

        if not col_map["emg"]:
            print(f"  [WARNUNG] {short_name}: keine EMG-Spalten – uebersprungen.")
            continue

        # Event-Zeiten lesen
        events = _read_event_times(data, header, col_map)

        # EMG-Rate aus erstem Trial mit Events ableiten;
        # fuer Trials ohne Events (z.B. Squats) wird fs_hint verwendet.
        if fs_emg is None:
            if events:
                fs_emg = _derive_fs_emg(data, col_map, events, fs_hint)
            else:
                fs_emg = fs_hint

            # Video-Rate ableiten aus EMG/Kin-Verhaeltnis
            n_emg_valid = pd.to_numeric(
                data.iloc[:, col_map["emg"][0]], errors="coerce"
            ).notna().sum()
            n_kin_valid = 0
            if col_map["kin"]:
                n_kin_valid = pd.to_numeric(
                    data.iloc[:, col_map["kin"][0]], errors="coerce"
                ).notna().sum()
            if n_kin_valid > 0:
                ratio = round(n_emg_valid / n_kin_valid)
                fs_video = fs_emg / ratio
            else:
                fs_video = fs_emg / 8   # Fallback

            print(f"\n=== {file_path.name} | {subject_id} | {phase} "
                  f"| fs_emg={fs_emg:.0f} Hz | fs_video={fs_video:.0f} Hz ===")

        # Start-Offset: fruehestes Event = Zeitpunkt von Zeile 0.
        # Ohne Events (z.B. Squats): Zeitachse beginnt bei 0.
        if events:
            t_start = min(events.values())
        else:
            t_start = 0.0
            print(f"  [INFO] {short_name}: keine Events – time_s beginnt bei 0.")

        # Output-Ordner
        out_dir = OUTPUT_DIR / subject_id / phase / movement / side
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- EMG-Datei ----
        emg_df = build_emg_dataframe(
            data, header, col_map, subject_id, fs_emg, t_start,
            phase=phase, movement=movement,
        )
        if not emg_df.empty:
            out_emg = out_dir / f"{short_name}_emg.csv"
            emg_df.to_csv(out_emg, index=False)
            emg_data_cols = [c for c in emg_df.columns
                            if c != "time_s"
                            and not c.startswith("event_")
                            and c not in [header.iloc[1].iloc[j].strip()
                                          for j in col_map["scalar"]]]
            print(f"  [EMG] {out_emg.relative_to(OUTPUT_DIR)}  "
                  f"({len(emg_data_cols)} Kanäle | {len(emg_df)} Samples "
                  f"| {len(emg_df)/fs_emg:.2f} s @ {fs_emg:.0f} Hz)")
            saved += 1
        else:
            print(f"  [WARNUNG] {short_name}: keine gueltigen EMG-Daten.")

        # ---- Kinematik-Datei ----
        if col_map["kin"]:
            kin_df = build_kin_dataframe(
                data, header, col_map, fs_emg, t_start,
            )
            if not kin_df.empty:
                out_kin = out_dir / f"{short_name}_kin.csv"
                kin_df.to_csv(out_kin, index=False)
                kin_data_cols = [c for c in kin_df.columns
                                if c != "time_s"
                                and not c.startswith("event_")
                                and c not in [header.iloc[1].iloc[j].strip()
                                              for j in col_map["scalar"]]]
                print(f"  [KIN] {out_kin.relative_to(OUTPUT_DIR)}  "
                      f"({len(kin_data_cols)} Kanäle | {len(kin_df)} Samples "
                      f"| {len(kin_df)/fs_video:.2f} s @ {fs_video:.0f} Hz)")
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

    print(f"\n{'='*65}")
    print("FERTIG")
    print(f"  Gespeicherte Dateien (EMG+KIN) : {total_saved}")
    print(f"  Dateien mit Fehler             : {total_errors}")
    print(f"  Output-Ordner                  : {OUTPUT_DIR}")
    print(f"{'='*65}")