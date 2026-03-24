from pathlib import Path
import pandas as pd
import numpy as np
import shutil


from scripts.utils.helpers import (
    get_sampling_rate_for_subject,
    get_subject_id_from_filename,
    detect_phase,
    detect_movement,
    get_bilateral_trials,
    get_left_trials,
    get_right_trials,
    apply_s08_scaling,
    make_short_trial_name,
)

# ============================================================
# EINSTELLUNGEN
# ============================================================
SOURCE_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\anonymized_csv_data")
OUTPUT_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\preprocessed_emg_data")

CSV_SEPARATOR = ","
HEADER_ROWS = [0, 1, 2, 3, 4]

# ============================================================
# FUNKTIONEN
# ============================================================

# unterschiedliche Header-Behandlung
def load_emg_csv(file_path: Path, subject_id: str) -> pd.DataFrame:
    """
    Liest eine CSV ein und unterdrückt DtypeWarnings.
    """
    # S08 bis S11 haben 4 Header-Zeilen, andere haben 5
    special_subjects = {"S08"} #"S09", "S10", "S11"
    h_rows = [0, 1, 2, 3] if subject_id in special_subjects else [0, 1, 2, 3, 4]
    
    return pd.read_csv(
        file_path, 
        sep=CSV_SEPARATOR, 
        header=h_rows, 
        low_memory=False  # <--- HIER wird die Fehlermeldung unterdrückt
    )


def remove_item_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entfernt ITEM-Spalten sicher, egal wie viele Header-Level existieren.
    """
    mask = np.ones(len(df.columns), dtype=bool)
    for level in range(df.columns.nlevels):
        # Nutze .str.upper(), um die Operation auf alle Elemente anzuwenden
        level_values = df.columns.get_level_values(level).astype(str).str.upper()
        mask &= (level_values != "ITEM")
    return df.loc[:, mask].copy()

def extract_emg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sucht ANALOG flexibel in allen Header-Ebenen. 
    Bei S01-S07 wird zusätzlich auf EMG_RAW geprüft, bei S08-S11 nur auf ANALOG.
    """
    column_masks = []
    for level in range(df.columns.nlevels):
        vals = df.columns.get_level_values(level).astype(str).str.upper()
        column_masks.append(vals)

    # Grundbedingung: ANALOG muss irgendwo stehen
    is_analog = np.any([v == "ANALOG" for v in column_masks], axis=0)
    
    # Check ob EMG_RAW irgendwo vorkommt (typisch für S01-S07)
    has_emg_raw_anywhere = np.any([v == "EMG_RAW" for v in column_masks])
    
    if has_emg_raw_anywhere:
        # Wenn EMG_RAW existiert (S01-S07), muss es auch in der Spalte sein
        is_emg = np.any([v == "EMG_RAW" for v in column_masks], axis=0)
        return df.loc[:, is_analog & is_emg].copy()
    else:
        # Wenn kein EMG_RAW im gesamten File (S08-S11), nehmen wir alle ANALOG Spalten
        # Das sind dann direkt die Muskeln
        return df.loc[:, is_analog].copy()

def get_trial_names(emg_df: pd.DataFrame) -> list[str]:
    return pd.Index(emg_df.columns.get_level_values(0)).dropna().unique().tolist()

def extract_trial_dataframe(emg_df: pd.DataFrame, trial_name: str, subject_id: str, fs: float) -> pd.DataFrame:
    # 1. Wähle die Spalten für das spezifische Trial aus
    trial_mask = emg_df.columns.get_level_values(0) == trial_name
    trial_df = emg_df.loc[:, trial_mask].copy()
    
    # 2. DER VORTEIL: Wir setzen die Namen fix auf Level 1 (die Muskelnamen)
    # Egal ob Subject S01 oder S08 - Level 1 enthält laut deiner Info immer die Muskeln.
    trial_df.columns = trial_df.columns.get_level_values(1)
    
    # 3. Bereinigung
    trial_df = trial_df.apply(pd.to_numeric, errors="coerce")
    trial_df = apply_s08_scaling(trial_df, subject_id) # Skalierung nur für S08
    
    # 4. Zeitspalte einfügen
    trial_df.insert(0, "time_s", np.arange(len(trial_df)) / fs)
    return trial_df

def build_trial_dictionary(emg_df, trial_names, subject_id, fs):
    data = {}
    for name in trial_names:
        df = extract_trial_dataframe(emg_df, name, subject_id, fs)
        short_name = make_short_trial_name(name)
        data[short_name] = df
    return data

def preprocess_emg_file(file_path: Path) -> dict:
    subject_id = get_subject_id_from_filename(file_path)
    phase = detect_phase(file_path)
    movement_type = detect_movement(file_path)
    fs = get_sampling_rate_for_subject(subject_id)

    # Kurzes Ordner-Kürzel für die Struktur (CMJ, DJ, SQ)
    folder_map = {"Counter-Movement": "CMJ", "Drop": "DJ", "Squat": "SQ"}
    folder_name = "OTHER"
    for key, val in folder_map.items():
        if key in str(movement_type):
            folder_name = val
            break

    print(f"\n=== {file_path.name} | {subject_id} | {phase} | {movement_type} ===")

    df = load_emg_csv(file_path, subject_id)
    df = remove_item_column(df)

    time = np.arange(len(df)) / fs
    df["time_s"] = time

    emg_df = extract_emg_columns(df)
    trials = get_trial_names(emg_df)

    bilateral = build_trial_dictionary(emg_df, get_bilateral_trials(trials), subject_id, fs)
    left = build_trial_dictionary(emg_df, get_left_trials(trials), subject_id, fs)
    right = build_trial_dictionary(emg_df, get_right_trials(trials), subject_id, fs)

    return {
        "subject_id": subject_id, "phase": phase, "movement_type": movement_type,
        "folder_name": folder_name, "bilateral": bilateral, "left": left, "right": right
    }

def save_preprocessed_trials(preprocessed: dict, out_dir: Path):
    """Speichert die Trials in die gewünschte Ordnerstruktur."""
    subject_id = preprocessed["subject_id"]
    phase = preprocessed["phase"]
    folder_name = preprocessed["folder_name"]
    movement_type = preprocessed["movement_type"]
    
    for group, trials in {
        "BILATERAL": preprocessed["bilateral"],
        "LEFT": preprocessed["left"],
        "RIGHT": preprocessed["right"]
    }.items():
        for name, df in trials.items():
            # Kurzer Name für den Ordner, voller Name für die Datei
            target_dir = out_dir / subject_id / phase / folder_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Beispiel: SQ_L_01_Squatting Left.csv
            file_name = f"{name}_{movement_type}.csv"
            out_path = target_dir / file_name
            
            df.to_csv(out_path, index=False)
            print(f"[SAVED] {out_path.relative_to(out_dir)}")

if __name__ == "__main__":
    all_files = list(SOURCE_DIR.rglob("*.csv"))
    print(f"Gefundene CSV-Dateien: {len(all_files)}")

    for file_path in sorted(all_files):
        try:
            preprocessed = preprocess_emg_file(file_path)
            save_preprocessed_trials(preprocessed, OUTPUT_DIR)
        except Exception as e:
            print(f"[ERROR] Fehler bei Datei {file_path.name}: {e}")

    print(f"\n>>> Alle Dateien erfolgreich in {OUTPUT_DIR} gespeichert.")
