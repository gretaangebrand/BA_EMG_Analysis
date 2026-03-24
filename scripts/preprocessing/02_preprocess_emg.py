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

# Beispiel für unterschiedliche Header-Behandlung
def load_emg_csv(file_path: Path, is_special_case=False) -> pd.DataFrame:
    """
    Liest eine CSV mit 5 Header-Zeilen ein, aber behandelt spezielle Fälle (z.B. S08 bis S11).
    """
    if is_special_case:
        # Hier könnte für S08 bis S11 die Header-Zeilen-Anpassung vorgenommen werden
        df = pd.read_csv(file_path, sep=CSV_SEPARATOR, header=4)  # Beispiel für 4 Header-Zeilen
    else:
        df = pd.read_csv(file_path, sep=CSV_SEPARATOR, header=HEADER_ROWS)
    
    return df


# ============================================================
# FUNKTIONEN
# ============================================================

def load_emg_csv(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path, sep=CSV_SEPARATOR, header=HEADER_ROWS)

def remove_item_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, df.columns.get_level_values(4) != "ITEM"].copy()

def extract_emg_columns(df: pd.DataFrame) -> pd.DataFrame:
    level_2 = df.columns.get_level_values(2).astype(str).str.upper()
    level_3 = df.columns.get_level_values(3).astype(str).str.upper()
    emg_mask = (level_2 == "ANALOG") & (level_3 == "EMG_RAW")
    return df.loc[:, emg_mask].copy()

def get_trial_names(emg_df: pd.DataFrame) -> list[str]:
    return pd.Index(emg_df.columns.get_level_values(0)).dropna().unique().tolist()

def extract_trial_dataframe(emg_df: pd.DataFrame, trial_name: str, subject_id: str, fs: float) -> pd.DataFrame:
    trial_mask = emg_df.columns.get_level_values(0) == trial_name
    trial_df = emg_df.loc[:, trial_mask].copy()
    trial_df.columns = trial_df.columns.get_level_values(1)
    trial_df = trial_df.apply(pd.to_numeric, errors="coerce")
    trial_df = apply_s08_scaling(trial_df, subject_id)
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

    print(f"\n=== {file_path.name} | {subject_id} | {phase} | {movement_type} ===")

    df = load_emg_csv(file_path)
    df = remove_item_column(df)
    time = np.arange(len(df)) / fs
    df["time_s"] = time

    emg_df = extract_emg_columns(df)
    trials = get_trial_names(emg_df)

    bilateral = build_trial_dictionary(emg_df, get_bilateral_trials(trials), subject_id, fs)
    left = build_trial_dictionary(emg_df, get_left_trials(trials), subject_id, fs)
    right = build_trial_dictionary(emg_df, get_right_trials(trials), subject_id, fs)

    return {
        "subject_id": subject_id,
        "phase": phase,
        "movement_type": movement_type,
        "bilateral": bilateral,
        "left": left,
        "right": right
    }

def save_preprocessed_trials(preprocessed: dict, out_dir: Path):
    subject_id = preprocessed["subject_id"]
    phase = preprocessed["phase"]
    movement_type = preprocessed["movement_type"]
    for group, trials in {
        "BILATERAL": preprocessed["bilateral"],
        "LEFT": preprocessed["left"],
        "RIGHT": preprocessed["right"]
    }.items():
        for name, df in trials.items():
            target_dir = out_dir / subject_id / phase / movement_type
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / f"{name}.csv"
            df.to_csv(out_path, index=False)
            print(f"[SAVED] {out_path.relative_to(out_dir)}")

# ============================================================
# HAUPTAUFRUF: alle Dateien verarbeiten
# ============================================================

if __name__ == "__main__":
    all_files = list(SOURCE_DIR.rglob("*.csv"))
    print(f"Gefundene CSV-Dateien: {len(all_files)}")

    for file_path in sorted(all_files):
        preprocessed = preprocess_emg_file(file_path)
        save_preprocessed_trials(preprocessed, OUTPUT_DIR)

    print(f"\n>>> Alle Dateien erfolgreich in {OUTPUT_DIR} gespeichert.")
