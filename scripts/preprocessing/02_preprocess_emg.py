from pathlib import Path
import pandas as pd
import sys

# Projekt-Root zum Python-Pfad hinzufügen
#sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.utils.helpers import (
    get_subject_id_from_filename,
    get_bilateral_trials,
    get_left_trials,
    get_right_trials,
    apply_s08_scaling,
    make_short_trial_name,
)

# ============================================================
# EINSTELLUNGEN
# ============================================================

TEST_FILE = Path(
    r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\anonymized_csv_data\01_PER\CMJ\S01_01_PER_CMJ.csv"
)

CSV_SEPARATOR = "\t"
HEADER_ROWS = [0, 1, 2, 3, 4]

# ============================================================
# EINLESEN
# ============================================================

def load_emg_csv(file_path: Path) -> pd.DataFrame:
    """
    Liest eine CSV mit 5 Header-Zeilen ein.
    """
    df = pd.read_csv(file_path, sep=CSV_SEPARATOR, header=HEADER_ROWS)
    return df


def remove_item_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entfernt die ITEM-Spalte.
    """
    return df.loc[:, df.columns.get_level_values(4) != "ITEM"].copy()


# ============================================================
# EMG-SPALTEN FILTERN
# ============================================================

def extract_emg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wählt nur echte EMG-Spalten aus:
    Ebene 2 = ANALOG
    Ebene 3 = EMG_RAW
    """
    level_2 = df.columns.get_level_values(2).astype(str).str.upper()
    level_3 = df.columns.get_level_values(3).astype(str).str.upper()

    emg_mask = (level_2 == "ANALOG") & (level_3 == "EMG_RAW")

    return df.loc[:, emg_mask].copy()


def get_trial_names(emg_df: pd.DataFrame) -> list[str]:
    """
    Liefert alle Trialnamen aus der ersten Header-Ebene.
    """
    trial_names = pd.Index(emg_df.columns.get_level_values(0)).dropna().unique().tolist()
    return trial_names


# ============================================================
# TRIALS EXTRAHIEREN
# ============================================================

def extract_trial_dataframe(emg_df: pd.DataFrame, trial_name: str, subject_id: str) -> pd.DataFrame:
    """
    Extrahiert für einen Trial ein sauberes DataFrame mit Muskelnamen
    als Spalten.
    """
    trial_mask = emg_df.columns.get_level_values(0) == trial_name
    trial_df = emg_df.loc[:, trial_mask].copy()

    # Muskelnamen als einfache Spaltennamen setzen
    muscle_names = trial_df.columns.get_level_values(1)
    trial_df.columns = muscle_names

    # numerisch umwandeln
    trial_df = trial_df.apply(pd.to_numeric, errors="coerce")

    # S08 skalieren
    trial_df = apply_s08_scaling(trial_df, subject_id)

    return trial_df


def build_trial_dictionary(emg_df: pd.DataFrame, trial_names: list[str], subject_id: str) -> dict[str, pd.DataFrame]:
    """
    Baut ein Dictionary: short_trial_name -> trial_df
    """
    trial_data = {}

    for trial_name in trial_names:
        trial_df = extract_trial_dataframe(emg_df, trial_name, subject_id)
        short_name = make_short_trial_name(trial_name)
        trial_data[short_name] = trial_df

    return trial_data


# ============================================================
# HAUPTVERARBEITUNG FÜR EINE DATEI
# ============================================================

def preprocess_emg_file(file_path: Path) -> dict:
    """
    Verarbeitet eine EMG-Datei vollständig.
    """
    subject_id = get_subject_id_from_filename(file_path)

    print("=" * 80)
    print(f"Verarbeite Datei: {file_path.name}")
    print(f"Subject ID: {subject_id}")
    print("=" * 80)

    df = load_emg_csv(file_path)
    print(f"Original Shape: {df.shape}")

    df = remove_item_column(df)
    print(f"Shape ohne ITEM-Spalte: {df.shape}")

    emg_df = extract_emg_columns(df)
    print(f"Shape nur EMG: {emg_df.shape}")

    all_trial_names = get_trial_names(emg_df)
    print(f"Anzahl aller EMG-Trials: {len(all_trial_names)}")

    bilateral_trials = get_bilateral_trials(all_trial_names)
    left_trials = get_left_trials(all_trial_names)
    right_trials = get_right_trials(all_trial_names)

    print(f"Bilaterale Trials: {len(bilateral_trials)}")
    print(f"Linke Trials: {len(left_trials)}")
    print(f"Rechte Trials: {len(right_trials)}")

    bilateral_data = build_trial_dictionary(emg_df, bilateral_trials, subject_id)
    left_data = build_trial_dictionary(emg_df, left_trials, subject_id)
    right_data = build_trial_dictionary(emg_df, right_trials, subject_id)

    return {
        "file_path": file_path,
        "subject_id": subject_id,
        "raw_df": df,
        "emg_df": emg_df,
        "all_trial_names": all_trial_names,
        "bilateral_trials": bilateral_trials,
        "left_trials": left_trials,
        "right_trials": right_trials,
        "bilateral_data": bilateral_data,
        "left_data": left_data,
        "right_data": right_data,
    }


# ============================================================
# TESTAUSGABE
# ============================================================

def print_trial_summary(trial_dict: dict[str, pd.DataFrame], title: str):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)

    if not trial_dict:
        print("Keine Trials gefunden.")
        return

    for trial_name, trial_df in trial_dict.items():
        print(f"\nTrial: {trial_name}")
        print(f"Shape: {trial_df.shape}")
        print("Spalten:")
        for col in trial_df.columns:
            print(f"  - {col}")


if __name__ == "__main__":
    result = preprocess_emg_file(TEST_FILE)

    print_trial_summary(result["bilateral_data"], "BILATERALE TRIALS")
    print_trial_summary(result["left_data"], "LEFT TRIALS")
    print_trial_summary(result["right_data"], "RIGHT TRIALS")