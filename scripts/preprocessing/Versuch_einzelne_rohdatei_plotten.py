import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# =========================================================
# EINSTELLUNGEN
# =========================================================
#BASE_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\Daten_BA-neu\_EMG_RAW\01_period")
BASE_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\Daten_BA-neu\_EMG_ORIGINAL\01_period")

SAVE_CLEAN_CSV = False   # True, wenn du bereinigte CSVs zusätzlich speichern willst
SHOW_PLOTS = True        # True nur wenn du Plotfenster direkt sehen willst

MUSCLE_COLUMNS = [
    "L_Biceps Femoris",
    "L_Gastrocnemius medial",
    "L_Gluteus Medius",
    "L_Semitendinosus",
    "L_Vastus Lateralis",
    "R_Biceps Femoris",
    "R_Gastrocnemius medial",
    "R_Gluteus Medius",
    "R_Semitendinosus",
    "R_Vastus Lateralis",
]

'''# nur diese Personen berücksichtigen
TARGET_SUBJECTS = {
    "P01_K",
    "P02_A",
    "P05_A",
    "P06_D",
    "P07_P",
    "P09_B",
    "P10_P",
}'''

# nur diese Personen berücksichtigen
TARGET_SUBJECTS = {
    "P01_Batzner",
    "P02_Lorenz",
    "P03_Feik",
    "P04_Platzer"
}

# nur .txt aus diesem Ordner verwenden bzw. mit diesen Exercise-Bezeichnungen
PHASES = ["01_period"]
EXERCISES = ["squatting"]


# =========================================================
# HILFSFUNKTIONEN
# =========================================================
def clean_text(s):
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_trial_index_from_level4(level4: str) -> int:
    """
    Trial-Nummer aus der 4. Header-Ebene lesen:
    X   -> 0
    X.1 -> 1
    X.2 -> 2
    """
    level4 = clean_text(level4)
    m = re.search(r"\.(\d+)$", level4)
    return int(m.group(1)) if m else 0


def detect_phase_and_exercise(file_path: Path):
    phase = None
    exercise = None

    for part in file_path.parts:
        if part in PHASES:
            phase = part
        if part.lower() in EXERCISES:
            exercise = part.lower()

    return phase, exercise


def file_matches_target_subject(file_path: Path) -> bool:
    """
    Prüft, ob einer der gewünschten Personencodes im Dateinamen vorkommt.
    """
    stem = file_path.stem
    return any(subject in stem for subject in TARGET_SUBJECTS)


# =========================================================
# TXT EINLESEN
# =========================================================
def read_raw_txt_with_multiheader(file_path: Path) -> pd.DataFrame:
    """
    Passend für das Exportformat mit mehrzeiligem Header.
    """
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=[1, 2, 3, 4],
        engine="python",
        on_bad_lines="skip"
    )
    return df


# =========================================================
# EMG-TRIALS EXTRAHIEREN
# =========================================================
def extract_emg_trials_from_raw_df(raw_df: pd.DataFrame):
    """
    Extrahiert EMG-Daten pro Trial aus dem MultiHeader-DataFrame.
    """
    trial_columns = {}

    for col in raw_df.columns:
        level0 = clean_text(col[0])  # Muskelname
        level1 = clean_text(col[1])  # ANALOG / METRIC
        level2 = clean_text(col[2])  # ORIGINAL / PROCESSED
        level3 = clean_text(col[3])  # X / X.1 / X.2 / ...

        if (
            level0 in MUSCLE_COLUMNS
            #and level1.upper() == "ANALOG"
            #and level2.upper() == "ORIGINAL"
        ):
            trial_idx = parse_trial_index_from_level4(level3)
            trial_columns.setdefault(trial_idx, {})[level0] = col

    trials = {}

    for trial_idx, muscle_map in sorted(trial_columns.items()):
        out = pd.DataFrame()

        for muscle in MUSCLE_COLUMNS:
            if muscle in muscle_map:
                out[muscle] = pd.to_numeric(raw_df[muscle_map[muscle]], errors="coerce")

        existing_muscles = [c for c in MUSCLE_COLUMNS if c in out.columns]
        if not existing_muscles:
            continue

        # Nur Zeilen behalten, in denen mindestens ein Muskel numerisch ist
        out = out[out[existing_muscles].notna().any(axis=1)].copy()
        out.reset_index(drop=True, inplace=True)

        if out.empty:
            continue

        trials[trial_idx] = out

    return trials


# =========================================================
# TRIALS ZU EINEM LANGEN VERLAUF ZUSAMMENSETZEN
# =========================================================
def combine_trials_to_long_emg(trials: dict):
    """
    Hängt alle Trials einer Datei untereinander.
    Ergebnis:
    Ein langer Verlauf pro Person/Datei.
    """
    if not trials:
        return None

    combined_parts = []

    for trial_idx in sorted(trials.keys()):
        df_trial = trials[trial_idx].copy()
        df_trial["trial"] = trial_idx + 1
        combined_parts.append(df_trial)

    combined_df = pd.concat(combined_parts, ignore_index=True)
    combined_df["sample_global"] = range(len(combined_df))

    return combined_df


# =========================================================
# OPTIONAL: BEREINIGTE CSV SPEICHERN
# =========================================================
def save_combined_csv(df_combined: pd.DataFrame, txt_file: Path):
    csv_name = f"{txt_file.stem}_combined.csv"
    csv_path = txt_file.with_name(csv_name)
    df_combined.to_csv(csv_path, index=False)


# =========================================================
# KOMBINIERTEN PLOT PRO PERSON SPEICHERN
# =========================================================
def plot_combined_emg(df_combined: pd.DataFrame, txt_file: Path, plot_dir: Path):
    muscle_cols = [c for c in MUSCLE_COLUMNS if c in df_combined.columns]
    if not muscle_cols:
        print(f"   -> Keine Muskelspalten zum Plotten in {txt_file.name}")
        return

    x = df_combined["sample_global"] if "sample_global" in df_combined.columns else df_combined.index

    plt.figure(figsize=(16, 8))

    for col in muscle_cols:
        plt.plot(x, df_combined[col], linewidth=0.8, label=col)

    plt.title(f"EMG-Rohdaten kombiniert: {txt_file.stem}")
    plt.xlabel("Sample")
    plt.ylabel("EMG Signal")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    plot_path = plot_dir / f"{txt_file.stem}_combined.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")

    print(f"   -> Plot gespeichert: {plot_path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


# =========================================================
# HAUPTPROGRAMM
# =========================================================
def main():
    txt_files = [
        f for f in BASE_DIR.rglob("*.txt")
        if file_matches_target_subject(f)
    ]

    combined_plot_dir = BASE_DIR / "_combined_person_plots"
    combined_plot_dir.mkdir(exist_ok=True)

    if not txt_files:
        print("Keine passenden .txt-Dateien gefunden.")
        return

    print(f"{len(txt_files)} passende .txt-Datei(en) gefunden.\n")

    for txt_file in sorted(txt_files):
        try:
            phase, exercise = detect_phase_and_exercise(txt_file)

            print(f"Verarbeite: {txt_file}")

            raw_df = read_raw_txt_with_multiheader(txt_file)
            trials = extract_emg_trials_from_raw_df(raw_df)

            if not trials:
                print("   -> Keine verwertbaren EMG-Trials gefunden\n")
                continue

            print(f"   -> {len(trials)} Trial(s) erkannt")

            combined_df = combine_trials_to_long_emg(trials)

            if combined_df is None or combined_df.empty:
                print("   -> Kombinierter Verlauf leer\n")
                continue

            if SAVE_CLEAN_CSV:
                save_combined_csv(combined_df, txt_file)
                print("   -> Kombinierte CSV gespeichert")

            plot_combined_emg(combined_df, txt_file, combined_plot_dir)
            print()

        except Exception as e:
            print(f"[FEHLER] bei Datei: {txt_file}")
            print(e)
            print()

    print("Fertig.")


if __name__ == "__main__":
    main()