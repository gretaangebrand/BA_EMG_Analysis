from pathlib import Path
import pandas as pd

# ============================================================
# PFAD ZUR TESTDATEI
# ============================================================

file_path = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\Daten_BA\anonymized_csv_data\01_PER\CMJ\S01_01_PER_CMJ.csv")


# ============================================================
# CSV EINLESEN
# ============================================================

# Bei deinen Dateien ist sehr wahrscheinlich Tab als Trennzeichen richtig.
# Falls es nicht funktioniert, kann man später auf sep="," umstellen.
df = pd.read_csv(file_path, sep="\t", header=[0, 1, 2, 3, 4])

print("=" * 70)
print("DATEI EINGELESEN")
print("=" * 70)
print(f"Datei: {file_path.name}")
print(f"Shape: {df.shape}")
print()

# ============================================================
# HEADER-EBENEN KURZ ANZEIGEN
# ============================================================

print("=" * 70)
print("HEADER-EBENEN BEISPIEL")
print("=" * 70)

for i, col in enumerate(df.columns[:10]):
    print(f"Spalte {i+1}: {col}")

print()

# ============================================================
# TRIALS ERKENNEN
# ============================================================

trial_names = pd.Index(df.columns.get_level_values(0)).dropna().unique()

print("=" * 70)
print("GEFUNDENE TRIALS")
print("=" * 70)

for i, trial in enumerate(trial_names, start=1):
    print(f"{i}: {trial}")

print()

# ============================================================
# EMG-SPALTEN ERKENNEN
# Annahme: Ebene 2 enthält Signaltyp, 'ANALOG' = EMG
# ============================================================

signal_type_level = 2
variable_name_level = 1
trial_level = 0

emg_mask = df.columns.get_level_values(signal_type_level).astype(str).str.upper() == "ANALOG"
emg_df = df.loc[:, emg_mask]

print("=" * 70)
print("EMG-SPALTEN")
print("=" * 70)
print(f"Anzahl EMG-Spalten insgesamt: {emg_df.shape[1]}")
print()

if emg_df.shape[1] == 0:
    print("Keine EMG-Spalten erkannt.")
else:
    # Muskelnamen aus Ebene 1
    muscle_names = pd.Index(emg_df.columns.get_level_values(variable_name_level)).dropna().unique()

    print("Gefundene Muskelnamen:")
    for muscle in muscle_names:
        print(f"- {muscle}")

print()

# ============================================================
# EMG-SPALTEN NACH TRIAL AUFLISTEN
# ============================================================

print("=" * 70)
print("EMG-SPALTEN PRO TRIAL")
print("=" * 70)

if emg_df.shape[1] > 0:
    for trial in pd.Index(emg_df.columns.get_level_values(trial_level)).dropna().unique():
        trial_mask = emg_df.columns.get_level_values(trial_level) == trial
        trial_emg = emg_df.loc[:, trial_mask]

        muscles_this_trial = pd.Index(
            trial_emg.columns.get_level_values(variable_name_level)
        ).dropna().unique()

        print(f"\nTrial: {trial}")
        print(f"  Anzahl EMG-Spalten: {trial_emg.shape[1]}")
        print(f"  Muskeln:")
        for muscle in muscles_this_trial:
            print(f"   - {muscle}")

print()

# ============================================================
# PRÜFEN, OB UNNAMED/SPASS-SPALTEN DABEI SIND
# ============================================================

print("=" * 70)
print("PRÜFUNG AUF UNNAMED-/LEERE HEADER")
print("=" * 70)

problem_cols = []
for col in df.columns:
    col_as_text = [str(x).strip() for x in col]
    if any("Unnamed" in x for x in col_as_text) or any(x == "" for x in col_as_text):
        problem_cols.append(col)

print(f"Anzahl problematischer Header-Spalten: {len(problem_cols)}")

if problem_cols:
    print("Beispiele:")
    for col in problem_cols[:10]:
        print(col)

print()

# ============================================================
# ERSTE DATENZEILEN DER EMG-DATEN ZEIGEN
# ============================================================

print("=" * 70)
print("ERSTE ZEILEN DER EMG-DATEN")
print("=" * 70)

if emg_df.shape[1] > 0:
    print(emg_df.head())
else:
    print("Keine EMG-Daten zum Anzeigen gefunden.")