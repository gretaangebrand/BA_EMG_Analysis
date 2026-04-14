"""
check_trials_over_100pct.py
============================
Findet alle Trials in CMJ und SQ (bilateral + einbeinig), bei denen
die EMG-Aktivierung (% Baseline) über 100 % hinausgeht.

Gibt eine sortierte Übersicht aus:
  - Welches Subject
  - Welche Übung / Phase
  - Welcher Muskel
  - Wie hoch der Peak-Wert ist

Verwendet die processed Daten (04_processed_emg_data) und die
beste_trials.csv aus Script 05.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, str(Path(r'C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG')))

from scripts.utils.config import MUSCLE_NAMES

# ============================================================
# PFADE
# ============================================================
BEST_TRIALS_CSV = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\05_best_trials_group_and_individual\beste_trials.csv")
PROCESSED_DIR   = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\04_processed_emg_data")

SIDE = "R"
PHASE_REVERSE = {"PER": "01_PER", "OVU": "02_OVU", "LUT": "03_LUT"}

EXERCISE_MAP = {
    "CMJ bilateral":   ("CMJ", "BILATERAL"),
    "CMJ einbeinig R": ("CMJ", "RIGHT"),
    "DJ bilateral":    ("DJ",  "BILATERAL"),
    "SQ bilateral":    ("SQ",  "BILATERAL"),
    "SQ einbeinig R":  ("SQ",  "RIGHT"),
}

# Nur CMJ und SQ untersuchen (DJ ist die Baseline-Übung)
EXERCISES_TO_CHECK = [
    "CMJ bilateral", "CMJ einbeinig R",
    "SQ bilateral", "SQ einbeinig R",
]


def find_processed_file(subject, phase_key, ex_folder, side_folder, trial_name):
    trial_dir = PROCESSED_DIR / subject / phase_key / ex_folder / side_folder
    if not trial_dir.exists():
        return None
    exact = trial_dir / f"{trial_name}_processed.csv"
    if exact.exists():
        return exact
    for f in trial_dir.glob("*_processed.csv"):
        if trial_name in f.stem:
            return f
    return None


def main():
    print("=" * 80)
    print("Trials mit EMG-Aktivierung > 100 % (CMJ + SQ)")
    print("=" * 80)

    df_best = pd.read_csv(BEST_TRIALS_CSV)
    print(f"Beste Trials geladen: {len(df_best)}")

    # Nur CMJ und SQ
    df_check = df_best[df_best["Übung"].isin(EXERCISES_TO_CHECK)].copy()
    print(f"Davon CMJ/SQ: {len(df_check)}\n")

    results = []

    for _, row in df_check.iterrows():
        subject     = row["Subject"]
        label       = row["Übung"]
        phase_short = row["Phase"]
        trial_name  = row["Bester Trial"]

        phase_key = PHASE_REVERSE.get(phase_short)
        mapping   = EXERCISE_MAP.get(label)
        if phase_key is None or mapping is None:
            continue

        ex_folder, side_folder = mapping
        proc_path = find_processed_file(
            subject, phase_key, ex_folder, side_folder, trial_name
        )
        if proc_path is None:
            continue

        try:
            df = pd.read_csv(proc_path, low_memory=False)
        except Exception:
            continue

        for muscle in MUSCLE_NAMES:
            col = f"{SIDE}_{muscle}"
            if col not in df.columns:
                continue

            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if vals.empty:
                continue

            peak = float(vals.max())
            mean_val = float(vals.mean())

            if peak > 100:
                results.append({
                    "Subject":    subject,
                    "Übung":      label,
                    "Phase":      phase_short,
                    "Trial":      trial_name,
                    "Muskel":     muscle,
                    "Peak (%)":   round(peak, 1),
                    "Mean (%)":   round(mean_val, 1),
                })

    if not results:
        print("✅ Keine Trials mit Werten > 100 % gefunden.")
        return

    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values(
        ["Peak (%)", "Subject", "Übung", "Phase"],
        ascending=[False, True, True, True]
    )

    # ── Konsolenausgabe ──
    print(f"\n{'='*80}")
    print(f"GEFUNDEN: {len(df_result)} Muskel × Trial Kombinationen > 100 %")
    print(f"{'='*80}\n")

    print(f"{'Subject':7s} | {'Übung':22s} | {'Phase':5s} | {'Trial':12s} | "
          f"{'Muskel':25s} | {'Peak':>8s} | {'Mean':>8s}")
    print("-" * 100)

    for _, r in df_result.iterrows():
        print(f"{r['Subject']:7s} | {r['Übung']:22s} | {r['Phase']:5s} | "
              f"{r['Trial']:12s} | {r['Muskel']:25s} | "
              f"{r['Peak (%)']:7.1f}% | {r['Mean (%)']:7.1f}%")

    # ── Zusammenfassung pro Muskel ──
    print(f"\n{'='*80}")
    print("ZUSAMMENFASSUNG: Häufigkeit pro Muskel")
    print(f"{'='*80}")
    counts = df_result.groupby("Muskel").agg(
        Anzahl=("Peak (%)", "count"),
        Max_Peak=("Peak (%)", "max"),
        Mean_Peak=("Peak (%)", "mean"),
    ).sort_values("Anzahl", ascending=False)
    for muscle, row in counts.iterrows():
        print(f"  {muscle:25s}: {int(row['Anzahl']):3d}x  "
              f"(max {row['Max_Peak']:.1f}%, Ø {row['Mean_Peak']:.1f}%)")

    # ── Zusammenfassung pro Subject ──
    print(f"\n{'='*80}")
    print("ZUSAMMENFASSUNG: Häufigkeit pro Subject")
    print(f"{'='*80}")
    counts_subj = df_result.groupby("Subject").agg(
        Anzahl=("Peak (%)", "count"),
        Max_Peak=("Peak (%)", "max"),
    ).sort_values("Max_Peak", ascending=False)
    for subj, row in counts_subj.iterrows():
        print(f"  {subj:7s}: {int(row['Anzahl']):3d}x  (max {row['Max_Peak']:.1f}%)")

    # ── CSV speichern ──
    out_csv = PROCESSED_DIR.parent / "trials_ueber_100pct.csv"
    df_result.to_csv(out_csv, index=False)
    print(f"\n  CSV gespeichert: {out_csv}")

    print(f"\n{'='*80}")
    print("FERTIG")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()