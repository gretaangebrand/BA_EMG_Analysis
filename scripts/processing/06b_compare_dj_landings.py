"""
06b_compare_dj_landings.py
===========================
Vergleicht Peak-EMG-Aktivierung zwischen Landing 1 (Bodenkontakt)
und Landing 2 (zweite Landung nach Absprung) beim Drop Jump.

Verwendet die zeitnormalisierten, amplitudennormalisierten Daten
aus 04_processed_emg_data (beste Trials).

Output:
  - LaTeX-Tabelle mit Peak-Werten pro Probandin × Phase × Muskel
  - Landung1 und Landung2 nebeneinander
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, str(Path(r'C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG')))

from scripts.utils.config import MUSCLE_NAMES, EXERCISE_MAP

# ============================================================
# PFADE
# ============================================================
BEST_TRIALS_CSV = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\05_best_trials_group_and_individual\beste_trials.csv")
PROCESSED_DIR   = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\04_processed_emg_data")
PREPROCESSED_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\03_preprocessed_emg_data")
OUTPUT_DIR      = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\outputs\statistics")

SIDE = "R"
PHASE_REVERSE = {"PER": "01_PER", "OVU": "02_OVU", "LUT": "03_LUT"}
PHASE_ORDER = ["PER", "OVU", "LUT"]


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

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


def find_kin_file(subject, phase_key, ex_folder, side_folder, trial_name):
    """Findet die _kin.csv für Event-Zeitpunkte."""
    trial_dir = PREPROCESSED_DIR / subject / phase_key / ex_folder / side_folder
    if not trial_dir.exists():
        return None
    exact = trial_dir / f"{trial_name}_kin.csv"
    if exact.exists():
        return exact
    for f in trial_dir.glob("*_kin.csv"):
        if trial_name in f.stem:
            return f
    return None


def get_landing_pct_ranges(kin_path):
    """
    Berechnet prozentuale Zeitfenster für Landung1 und Landung2 beim DJ.
    
    Returns: dict {"Landung1": (start_pct, end_pct), "Landung2": (start_pct, end_pct)}
    """
    try:
        kin = pd.read_csv(kin_path, nrows=1, low_memory=False)
    except Exception:
        return {}

    cols_needed = ["event_landing1_s", "event_take_off_s",
                   "event_landing2_s", "event_end_jump_s"]
    if not all(c in kin.columns for c in cols_needed):
        return {}

    t_land1   = float(kin["event_landing1_s"].iloc[0])
    t_takeoff = float(kin["event_take_off_s"].iloc[0])
    t_land2   = float(kin["event_landing2_s"].iloc[0])
    t_end     = float(kin["event_end_jump_s"].iloc[0])
    total     = t_end - t_land1

    if total <= 0:
        return {}

    # Bodenkontakt = Landing1 → Take-off
    contact_start_pct = 0.0
    contact_end_pct   = ((t_takeoff - t_land1) / total) * 100
    
    # Landung2 = Landing2 → End
    land2_start_pct   = ((t_land2 - t_land1) / total) * 100
    land2_end_pct     = 100.0

    return {
        "Landung1": (contact_start_pct, contact_end_pct),
        "Landung2": (land2_start_pct, land2_end_pct),
    }


def extract_peak_in_range(pct, signal, pct_start, pct_end):
    """Extrahiert Peak-Wert aus einem Abschnitt der Kurve."""
    mask = (pct >= pct_start) & (pct <= pct_end)
    segment = signal[mask]
    if len(segment) == 0:
        return np.nan
    return float(np.max(segment))


# ============================================================
# HAUPTANALYSE
# ============================================================

def build_comparison_table():
    """
    Baut Vergleichstabelle: Subject × Phase × Muskel → Peak Landung1 vs Landung2
    """
    df_best = pd.read_csv(BEST_TRIALS_CSV)
    df_dj = df_best[df_best["Übung"] == "DJ bilateral"].copy()
    
    if df_dj.empty:
        print("[FEHLER] Keine DJ bilateral Trials in beste_trials.csv gefunden")
        return None

    records = []

    for _, row in df_dj.iterrows():
        subject     = row["Subject"]
        phase_short = row["Phase"]
        trial_name  = row["Bester Trial"]

        phase_key = PHASE_REVERSE.get(phase_short)
        if phase_key is None:
            continue

        ex_folder, side_folder = EXERCISE_MAP.get("DJ bilateral", (None, None))
        if ex_folder is None:
            continue

        # Processed CSV laden (zeitnormalisiert 0-100%)
        proc_path = find_processed_file(
            subject, phase_key, ex_folder, side_folder, trial_name
        )
        if proc_path is None:
            print(f"  [SKIP] {subject} | {phase_short} | {trial_name} – keine processed-Datei")
            continue

        try:
            df_proc = pd.read_csv(proc_path, low_memory=False)
        except Exception as e:
            print(f"  [FEHLER] {proc_path.name}: {e}")
            continue

        if "pct" not in df_proc.columns:
            print(f"  [SKIP] {proc_path.name} – keine pct-Spalte")
            continue

        pct = df_proc["pct"].values

        # Event-Zeitfenster aus KIN-Datei holen
        kin_path = find_kin_file(
            subject, phase_key, ex_folder, side_folder, trial_name
        )
        if kin_path is None:
            print(f"  [SKIP] {subject} | {phase_short} | {trial_name} – keine kin-Datei")
            continue

        landing_ranges = get_landing_pct_ranges(kin_path)
        if not landing_ranges:
            print(f"  [SKIP] {subject} | {phase_short} | {trial_name} – Events fehlen")
            continue

        pct_land1 = landing_ranges.get("Landung1")
        pct_land2 = landing_ranges.get("Landung2")

        if pct_land1 is None or pct_land2 is None:
            continue

        # Peak pro Muskel extrahieren
        for muscle in MUSCLE_NAMES:
            col = f"{SIDE}_{muscle}"
            if col not in df_proc.columns:
                continue

            signal = df_proc[col].values
            
            peak_land1 = extract_peak_in_range(pct, signal, pct_land1[0], pct_land1[1])
            peak_land2 = extract_peak_in_range(pct, signal, pct_land2[0], pct_land2[1])

            records.append({
                "Subject":      subject,
                "Phase":        phase_short,
                "Muskel":       muscle,
                "Peak_Landung1": peak_land1,
                "Peak_Landung2": peak_land2,
            })

        print(f"  ✅ {subject} | {phase_short} | {trial_name}")

    return pd.DataFrame(records)


# ============================================================
# LATEX-AUSGABE
# ============================================================

def generate_latex_table(df, out_path):
    """
    Generiert LaTeX-Tabelle mit Peak-Werten Landung1 vs Landung2.
    Pro Probandin eine Zeile, Spalten: Subject | Muskel | PER_L1 | PER_L2 | OVU_L1 | OVU_L2 | LUT_L1 | LUT_L2
    """
    lines = [
        "% Automatisch generiert von 06b_compare_dj_landings.py",
        "% Peak-EMG beim DJ: Landung 1 (Bodenkontakt) vs. Landung 2 (nach Absprung)",
        "% Werte in % Baseline (normalisiert auf DJ bilateral)",
        "",
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Peak-EMG-Aktivierung beim Drop Jump: Vergleich Landung 1 (Bodenkontakt) vs. Landung 2 (zweite Landung nach Absprung). Werte in \% Baseline (normalisiert auf DJ bilateral, beste Trials pro Probandin und Phase).}",
        r"  \label{tab:dj_landing_comparison}",
        r"  \small",
        r"  \begin{tabular}{l l c c c c c c}",
        r"    \hline",
        r"    \textbf{Subject} & \textbf{Muskel} & ",
        r"    \multicolumn{2}{c}{\textbf{PER}} & ",
        r"    \multicolumn{2}{c}{\textbf{OVU}} & ",
        r"    \multicolumn{2}{c}{\textbf{LUT}} \\",
        r"    \cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
        r"    & & L1 & L2 & L1 & L2 & L1 & L2 \\",
        r"    \hline",
    ]

    # Pivot: Subject × Muskel als Index, Phase als Spalten
    for subject in sorted(df["Subject"].unique()):
        df_subj = df[df["Subject"] == subject]
        
        first_row = True
        for muscle in MUSCLE_NAMES:
            df_muscle = df_subj[df_subj["Muskel"] == muscle]
            if df_muscle.empty:
                continue

            subj_str = subject if first_row else ""
            first_row = False

            # Werte pro Phase holen
            def get_val(phase, landing):
                row = df_muscle[df_muscle["Phase"] == phase]
                if row.empty:
                    return "--"
                val = row.iloc[0][f"Peak_{landing}"]
                if pd.isna(val):
                    return "--"
                return f"{val:.1f}".replace(".", "{,}")

            per_l1 = get_val("PER", "Landung1")
            per_l2 = get_val("PER", "Landung2")
            ovu_l1 = get_val("OVU", "Landung1")
            ovu_l2 = get_val("OVU", "Landung2")
            lut_l1 = get_val("LUT", "Landung1")
            lut_l2 = get_val("LUT", "Landung2")

            lines.append(
                f"    {subj_str} & M.~{muscle.lower()} & "
                f"{per_l1} & {per_l2} & "
                f"{ovu_l1} & {ovu_l2} & "
                f"{lut_l1} & {lut_l2} \\\\"
            )
        
        lines.append(r"    \hline")

    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  LaTeX-Tabelle: {out_path.name}")


# ============================================================
# HAUPTFUNKTION
# ============================================================

def main():
    print("=" * 70)
    print("06b_compare_dj_landings.py  –  Peak EMG Landung1 vs Landung2")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not BEST_TRIALS_CSV.exists():
        print(f"[FEHLER] Datei nicht gefunden:\n  {BEST_TRIALS_CSV}")
        return

    print("\nExtrahiere Peak-Werte aus besten DJ-Trials...")
    df = build_comparison_table()

    if df is None or df.empty:
        print("[FEHLER] Keine Daten extrahiert.")
        return

    print(f"\n  Extrahierte Datensätze: {len(df)}")

    # CSV-Ausgabe (optional, für Kontrolle)
    csv_out = OUTPUT_DIR / "dj_landing_comparison.csv"
    df.to_csv(csv_out, index=False)
    print(f"  CSV: {csv_out.name}")

    # LaTeX-Tabelle
    tex_out = OUTPUT_DIR / "dj_landing_comparison.tex"
    generate_latex_table(df, tex_out)

    # Zusammenfassung
    print(f"\n{'='*70}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*70}")
    
    # Gruppenmittelwerte pro Phase × Landung
    for phase in PHASE_ORDER:
        df_phase = df[df["Phase"] == phase]
        if df_phase.empty:
            continue
        
        land1_mean = df_phase["Peak_Landung1"].mean()
        land2_mean = df_phase["Peak_Landung2"].mean()
        
        print(f"\n  {phase}:")
        print(f"    Landung 1 (Bodenkontakt): {land1_mean:.1f} % BL (Mittelwert über alle Muskeln)")
        print(f"    Landung 2 (nach Absprung): {land2_mean:.1f} % BL")
        print(f"    Differenz: {land2_mean - land1_mean:+.1f} %")

    print(f"\n{'='*70}")
    print(f"FERTIG – Output: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()