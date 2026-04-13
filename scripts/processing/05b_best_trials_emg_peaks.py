"""
05b_best_trials_with_emg_peaks.py
==================================
Erweitert die beste_trials-Übersicht um die Peak-EMG-Aktivierung
(% Baseline) pro Muskel aus den processed EMG-Daten.

Erzeugt:
  - CSV:  beste_trials_mit_emg_peaks.csv
  - PDF:  beste_trials_mit_emg_peaks.pdf
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(r'C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG')))

from scripts.utils.config import MUSCLE_NAMES

# ============================================================
# PFADE
# ============================================================
BEST_TRIALS_CSV = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\05_best_trials_group_and_individual\beste_trials.csv")
PROCESSED_DIR   = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\04_processed_emg_data")
OUTPUT_DIR      = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\05_best_trials_group_and_individual")
FIGURES_DIR     = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\outputs\figures\best_trials_group_individual")

SIDE = "R"
PHASE_REVERSE = {"PER": "01_PER", "OVU": "02_OVU", "LUT": "03_LUT"}

# Mapping: Übungs-Label → (Ordner, Seitenordner)
EXERCISE_MAP = {
    "CMJ bilateral":   ("CMJ", "BILATERAL"),
    "CMJ einbeinig R": ("CMJ", "RIGHT"),
    "DJ bilateral":    ("DJ",  "BILATERAL"),
    "SQ bilateral":    ("SQ",  "BILATERAL"),
    "SQ einbeinig R":  ("SQ",  "RIGHT"),
}


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def find_processed_file(subject, phase_key, ex_folder, side_folder, trial_name):
    """Findet die processed CSV-Datei für einen Trial."""
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


def get_emg_peaks(proc_path):
    """
    Liest eine processed CSV und gibt den Peak-Wert (max) pro Muskel zurück.
    Returns: dict {Muskelname: Peak-Wert} oder leeres dict bei Fehler.
    """
    try:
        df = pd.read_csv(proc_path, low_memory=False)
    except Exception:
        return {}

    peaks = {}
    for muscle in MUSCLE_NAMES:
        col = f"{SIDE}_{muscle}"
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            peaks[muscle] = float(vals.max()) if not vals.empty else np.nan
        else:
            peaks[muscle] = np.nan
    return peaks


# ============================================================
# PDF-ERSTELLUNG
# ============================================================

def create_pdf(df_individual, df_group, out_path):
    """Erstellt eine PDF mit Gruppen- und Individualtabelle."""

    # ── Spalten für die Darstellung ──
    muscle_short = {
        "Gluteus Medius":       "Glut Med",
        "Vastus Lateralis":     "Vast Lat",
        "Biceps Femoris":       "Bic Fem",
        "Semitendinosus":       "Semitend",
        "Gastrocnemius medial": "Gastroc",
    }

    # Tabelle 1: Gruppenmittelwerte
    group_cols = ["Übung", "Phase", "Wert (M ± SD)"]
    for m in MUSCLE_NAMES:
        group_cols.append(muscle_short[m])
    tab1_data = []
    for _, row in df_group.iterrows():
        line = [row["Übung"], row["Phase"], row["Wert (M ± SD)"]]
        for m in MUSCLE_NAMES:
            m_mean = row.get(f"{m}_mean", np.nan)
            m_sd   = row.get(f"{m}_sd", np.nan)
            if pd.notna(m_mean) and pd.notna(m_sd):
                line.append(f"{m_mean:.1f} ± {m_sd:.1f}")
            else:
                line.append("–")
        tab1_data.append(line)

    # Tabelle 2: Individuelle Trials
    ind_cols = ["Subject", "Übung", "Phase", "Bester Trial", "Wert"]
    for m in MUSCLE_NAMES:
        ind_cols.append(muscle_short[m])
    tab2_data = []
    for _, row in df_individual.iterrows():
        line = [
            row["Subject"], row["Übung"], row["Phase"],
            row["Bester Trial"],
            f"{row['Wert']:.3f} {row['Einheit']}",
        ]
        for m in MUSCLE_NAMES:
            val = row.get(f"Peak_{m}", np.nan)
            if pd.notna(val):
                line.append(f"{val:.1f}")
            else:
                line.append("–")
        tab2_data.append(line)

    # ── Layout ──
    n1, n2 = len(tab1_data), len(tab2_data)
    fig_h = max(6, 2.0 + 0.35 * n1 + 0.30 * n2)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, fig_h),
        gridspec_kw={"height_ratios": [max(1, 1 + n1), max(1, 1 + n2)]},
    )

    for ax, title, col_labels, data in [
        (ax1, "Gruppenmittelwerte – Kinematik + EMG-Peak [% BL] (beste Trials)", group_cols, tab1_data),
        (ax2, "Individuelle beste Trials – Kinematik + EMG-Peak [% BL]", ind_cols, tab2_data),
    ]:
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=10)

        tbl = ax.table(
            cellText=data,
            colLabels=col_labels,
            cellLoc="center",
            loc="upper left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.0, 1.30)

        # Header-Stil
        for col_idx in range(len(col_labels)):
            tbl[0, col_idx].set_facecolor("#4472C4")
            tbl[0, col_idx].set_text_props(color="white", fontweight="bold")

        # Zeilenfarben
        for row_idx in range(1, len(data) + 1):
            bg = "#F2F2F2" if row_idx % 2 == 0 else "white"
            for col_idx in range(len(col_labels)):
                tbl[row_idx, col_idx].set_facecolor(bg)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PDF: {out_path}")


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():
    print("=" * 70)
    print("05b – Beste Trials mit EMG-Peak-Aktivierung pro Muskel")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── beste_trials.csv laden ──
    df_best = pd.read_csv(BEST_TRIALS_CSV)
    print(f"  {len(df_best)} beste Trials geladen")

    # ── EMG-Peaks pro Trial extrahieren ──
    peak_records = []
    for _, row in df_best.iterrows():
        subject     = row["Subject"]
        label       = row["Übung"]
        phase_short = row["Phase"]
        trial_name  = row["Bester Trial"]

        phase_key = PHASE_REVERSE.get(phase_short)
        mapping   = EXERCISE_MAP.get(label)

        if phase_key is None or mapping is None:
            peaks = {}
        else:
            ex_folder, side_folder = mapping
            proc_path = find_processed_file(
                subject, phase_key, ex_folder, side_folder, trial_name
            )
            if proc_path is not None:
                peaks = get_emg_peaks(proc_path)
                print(f"  ✅ {subject} | {label:22s} | {phase_short} | {trial_name}")
            else:
                peaks = {}
                print(f"  ⚠️  {subject} | {label:22s} | {phase_short} | {trial_name} – nicht gefunden")

        record = {
            "Subject":     subject,
            "Übung":       label,
            "Phase":       phase_short,
            "Bester Trial": trial_name,
            "Wert":        row["Wert"],
            "Einheit":     row["Einheit"],
        }
        for m in MUSCLE_NAMES:
            record[f"Peak_{m}"] = peaks.get(m, np.nan)
        peak_records.append(record)

    df_result = pd.DataFrame(peak_records)

    # ── CSV speichern ──
    csv_path = OUTPUT_DIR / "beste_trials_mit_emg_peaks.csv"
    df_result.to_csv(csv_path, index=False)
    print(f"\n  CSV: {csv_path}")

    # ── Gruppenmittelwerte berechnen ──
    group_agg = {"Wert": ["mean", "std", "count"]}
    for m in MUSCLE_NAMES:
        group_agg[f"Peak_{m}"] = ["mean", "std"]

    df_grouped = df_result.groupby(["Übung", "Phase"]).agg(group_agg).reset_index()
    # Spalten flach machen
    df_grouped.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in df_grouped.columns
    ]

    # Formatierte Spalte für Kinematik-Wert
    df_grouped["Wert (M ± SD)"] = df_grouped.apply(
        lambda r: f"{r['Wert_mean']:.3f} ± {r['Wert_std']:.3f}", axis=1
    )

    # Spalten umbenennen für sauberen Zugriff
    for m in MUSCLE_NAMES:
        df_grouped.rename(columns={
            f"Peak_{m}_mean": f"{m}_mean",
            f"Peak_{m}_std":  f"{m}_sd",
        }, inplace=True)

    # ── PDF erstellen ──
    pdf_path = FIGURES_DIR / "beste_trials_mit_emg_peaks.pdf"
    create_pdf(df_result, df_grouped, pdf_path)

    print(f"\n{'='*70}")
    print("FERTIG")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()