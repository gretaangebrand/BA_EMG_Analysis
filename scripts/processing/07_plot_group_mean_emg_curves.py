"""
07_plot_group_mean_emg_curves.py
=================================
Plottet die gemittelten zeitnormalisierten EMG-Verläufe (0–100 %)
über alle Probandinnen, für die besten Trials.

Pro Übung entsteht ein Plot mit:
  - 3 Spalten = 3 Zyklusphasen (PER, OVU, LUT)
  - Alle 5 Muskeln überlagert in jedem Subplot
  - Mittelwert als Linie + SD als Band

Zusätzlich ein Overlay-Plot pro Übung:
  - 5 Zeilen = 5 Muskeln
  - Alle 3 Phasen überlagert in jedem Subplot
  -> Direkt vergleichbar, ob sich die Kurvenform zwischen Phasen unterscheidet
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# PFADE
# ============================================================
PROCESSED_DIR   = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\04_processed_emg_data")
BEST_TRIALS_CSV = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\05_best_trials_group_and_individual\beste_trials.csv")
OUTPUT_DIR      = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\outputs\figures\plots_group_mean_emg")


# ============================================================
# EINSTELLUNGEN
# ============================================================
PHASES = ["01_PER", "02_OVU", "03_LUT"]
PHASE_LABELS = {"01_PER": "PER", "02_OVU": "OVU", "03_LUT": "LUT"}
PHASE_REVERSE = {"PER": "01_PER", "OVU": "02_OVU", "LUT": "03_LUT"}

SIDE = "R"

MUSCLE_NAMES = [
    "Vastus Lateralis",
    "Biceps Femoris",
    "Semitendinosus",
    "Gluteus Medius",
    "Gastrocnemius medial",
]

MUSCLE_COLORS = {
    "Vastus Lateralis":     "#f4a261",
    "Biceps Femoris":       "#e63946",
    "Semitendinosus":       "#e9c46a",
    "Gluteus Medius":       "#2a9d8f",
    "Gastrocnemius medial": "#457b9d",
}

PHASE_COLORS = {"PER": "#C44462", "OVU": "#0EB55F", "LUT": "#FAD758"}

EXERCISE_MAP = {
    "CMJ bilateral":   ("CMJ", "BILATERAL"),
    "CMJ einbeinig R": ("CMJ", "RIGHT"),
    "DJ bilateral":    ("DJ",  "BILATERAL"),
    "SQ einbeinig R":  ("SQ",  "RIGHT"),
}

N_POINTS = 101  # 0–100 %


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


def collect_curves(df_best):
    """
    Sammelt die zeitnormalisierten EMG-Kurven der besten Trials.

    Returns:
        dict: { exercise_label: { phase_short: { muscle: [array, ...] } } }
    """
    data = {}

    for _, row in df_best.iterrows():
        subject     = row["Subject"]
        label       = row["Übung"]
        phase_short = row["Phase"]
        trial_name  = row["Bester Trial"]

        phase_key = PHASE_REVERSE.get(phase_short)
        if phase_key is None or label not in EXERCISE_MAP:
            continue

        ex_folder, side_folder = EXERCISE_MAP[label]
        proc_path = find_processed_file(
            subject, phase_key, ex_folder, side_folder, trial_name
        )
        if proc_path is None:
            continue

        try:
            df = pd.read_csv(proc_path, low_memory=False)
        except Exception:
            continue

        if "pct" not in df.columns:
            continue

        if label not in data:
            data[label] = {}
        if phase_short not in data[label]:
            data[label][phase_short] = {m: [] for m in MUSCLE_NAMES}

        for muscle in MUSCLE_NAMES:
            col = f"{SIDE}_{muscle}"
            if col in df.columns:
                vals = df[col].values
                if len(vals) == N_POINTS:
                    data[label][phase_short][muscle].append(vals)

    return data


# ============================================================
# PLOT 1: Alle Muskeln überlagert, getrennt nach Phase
# ============================================================

def plot_all_muscles_by_phase(curves, out_dir):
    """
    Pro Übung: 1 Zeile × 3 Spalten (PER | OVU | LUT).
    Alle 5 Muskeln überlagert in jedem Subplot.
    """
    pct = np.linspace(0, 100, N_POINTS)
    phase_order = ["PER", "OVU", "LUT"]

    for exercise, phase_data in sorted(curves.items()):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

        for col_idx, phase in enumerate(phase_order):
            ax = axes[col_idx]
            muscle_dict = phase_data.get(phase, {})

            for muscle in MUSCLE_NAMES:
                arrays = muscle_dict.get(muscle, [])
                if not arrays:
                    continue

                stacked = np.vstack(arrays)
                mean = np.mean(stacked, axis=0)
                sd = np.std(stacked, axis=0, ddof=1) if len(arrays) > 1 \
                    else np.zeros_like(mean)

                color = MUSCLE_COLORS[muscle]

                ax.fill_between(pct, mean - sd, mean + sd,
                                color=color, alpha=0.15)
                ax.plot(pct, mean, color=color, linewidth=2.0,
                        label=muscle)

            ax.axhline(y=100, color="black", linewidth=0.6,
                       linestyle="--", alpha=0.4)

            ax.set_title(
                f"{PHASE_LABELS.get(PHASE_REVERSE.get(phase, ''), phase)} "
                f"(n={len(next(iter(muscle_dict.values()), []))})",
                fontsize=11, fontweight="bold",
            )
            ax.set_xlabel("Bewegungszyklus [%]", fontsize=9)
            ax.set_xlim(-1, 101)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)

        axes[0].set_ylabel("EMG-Amplitude (% Baseline)", fontsize=10)

        # Gemeinsame Legende
        handles, labels = axes[0].get_legend_handles_labels()
        baseline_handle = Line2D([0], [0], color="black", linewidth=0.8,
                                 linestyle="--", alpha=0.5, label="100 % BL")
        handles.append(baseline_handle)
        labels.append("100 % BL")

        fig.legend(handles, labels, loc="lower center",
                   ncol=len(handles), fontsize=9, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(
            f"{exercise}  –  Gruppenmittel aller Muskeln (beste Trials)",
            fontsize=13, fontweight="bold",
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        out_path = out_dir / f"group_mean_{exercise.replace(' ', '_')}_by_phase.pdf"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out_path.name}")


# ============================================================
# PLOT 2: Phasen überlagert, getrennt nach Muskel
# ============================================================

def plot_phases_overlay_by_muscle(curves, out_dir):
    """
    Pro Übung: 5 Zeilen (Muskeln) × 1 Spalte.
    Alle 3 Phasen überlagert in jedem Subplot.
    -> Zeigt direkt, ob sich Kurven zwischen Phasen unterscheiden.
    """
    pct = np.linspace(0, 100, N_POINTS)
    phase_order = ["PER", "OVU", "LUT"]

    for exercise, phase_data in sorted(curves.items()):
        n_mus = len(MUSCLE_NAMES)
        fig, axes = plt.subplots(n_mus, 1, figsize=(14, 3 * n_mus), sharex=True)

        for row_idx, muscle in enumerate(MUSCLE_NAMES):
            ax = axes[row_idx]

            for phase in phase_order:
                muscle_dict = phase_data.get(phase, {})
                arrays = muscle_dict.get(muscle, [])
                if not arrays:
                    continue

                stacked = np.vstack(arrays)
                mean = np.mean(stacked, axis=0)
                sd = np.std(stacked, axis=0, ddof=1) if len(arrays) > 1 \
                    else np.zeros_like(mean)

                color = PHASE_COLORS[phase]

                ax.fill_between(pct, mean - sd, mean + sd,
                                color=color, alpha=0.12)
                ax.plot(pct, mean, color=color, linewidth=2.0,
                        label=f"{phase} (n={len(arrays)})")

            ax.axhline(y=100, color="black", linewidth=0.6,
                       linestyle="--", alpha=0.4)

            ax.set_ylabel(f"{muscle}\n(% BL)", fontsize=8)
            ax.set_xlim(-1, 101)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

        axes[-1].set_xlabel("Bewegungszyklus [%]", fontsize=10)

        fig.suptitle(
            f"{exercise}  –  Phasenvergleich pro Muskel (Gruppenmittel, beste Trials)",
            fontsize=13, fontweight="bold",
        )

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = out_dir / f"group_mean_{exercise.replace(' ', '_')}_phase_overlay.pdf"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out_path.name}")


# ============================================================
# HAUPTFUNKTION
# ============================================================

def main():
    print("=" * 70)
    print("07_plot_group_mean_emg_curves.py  –  Gruppen-EMG-Kurven")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not BEST_TRIALS_CSV.exists():
        print(f"[FEHLER] Beste-Trials-CSV nicht gefunden:\n  {BEST_TRIALS_CSV}")
        return

    df_best = pd.read_csv(BEST_TRIALS_CSV)
    print(f"\nBeste Trials geladen: {len(df_best)} Eintraege")

    print("\nSammle EMG-Kurven der besten Trials ...")
    curves = collect_curves(df_best)

    if not curves:
        print("[FEHLER] Keine Kurven gefunden.")
        return

    for ex, phases in curves.items():
        for phase, muscles in phases.items():
            n = len(next(iter(muscles.values()), []))
            print(f"  {ex:22s} | {phase:4s} | {n} Probandinnen")

    print("\nErstelle Plots ...")
    plot_all_muscles_by_phase(curves, OUTPUT_DIR)
    plot_phases_overlay_by_muscle(curves, OUTPUT_DIR)

    print(f"\n{'='*70}")
    print("FERTIG")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()