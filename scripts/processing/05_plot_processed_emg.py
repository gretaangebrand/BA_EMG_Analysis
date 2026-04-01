"""
05_plot_processed_emg.py
========================
Plottet die verarbeiteten EMG-Daten (nach Pipeline) pro Subject und Übungstyp.

Liest die *_processed.csv Dateien aus dem processed_emg_pipeline-Ordner
(zeitnormalisiert auf 0–100 %, normalisiert auf SQ-Baseline).

Layout pro Plot-Datei:
  - 5 Zeilen  = 5 Muskeln (Vastus Lateralis, Biceps Femoris,
                            Semitendinosus, Gluteus Medius, Gastrocnemius medial)
  - 3 Spalten = 3 Zyklusphasen (01_PER | 02_OVU | 03_LUT)
  -> 15 Subplots pro Datei

Pro Subject + Übungstyp wird eine Plot-Datei erzeugt (nur R-Seite):
  S01_CMJ_BILATERAL_processed.pdf

Trials werden übereinandergelegt + Mittelwert ± SD als Band.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D


# ============================================================
# PFADE
# ============================================================
DATA_DIR   = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\processed_emg_data")
OUTPUT_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\outputs\figures\plots_processed_emg")


# ============================================================
# EINSTELLUNGEN
# ============================================================
PHASES       = ["01_PER", "02_OVU", "03_LUT"]
PHASE_LABELS = {
    "01_PER": "Follikelphase (PER)",
    "02_OVU": "Ovulationsphase (OVU)",
    "03_LUT": "Lutealphase (LUT)",
}

SIDE = "R"

# Muskelreihenfolge im Plot
MUSCLE_ORDER = [
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

TRIAL_COLORS = plt.cm.tab10.colors
TRIAL_ALPHA  = 0.35
TRIAL_LW     = 0.8
MEAN_LW      = 2.0
FIG_WIDTH    = 18
FIG_HEIGHT   = 14


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def load_processed_trials(phase_dir: Path, exercise: str,
                          side_folder: str) -> list[pd.DataFrame]:
    """
    Laedt alle *_processed.csv Dateien fuer eine Phase/Übung/Seite.
    """
    trial_dir = phase_dir / exercise / side_folder
    if not trial_dir.exists():
        return []

    dfs = []
    for csv in sorted(trial_dir.glob("*_processed.csv")):
        try:
            df = pd.read_csv(csv, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"    [WARNUNG] {csv.name}: {e}")
    return dfs


def compute_mean_sd(trials: list[pd.DataFrame],
                    col_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Berechnet Mittelwert und SD ueber mehrere Trials fuer eine Spalte.
    Gibt (pct, mean, sd) zurueck.
    """
    if not trials:
        return np.array([]), np.array([]), np.array([])

    # Alle Trials sollten 101 Punkte haben (pct 0–100)
    arrays = []
    for df in trials:
        if col_name in df.columns and "pct" in df.columns:
            arrays.append(df[col_name].values)

    if not arrays:
        return np.array([]), np.array([]), np.array([])

    stacked = np.vstack(arrays)
    pct  = trials[0]["pct"].values
    mean = np.mean(stacked, axis=0)
    sd   = np.std(stacked, axis=0, ddof=1) if len(arrays) > 1 else np.zeros_like(mean)

    return pct, mean, sd


# ============================================================
# PLOT-FUNKTION
# ============================================================

def create_plot(
    subject_id:       str,
    exercise:         str,
    side_folder:      str,
    trials_per_phase: dict[str, list[pd.DataFrame]],
    out_path:         Path,
):
    """
    Erstellt eine Abbildung: 5 Muskeln x 3 Phasen = 15 Subplots.
    Jeder Subplot zeigt Einzeltrials (transparent) + Mittelwert ± SD.
    """
    n_muscles = len(MUSCLE_ORDER)
    n_phases  = len(PHASES)

    fig, axes = plt.subplots(
        n_muscles, n_phases,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        sharex=True,
        sharey="row",
    )

    for col_idx, phase in enumerate(PHASES):
        trials = trials_per_phase.get(phase, [])

        for row_idx, muscle in enumerate(MUSCLE_ORDER):
            ax       = axes[row_idx, col_idx]
            col_name = f"{SIDE}_{muscle}"

            # Spalten-Titel (Phase) – nur erste Zeile
            if row_idx == 0:
                ax.set_title(
                    PHASE_LABELS.get(phase, phase),
                    fontsize=9.5, fontweight="bold", pad=5,
                )

            # Zeilen-Label (Muskel) – nur erste Spalte
            if col_idx == 0:
                ax.set_ylabel(muscle, fontsize=8, labelpad=3)

            # Kein Daten-Fall
            if not trials:
                ax.text(0.5, 0.5,
                        f"Keine Daten\n"
                        f"für {exercise} / {side_folder}\n"
                        f"in {PHASE_LABELS.get(phase, phase)}",
                        ha="center", va="center",
                        transform=ax.transAxes,
                        fontsize=7, color="#999999",
                        linespacing=1.4)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Einzeltrials zeichnen (transparent)
            for t_idx, df in enumerate(trials):
                if col_name not in df.columns or "pct" not in df.columns:
                    continue

                pct = df["pct"].values
                sig = df[col_name].values
                color = TRIAL_COLORS[t_idx % len(TRIAL_COLORS)]

                ax.plot(
                    pct, sig,
                    color=color,
                    alpha=TRIAL_ALPHA,
                    linewidth=TRIAL_LW,
                    rasterized=True,
                    label=f"Trial {t_idx + 1}" if row_idx == 0 and col_idx == 0 else None,
                )

            # Mittelwert ± SD
            pct_m, mean, sd = compute_mean_sd(trials, col_name)
            if len(pct_m) > 0:
                muscle_color = MUSCLE_COLORS.get(muscle, "#333333")

                ax.fill_between(
                    pct_m, mean - sd, mean + sd,
                    color=muscle_color, alpha=0.20, zorder=2,
                )
                ax.plot(
                    pct_m, mean,
                    color=muscle_color, linewidth=MEAN_LW,
                    zorder=3,
                    label="Mittelwert" if row_idx == 0 and col_idx == 0 else None,
                )

            # 100%-Baseline-Linie
            ax.axhline(y=100, color="black", linewidth=0.6,
                       linestyle="--", alpha=0.4, zorder=1)

            # Achsen-Formatierung
            ax.tick_params(axis="both", labelsize=6)
            ax.set_xlim(-1, 101)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # X-Achsen-Label nur unterste Zeile
            if row_idx == n_muscles - 1:
                ax.set_xlabel("Bewegungszyklus [%]", fontsize=7, labelpad=2)

    # ---- Legende ----
    max_trials = max((len(t) for t in trials_per_phase.values()), default=0)

    trial_handles = [
        Line2D([0], [0],
               color=TRIAL_COLORS[i % len(TRIAL_COLORS)],
               linewidth=1.5, alpha=0.6,
               label=f"Trial {i + 1}")
        for i in range(min(max_trials, 9))
    ]

    mean_handle = Line2D([0], [0], color="#333333", linewidth=2.2,
                         label="Mittelwert")
    sd_handle = plt.Rectangle((0, 0), 1, 1, fc="#333333", alpha=0.20,
                              label="± SD")
    baseline_handle = Line2D([0], [0], color="black", linewidth=0.8,
                             linestyle="--", alpha=0.5,
                             label="100 % Baseline (SQ)")

    all_handles = trial_handles + [mean_handle, sd_handle, baseline_handle]
    fig.legend(
        handles=all_handles,
        loc="lower center",
        ncol=min(len(all_handles), 8),
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.005),
    )

    # Haupt-Titel und y-Achsen-Label
    fig.suptitle(
        f"{subject_id}  |  {exercise}  |  {side_folder}  |  "
        f"Rechtes Bein  –  Processed EMG",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.text(
        0.005, 0.5, "EMG-Amplitude (% Baseline)",
        va="center", rotation="vertical", fontsize=9,
    )

    plt.tight_layout(rect=[0.025, 0.055, 1.0, 0.975])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {out_path.name}")


# ============================================================
# HAUPTFUNKTION
# ============================================================

def main():
    print("=" * 62)
    print("05_plot_processed_emg.py  –  Processed EMG Plots")
    print("=" * 62)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        print(f"[FEHLER] DATA_DIR nicht gefunden:\n  {DATA_DIR}")
        return

    subject_dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())
    if not subject_dirs:
        print(f"[FEHLER] Keine Subject-Ordner in {DATA_DIR}")
        return

    total_files = 0

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        print(f"\n--- {subject_id} ---")

        # Alle einzigartigen (exercise, side_folder)-Kombis ermitteln
        combos: set[tuple[str, str]] = set()
        for phase in PHASES:
            phase_dir = subject_dir / phase
            if not phase_dir.exists():
                continue
            for ex_dir in phase_dir.iterdir():
                if not ex_dir.is_dir():
                    continue
                for side_dir in ex_dir.iterdir():
                    if side_dir.is_dir():
                        combos.add((ex_dir.name, side_dir.name))

        if not combos:
            print("  Keine Daten gefunden.")
            continue

        for exercise, side_folder in sorted(combos):
            print(f"  {exercise}/{side_folder}")

            # Trials pro Phase laden
            trials_per_phase: dict[str, list[pd.DataFrame]] = {}
            for phase in PHASES:
                phase_dir = subject_dir / phase
                trials_per_phase[phase] = load_processed_trials(
                    phase_dir, exercise, side_folder
                )

            # Output-Ordner
            out_subdir = OUTPUT_DIR / subject_id / exercise
            out_subdir.mkdir(parents=True, exist_ok=True)

            out_path = out_subdir / f"{subject_id}_{exercise}_{side_folder}_processed.pdf"
            create_plot(
                subject_id       = subject_id,
                exercise         = exercise,
                side_folder      = side_folder,
                trials_per_phase = trials_per_phase,
                out_path         = out_path,
            )
            total_files += 1

    print(f"\n{'='*62}")
    print("FERTIG")
    print(f"  Erzeugte Plot-Dateien : {total_files}")
    print(f"  Gespeichert in        : {OUTPUT_DIR}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
