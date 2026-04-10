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
import sys
sys.path.insert(0, str(Path(r'C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG')))

from scripts.utils.config import EXERCISE_MAP

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


N_POINTS = 101  # 0–100 %

# ── Darstellungsoptionen ──────────────────────────────────────
SHOW_SD = False   # auf True oder False setzen, um die SD-Bänder ein- oder eben auszublenden

# ── Pfad zu den preprocessed Daten (für KIN-Events) ──────────
PREPROCESSED_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\03_preprocessed_emg_data")

# ── Event-Konfiguration pro Übungstyp ─────────────────────────
# Jedes Event: (event_start_col, event_time_col, label, farbe, linestyle)
EVENT_CONFIG = {
    "CMJ": [
        ("event_start_s",    "Start",     "#888888", ":"),
        ("event_take_off_s", "Take-off",  "#2196F3", "-"),
        ("event_landing_s",  "Landing",   "#FF5722", "-"),
        ("event_end_jump_s", "End",       "#888888", ":"),
    ],
    "DJ": [
        ("event_landing1_s", "Landing 1", "#FF5722", "-"),
        ("event_take_off_s", "Take-off",  "#2196F3", "-"),
        ("event_landing2_s", "Landing 2", "#FF5722", "--"),
        ("event_end_jump_s", "End",       "#888888", ":"),
    ],
}


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
    """Findet die _kin.csv fuer Event-Zeitpunkte."""
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


def get_event_pct(kin_path, exercise_label):
    """
    Berechnet die prozentualen Positionen der Events im Bewegungszyklus.
    Gibt eine Liste von (pct, label, color, linestyle) zurueck.
    """
    try:
        kin = pd.read_csv(kin_path, nrows=1, low_memory=False)
    except Exception:
        return []

    # Übungstyp bestimmen (CMJ oder DJ)
    ex_type = None
    for key in EVENT_CONFIG:
        if key in exercise_label:
            ex_type = key
            break
    if ex_type is None:
        return []

    events_cfg = EVENT_CONFIG[ex_type]

    # Alle Event-Zeiten lesen
    event_times = {}
    for col_name, label, color, ls in events_cfg:
        if col_name in kin.columns:
            val = pd.to_numeric(kin[col_name].iloc[0], errors="coerce")
            if pd.notna(val):
                event_times[col_name] = (float(val), label, color, ls)

    if len(event_times) < 2:
        return []

    # Gesamtdauer: erstes bis letztes Event
    all_times = [v[0] for v in event_times.values()]
    t_start = min(all_times)
    t_end = max(all_times)
    duration = t_end - t_start

    if duration <= 0:
        return []

    # Prozentuale Position berechnen
    result = []
    for col_name, (t, label, color, ls) in event_times.items():
        pct_pos = ((t - t_start) / duration) * 100
        result.append((pct_pos, label, color, ls))

    return result


def collect_curves(df_best):
    """
    Sammelt die zeitnormalisierten EMG-Kurven der besten Trials.

    Returns:
        curves: { exercise_label: { phase_short: { muscle: [array, ...] } } }
        events: { exercise_label: { phase_short: [ [(pct, label, color, ls), ...], ... ] } }
    """
    data = {}
    event_data = {}

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
            event_data[label] = {}
        if phase_short not in data[label]:
            data[label][phase_short] = {m: [] for m in MUSCLE_NAMES}
            event_data[label][phase_short] = []

        for muscle in MUSCLE_NAMES:
            col = f"{SIDE}_{muscle}"
            if col in df.columns:
                vals = df[col].values
                if len(vals) == N_POINTS:
                    data[label][phase_short][muscle].append(vals)

        # Events aus KIN-Datei sammeln
        kin_path = find_kin_file(
            subject, phase_key, ex_folder, side_folder, trial_name
        )
        if kin_path is not None:
            evts = get_event_pct(kin_path, label)
            if evts:
                event_data[label][phase_short].append(evts)

    return data, event_data


def draw_event_lines(ax, event_lists, add_label=True):
    """
    Zeichnet gemittelte Event-Linien in einen Subplot.
    event_lists: Liste von Event-Listen (eine pro Probandin).
    Mittelt die prozentualen Positionen über alle Probandinnen.
    """
    if not event_lists:
        return []

    # Events nach Label gruppieren und mitteln
    from collections import defaultdict
    grouped = defaultdict(lambda: {"pcts": [], "color": "", "ls": ""})
    for trial_events in event_lists:
        for pct_pos, label, color, ls in trial_events:
            grouped[label]["pcts"].append(pct_pos)
            grouped[label]["color"] = color
            grouped[label]["ls"] = ls

    handles = []
    for label, info in grouped.items():
        mean_pct = np.mean(info["pcts"])
        # Start und End nicht zeichnen (liegen bei 0% und 100%)
        if mean_pct < 1 or mean_pct > 99:
            continue
        line = ax.axvline(
            x=mean_pct, color=info["color"], linewidth=1.2,
            linestyle=info["ls"], alpha=0.7, zorder=1,
        )
        if add_label:
            ax.text(
                mean_pct, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100,
                f" {label}", fontsize=6, color=info["color"],
                rotation=90, va="top", ha="left", alpha=0.8,
            )
            handles.append(Line2D([0], [0], color=info["color"],
                                  linewidth=1.2, linestyle=info["ls"],
                                  alpha=0.7, label=label))
    return handles


# ============================================================
# PLOT 1: Alle Muskeln überlagert, getrennt nach Phase
# ============================================================

def plot_all_muscles_by_phase(curves, events, out_dir):
    """
    Pro Übung: 1 Zeile × 3 Spalten (PER | OVU | LUT).
    Alle 5 Muskeln überlagert in jedem Subplot.
    """
    pct = np.linspace(0, 100, N_POINTS)
    phase_order = ["PER", "OVU", "LUT"]

    for exercise, phase_data in sorted(curves.items()):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        event_handles = []

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

                if SHOW_SD:
                    ax.fill_between(pct, mean - sd, mean + sd,
                                    color=color, alpha=0.15)
                ax.plot(pct, mean, color=color, linewidth=2.0,
                        label=muscle)

            ax.axhline(y=100, color="black", linewidth=0.6,
                       linestyle="--", alpha=0.4)

            # Event-Linien zeichnen
            evt_lists = events.get(exercise, {}).get(phase, [])
            if evt_lists:
                eh = draw_event_lines(ax, evt_lists, add_label=(col_idx == 0))
                if col_idx == 0:
                    event_handles = eh

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

        # Event-Handles hinzufuegen
        for eh in event_handles:
            handles.append(eh)
            labels.append(eh.get_label())

        fig.legend(handles, labels, loc="lower center",
                   ncol=len(handles), fontsize=9, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(
            f"{exercise}  –  Gruppenmittel aller Muskeln (beste Trials)",
            fontsize=13, fontweight="bold",
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        out_path = out_dir / f"group_mean_{exercise.replace(' ', '_')}_by_phase.svg"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out_path.name}")


# ============================================================
# PLOT 2: Phasen überlagert, getrennt nach Muskel
# ============================================================

def plot_phases_overlay_by_muscle(curves, events, out_dir):
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

                if SHOW_SD:
                    ax.fill_between(pct, mean - sd, mean + sd,
                                    color=color, alpha=0.12)
                ax.plot(pct, mean, color=color, linewidth=2.0,
                        label=f"{phase} (n={len(arrays)})")

            ax.axhline(y=100, color="black", linewidth=0.6,
                       linestyle="--", alpha=0.4)

            # Event-Linien: Mittelwert über alle Phasen
            all_evt_lists = []
            for phase in phase_order:
                all_evt_lists.extend(events.get(exercise, {}).get(phase, []))
            if all_evt_lists:
                draw_event_lines(ax, all_evt_lists, add_label=(row_idx == 0))

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
        out_path = out_dir / f"group_mean_{exercise.replace(' ', '_')}_phase_overlay.svg"
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
    curves, events = collect_curves(df_best)

    if not curves:
        print("[FEHLER] Keine Kurven gefunden.")
        return

    for ex, phases in curves.items():
        for phase, muscles in phases.items():
            n = len(next(iter(muscles.values()), []))
            n_evt = len(events.get(ex, {}).get(phase, []))
            evt_info = f"  ({n_evt} mit Events)" if n_evt > 0 else ""
            print(f"  {ex:22s} | {phase:4s} | {n} Probandinnen{evt_info}")

    print(f"\nSHOW_SD = {SHOW_SD}")
    print("\nErstelle Plots ...")
    plot_all_muscles_by_phase(curves, events, OUTPUT_DIR)
    plot_phases_overlay_by_muscle(curves, events, OUTPUT_DIR)

    print(f"\n{'='*70}")
    print("FERTIG")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()