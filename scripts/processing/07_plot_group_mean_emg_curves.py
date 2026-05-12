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
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D
import sys
sys.path.insert(0, str(Path(r'C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG')))

from scripts.utils.config import (
    EXERCISE_MAP, MUSCLE_NAMES, MUSCLE_COLORS, MUSCLE_LINESTYLES,
    PHASE_COLORS, PHASE_LINESTYLES, PHASE_HATCHES
)

# ============================================================
# PFADE
# ============================================================
PROCESSED_DIR   = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\04_processed_emg_data")
BEST_TRIALS_CSV = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\05_best_trials_group_and_individual\beste_trials.csv")
STATS_CSV       = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\outputs\statistics\statistische_ergebnisse.csv")
OUTPUT_DIR      = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\outputs\figures\plots_group_mean_emg")


# ============================================================
# EINSTELLUNGEN
# ============================================================
PHASES = ["01_PER", "02_OVU", "03_LUT"]
PHASE_LABELS = {"01_PER": "PER", "02_OVU": "OVU", "03_LUT": "LUT"}
PHASE_REVERSE = {"PER": "01_PER", "OVU": "02_OVU", "LUT": "03_LUT"}

SIDE = "R"

N_POINTS = 101  # 0–100 %

# ── Darstellungsoptionen ──────────────────────────────────────
SHOW_SD = True   # auf True oder False setzen, um die SD-Bänder ein- oder eben auszublenden

# ── Einheitliche Typografie fuer alle Plots in diesem Script ─
# Zentral definiert, damit alle Abbildungen im Thesis-Dokument
# konsistent aussehen.
FONT = {
    "suptitle":   {"fontsize": 14, "fontweight": "bold"},  # Figur-Titel
    "subtitle":   {"fontsize": 12, "fontweight": "bold"},  # Subplot-Titel
    "axis_label": {"fontsize": 12},                         # x-/y-Achsentitel
    "tick_label": {"fontsize": 10},                         # fuer set_xlabel/set_ylabel mit kleinerer Schrift
    "tick":       {"labelsize": 10},                         # fuer ax.tick_params()
    "legend":     {"fontsize": 10},                         # Legende
    "annotation": {"fontsize": 10},                         # Text-Annotationen im Plot
}
# Globale rcParams als Fallback fuer alle nicht explizit gesetzten Werte
plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# ── Pfad zu den preprocessed Daten (für KIN-Events) ──────────
PREPROCESSED_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\03_preprocessed_emg_data")

# ── Strichmaennchen-Konfiguration fuer Overlay-Plots ─────────
# Strichmaennchen werden oberhalb des obersten Muskel-Subplots
# platziert und visualisieren den Bewegungsablauf.
STICKFIGURE_DIR = Path(
    r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG"
    r"\figures\stickfigures"
)

# Zoom-Faktor (kalibriert fuer hochaufgeloeste PNGs, 1200 px breit).
# Bei niedriger aufgeloesten Quell-PNGs entsprechend groesseren Wert waehlen.
STICKFIGURE_ZOOM_OVERLAY = 0.015

# Konfiguration der Strichmaennchen-Positionierung pro Uebungstyp.
# "at_event":  Position = prozentualer Zeitpunkt eines Events im Bewegungszyklus
# "between":   Position = Interpolation zwischen zwei Events (factor in [0, 1])
# "at_pct":    Position = fester prozentualer Punkt (fuer Squat)
STICKFIGURES_OVERLAY = {
    "CMJ": [
        {"filename": "cmj_start.png",   "at_event": "Start"},
        {"filename": "cmj_tief.png",    "between": ("Start", "Take-off", 0.5)},
        {"filename": "cmj_takeoff.png", "at_event": "Take-off"},
        {"filename": "cmj_in_luft.png", "between": ("Take-off", "Landing", 0.5)},
        {"filename": "cmj_landing.png", "at_event": "Landing"},
        {"filename": "cmj_endjump.png", "at_event": "End"},
    ],
    "DJ": [
        {"filename": "dj_auf_box.png",     "at_pct": -5.0},
        {"filename": "dj_landing1.png",    "at_event": "Landing 1"},
        {"filename": "dj_exzentrisch.png", "between": ("Landing 1", "Take-off", 0.5)},
        {"filename": "dj_takeoff.png",     "at_event": "Take-off"},
        {"filename": "dj_flug.png",        "between": ("Take-off", "Landing 2", 0.5)},
        {"filename": "dj_landing2.png",    "at_event": "Landing 2"},
        {"filename": "dj_endjump.png",     "at_event": "End"},
    ],
    "SQ": [
        {"filename": "sq_start.png",  "at_pct": 0.0},
        {"filename": "sq_bottom.png", "at_pct": 50.0},
        {"filename": "sq_end.png",    "at_pct": 100.0},
    ],
}

# ── Event-Konfiguration pro Übungstyp ─────────────────────────
# Jedes Event: (event_start_col, event_time_col, label, farbe, linestyle)

EVENT_CONFIG = {
    "CMJ": [
        ("event_start_s",    "Start",     "#888888", ":"),
        ("event_take_off_s", "Take-off",  "#0072B2", "-"),
        ("event_landing_s",  "Landing",   "#D55E00", "-"),
        ("event_end_jump_s", "End",       "#888888", ":"),
    ],
    "DJ": [
        ("event_landing1_s", "Landing 1", "#D55E00", "-"),
        ("event_take_off_s", "Take-off",  "#0072B2", "-"),
        ("event_landing2_s", "Landing 2", "#D55E00", "--"),
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

def load_stats(stats_path):
    """Laedt statistische Ergebnisse und gibt Dict zurueck."""
    if not stats_path.exists():
        return {}
    df = pd.read_csv(stats_path)
    stats = {}
    for _, row in df.iterrows():
        key = (row["Uebung"], row["Muskel"], row["Kennwert"])
        stats[key] = {
            "signifikant": row["Signifikant"] == "ja",
            "posthoc_PER_OVU_p": row.get("posthoc_PER_OVU_p", np.nan),
            "posthoc_PER_LUT_p": row.get("posthoc_PER_LUT_p", np.nan),
            "posthoc_OVU_LUT_p": row.get("posthoc_OVU_LUT_p", np.nan),
        }
    return stats


def get_sig_symbol(p):
    """Gibt Signifikanz-Symbol zurueck: ***, **, *, ns"""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

def collect_curves(df_best):
    """
    Sammelt die zeitnormalisierten EMG-Kurven der besten Trials.

    Returns:
        curves: { exercise_label: { phase_short: { muscle: [array, ...] } } }
        events: { exercise_label: { phase_short: [ [(pct, label, color, ls), ...], ... ] } }
        trial_metadata: { exercise_label: { phase_short: { muscle: [
            {"subject": str, "trial": str, "events": [(pct, label), ...]}, ...
        ] } } }
    """
    data = {}
    event_data = {}
    trial_metadata = {}

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
            trial_metadata[label] = {}
        if phase_short not in data[label]:
            data[label][phase_short] = {m: [] for m in MUSCLE_NAMES}
            event_data[label][phase_short] = []
            trial_metadata[label][phase_short] = {m: [] for m in MUSCLE_NAMES}

        # Events aus KIN-Datei lesen (für diesen spezifischen Trial)
        kin_path = find_kin_file(
            subject, phase_key, ex_folder, side_folder, trial_name
        )
        trial_events = []
        if kin_path is not None:
            evts = get_event_pct(kin_path, label)
            if evts:
                event_data[label][phase_short].append(evts)
                # Nur (pct, label) für Metadata speichern
                trial_events = [(pct, lbl) for pct, lbl, _, _ in evts]

        for muscle in MUSCLE_NAMES:
            col = f"{SIDE}_{muscle}"
            if col in df.columns:
                vals = df[col].values
                if len(vals) == N_POINTS and not np.all(np.isnan(vals)):
                    data[label][phase_short][muscle].append(vals)
                    trial_metadata[label][phase_short][muscle].append({
                        "subject": subject,
                        "trial": trial_name,
                        "events": trial_events
                    })

    return data, event_data, trial_metadata


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
            offset_x = 0.25  # <--- Wert anpassen, je nachdem wie viel Abstand gewünscht ist
            ax.text(
                mean_pct + offset_x, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100,
                f"{label}", **FONT["annotation"], color=info["color"],
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
                mean = np.nanmean(stacked, axis=0)
                sd = np.nanstd(stacked, axis=0, ddof=1) if len(arrays) > 1 \
                    else np.zeros_like(mean)

                color = MUSCLE_COLORS[muscle]
                ls = MUSCLE_LINESTYLES.get(muscle, "-")

                if SHOW_SD:
                    ax.fill_between(pct, mean - sd, mean + sd,
                                    color=color, alpha=0.15)
                ax.plot(pct, mean, color=color, linewidth=2.0,
                        linestyle=ls, label=muscle)

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
                **FONT["subtitle"],
            )
            ax.set_xlabel("Bewegungszyklus in %", **FONT["axis_label"])
            ax.set_xlim(-1, 101)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)

        axes[0].set_ylabel("EMG-Amplitude in % Baseline)", **FONT["axis_label"])

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
                   ncol=len(handles), **FONT["legend"], framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.02))

        """
        fig.suptitle(
            f"{exercise}  –  Gruppenmittel aller Muskeln (beste Trials)",
            **FONT["suptitle"], y=1.01
        )
        """

        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        out_path = out_dir / f"group_mean_{exercise.replace(' ', '_')}_by_phase.svg"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out_path.name}")


# ============================================================
# PLOT 2: Phasen überlagert, getrennt nach Muskel
# ============================================================

def _get_landing_pct_ranges(events, exercise, phase_order):
    """
    Berechnet die gemittelten prozentualen Landungsphasen-Bereiche
    aus den Event-Daten aller Probandinnen.

    CMJ: Landing → End  (eine Landungsphase)
    DJ:  Landing 2 → End

    Returns: list of (start_pct, end_pct, label, color)
    """
    from collections import defaultdict

    # Alle Events über alle Phasen sammeln
    all_event_lists = []
    for phase in phase_order:
        all_event_lists.extend(events.get(exercise, {}).get(phase, []))

    if not all_event_lists:
        return []

    # Events nach Label gruppieren und mitteln
    grouped = defaultdict(list)
    for trial_events in all_event_lists:
        for pct_pos, label, color, ls in trial_events:
            grouped[label].append(pct_pos)

    mean_events = {label: np.mean(pcts) for label, pcts in grouped.items()}

    ranges = []
    if "CMJ" in exercise:
        if "Landing" in mean_events and "End" in mean_events:
            ranges.append((mean_events["Landing"], 100.0,
                           "Landing", "#D55E00"))
    elif "DJ" in exercise:
        if "Landing 2" in mean_events and "End" in mean_events:
            ranges.append((mean_events["Landing 2"], 100.0,
                           "Landing 2", "#D55E00"))

    return ranges


def _compute_phase_means_in_range(curves_exercise, trial_meta_exercise, 
                                  phase_order, muscle, exercise_label):
    """
    Berechnet mean und SD der EMG-Amplitude innerhalb der INDIVIDUELLEN
    Landungsphase jedes Trials (statt Gruppenmittelwert-Zeitfenster).

    Returns: dict {phase: (mean, sd, n)}
    """
    pct = np.linspace(0, 100, N_POINTS)

    # Event-Labels für Landing je nach Übung
    if "CMJ" in exercise_label:
        landing_label = "Landing"
    elif "DJ" in exercise_label:
        landing_label = "Landing 2"
    else:
        # SQ hat keine Landungsphase
        return {p: (np.nan, np.nan, 0) for p in phase_order}

    result = {}
    for phase in phase_order:
        muscle_dict = curves_exercise.get(phase, {})
        arrays = muscle_dict.get(muscle, [])
        metadata = trial_meta_exercise.get(phase, {}).get(muscle, [])
        
        if not arrays or len(arrays) != len(metadata):
            result[phase] = (np.nan, np.nan, 0)
            continue

        trial_means = []
        missing_events = []
        
        for arr, meta in zip(arrays, metadata):
            events = meta.get("events", [])
            # Finde Landing-Event für diesen Trial
            landing_pct = None
            for event_pct, event_label in events:
                if event_label == landing_label:
                    landing_pct = event_pct
                    break
            
            if landing_pct is None:
                # Landing-Event fehlt → Trial überspringen (Variante A)
                missing_events.append(f"{meta['subject']}/{meta['trial']}")
                continue
            
            # Individuelles Zeitfenster: Landing → End (100%)
            mask = (pct >= landing_pct) & (pct <= 100)
            segment = arr[mask]
            if len(segment) > 0:
                trial_means.append(np.nanmean(segment))
        
        if missing_events:
            print(f"    [WARNUNG] {exercise_label} | {muscle} | {phase}: "
                  f"{len(missing_events)} Trial(s) ohne '{landing_label}'-Event übersprungen: "
                  f"{', '.join(missing_events)}")

        if trial_means:
            result[phase] = (
                np.nanmean(trial_means),
                np.nanstd(trial_means, ddof=1) if len(trial_means) > 1 else 0.0,
                len(trial_means),
            )
        else:
            result[phase] = (np.nan, np.nan, 0)

    return result

def _compute_phase_peaks_in_range(curves_exercise, trial_meta_exercise,
                                  phase_order, muscle, exercise_label):
    """
    Berechnet Peak (Maximum) und SD der EMG-Amplitude innerhalb der 
    INDIVIDUELLEN Landungsphase jedes Trials.

    Returns: dict {phase: (peak_mean, peak_sd, n)}
    """
    pct = np.linspace(0, 100, N_POINTS)

    # Event-Labels für Landing je nach Übung
    if "CMJ" in exercise_label:
        landing_label = "Landing"
    elif "DJ" in exercise_label:
        landing_label = "Landing 2"
    else:
        # SQ hat keine Landungsphase
        return {p: (np.nan, np.nan, 0) for p in phase_order}

    result = {}
    for phase in phase_order:
        muscle_dict = curves_exercise.get(phase, {})
        arrays = muscle_dict.get(muscle, [])
        metadata = trial_meta_exercise.get(phase, {}).get(muscle, [])
        
        if not arrays or len(arrays) != len(metadata):
            result[phase] = (np.nan, np.nan, 0)
            continue

        trial_peaks = []
        missing_events = []
        
        for arr, meta in zip(arrays, metadata):
            events = meta.get("events", [])
            # Finde Landing-Event für diesen Trial
            landing_pct = None
            for event_pct, event_label in events:
                if event_label == landing_label:
                    landing_pct = event_pct
                    break
            
            if landing_pct is None:
                # Landing-Event fehlt → Trial überspringen (Variante A)
                missing_events.append(f"{meta['subject']}/{meta['trial']}")
                continue
            
            # Individuelles Zeitfenster: Landing → End (100%)
            mask = (pct >= landing_pct) & (pct <= 100)
            segment = arr[mask]
            if len(segment) > 0:
                trial_peaks.append(np.nanmax(segment))
        
        if missing_events:
            print(f"    [WARNUNG] {exercise_label} | {muscle} | {phase}: "
                  f"{len(missing_events)} Trial(s) ohne '{landing_label}'-Event übersprungen: "
                  f"{', '.join(missing_events)}")

        if trial_peaks:
            result[phase] = (
                np.nanmean(trial_peaks),
                np.nanstd(trial_peaks, ddof=1) if len(trial_peaks) > 1 else 0.0,
                len(trial_peaks),
            )
        else:
            result[phase] = (np.nan, np.nan, 0)

    return result

def _get_exercise_type(exercise: str) -> str:
    """Ermittelt den Uebungstyp (CMJ, DJ, SQ) aus dem Uebungs-Label."""
    if "CMJ" in exercise:
        return "CMJ"
    if "DJ" in exercise:
        return "DJ"
    if "SQ" in exercise:
        return "SQ"
    return ""


def _mean_event_positions_pct(events, exercise, phase_order):
    """
    Ermittelt die gemittelte prozentuale Position jedes Events
    ueber alle Phasen und Probandinnen.

    Returns: dict {label: pct_position}
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for phase in phase_order:
        for trial_events in events.get(exercise, {}).get(phase, []):
            for pct_pos, label, color, ls in trial_events:
                grouped[label].append(pct_pos)

    return {label: float(np.mean(pcts)) for label, pcts in grouped.items()}


def _resolve_stickfigure_x_pct(sf_cfg, event_positions):
    """
    Berechnet die prozentuale x-Position eines Strichmaennchens im
    Bewegungszyklus.

    Rueckgabe: x-Position (0-100) oder None, wenn Event fehlt.
    """
    if "at_event" in sf_cfg:
        return event_positions.get(sf_cfg["at_event"])

    if "between" in sf_cfg:
        ev_a, ev_b, factor = sf_cfg["between"]
        t_a = event_positions.get(ev_a)
        t_b = event_positions.get(ev_b)
        if t_a is None or t_b is None:
            return None
        return t_a + (t_b - t_a) * factor

    if "at_pct" in sf_cfg:
        return sf_cfg["at_pct"]

    return None


def _add_overlay_stickfigures(ax, exercise, events, phase_order):
    """
    Platziert Strichmaennchen oberhalb des uebergebenen Subplots.
    Aufrufer muss sicherstellen, dass genug Platz oberhalb der Ax
    reserviert ist (z.B. via fig.subplots_adjust(top=...)).
    """
    ex_type = _get_exercise_type(exercise)
    sf_cfgs = STICKFIGURES_OVERLAY.get(ex_type, [])
    if not sf_cfgs:
        return

    event_positions = _mean_event_positions_pct(events, exercise, phase_order)

    for sf in sf_cfgs:
        x_pct = _resolve_stickfigure_x_pct(sf, event_positions)
        if x_pct is None:
            continue
 
        img_path = STICKFIGURE_DIR / sf["filename"]
        if not img_path.exists():
            print(f"    [WARNUNG] Stichmaennchen fehlt: {img_path.name}")
            continue
 
        img = mpimg.imread(img_path)
        imagebox = OffsetImage(
            img, zoom=STICKFIGURE_ZOOM_OVERLAY,
            resample=True, interpolation="hermite",
        )
        ab = AnnotationBbox(
            imagebox,
            xy=(x_pct, 1.05),
            xycoords=("data", "axes fraction"),
            frameon=False,
            box_alignment=(0.5, 0.0),
        )
        ax.add_artist(ab)
 

def plot_phases_overlay_by_muscle(curves, events, trial_metadata, out_dir, stats_dict):
    """
    Pro Übung: 5 Zeilen (Muskeln).
    Alle 3 Phasen überlagert in jedem Subplot.
 
    Bei CMJ und DJ: zwei zusätzliche Spalten rechts mit Balkendiagrammen
    für Mean- und Peak-EMG-Amplitude in der Landungsphase.
    Landungsphase im Kurvenplot einheitlich orange hinterlegt.
    """
    pct = np.linspace(0, 100, N_POINTS)
    phase_order = ["PER", "OVU", "LUT"]
 
    for exercise, phase_data in sorted(curves.items()):
        n_mus = len(MUSCLE_NAMES)
 
        # Landungsphasen bestimmen
        landing_ranges = _get_landing_pct_ranges(events, exercise, phase_order)
        has_bars = len(landing_ranges) > 0
 
        # Layout: Kurvenplot + 2 Balkendiagramme (Mean + Peak)
        if has_bars:
            n_cols = 3  # Kurve + Mean + Peak
            width_ratios = [4, 1, 1]
            fig, axes_all = plt.subplots(
                n_mus, n_cols, figsize=(14 + 7, 3 * n_mus),
                gridspec_kw={"width_ratios": width_ratios},
            )
        else:
            n_cols = 1
            fig, axes_col = plt.subplots(
                n_mus, 1, figsize=(14, 3 * n_mus), sharex=True,
            )
            axes_all = axes_col.reshape(-1, 1)
 
        for row_idx, muscle in enumerate(MUSCLE_NAMES):
            # ── Kurvenplot (linke Spalte) ─────────────────────────
            ax = axes_all[row_idx, 0]
 
            # Landungsphase einheitlich orange hinterlegen
            for pct_s, pct_e, lbl, col in landing_ranges:
                ax.axvspan(pct_s, pct_e, color=col, alpha=0.08, zorder=0)
 
            for phase in phase_order:
                muscle_dict = phase_data.get(phase, {})
                arrays = muscle_dict.get(muscle, [])
                if not arrays:
                    continue
 
                stacked = np.vstack(arrays)
                mean = np.nanmean(stacked, axis=0)
                sd = np.nanstd(stacked, axis=0, ddof=1) if len(arrays) > 1 \
                    else np.zeros_like(mean)
 
                color = PHASE_COLORS[phase]
                ls = PHASE_LINESTYLES.get(phase, "-")
 
                if SHOW_SD:
                    ax.fill_between(pct, mean - sd, mean + sd,
                                    color=color, alpha=0.10)
                ax.plot(pct, mean, color=color, linewidth=2.0,
                        linestyle=ls,
                        label=f"{phase} (n={len(arrays)})")
 
            ax.axhline(y=100, color="black", linewidth=0.6,
                       linestyle="--", alpha=0.4)
 
            # ── Strichmaennchen oberhalb des obersten Muskel-Subplots ──
            _add_overlay_stickfigures(
                axes_all[0, 0], exercise, events, phase_order,
            )

            # Event-Linien
            all_evt_lists = []
            for phase in phase_order:
                all_evt_lists.extend(events.get(exercise, {}).get(phase, []))
            if all_evt_lists:
                draw_event_lines(ax, all_evt_lists, add_label=(row_idx == 0))
 
            ax.set_ylabel(f"{muscle}\nin % BL", **FONT["tick_label"])
            ax.set_xlim(-1, 101)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(**FONT["legend"], loc="upper right", framealpha=0.9)

            # SQ ohne Balken: Y-Achse anpassen
            if not has_bars and "SQ" in exercise:
                if "bilateral" in exercise.lower():
                    # Bilateral ist sehr niedrig -> auf 60% ranzoomen
                    ax.set_ylim(0, 60)
                else:
                    # Einbeinig (höhere Ausschläge) -> Nicht kappen!
                    # Wir zwingen die Achse nur dazu, bei 0 zu starten.
                    # Die Obergrenze passt Matplotlib automatisch an die Daten an.
                    ax.set_ylim(bottom=0)
 
            if row_idx == n_mus - 1:
                ax.set_xlabel("Bewegungszyklus in %", **FONT["axis_label"])
 
            # ── Balkendiagramme (Mean + Peak) ─────────────────────
            if has_bars:
                pct_s, pct_e, bar_label, bar_color = landing_ranges[0]
 
                # Mean-Aktivierung (individuell pro Trial berechnet)
                mean_stats = _compute_phase_means_in_range(
                    phase_data, 
                    trial_metadata.get(exercise, {}),
                    phase_order, 
                    muscle, 
                    exercise,
                )
                # Peak-Aktivierung (individuell pro Trial berechnet)
                peak_stats = _compute_phase_peaks_in_range(
                    phase_data,
                    trial_metadata.get(exercise, {}),
                    phase_order,
                    muscle,
                    exercise,
                )
 
                # ── Gemeinsame Y-Achse berechnen (über Kurve + beide Balken) ──
                # Kurvenplot: aktuelles ylim nach auto-scaling
                curve_ylo, curve_yhi = ax.get_ylim()
                
                # Maximale Balkenhöhe (Mean + Peak + SD)
                all_bar_maxima = []
                for stats in [mean_stats, peak_stats]:
                    for phase in phase_order:
                        m, s, n = stats[phase]
                        if not np.isnan(m) and not np.isnan(s):
                            all_bar_maxima.append(m + s)
                
                if all_bar_maxima:
                    bar_max = max(all_bar_maxima)
                    # 8% Puffer oberhalb höchstem Balken
                    needed_yhi = bar_max * 1.08
                else:
                    needed_yhi = curve_yhi
                
                # Y-Obergrenze = max(Kurve, Balken)
                unified_yhi = max(curve_yhi, needed_yhi)
                
                # SQ: Y-Achse auf 60% begrenzen (bessere Auflösung)
                if "SQ" in exercise:
                    unified_yhi = min(unified_yhi, 60)
                
                # Untergrenze = 0 (Balken stehen am Boden)
                unified_ylo = 0
                
                # Y-Achse auf Kurvenplot UND beide Balkendiagramme anwenden
                ax.set_ylim(unified_ylo, unified_yhi)

                for bar_col_idx, (stats, title_text) in enumerate([
                    (mean_stats, "Mittlere Aktivierung"),
                    (peak_stats, "Maximale Aktivierung"),
                ]):
                    ax_bar = axes_all[row_idx, 1 + bar_col_idx]
 
                    phases_present = [p for p in phase_order
                                      if not np.isnan(stats[p][0])]
                    means = [stats[p][0] for p in phases_present]
                    sds = [stats[p][1] for p in phases_present]
                    colors = [PHASE_COLORS[p] for p in phases_present]
 
                    x_pos = np.arange(len(phases_present))
                    bars = ax_bar.bar(
                        x_pos, means, yerr=sds, capsize=3,
                        color=colors, alpha=0.85,
                        edgecolor="white", linewidth=1.0,
                        zorder=2,
                    )
                    for bar, p in zip(bars, phases_present):
                        bar.set_hatch(PHASE_HATCHES.get(p, ""))
 
                    ax_bar.set_xticks(x_pos)
                    ax_bar.set_xticklabels(phases_present, **FONT["annotation"])
                    ax_bar.spines["top"].set_visible(False)
                    ax_bar.spines["right"].set_visible(False)
                    ax_bar.grid(axis="y", alpha=0.2)
 
                    # Einheitliche Y-Achse für alle 3 Subplots
                    ax_bar.set_ylim(unified_ylo, unified_yhi)
                    ax_bar.tick_params(axis="y", labelsize=9)
 
                    # Titel nur in erster Zeile
                    if row_idx == 0:
                        # Übersetzt "Landing" zu "Landung" (funktioniert auch bei "Landing 2")
                        deutsches_label = bar_label.replace("Landing", "Landung")

                        ax_bar.set_title(
                            f"{title_text}\n{deutsches_label}",
                            **FONT["subtitle"],
                            color=bar_color,
                        )

                    # ── Signifikanz-Brackets ──────────────────────────────
                    kennwert = "mean_emg" if title_text == "Mittlere Aktivierung" else "peak_emg"
                    stat_key = (exercise, muscle, kennwert)
                    if stat_key in stats_dict and stats_dict[stat_key]["signifikant"]:
                        sig_info = stats_dict[stat_key]
                        
                        # Den höchsten Punkt der Balken in DIESEM Diagramm finden (Mean + SD)
                        local_bar_max = max([m + s for m, s in zip(means, sds)]) if means else 0
                        y_range = ax_bar.get_ylim()[1] - ax_bar.get_ylim()[0]
                        
                        # Höhe für Brackets direkt über dem höchsten Balken starten
                        bracket_height = local_bar_max + 0.05 * y_range
                        bracket_offset = 0.08 * y_range
                        
                        # Post-hoc-Vergleiche prüfen
                        comparisons = []
                        if "PER" in phases_present and "OVU" in phases_present:
                            p = sig_info["posthoc_PER_OVU_p"]
                            sym = get_sig_symbol(p)
                            if sym:
                                i_per = phases_present.index("PER")
                                i_ovu = phases_present.index("OVU")
                                comparisons.append((i_per, i_ovu, sym))
                        
                        if "PER" in phases_present and "LUT" in phases_present:
                            p = sig_info["posthoc_PER_LUT_p"]
                            sym = get_sig_symbol(p)
                            if sym:
                                i_per = phases_present.index("PER")
                                i_lut = phases_present.index("LUT")
                                comparisons.append((i_per, i_lut, sym))
                        
                        if "OVU" in phases_present and "LUT" in phases_present:
                            p = sig_info["posthoc_OVU_LUT_p"]
                            sym = get_sig_symbol(p)
                            if sym:
                                i_ovu = phases_present.index("OVU")
                                i_lut = phases_present.index("LUT")
                                comparisons.append((i_ovu, i_lut, sym))
                        
                        # Brackets zeichnen (gestaffelt bei mehreren)
                        for level, (i1, i2, symbol) in enumerate(comparisons):
                            y_bracket = bracket_height + level * bracket_offset
                            x1, x2 = x_pos[i1], x_pos[i2]
                            
                            # Horizontale Linie (clip_on=False verhindert das Abschneiden am Rand)
                            ax_bar.plot([x1, x2], [y_bracket, y_bracket],
                                       color="black", linewidth=1.0, zorder=10, clip_on=False)
                            # Vertikale Endstriche
                            tick_len = 0.02 * y_range
                            ax_bar.plot([x1, x1], [y_bracket - tick_len, y_bracket],
                                       color="black", linewidth=1.0, zorder=10, clip_on=False)
                            ax_bar.plot([x2, x2], [y_bracket - tick_len, y_bracket],
                                       color="black", linewidth=1.0, zorder=10, clip_on=False)
                            # Signifikanz-Symbol
                            ax_bar.text((x1 + x2) / 2, y_bracket + 0.01 * y_range,
                                       symbol, ha="center", va="bottom",
                                       fontsize=12, fontweight="bold", zorder=11, clip_on=False)
 
        """
        fig.suptitle(
            f"{exercise}  –  Phasenvergleich pro Muskel (Gruppenmittel, beste Trials)",
            **FONT["suptitle"], y=1.01
        )"""

        # ── Strichmaennchen oberhalb des obersten Muskel-Subplots ──
        _add_overlay_stickfigures(
            axes_all[0, 0], exercise, events, phase_order,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # Zusaetzlicher Platz oberhalb der ersten Zeile fuer die Strichmaennchen
        fig.subplots_adjust(top=0.92)
        out_path = out_dir / f"group_mean_{exercise.replace(' ', '_')}_phase_overlay.svg"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out_path.name}")


# ============================================================
# PLOT 3: Alle Muskeln by Phase – MIT Einzeltrials
# ============================================================

def plot_all_muscles_by_phase_with_trials(curves, events, out_dir):
    """
    Wie plot_all_muscles_by_phase, aber zusätzlich werden die
    einzelnen Probandinnen-Kurven als dünne Linien im Hintergrund
    in der jeweiligen Muskelfarbe dargestellt.
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

                color = MUSCLE_COLORS[muscle]
                ls = MUSCLE_LINESTYLES.get(muscle, "-")

                # Einzelne Trials als Linien
                for i, arr in enumerate(arrays):
                    ax.plot(pct, arr, color=color, linewidth=1,
                            linestyle=ls, alpha=0.45, zorder=1,
                            label=muscle if i == 0 else None)

            ax.axhline(y=100, color="black", linewidth=0.6,
                       linestyle="--", alpha=0.4)

            # Event-Linien
            evt_lists = events.get(exercise, {}).get(phase, [])
            if evt_lists:
                eh = draw_event_lines(ax, evt_lists, add_label=(col_idx == 0))
                if col_idx == 0:
                    event_handles = eh

            ax.set_title(
                f"{PHASE_LABELS.get(PHASE_REVERSE.get(phase, ''), phase)} "
                f"(n={len(next(iter(muscle_dict.values()), []))})",
                **FONT["subtitle"],
            )
            ax.set_xlabel("Bewegungszyklus in %", **FONT["axis_label"])
            ax.set_xlim(-1, 101)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)

        axes[0].set_ylabel("EMG-Amplitude in % Baseline", **FONT["axis_label"])

        # Legende
        handles, labels = axes[0].get_legend_handles_labels()
        baseline_handle = Line2D([0], [0], color="black", linewidth=0.8,
                                 linestyle="--", alpha=0.5, label="100 % BL")
        handles += [baseline_handle]
        labels += ["100 % BL"]

        for eh in event_handles:
            handles.append(eh)
            labels.append(eh.get_label())

        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(handles), 9), **FONT["legend"], framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.02))

        """
        fig.suptitle(
            f"{exercise}  –  Einzeltrials aller Muskeln (beste Trials, alle Probandinnen)",
            **FONT["suptitle"], y=1.01
        )"""

        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        out_path = out_dir / f"group_mean_{exercise.replace(' ', '_')}_by_phase_with_trials.svg"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out_path.name}")


# ============================================================
# PLOT 4: Phasen-Overlay by Muscle – MIT Einzeltrials
# ============================================================

def plot_phases_overlay_by_muscle_with_trials(curves, events, out_dir):
    """
    Wie plot_phases_overlay_by_muscle, aber zusätzlich werden die
    einzelnen Probandinnen-Kurven als dünne Linien im Hintergrund
    in der jeweiligen Phasenfarbe dargestellt.
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

                color = PHASE_COLORS[phase]
                ls = PHASE_LINESTYLES.get(phase, "-")

                # Einzelne Trials als Linien
                for i, arr in enumerate(arrays):
                    ax.plot(pct, arr, color=color, linewidth=1,
                            linestyle=ls, alpha=0.40, zorder=1,
                            label=f"{phase} (n={len(arrays)})" if i == 0 else None)

            ax.axhline(y=100, color="black", linewidth=0.6,
                       linestyle="--", alpha=0.4)

            # Event-Linien
            all_evt_lists = []
            for phase in phase_order:
                all_evt_lists.extend(events.get(exercise, {}).get(phase, []))
            if all_evt_lists:
                draw_event_lines(ax, all_evt_lists, add_label=(row_idx == 0))

            ax.set_ylabel(f"{muscle}\nin % BL", **FONT["tick_label"])
            ax.set_xlim(-1, 101)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(**FONT["legend"], loc="upper right", framealpha=0.8)


        axes[-1].set_xlabel("Bewegungszyklus in %", **FONT["axis_label"])

        """
        fig.suptitle(
            f"{exercise}  –  Phasenvergleich pro Muskel, Einzeltrials (alle Probandinnen)",
            **FONT["suptitle"], y=1.01
        )
        """

        plt.tight_layout(rect=[0, 0, 1, 0.88])
        
        # top=0.78 gibt oben 22% des Platzes für die Männchen und den Titel frei.
        # So schneidet bbox_inches="tight" nichts mehr ab.
        #fig.subplots_adjust(top=0.78, left=0.08, right=0.98)

        out_path = out_dir / f"group_mean_{exercise.replace(' ', '_')}_phase_muscle_trials.svg"
        fig.savefig(out_path, format="svg", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out_path.name}")

# ============================================================
# PLOT 5: Nur Balkendiagramme für Poster
# ============================================================

def plot_landing_bars_only(curves, events, trial_metadata, out_dir, stats_dict):
    """
    Erstellt separate SVGs, die NUR die Balkendiagramme (Mittlere Aktivierung & Peak)
    für die Landungsphase enthalten. Wird nur für Sprungübungen (CMJ, DJ) generiert.
    """
    phase_order = ["PER", "OVU", "LUT"]

    for exercise, phase_data in sorted(curves.items()):
        n_mus = len(MUSCLE_NAMES)

        # Landungsphasen bestimmen
        landing_ranges = _get_landing_pct_ranges(events, exercise, phase_order)
        has_bars = len(landing_ranges) > 0

        # Wenn die Übung keine Landungsphase hat (z. B. Squat), überspringen wir sie hier
        if not has_bars:
            continue

        # Layout: Nur 2 Spalten (Mean + Peak)
        fig, axes_all = plt.subplots(
            n_mus, 2, figsize=(8, 3 * n_mus), # Schmaler als vorher, da die Kurve fehlt
        )

        for row_idx, muscle in enumerate(MUSCLE_NAMES):
            pct_s, pct_e, bar_label, bar_color = landing_ranges[0]
            deutsches_label = bar_label.replace("Landing", "Landung")

            # Mean-Aktivierung berechnen
            mean_stats = _compute_phase_means_in_range(
                phase_data, 
                trial_metadata.get(exercise, {}),
                phase_order, 
                muscle, 
                exercise,
            )
            # Peak-Aktivierung berechnen
            peak_stats = _compute_phase_peaks_in_range(
                phase_data,
                trial_metadata.get(exercise, {}),
                phase_order,
                muscle,
                exercise,
            )

            # ── Y-Achse für diese Muskel-Zeile berechnen ──
            all_bar_maxima = []
            for stats in [mean_stats, peak_stats]:
                for phase in phase_order:
                    m, s, n = stats[phase]
                    if not np.isnan(m) and not np.isnan(s):
                        all_bar_maxima.append(m + s)
            
            if all_bar_maxima:
                bar_max = max(all_bar_maxima)
                # Etwas mehr Puffer (15%) für die Signifikanz-Klammern
                unified_yhi = bar_max * 1.15 
            else:
                unified_yhi = 100
            
            unified_ylo = 0

            # ── Balkendiagramme zeichnen ──
            for bar_col_idx, (stats, title_text) in enumerate([
                (mean_stats, "Mittlere Aktivierung"),
                (peak_stats, "Peak-Aktivierung"),
            ]):
                ax_bar = axes_all[row_idx, bar_col_idx]

                phases_present = [p for p in phase_order if not np.isnan(stats[p][0])]
                means = [stats[p][0] for p in phases_present]
                sds = [stats[p][1] for p in phases_present]
                colors = [PHASE_COLORS[p] for p in phases_present]

                x_pos = np.arange(len(phases_present))
                bars = ax_bar.bar(
                    x_pos, means, yerr=sds, capsize=3,
                    color=colors, alpha=0.85,
                    edgecolor="white", linewidth=1.0,
                    zorder=2,
                )
                for bar, p in zip(bars, phases_present):
                    bar.set_hatch(PHASE_HATCHES.get(p, ""))

                ax_bar.set_xticks(x_pos)
                ax_bar.set_xticklabels(phases_present, **FONT["annotation"])
                ax_bar.spines["top"].set_visible(False)
                ax_bar.spines["right"].set_visible(False)
                ax_bar.grid(axis="y", alpha=0.2)

                # Y-Achse anwenden
                ax_bar.set_ylim(unified_ylo, unified_yhi)
                ax_bar.tick_params(axis="y", labelsize=9)

                # Y-Achsen-Beschriftung nur links (bei der Mean-Spalte)
                if bar_col_idx == 0:
                    ax_bar.set_ylabel(f"{muscle}\nin % BL", **FONT["tick_label"])

                # Titel nur in der obersten Zeile
                if row_idx == 0:
                    ax_bar.set_title(
                        f"{title_text}\n{deutsches_label}",
                        **FONT["subtitle"],
                        color=bar_color,
                    )

                # ── Signifikanz-Brackets ──
                kennwert = "mean_emg" if title_text == "Mittlere Aktivierung" else "peak_emg"
                stat_key = (exercise, muscle, kennwert)
                if stat_key in stats_dict and stats_dict[stat_key]["signifikant"]:
                    sig_info = stats_dict[stat_key]
                    
                    local_bar_max = max([m + s for m, s in zip(means, sds)]) if means else 0
                    y_range = ax_bar.get_ylim()[1] - ax_bar.get_ylim()[0]
                    
                    bracket_height = local_bar_max + 0.05 * y_range
                    bracket_offset = 0.08 * y_range
                    
                    comparisons = []
                    if "PER" in phases_present and "OVU" in phases_present:
                        p = sig_info["posthoc_PER_OVU_p"]
                        sym = get_sig_symbol(p)
                        if sym:
                            comparisons.append((phases_present.index("PER"), phases_present.index("OVU"), sym))
                    
                    if "PER" in phases_present and "LUT" in phases_present:
                        p = sig_info["posthoc_PER_LUT_p"]
                        sym = get_sig_symbol(p)
                        if sym:
                            comparisons.append((phases_present.index("PER"), phases_present.index("LUT"), sym))
                    
                    if "OVU" in phases_present and "LUT" in phases_present:
                        p = sig_info["posthoc_OVU_LUT_p"]
                        sym = get_sig_symbol(p)
                        if sym:
                            comparisons.append((phases_present.index("OVU"), phases_present.index("LUT"), sym))
                    
                    for level, (i1, i2, symbol) in enumerate(comparisons):
                        y_bracket = bracket_height + level * bracket_offset
                        x1, x2 = x_pos[i1], x_pos[i2]
                        
                        ax_bar.plot([x1, x2], [y_bracket, y_bracket], color="black", linewidth=1.0, zorder=10, clip_on=False)
                        tick_len = 0.02 * y_range
                        ax_bar.plot([x1, x1], [y_bracket - tick_len, y_bracket], color="black", linewidth=1.0, zorder=10, clip_on=False)
                        ax_bar.plot([x2, x2], [y_bracket - tick_len, y_bracket], color="black", linewidth=1.0, zorder=10, clip_on=False)
                        ax_bar.text((x1 + x2) / 2, y_bracket + 0.01 * y_range, symbol, ha="center", va="bottom", fontsize=12, fontweight="bold", zorder=11, clip_on=False)

        plt.tight_layout()
        out_path = out_dir / f"group_mean_{exercise.replace(' ', '_')}_landing_bars_only.svg"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
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

    # Statistik laden
    stats_dict = load_stats(STATS_CSV)
    if stats_dict:
        print(f"Statistische Ergebnisse geladen: {len(stats_dict)} Eintraege")
    else:
        print("[WARNUNG] Keine statistischen Ergebnisse gefunden – keine Signifikanz-Brackets")

    print("\nSammle EMG-Kurven der besten Trials ...")
    curves, events, trial_metadata = collect_curves(df_best)

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
    plot_phases_overlay_by_muscle(curves, events, trial_metadata, OUTPUT_DIR, stats_dict)
    #Aufruf für die reinen Balken-Plots
    plot_landing_bars_only(curves, events, trial_metadata, OUTPUT_DIR, stats_dict)

    print("\nErstelle Plots mit Einzeltrials ...")
    plot_all_muscles_by_phase_with_trials(curves, events, OUTPUT_DIR)
    plot_phases_overlay_by_muscle_with_trials(curves, events, OUTPUT_DIR)

    print(f"\n{'='*70}")
    print("FERTIG")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()