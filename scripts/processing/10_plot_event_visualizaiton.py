"""
plot_event_definitions.py
-------------------------
Erstellt Abbildungen, die die Event-Definitionen fuer CMJ, DJ und Squat-Uebungen
visualisieren.

Darstellung:
  - CMJ: vGRF (summiert, in N) mit 4 Events
  - DJ:  vGRF (summiert, in N) mit 4 Events
  - SQ (bilateral):            vGRF (summiert, in N) + Kniewinkel rechts
  - SQ (unilateral rechts):    vGRF (rechts, in N)   + Kniewinkel rechts

Hinweise:
  - GRF-Rohdaten liegen in BW (Body Weight) vor und werden mit
    MASS * 9.81 in Newton zurueckgerechnet.
  - Ausgewertet wird immer die rechte Seite (siehe Methodik).
  - Fuer SQ werden keine Event-Markierungen gesetzt; der gesamte Trial
    wird analysiert. Stattdessen werden Strichmaennchen an drei
    repraesentativen Zeitpunkten (Start, Tiefster Punkt, Ende) gezeigt.

INPUT:  _kin.csv-Dateien aus 03_preprocessed_emg_data
        PNG-Strichmaennchen aus STICKFIGURE_DIR (transparenter Hintergrund)
OUTPUT: SVG-Dateien fuer den Methodenteil
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# =============================================================================
# KONFIGURATION
# =============================================================================

# Pfade zu den Kinematik-CSVs (jeweils ein repräsentatives Beispiel)
PREPROCESSED_DIR = Path(
    r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG"
    r"\data\03_preprocessed_emg_data"
)

# beste Trials von S01 verwendet zum darstellen
BEISPIEL_TRIALS = {
    "CMJ": {
        "subject": "S01",
        "phase":   "01_PER",
        "exercise": "CMJ",
        "side":    "BILATERAL",
        "trial":   "CMJ_02",        # Name ohne _kin.csv
    },
    "DJ": {
        "subject": "S01",
        "phase":   "01_PER",
        "exercise": "DJ",
        "side":    "BILATERAL",
        "trial":   "DJ_01",
    },
    "SQ": {
        "subject": "S01",
        "phase":   "03_LUT",
        "exercise": "SQ",
        "side":    "BILATERAL",
        "trial":   "SQ_03",
    },
}

OUTPUT_DIR = Path(
    r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\outputs\figures\event_definitions")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ordner mit den Strichmaennchen-PNGs (transparenter Hintergrund!)
STICKFIGURE_DIR = Path(
    r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG"
    r"\figures\stickfigures"
)

# Zoom-Faktor fuer Strichmaennchen (je kleiner, desto kleiner die Figur)
STICKFIGURE_ZOOM = 0.25

# Events pro Übung: (Spaltenname, Label, Farbe, Linienstil, Stickfigure)
EVENTS_CONFIG = {
    "CMJ": [
        ("event_start_s",    "Start",     "#4CAF50", "-",  "cmj_start.png"),
        ("event_take_off_s", "Take-off",  "#2196F3", "-",  "cmj_takeoff.png"),
        ("event_landing_s",  "Landing",   "#FF5722", "-",  "cmj_landing.png"),
        ("event_end_jump_s", "End Jump",  "#888888", "--", "cmj_endjump.png"),
    ],
    "DJ": [
        ("event_landing1_s", "Landing 1", "#FF5722", "-",  "dj_landing1.png"),
        ("event_take_off_s", "Take-off",  "#2196F3", "-",  "dj_takeoff.png"),
        ("event_landing2_s", "Landing 2", "#FF5722", "--", "dj_landing2.png"),
        ("event_end_jump_s", "End Jump",  "#888888", ":",  "dj_endjump.png"),
    ],
}

# Fuer Squat: (relative Position im Trial 0..1, Label, Stickfigure-Datei)
# 0.0 = Trial-Anfang, 0.5 = Mitte, 1.0 = Trial-Ende
SQUAT_STICKFIGURES = [
    (0.00, "Start",           "sq_start.png"),
    (0.50, "Tiefster Punkt",  "sq_bottom.png"),
    (1.00, "Ende",            "sq_end.png"),
]

# Farben
COLOR_GRF   = "#1f4e79"
COLOR_KNEE  = "#c44e52"

# =============================================================================
# DATEN LADEN
# =============================================================================

def load_kin(subject, phase, exercise, side, trial):
    """Lädt die _kin.csv für Trial."""
    path = (
        PREPROCESSED_DIR / subject / phase / exercise / side
        / f"{trial}_kin.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    return pd.read_csv(path, low_memory=False)


def get_vgrf_in_newton(df, bilateral=True):
    """vGRF in Newton (BW * MASS * 9.81)."""
    mass = pd.to_numeric(df["MASS"], errors="coerce").iloc[0]
    if np.isnan(mass):
        raise ValueError("MASS nicht in den Metadaten gefunden")
    factor = mass * 9.81


    right = df["Right GRF_Z"] if "Right GRF_Z" in df.columns else None
    left  = df["Left GRF_Z"]  if "Left GRF_Z"  in df.columns else None

    if bilateral and left is not None and right is not None:
        left_num  = pd.to_numeric(left,  errors="coerce")
        right_num = pd.to_numeric(right, errors="coerce")
        mask_both_nan = left_num.isna() & right_num.isna()
        summed = left_num.fillna(0) + right_num.fillna(0)
        summed[mask_both_nan] = np.nan
        return summed * factor
    elif right is not None:
        return pd.to_numeric(right, errors="coerce") * factor
    elif left is not None:
        return pd.to_numeric(left, errors="coerce") * factor
    else:
        raise ValueError("Keine GRF_Z-Spalte gefunden")

def get_knee_angle_right(df):
    """Gibt die rechte Kniewinkel-Zeitreihe zurueck."""
    if "Right Knee Angles" in df.columns:
        return pd.to_numeric(df["Right Knee Angles"], errors="coerce")
    raise ValueError("'Right Knee Angles' nicht gefunden")


def get_events(df, exercise):
    """Liste von (t, label, color, linestyle, stickfigure_filename)."""
    events = []
    for col, label, color, ls, png in EVENTS_CONFIG.get(exercise, []):
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").iloc[0]
            if not np.isnan(val):
                events.append((val, label, color, ls, png))
    return events


def clean_timeseries(time, values):
    """Entfernt NaN-Zeilen aus einer Zeitreihe."""
    values = pd.to_numeric(values, errors="coerce")
    mask = values.notna()
    return time[mask].values, values[mask].values

# =============================================================================
# STRICHMAENNCHEN-HILFSFUNKTIONEN
# =============================================================================

def _load_stickfigure(filename):
    """Laedt ein Stickfigure-PNG. Gibt None zurueck, wenn nicht gefunden."""
    path = STICKFIGURE_DIR / filename
    if not path.exists():
        print(f"    [WARNUNG] Stickfigure fehlt: {path.name}")
        return None
    return mpimg.imread(path)


def _add_stickfigure_row(ax_fig, ax_data, positions_labels_files):
    """
    Platziert Strichmaennchen ueber dem eigentlichen Plot-Bereich.

    ax_fig   : die Figure (fuer koordinaten-unabhaengige Platzierung)
    ax_data  : der Plot-Axes (fuer x-Daten-Koordinaten)
    positions_labels_files : Liste von (x_data, label, filename)
    """
    for x_data, label, filename in positions_labels_files:
        img = _load_stickfigure(filename)
        if img is None:
            continue

        # Umrechnung: Daten-x -> Display-x -> Figure-y liegt fix oben
        imagebox = OffsetImage(img, zoom=STICKFIGURE_ZOOM)
        # y in ax_data-Koordinaten: oberhalb der Ax-Grenze (axes fraction 1.15)
        ab = AnnotationBbox(
            imagebox,
            xy=(x_data, 1.18),
            xycoords=("data", "axes fraction"),
            frameon=False,
            box_alignment=(0.5, 0.0),  # Mitte-unten
        )
        ax_data.add_artist(ab)

# =============================================================================
# PLOT-FUNKTIONEN
# =============================================================================

def plot_jump(df, exercise, out_path):
    """Plot fuer CMJ / DJ: vGRF mit Events + Strichmaennchen-Reihe oben."""
    time = df["time_s"]
    vgrf = get_vgrf_in_newton(df, bilateral=True)
    t, v = clean_timeseries(time, vgrf)
    events = get_events(df, exercise)

    t0 = events[0][0] if events else t[0]
    t_rel = t - t0

    # Etwas hoehere Figur wegen Strichmaennchen-Reihe
    fig, ax = plt.subplots(figsize=(8, 5.0))
    ax.plot(t_rel, v, color=COLOR_GRF, linewidth=1.2,
            label="vGRF (summiert)")
    ax.set_ylabel("vGRF [N]")
    ax.set_xlabel("Zeit [s]")
    ax.grid(alpha=0.3)

    # Extra Platz oben fuer die Strichmaennchen-Reihe
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi)  # Plot-Bereich bleibt gleich

    # Event-Linien (innerhalb Plot) + Labels direkt am oberen Rand der Ax
    for t_evt, label, color, ls, _png in events:
        t_ev_rel = t_evt - t0
        ax.axvline(t_ev_rel, color=color, linestyle=ls, linewidth=1.2,
                   alpha=0.85)
        ax.text(t_ev_rel, y_hi * 0.97, label,
                rotation=90, va="top", ha="right",
                fontsize=8, color=color,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.75, pad=1.5))

    # Strichmaennchen-Reihe oberhalb der Ax
    positions_labels_files = [
        (t_evt - t0, label, png)
        for t_evt, label, _color, _ls, png in events
    ]
    _add_stickfigure_row(fig, ax, positions_labels_files)

    # Platz oben schaffen, damit die Strichmaennchen nicht abgeschnitten werden
    fig.subplots_adjust(top=0.72)

    ax.legend(loc="center right", fontsize=8)
    fig.suptitle(f"Event-Definition: {exercise}",
                 fontsize=11, y=0.98)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path.name}")


def plot_squat(df, bilateral, out_path, title):
    """Plot fuer Squat: vGRF + Kniewinkel rechts + 3 Strichmaennchen oben."""
    time = df["time_s"]
    vgrf = get_vgrf_in_newton(df, bilateral=bilateral)
    knee = get_knee_angle_right(df)

    t_v, v = clean_timeseries(time, vgrf)
    t_k, k = clean_timeseries(time, knee)

    t0 = min(t_v[0], t_k[0])
    t_v = t_v - t0
    t_k = t_k - t0
    t_max = max(t_v[-1], t_k[-1])

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True)

    vgrf_label = "vGRF (summiert)" if bilateral else "vGRF (rechts)"
    axes[0].plot(t_v, v, color=COLOR_GRF, linewidth=1.2, label=vgrf_label)
    axes[0].set_ylabel("vGRF [N]")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_k, k, color=COLOR_KNEE, linewidth=1.2,
                 label="Kniewinkel rechts")
    axes[1].set_ylabel("Kniewinkel [°]")
    axes[1].set_xlabel("Zeit [s]")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[0].text(0.02, 0.95,
                 "Keine Event-Markierungen –\ngesamter Trial wird analysiert",
                 transform=axes[0].transAxes, fontsize=8, va="top",
                 bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))

    # Strichmaennchen-Reihe ueber oberem Subplot
    positions_labels_files = [
        (rel * t_max, label, png)
        for rel, label, png in SQUAT_STICKFIGURES
    ]
    _add_stickfigure_row(fig, axes[0], positions_labels_files)

    # Mehr Platz oben fuer Strichmaennchen
    fig.subplots_adjust(top=0.78, hspace=0.15)

    fig.suptitle(title, fontsize=11, y=0.98)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path.name}")


# =============================================================================
# HAUPTPROGRAMM
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output-Verzeichnis:    {OUTPUT_DIR}")
    print(f"Strichmaennchen aus:   {STICKFIGURE_DIR}\n")

    for key, cfg in BEISPIEL_TRIALS.items():
        print(f"-> {key}: {cfg['subject']}/{cfg['phase']}/"
              f"{cfg['exercise']}/{cfg['side']}/{cfg['trial']}")
        try:
            df = load_kin(**cfg)
        except FileNotFoundError as e:
            print(f"  [FEHLER] {e}\n")
            continue

        out_path = OUTPUT_DIR / f"event_definition_{key}.svg"

        if key in ("CMJ", "DJ"):
            plot_jump(df, key, out_path)
        elif key == "SQ":
            plot_squat(df, bilateral=True, out_path=out_path,
                       title="Event-Definition: Squat (bilateral)")
        elif key == "SQ_R":
            plot_squat(df, bilateral=False, out_path=out_path,
                       title="Event-Definition: Squat (unilateral rechts)")

    print("\nFertig.")


if __name__ == "__main__":
    main()