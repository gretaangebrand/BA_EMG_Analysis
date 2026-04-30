"""
plot_event_definitions.py
-------------------------
Erstellt Abbildungen, die die Event-Definitionen fuer CMJ, DJ und Squat-Uebungen
visualisieren. Strichmaennchen werden oberhalb des Plots platziert:
  - an Event-Zeitpunkten (z.B. Take-off, Landing)
  - an relativen Positionen zwischen Events (z.B. tiefster Punkt
    im Countermovement, Flugphase)

Darstellung:
  - CMJ: vGRF (Gesamt, in N) mit 4 Events, 6 Strichmaennchen
  - DJ:  vGRF (Gesamt, in N) mit 4 Events, 7 Strichmaennchen
  - SQ (bilateral):          vGRF (Gesamt, in N) + Kniewinkel rechts
  - SQ (unilateral rechts):  vGRF (rechts, in N)   + Kniewinkel rechts

Hinweise:
  - GRF-Rohdaten liegen in BW normalisiert vor und werden mit
    MASS * 9.81 in Newton zurueckgerechnet.
  - Ausgewertet wird immer die rechte Seite (siehe Methodik).
  - Fuer SQ werden keine Event-Markierungen gesetzt; der gesamte Trial
    wird analysiert.

INPUT:  _kin.csv-Dateien aus 03_preprocessed_emg_data
        SVG-Strichmaennchen aus STICKFIGURE_DIR (transparenter Hintergrund)
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

# Ordner mit den Strichmaennchen-SVGs (transparenter Hintergrund!)
STICKFIGURE_DIR = Path(
    r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG"
    r"\figures\stickfigures"
)

# Zoom-Faktor fuer Strichmaennchen (je kleiner, desto kleiner die Figur).
# Wert ist fuer die hochaufgeloesten PNGs (1200 px breit) kalibriert;
# bei niedrig aufgeloesten Quellen entsprechend hoeheren Wert waehlen.
STICKFIGURE_ZOOM = 0.015

# Events (werden als vertikale Linien im Plot gezeichnet)
#   Format: (Spaltenname, Label, Farbe, Linienstil)
EVENTS_CONFIG = {
    "CMJ": [
        ("event_start_s",    "Start",     "#009E73", "-"),
        ("event_take_off_s", "Take-off",  "#0072B2", "-"),
        ("event_landing_s",  "Landing",   "#D55E00", "-"),
        ("event_end_jump_s", "End Jump",  "#888888", "--"),
    ],
    "DJ": [
        ("event_landing1_s", "Landing 1", "#D55E00", "-"),
        ("event_take_off_s", "Take-off",  "#0072B2", "-"),
        ("event_landing2_s", "Landing 2", "#D55E00", "--"),
        ("event_end_jump_s", "End Jump",  "#888888", ":"),
    ],
}
 
# Strichmaennchen-Konfiguration.
# Ein Eintrag ist ein Dict mit Feldern:
#   "filename":   Name der SVG-Datei in STICKFIGURE_DIR
#   "at_event":   (optional) Name der Event-Spalte, z.B. "event_start_s".
#                 Wenn gesetzt, wird das Strichmaennchen an dieses Event platziert.
#   "between":    (optional) Tuple (event_a, event_b, factor) mit factor in [0, 1].
#                 Das Strichmaennchen wird zwischen event_a und event_b platziert,
#                 factor=0.5 = mittig, factor=0.3 = eher bei a.
#   "before":     (optional) Event-Spalte + Offset in Sekunden, fuer Position
#                 VOR dem ersten Event (z.B. "dj_auf_box" vor "event_landing1_s").
STICKFIGURES_CMJ = [
    {"filename": "cmj_start.png",   "at_event": "event_start_s"},
    {"filename": "cmj_tief.png",    "between": ("event_start_s",    "event_take_off_s", 0.5)},
    {"filename": "cmj_takeoff.png", "at_event": "event_take_off_s"},
    {"filename": "cmj_in_luft.png", "between": ("event_take_off_s", "event_landing_s",  0.5)},
    {"filename": "cmj_landing.png", "at_event": "event_landing_s"},
    {"filename": "cmj_endjump.png", "at_event": "event_end_jump_s"},
]
 
STICKFIGURES_DJ = [
    {"filename": "dj_auf_box.png",     "before": ("event_landing1_s", 0.05)},
    {"filename": "dj_landing1.png",    "at_event": "event_landing1_s"},
    {"filename": "dj_exzentrisch.png", "between": ("event_landing1_s", "event_take_off_s", 0.5)},
    {"filename": "dj_takeoff.png",     "at_event": "event_take_off_s"},
    {"filename": "dj_flug.png",        "between": ("event_take_off_s", "event_landing2_s", 0.5)},
    {"filename": "dj_landing2.png",    "at_event": "event_landing2_s"},
    {"filename": "dj_endjump.png",     "at_event": "event_end_jump_s"},
]
 
STICKFIGURES_CONFIG = {
    "CMJ": STICKFIGURES_CMJ,
    "DJ":  STICKFIGURES_DJ,
}
 
# Fuer Squat: (relative Position im Trial 0..1, Dateiname)
SQUAT_STICKFIGURES = [
    (0.00, "sq_start.png"),
    (0.50, "sq_bottom.png"),
    (1.00, "sq_end.png"),
]
 
# Farben
COLOR_GRF   = "#1f4e79"
COLOR_KNEE  = "#c44e52"
 
 
# =============================================================================
# DATEN LADEN
# =============================================================================
 
def load_kin(subject, phase, exercise, side, trial):
    path = (
        PREPROCESSED_DIR / subject / phase / exercise / side
        / f"{trial}_kin.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    return pd.read_csv(path, low_memory=False)
 
 
def get_vgrf_in_bw(df, bilateral=True):
    """
    vGRF in BW (Body Weight) – Rohwerte direkt aus der _kin.csv.
    Standwert liegt bei ~1.0 BW, Sprung-Peaks bei ~2-5 BW.
    """
    right = df["Right GRF_Z"] if "Right GRF_Z" in df.columns else None
    left  = df["Left GRF_Z"]  if "Left GRF_Z"  in df.columns else None
 
    if bilateral and left is not None and right is not None:
        left_num  = pd.to_numeric(left,  errors="coerce")
        right_num = pd.to_numeric(right, errors="coerce")
        mask_both_nan = left_num.isna() & right_num.isna()
        summed = left_num.fillna(0) + right_num.fillna(0)
        summed[mask_both_nan] = np.nan
        return summed
    elif right is not None:
        return pd.to_numeric(right, errors="coerce")
    elif left is not None:
        return pd.to_numeric(left, errors="coerce")
    else:
        raise ValueError("Keine GRF_Z-Spalte gefunden")
 
 
def get_knee_angle_right(df):
    if "Right Knee Angles" in df.columns:
        return pd.to_numeric(df["Right Knee Angles"], errors="coerce")
    raise ValueError("'Right Knee Angles' nicht gefunden")
 
 
def get_event_times(df, exercise):
    """Gibt Dict mit {column_name: time_value} aus EVENTS_CONFIG zurueck."""
    events = {}
    for col, *_rest in EVENTS_CONFIG.get(exercise, []):
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").iloc[0]
            if not np.isnan(val):
                events[col] = val
    return events
 
 
def get_event_lines(df, exercise):
    """Liste (t, label, color, linestyle) fuer die vertikalen Event-Linien."""
    lines = []
    for col, label, color, ls in EVENTS_CONFIG.get(exercise, []):
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").iloc[0]
            if not np.isnan(val):
                lines.append((val, label, color, ls))
    return lines
 
 
def clean_timeseries(time, values):
    values = pd.to_numeric(values, errors="coerce")
    mask = values.notna()
    return time[mask].values, values[mask].values
 
 
# =============================================================================
# STRICHMAENNCHEN
# =============================================================================
 
def _load_stickfigure(filename):
    """Laedt ein Strichmaennchen (PNG)."""
    path = STICKFIGURE_DIR / filename
    if not path.exists():
        print(f"    [WARNUNG] Stichmaennchen fehlt: {path.name}")
        return None
    return mpimg.imread(path)
 
 
def _resolve_stickfigure_x(sf_cfg, event_times):
    """
    Berechnet die x-Position eines Strichmaennchens in absoluter Zeit.
 
    sf_cfg       : Dict aus STICKFIGURES_CMJ / STICKFIGURES_DJ
    event_times  : Dict {event_column: time_value}
 
    Rueckgabe: x-Position in Sekunden oder None (wenn Event fehlt).
    """
    if "at_event" in sf_cfg:
        ev = sf_cfg["at_event"]
        return event_times.get(ev)
 
    if "between" in sf_cfg:
        ev_a, ev_b, factor = sf_cfg["between"]
        t_a = event_times.get(ev_a)
        t_b = event_times.get(ev_b)
        if t_a is None or t_b is None:
            return None
        return t_a + (t_b - t_a) * factor
 
    if "before" in sf_cfg:
        ev, offset = sf_cfg["before"]
        t = event_times.get(ev)
        if t is None:
            return None
        return t - offset
 
    return None
 
 
def _add_stickfigures(ax, positions_files):
    """
    Platziert Strichmaennchen oberhalb des Plot-Bereichs.
 
    positions_files : Liste von (x_data, filename)
    """
    for x_data, filename in positions_files:
        img = _load_stickfigure(filename)
        if img is None:
            continue
        # resample=True + hermite = saubere Verkleinerung (kein Pixelraster)
        imagebox = OffsetImage(img, zoom=STICKFIGURE_ZOOM,
                               resample=True, interpolation="hermite")
        ab = AnnotationBbox(
            imagebox,
            xy=(x_data, 1.05),
            xycoords=("data", "axes fraction"),
            frameon=False,
            box_alignment=(0.5, 0.0),
        )
        ax.add_artist(ab)
 
 
# =============================================================================
# PLOT-FUNKTIONEN
# =============================================================================
 
def plot_jump(df, exercise, out_path):
    """Plot fuer CMJ / DJ: vGRF mit Events + Strichmaennchen-Reihe oben."""
    time = df["time_s"]
    vgrf = get_vgrf_in_bw(df, bilateral=True)
    t, v = clean_timeseries(time, vgrf)
 
    event_times = get_event_times(df, exercise)
    event_lines = get_event_lines(df, exercise)
 
    # Zeit auf 0 normalisieren (relativ zum ersten Event)
    t0 = event_lines[0][0] if event_lines else t[0]
    t_rel = t - t0
 
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(t_rel, v, color=COLOR_GRF, linewidth=1.2,
            label="vGRF (Gesamt)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.9,
               alpha=0.7, label="1 BW")
    ax.set_ylabel("vGRF in BW")
    ax.set_xlabel("Zeit in s")
    ax.grid(alpha=0.3)
 
    y_lo, y_hi = ax.get_ylim()
 
    # Event-Linien + Labels
    for t_evt, label, color, ls in event_lines:
        t_ev_rel = t_evt - t0
        ax.axvline(t_ev_rel, color=color, linestyle=ls, linewidth=1.2,
                   alpha=0.85)
        ax.text(t_ev_rel, y_hi * 0.97, label,
                rotation=90, va="top", ha="right",
                fontsize=8, color=color,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.75, pad=1.5))
 
    # Strichmaennchen vorbereiten: x-Position aus Konfiguration berechnen
    sf_cfgs = STICKFIGURES_CONFIG.get(exercise, [])
    positions_files = []
    for sf in sf_cfgs:
        x_abs = _resolve_stickfigure_x(sf, event_times)
        if x_abs is None:
            print(f"    [WARNUNG] Position fuer {sf['filename']} "
                  f"konnte nicht bestimmt werden")
            continue
        x_rel = x_abs - t0
        positions_files.append((x_rel, sf["filename"]))
 
    _add_stickfigures(ax, positions_files)
 
    # Platz oben fuer Strichmaennchen (weniger als vorher = naeher am Plot)
    fig.subplots_adjust(top=0.80)
 
    ax.legend(loc="center right", fontsize=8)
    #fig.suptitle(f"vGRF und Events beim {exercise}", fontsize=11, y=1.02)
    fig.savefig(out_path, format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [OK] {out_path.name}")
 
 
def plot_squat(df, bilateral, out_path, title):
    """Plot fuer Squat: vGRF + Kniewinkel rechts + 3 Strichmaennchen oben."""
    time = df["time_s"]
    vgrf = get_vgrf_in_bw(df, bilateral=bilateral)
    knee = get_knee_angle_right(df)
 
    t_v, v = clean_timeseries(time, vgrf)
    t_k, k = clean_timeseries(time, knee)
 
    t0 = min(t_v[0], t_k[0])
    t_v = t_v - t0
    t_k = t_k - t0
    t_max = max(t_v[-1], t_k[-1])
 
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
 
    vgrf_label = "vGRF (Gesamt)" if bilateral else "vGRF (rechts)"
    axes[0].plot(t_v, v, color=COLOR_GRF, linewidth=1.2, label=vgrf_label)
    axes[0].axhline(1.0, color="gray", linestyle=":", linewidth=0.9,
                    alpha=0.7, label="1 BW")
    axes[0].set_ylabel("vGRF in BW")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)
 
    axes[1].plot(t_k, k, color=COLOR_KNEE, linewidth=1.2,
                 label="Kniewinkel rechts")
    axes[1].set_ylabel("Kniewinkel in °")
    axes[1].set_xlabel("Zeit in s")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)
 
    axes[0].text(0.02, 0.95,
                 "", #Keine Event-Markierungen –\ngesamter Trial wird analysiert
                 transform=axes[0].transAxes, fontsize=8, va="top",
                 bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))
 
    # Strichmaennchen (relative Positionen)
    positions_files = [
        (rel * t_max, filename)
        for rel, filename in SQUAT_STICKFIGURES
    ]
    _add_stickfigures(axes[0], positions_files)
 
    fig.subplots_adjust(top=0.85, hspace=0.15)
    #fig.suptitle(title, fontsize=11, y=1.02)
    fig.savefig(out_path, format="svg", bbox_inches="tight", dpi=300)
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
                       title="") #vGRF, Kniewinkel und Events beim Squat
        elif key == "SQ_R":
            plot_squat(df, bilateral=False, out_path=out_path,
                       title="") #vGRF und Events beim Squat (unilateral rechts)
 
    print("\nFertig.")
 
 
if __name__ == "__main__":
    main()