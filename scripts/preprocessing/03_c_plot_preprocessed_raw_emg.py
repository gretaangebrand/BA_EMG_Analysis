"""
03_plot_raw_emg.py
==================
Plottet die Roh-EMG-Daten pro Subject, Übungstyp und Seite.

Layout pro Plot-Datei (wie von Betreuerin vorgegeben):
  - 5 Zeilen  = 5 Muskeln (Vastus Lateralis, Biceps Femoris,
                            Semitendinosus, Gluteus Medius, Gastrocnemius medial)
  - 3 Spalten = 3 Zyklusphasen (01_PER | 02_OVU | 03_LUT)
  -> 15 Subplots pro Datei

Pro Subject + Übungstyp + Seite werden zwei Dateien erzeugt:
  S01_CMJ_BILATERAL_L_leg.pdf   (linkes Bein)
  S01_CMJ_BILATERAL_R_leg.pdf   (rechtes Bein)

Wenn mehrere Trials pro Phase vorhanden sind (z.B. CMJ_01, CMJ_02, CMJ_03),
werden alle Trials übereinandergelegt — jeder Trial als eigene Linie, leicht
transparent — damit Konsistenz und Ausreißer sichtbar werden.

Vertikale Linien markieren die Events (start, take_off, landing, end_jump etc.)
aus den event_*_s Spalten.

Hinweis zur unterschiedlichen Aufnahmefrequenz:
  EMG:       2000 / 2100 Hz  -> dicht, jede Zeile hat Werte
  Kinematik: 200 Hz          -> nur jede ~10. Zeile hat Werte, Rest NaN
  Für EMG-Plots irrelevant: es werden nur die EMG-Spalten (L_*/R_*) geplottet.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ============================================================
# PFADE – anpassen!
# ============================================================
DATA_DIR   = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\preprocessed_emg_data")
OUTPUT_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\outputs\figures\plots_raw_emg")

# ============================================================
# EINSTELLUNGEN
# ============================================================
PHASES       = ["01_PER", "02_OVU", "03_LUT"]
PHASE_LABELS = {
    "01_PER": "Follikelphase (PER)",
    "02_OVU": "Ovulationsphase (OVU)",
    "03_LUT": "Lutealphase (LUT)",
}

# Muskelreihenfolge im Plot – Präfix (L_ / R_) wird automatisch ergänzt
MUSCLE_ORDER = [
    "Vastus Lateralis",
    "Biceps Femoris",
    "Semitendinosus",
    "Gluteus Medius",
    "Gastrocnemius medial",
]

# Farbe und Linienstil pro Event-Spaltenname
EVENT_STYLE = {
    "event_start_s":    {"color": "#2196F3", "label": "Start",      "ls": "--"},
    "event_take_off_s": {"color": "#FF9800", "label": "Take-off",   "ls": "-"},
    "event_landing_s":  {"color": "#F44336", "label": "Landing",    "ls": "-"},
    "event_landing1_s": {"color": "#F44336", "label": "Landing 1",  "ls": "-"},
    "event_landing2_s": {"color": "#9C27B0", "label": "Landing 2",  "ls": "-"},
    "event_end_jump_s": {"color": "#4CAF50", "label": "End jump",   "ls": "--"},
}

TRIAL_COLORS = plt.cm.tab10.colors
TRIAL_ALPHA  = 0.60
TRIAL_LW     = 0.7
FIG_WIDTH    = 18
FIG_HEIGHT   = 14


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def get_emg_columns(df: pd.DataFrame, side_prefix: str) -> list[str]:
    """
    Gibt EMG-Spaltennamen für ein Bein zurück, in Reihenfolge von MUSCLE_ORDER.
    side_prefix: 'L' oder 'R'
    """
    result = []
    for muscle in MUSCLE_ORDER:
        col = f"{side_prefix}_{muscle}"
        if col in df.columns:
            result.append(col)
    # Alle weiteren Spalten mit diesem Präfix die nicht in MUSCLE_ORDER stehen
    extras = [c for c in df.columns
              if c.startswith(f"{side_prefix}_") and c not in result]
    return result + extras


def get_event_times(df: pd.DataFrame) -> dict[str, float]:
    """
    Liest alle Event-Zeitpunkte aus event_*_s Spalten (Zeile 0).
    """
    events = {}
    for col in df.columns:
        if col.startswith("event_") and col.endswith("_s"):
            val = pd.to_numeric(df[col].iloc[0], errors="coerce")
            if not np.isnan(val):
                events[col] = float(val)
    return events


def load_trials(phase_dir: Path, exercise: str, side: str) -> list[pd.DataFrame]:
    """
    Laedt alle Trial-CSVs für phase/exercise/side.
    NaN-Zeilen (Auffüll-Zeilen von Vicon) werden entfernt.
    """
    trial_dir = phase_dir / exercise / side
    if not trial_dir.exists():
        return []

    dfs = []
    for csv in sorted(trial_dir.glob("*.csv")):
        try:
            df      = pd.read_csv(csv, low_memory=False)
            emg_all = [c for c in df.columns if c.startswith("L_") or c.startswith("R_")]
            if emg_all:
                # Nur Zeilen behalten wo mindestens eine EMG-Spalte einen Wert hat
                df = df.dropna(subset=emg_all, how="all").reset_index(drop=True)
            dfs.append(df)
        except Exception as e:
            print(f"    [WARNUNG] {csv.name}: {e}")
    return dfs


# ============================================================
# PLOT-FUNKTION
# ============================================================

def create_plot(
    subject_id:       str,
    exercise:         str,
    side:             str,    # Ordnername: BILATERAL / LEFT / RIGHT
    side_prefix:      str,    # 'L' oder 'R'
    trials_per_phase: dict[str, list[pd.DataFrame]],
    out_path:         Path,
):
    """
    Erstellt eine Abbildung: 5 Muskeln x 3 Phasen = 15 Subplots.
    """
    side_label = "Linkes Bein" if side_prefix == "L" else "Rechtes Bein"
    n_muscles  = len(MUSCLE_ORDER)
    n_phases   = len(PHASES)

    fig, axes = plt.subplots(
        n_muscles, n_phases,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        sharex=False,   # Trials können unterschiedlich lang sein
        sharey=False,   # Amplituden können zwischen Muskeln stark variieren
    )

    # Für die Legende alle Events sammeln die in diesen Daten vorkommen
    all_event_names: set[str] = set()

    for col_idx, phase in enumerate(PHASES):
        trials = trials_per_phase.get(phase, [])

        for row_idx, muscle in enumerate(MUSCLE_ORDER):
            ax       = axes[row_idx, col_idx]
            col_name = f"{side_prefix}_{muscle}"

            # Spalten-Titel (Phase) – nur erste Zeile
            if row_idx == 0:
                ax.set_title(
                    PHASE_LABELS.get(phase, phase),
                    fontsize=9.5, fontweight="bold", pad=5
                )

            # Zeilen-Label (Muskel) – nur erste Spalte
            if col_idx == 0:
                ax.set_ylabel(muscle, fontsize=8, labelpad=3)

            # Kein Daten-Fall
            if not trials:
                ax.text(0.5, 0.5,
                        f"Keine preprocessed EMG-Daten\n"
                        f"für {exercise} / {side}\n"
                        f"in {PHASE_LABELS.get(phase, phase)}",
                        ha="center", va="center",
                        transform=ax.transAxes,
                        fontsize=7, color="#999999",
                        linespacing=1.4)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Alle Trials zeichnen
            trial_events: dict[str, float] = {}
            for t_idx, df in enumerate(trials):
                if col_name not in df.columns:
                    continue

                t_vals  = df["time_s"].values
                sig     = pd.to_numeric(df[col_name], errors="coerce").values
                color   = TRIAL_COLORS[t_idx % len(TRIAL_COLORS)]

                ax.plot(
                    t_vals, sig,
                    color=color,
                    alpha=TRIAL_ALPHA,
                    linewidth=TRIAL_LW,
                    rasterized=True,   # schnelleres Rendern bei vielen Punkten
                    label=f"Trial {t_idx + 1}",
                )

                # Events nur aus erstem Trial lesen
                if t_idx == 0:
                    trial_events = get_event_times(df)
                    all_event_names.update(trial_events.keys())

            # # Vertikale Linien vorerst ausblenden
            """for evt_col, t_val in trial_events.items():
                style = EVENT_STYLE.get(
                    evt_col,
                    {"color": "gray", "label": evt_col, "ls": ":"}
                )
                ax.axvline(
                    x=t_val,
                    color=style["color"],
                    linestyle=style["ls"],
                    linewidth=1.3,
                    alpha=0.9,
                    zorder=5,
                )"""

            # Nulllinie
            ax.axhline(y=0, color="black", linewidth=0.4, alpha=0.35, zorder=1)

            # Achsen-Formatierung
            ax.tick_params(axis="both", labelsize=6)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            ax.set_xlabel("Zeit (s)", fontsize=6, labelpad=1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # ---- Legende ----
    max_trials = max((len(t) for t in trials_per_phase.values()), default=0)

    trial_handles = [
        Line2D([0], [0],
               color=TRIAL_COLORS[i % len(TRIAL_COLORS)],
               linewidth=1.8, alpha=0.85,
               label=f"Trial {i + 1}")
        for i in range(min(max_trials, 9))
    ]

    # Vertikale Linien vorerst ausblenden
    """event_handles = [
        Line2D([0], [0],
               color=EVENT_STYLE.get(e, {"color": "gray"})["color"],
               linewidth=1.8,
               linestyle=EVENT_STYLE.get(e, {"ls": ":"})["ls"],
               label=EVENT_STYLE.get(e, {"label": e})["label"])
        for e in sorted(all_event_names)
        if e in EVENT_STYLE
    ]"""

    all_handles = trial_handles #+ event_handles # Vertikale Linien vorerst ausblenden
    if all_handles:
        fig.legend(
            handles=all_handles,
            loc="lower center",
            ncol=min(len(all_handles), 9),
            fontsize=8,
            frameon=True,
            framealpha=0.9,
            bbox_to_anchor=(0.5, 0.005),
        )

    # Haupt-Titel und y-Achsen-Label
    fig.suptitle(
        f"{subject_id}  |  {exercise}  |  {side}  |  {side_label}  –  Roh-EMG",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.text(
        0.005, 0.5, "EMG-Amplitude (µV)",
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
    print("03_plot_raw_emg.py  –  Roh-EMG Plots erstellen")
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

    # Fehlende Daten sammeln fuer Uebersichtstabelle
    missing_data_records: list[dict[str, str]] = []
    # Trial-Anzahl pro Kombination sammeln (fuer <3 Trials Warnung)
    trial_count_records: list[dict] = []

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        print(f"\n--- {subject_id} ---")

        # Alle einzigartigen (exercise, side)-Kombis ermitteln
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

        for exercise, side in sorted(combos):
            print(f"  {exercise}/{side}")

            # Trials pro Phase laden
            trials_per_phase: dict[str, list[pd.DataFrame]] = {}
            for phase in PHASES:
                phase_dir = subject_dir / phase
                trials_per_phase[phase] = load_trials(phase_dir, exercise, side)

            # Fehlende Phasen und Trial-Anzahl erfassen
            for phase in PHASES:
                n_trials = len(trials_per_phase[phase])
                trial_count_records.append({
                    "Subject":    subject_id,
                    "Übung":      exercise,
                    "Seite":      side,
                    "Phase":      PHASE_LABELS.get(phase, phase),
                    "Anzahl Trials": n_trials,
                })
                if n_trials == 0:
                    missing_data_records.append({
                        "Subject":  subject_id,
                        "Übung":    exercise,
                        "Seite":    side,
                        "Phase":    PHASE_LABELS.get(phase, phase),
                    })

            # Output-Ordner
            out_subdir = OUTPUT_DIR / subject_id / exercise
            out_subdir.mkdir(parents=True, exist_ok=True)

            # Linkes und rechtes Bein je als eigene Datei
            for side_prefix in ("L", "R"):
                out_path = out_subdir / f"{subject_id}_{exercise}_{side}_{side_prefix}_leg.pdf"
                create_plot(
                    subject_id       = subject_id,
                    exercise         = exercise,
                    side             = side,
                    side_prefix      = side_prefix,
                    trials_per_phase = trials_per_phase,
                    out_path         = out_path,
                )
                total_files += 1

    print(f"\n{'='*62}")
    print(f"FERTIG")
    print(f"  Erzeugte Plot-Dateien : {total_files}")
    print(f"  Gespeichert in        : {OUTPUT_DIR}")

    # Uebersichtstabelle fehlender Daten erzeugen
    if missing_data_records or trial_count_records:
        n_incomplete = sum(
            1 for r in trial_count_records
            if 0 < r["Anzahl Trials"] < 3
        )
        print(f"  Fehlende Phasen       : {len(missing_data_records)}")
        print(f"  Phasen mit < 3 Trials : {n_incomplete}")
        table_path = OUTPUT_DIR / "03_preprocessed_plots_emg_daten_uebersicht.pdf"
        _export_missing_data_table(
            missing_data_records, trial_count_records, table_path
        )
    else:
        print("  Fehlende Phasen       : keine")

    print(f"{'='*62}")


# ============================================================
# TABELLE: FEHLENDE DATEN
# ============================================================

def _export_missing_data_table(
    records: list[dict[str, str]],
    trial_counts: list[dict],
    out_path: Path,
) -> None:
    """
    Exportiert eine Uebersichtstabelle der fehlenden preprocessed
    EMG-Daten als Vektorgrafik (PDF).

    Drei Teile:
      1. Häufigkeitstabelle – Anzahl fehlender Phasen pro Übung × Phase
      2. Unvollständige Trials – Kombinationen mit weniger als 3 Trials
      3. Detailtabelle – jede fehlende Kombination einzeln
    """
    df_missing = pd.DataFrame(records)
    df_counts  = pd.DataFrame(trial_counts)

    # --- Häufigkeitstabelle: Übung × Phase (nur fehlende) ---
    if not df_missing.empty:
        freq = (
            df_missing.groupby(["Übung", "Phase"])
            .size()
            .reset_index(name="Anzahl fehlend")
            .sort_values(["Übung", "Phase"])
        )
    else:
        freq = pd.DataFrame(columns=["Übung", "Phase", "Anzahl fehlend"])

    # --- Unvollständige Trials: 0 < n < 3 ---
    df_incomplete = (
        df_counts[
            (df_counts["Anzahl Trials"] > 0)
            & (df_counts["Anzahl Trials"] < 3)
        ]
        .sort_values(["Subject", "Übung", "Seite", "Phase"])
        .reset_index(drop=True)
    )

    # --- Anzahl Tabellen bestimmen ---
    tables = []
    if not freq.empty:
        tables.append(("Fehlende preprocessed EMG-Daten – Zusammenfassung",
                        freq, 9))
    if not df_incomplete.empty:
        tables.append(("Phasen mit weniger als 3 Trials",
                        df_incomplete, 8))
    if not df_missing.empty:
        tables.append(("Detailauflistung aller fehlenden Kombinationen",
                        df_missing, 8))

    if not tables:
        return

    n_tables = len(tables)
    total_rows = sum(len(t[1]) for t in tables)
    fig_height = max(4, 1.5 * n_tables + 0.35 * total_rows)
    height_ratios = [max(1, 1 + len(t[1])) for t in tables]

    fig, axes = plt.subplots(
        n_tables, 1,
        figsize=(12, fig_height),
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
    )

    for idx, (title, data_df, fontsize) in enumerate(tables):
        ax = axes[idx, 0]
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=10)

        tbl = ax.table(
            cellText=data_df.values,
            colLabels=data_df.columns,
            cellLoc="center",
            loc="upper left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)
        tbl.scale(1.0, 1.4)

        # Header
        for col_idx in range(len(data_df.columns)):
            tbl[0, col_idx].set_facecolor("#767676")
            tbl[0, col_idx].set_text_props(color="white", fontweight="bold")

        # Zebrastreifen
        for row_idx in range(1, len(data_df) + 1):
            bg = "#F2F2F2" if row_idx % 2 == 0 else "white"
            for col_idx in range(len(data_df.columns)):
                tbl[row_idx, col_idx].set_facecolor(bg)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Übersichtstabelle     : {out_path.name}")


if __name__ == "__main__":
    main()