"""
05_select_best_trials.py
========================
Bestimmt den besten Trial pro Subject, Phase und Übung.

Kriterien:
  - CMJ / DJ:  Höchste Sprunghöhe (Jumpheight aus Scalar-Metadaten)
  - SQ RIGHT:  Größter maximaler Kniewinkel (tiefster Squat) aus Right Knee Angles in _kin.csv.

Liest die preprocessed *_emg.csv und *_kin.csv Dateien.
Gibt eine Übersichtstabelle als PDF und CSV aus (individuell + Gruppenmittelwert).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import sys
sys.path.insert(0, str(Path(r'C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG')))

from scripts.utils.config import TRIAL_SELECTION_CONFIGS as EXERCISE_CONFIGS

# ============================================================
# PFADE
# ============================================================
DATA_DIR    = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\03_preprocessed_emg_data")
OUTPUT_DIR  = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\05_best_trials_group_and_individual")
FIGURES_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\outputs\figures\best_trials_group_individual")


# ============================================================
# EINSTELLUNGEN
# ============================================================
PHASES = ["01_PER", "02_OVU", "03_LUT"]
PHASE_LABELS = {
    "01_PER": "PER",
    "02_OVU": "OVU",
    "03_LUT": "LUT",
}

# Ausschluss von Trial in S07 wegen implausibiler Werte.
# Format: (Subject, Übungs-Label, Phase-Label, Trial-Name-Teilstring)
# Begründung wird als Kommentar dokumentiert.
EXCLUDED_TRIALS = [
    # S07, CMJ bilateral, PER, Trial 02: Jumpheight 0.768 m – unrealistisch hoch, Messfehler
    ("S07", "CMJ bilateral", "PER", "CMJ_02"),

    # S06, CMJ bilateral, OVU, Trial 01: unsaubere Landung auf Kraftmessplatte
    ("S06", "CMJ bilateral", "OVU", "CMJ_01"),

    # S09, CMJ bilateral: alle Phasen – Kraftmessplattendaten nicht verwertbar
    ("S09", "CMJ bilateral", "PER", "CMJ_01"),
    ("S09", "CMJ bilateral", "PER", "CMJ_02"),
    ("S09", "CMJ bilateral", "PER", "CMJ_03"),
    ("S09", "CMJ bilateral", "OVU", "CMJ_01"),
    ("S09", "CMJ bilateral", "OVU", "CMJ_02"),
    ("S09", "CMJ bilateral", "OVU", "CMJ_03"),
    ("S09", "CMJ bilateral", "LUT", "CMJ_01"),
    ("S09", "CMJ bilateral", "LUT", "CMJ_02"),
    ("S09", "CMJ bilateral", "LUT", "CMJ_03"),

    # S02, CMJ einbeinig R, LUT, Trials 01+02: fehlerhafte Aufnahme rechtes Bein
    ("S02", "CMJ einbeinig R", "LUT", "CMJ_01"),
    ("S02", "CMJ einbeinig R", "LUT", "CMJ_02"),
]


def _is_excluded(subject: str, label: str, phase: str, trial_name: str) -> bool:
    """Prüft, ob ein Trial in der Ausschlussliste steht."""
    for ex_subj, ex_label, ex_phase, ex_trial in EXCLUDED_TRIALS:
        if (subject == ex_subj and label == ex_label
                and phase == ex_phase and ex_trial in trial_name):
            return True
    return False


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def get_jumpheight(emg_path: Path) -> float:
    """
    Liest die Sprunghöhe (Jumpheight) aus der ersten Zeile der _emg.csv.
    Die Scalars stehen als Metadaten in Zeile 0.
    """
    df = pd.read_csv(emg_path, nrows=1, low_memory=False)
    if "Jumpheight" in df.columns:
        val = pd.to_numeric(df["Jumpheight"].iloc[0], errors="coerce")
        if pd.notna(val):
            return float(val)
    return np.nan


def get_max_knee_angle(kin_path: Path) -> float:
    """
    Liest den maximalen rechten Kniewinkel aus der _kin.csv (Zeitreihe).
    Sucht die Spalte 'Right Knee Angles' (X-Achse = Hauptkomponente).
    """
    if not kin_path.exists():
        return np.nan

    df = pd.read_csv(kin_path, low_memory=False)

    # Spalte finden: 'Right Knee Angles' (ohne _Y / _Z Suffix), da das die Extension/Flexion Spalte ist
    kin_col = None
    for c in df.columns:
        if c == "Right Knee Angles":
            kin_col = c
            break

    if kin_col is None:
        return np.nan

    vals = pd.to_numeric(df[kin_col], errors="coerce").dropna()
    if vals.empty:
        return np.nan

    return float(vals.max())


def select_best_sq(trials: list[dict]) -> dict | None:
    """
    Wählt den besten Squat-Trial: größter maximaler Kniewinkel (= tiefster Squat).
    """
    if not trials:
        return None

    valid = [t for t in trials if pd.notna(t["value"])]
    if not valid:
        return None

    return max(valid, key=lambda t: t["value"])


def select_best_jump(trials: list[dict]) -> dict | None:
    """Wählt den Trial mit der höchsten Sprunghöhe."""
    if not trials:
        return None

    valid = [t for t in trials if pd.notna(t["value"])]
    if not valid:
        return None

    return max(valid, key=lambda t: t["value"])


# ============================================================
# HAUPTFUNKTION
# ============================================================

def main():
    print("=" * 70)
    print("05_select_best_trials.py  –  Beste Trials auswählen")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        print(f"[FEHLER] DATA_DIR nicht gefunden:\n  {DATA_DIR}")
        return

    subjects = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    if not subjects:
        print("[FEHLER] Keine Subject-Ordner gefunden.")
        return

    # ── Daten sammeln ──────────────────────────────────────────────────────
    all_records     = []   # Alle Trials (für Detailtabelle)
    best_records    = []   # Nur die besten (für Ergebnistabelle)

    for subject in subjects:
        for cfg in EXERCISE_CONFIGS:
            ex    = cfg["exercise"]
            side  = cfg["side"]
            method = cfg["method"]
            label  = cfg["label"]

            for phase in PHASES:
                trial_dir = DATA_DIR / subject / phase / ex / side
                if not trial_dir.exists():
                    continue

                emg_files = sorted(trial_dir.glob("*_emg.csv"))
                if not emg_files:
                    continue

                trials_info = []
                for emg_path in emg_files:
                    trial_name = emg_path.stem.replace("_emg", "")

                    if method == "jumpheight":
                        value = get_jumpheight(emg_path)
                        unit  = "m"
                    elif method == "knee_angle":
                        kin_path = emg_path.parent / emg_path.name.replace("_emg", "_kin")
                        value = get_max_knee_angle(kin_path)
                        unit  = "°"
                    else:
                        value = np.nan
                        unit  = "?"

                    info = {
                        "Subject":  subject,
                        "Übung":    label,
                        "Phase":    PHASE_LABELS.get(phase, phase),
                        "Trial":    trial_name,
                        "Wert":     value,
                        "Einheit":  unit,
                        "Methode":  method,
                    }
                    all_records.append(info)

                    if _is_excluded(subject, label, PHASE_LABELS.get(phase, phase), trial_name):
                        print(f"  [AUSSCHLUSS] {subject} | {label} | {trial_name} "
                              f"| Wert={value} {unit} – aus Selektion ausgeschlossen")
                        continue

                    trials_info.append(info)

                # Besten Trial auswählen
                if method == "jumpheight":
                    best = select_best_jump(
                        [{"trial": t["Trial"], "value": t["Wert"]} for t in trials_info]
                    )
                elif method == "knee_angle":
                    best = select_best_sq(
                        [{"trial": t["Trial"], "value": t["Wert"]} for t in trials_info]
                    )
                else:
                    best = None

                if best:
                    best_info = next(
                        t for t in trials_info if t["Trial"] == best["trial"]
                    )
                    best_records.append({
                        "Subject":    subject,
                        "Übung":      label,
                        "Phase":      PHASE_LABELS.get(phase, phase),
                        "Bester Trial": best["trial"],
                        "Wert":       best["value"],
                        "Einheit":    unit,
                    })

    # ── DataFrames ────────────────────────────────────────────────────────
    df_all  = pd.DataFrame(all_records)
    df_best = pd.DataFrame(best_records)

    if df_best.empty:
        print("[WARNUNG] Keine Trials gefunden.")
        return

    print(f"\n  Trials gesamt   : {len(df_all)}")
    print(f"  Beste Trials    : {len(df_best)}")

    # ── Detailtabelle als CSV speichern ────────────────────────────────────
    csv_all  = OUTPUT_DIR / "alle_trials_uebersicht.csv"
    csv_best = OUTPUT_DIR / "beste_trials.csv"
    df_all.to_csv(csv_all, index=False)
    df_best.to_csv(csv_best, index=False)
    print(f"\n  CSV (alle)      : {csv_all.name}")
    print(f"  CSV (beste)     : {csv_best.name}")

    # ── Gruppenmittelwerte ────────────────────────────────────────────────
    df_group = (
        df_best
        .groupby(["Übung", "Phase"])["Wert"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    df_group.columns = ["Übung", "Phase", "Mittelwert", "SD", "n"]
    csv_group = OUTPUT_DIR / "gruppenmittelwerte.csv"
    df_group.to_csv(csv_group, index=False)
    print(f"  CSV (Gruppen)   : {csv_group.name}")

    # ── Konsolenausgabe: Gruppenmittelwerte ────────────────────────────────
    print(f"\n{'='*70}")
    print("GRUPPENMITTELWERTE (beste Trials)")
    print(f"{'='*70}")
    for _, row in df_group.iterrows():
        unit = "m" if "CMJ" in row["Übung"] or "DJ" in row["Übung"] else "°"
        print(f"  {row['Übung']:22s} | {row['Phase']:4s} | "
              f"{row['Mittelwert']:7.3f} ± {row['SD']:6.3f} {unit}  (n={int(row['n'])})")

    # ── Konsolenausgabe: Individuelle beste Trials ─────────────────────────
    print(f"\n{'='*70}")
    print("INDIVIDUELLE BESTE TRIALS")
    print(f"{'='*70}")
    for _, row in df_best.iterrows():
        print(f"  {row['Subject']:5s} | {row['Übung']:22s} | {row['Phase']:4s} | "
              f"{row['Bester Trial']:15s} | {row['Wert']:7.3f} {row['Einheit']}")

    # ── PDF: Übersichtstabelle ─────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _create_overview_pdf(df_best, df_group, FIGURES_DIR / "beste_trials_uebersicht.pdf")

    # ── Plot: Balkendiagramm Gruppenmittelwerte ────────────────────────────
    _create_group_barplot(df_group, FIGURES_DIR / "gruppenmittelwerte_plot.svg")

    # ── Plot: Individuelle Werte pro Übung ─────────────────────────────────
    _create_individual_plot(df_best, FIGURES_DIR / "individuelle_beste_trials_plot.svg")

    # ── Plot: Verfügbarkeit bester Trials pro Übung × Phase ───────────────
    #_create_availability_plot(df_best, FIGURES_DIR / "verfuegbarkeit_beste_trials.svg")

    # ── Plot: Bestleistung pro Probandin – in welcher Phase? ──────────────
    _create_peak_phase_plot(df_best, FIGURES_DIR / "bestleistung_pro_probandin.svg")

    print(f"\n{'='*70}")
    print("FERTIG")
    print(f"  CSV-Ordner      : {OUTPUT_DIR}")
    print(f"  Figures-Ordner  : {FIGURES_DIR}")
    print(f"{'='*70}")


# ============================================================
# VISUALISIERUNGEN
# ============================================================

def _create_overview_pdf(df_best: pd.DataFrame, df_group: pd.DataFrame,
                         out_path: Path):
    """Übersichtstabelle mit individuellen besten Trials + Gruppenmittelwerten."""

    # Tabelle 1: Gruppenmittelwerte
    group_display = df_group.copy()
    group_display["Mittelwert ± SD"] = group_display.apply(
        lambda r: f"{r['Mittelwert']:.3f} ± {r['SD']:.3f}", axis=1
    )
    tab1 = group_display[["Übung", "Phase", "Mittelwert ± SD", "n"]]

    # Tabelle 2: Individuelle beste Trials
    best_display = df_best[["Subject", "Übung", "Phase", "Bester Trial", "Wert", "Einheit"]].copy()
    best_display["Wert"] = best_display.apply(
        lambda r: f"{r['Wert']:.3f} {r['Einheit']}", axis=1
    )
    tab2 = best_display[["Subject", "Übung", "Phase", "Bester Trial", "Wert"]]

    # Layout
    n1, n2 = len(tab1), len(tab2)
    fig_h = max(5, 1.5 + 0.35 * n1 + 0.35 * n2)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, fig_h),
        gridspec_kw={"height_ratios": [max(1, 1 + n1), max(1, 1 + n2)]},
    )

    for ax, title, data in [
        (ax1, "Gruppenmittelwerte (beste Trials pro Subject × Phase)", tab1),
        (ax2, "Individuelle beste Trials", tab2),
    ]:
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=10)

        tbl = ax.table(
            cellText=data.values,
            colLabels=data.columns,
            cellLoc="center",
            loc="upper left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.35)

        for col_idx in range(len(data.columns)):
            tbl[0, col_idx].set_facecolor("#4472C4")
            tbl[0, col_idx].set_text_props(color="white", fontweight="bold")

        for row_idx in range(1, len(data) + 1):
            bg = "#F2F2F2" if row_idx % 2 == 0 else "white"
            for col_idx in range(len(data.columns)):
                tbl[row_idx, col_idx].set_facecolor(bg)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PDF (Übersicht) : {out_path.name}")


def _create_group_barplot(df_group: pd.DataFrame, out_path: Path):
    """Balkendiagramm: Mittelwert ± SD pro Übung und Phase. Layout 2×2."""
    exercise_order = [
        "CMJ bilateral", "CMJ einbeinig R",
        "DJ bilateral",  "SQ einbeinig R",
    ]
    exercise_order = [e for e in exercise_order if e in df_group["Übung"].values]
    n_ex = len(exercise_order)

    if n_ex <= 2:
        n_rows, n_cols = 1, n_ex
    else:
        n_rows, n_cols = 2, 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows),
                             sharey=False)
    axes_flat = axes.flatten() if n_ex > 1 else [axes]

    phase_colors = {"PER": "#C44462", "OVU": "#0EB55F", "LUT": "#FAD758"}
    phase_order  = ["PER", "OVU", "LUT"]

    for idx, ex_name in enumerate(exercise_order):
        ax = axes_flat[idx]
        sub = df_group[df_group["Übung"] == ex_name]
        sub = sub.set_index("Phase").reindex(phase_order).reset_index()
        sub = sub.dropna(subset=["Mittelwert"])

        phases = sub["Phase"].values
        means  = sub["Mittelwert"].values
        sds    = sub["SD"].values

        bars = ax.bar(
            phases, means,
            yerr=sds, capsize=5,
            color=[phase_colors.get(p, "gray") for p in phases],
            edgecolor="white", linewidth=1.2,
            alpha=0.85, zorder=2,
        )

        for bar, m, s in zip(bars, means, sds):
            unit = "m" if "CMJ" in ex_name or "DJ" in ex_name else "°"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.002,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(ex_name, fontsize=12, fontweight="bold")
        unit = "Sprunghöhe [m]" if "CMJ" in ex_name or "DJ" in ex_name else "Max. Kniewinkel [°]"
        ax.set_ylabel(unit, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    for idx in range(n_ex, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Gruppenmittelwerte der besten Trials", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  SVG (Balken)    : {out_path.name}")


def _create_individual_plot(df_best: pd.DataFrame, out_path: Path):
    """Stripplot: individuelle Werte der besten Trials pro Übung × Phase.
    Layout 2×2: CMJs oben, DJ + SQ unten."""
    # Feste Reihenfolge: CMJs oben, DJ + SQ unten
    exercise_order = [
        "CMJ bilateral", "CMJ einbeinig R",
        "DJ bilateral",  "SQ einbeinig R",
    ]
    # Nur vorhandene Übungen behalten
    exercise_order = [e for e in exercise_order if e in df_best["Übung"].values]
    n_ex = len(exercise_order)

    if n_ex <= 2:
        n_rows, n_cols = 1, n_ex
    else:
        n_rows, n_cols = 2, 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows),
                             sharey=False)
    axes_flat = axes.flatten() if n_ex > 1 else [axes]

    phase_colors = {"PER": "#C44462", "OVU": "#0EB55F", "LUT": "#FAD758"}
    phase_order  = ["PER", "OVU", "LUT"]

    for idx, ex_name in enumerate(exercise_order):
        ax = axes_flat[idx]
        sub = df_best[df_best["Übung"] == ex_name]

        for p_idx, phase in enumerate(phase_order):
            phase_data = sub[sub["Phase"] == phase]
            if phase_data.empty:
                continue

            vals = phase_data["Wert"].values
            # Jittered x-Positionen
            x = np.full_like(vals, p_idx, dtype=float)
            x += np.random.uniform(-0.15, 0.15, size=len(vals))

            ax.scatter(x, vals, color=phase_colors.get(phase, "gray"),
                       s=45, alpha=0.75, edgecolors="white", linewidth=0.5,
                       zorder=3)

            # Mittelwert als horizontale Linie
            mean_val = vals.mean()
            ax.hlines(mean_val, p_idx - 0.3, p_idx + 0.3,
                      color=phase_colors.get(phase, "gray"),
                      linewidth=2.5, zorder=4)

        ax.set_xticks(range(len(phase_order)))
        ax.set_xticklabels(phase_order)
        ax.set_title(ex_name, fontsize=12, fontweight="bold")
        unit = "Sprunghöhe [m]" if "CMJ" in ex_name or "DJ" in ex_name else "Max. Kniewinkel [°]"
        ax.set_ylabel(unit, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    # Leere Subplots ausblenden (falls n_ex < 4)
    for idx in range(n_ex, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Individuelle beste Trials (pro Probandin × Phase)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  SVG (Stripplot) : {out_path.name}")


def _create_availability_plot(df_best: pd.DataFrame, out_path: Path):
    """
    Balkendiagramm: Anzahl Probandinnen mit bestem Trial pro Übung × Phase.
    Zeigt auf einen Blick, wo Daten vollständig sind und wo nicht.
    """
    phase_order  = ["PER", "OVU", "LUT"]
    phase_colors = {"PER": "#C44462", "OVU": "#0EB55F", "LUT": "#FAD758"}

    # Zähle unique Subjects pro Übung × Phase
    counts = (
        df_best
        .groupby(["Übung", "Phase"])["Subject"]
        .nunique()
        .reset_index(name="n_subjects")
    )

    # Gesamtzahl Subjects (für Referenzlinie)
    n_total = df_best["Subject"].nunique()

    exercises = sorted(df_best["Übung"].unique())
    n_ex = len(exercises)

    bar_width = 0.25
    x = np.arange(n_ex)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, phase in enumerate(phase_order):
        vals = []
        for ex in exercises:
            row = counts[(counts["Übung"] == ex) & (counts["Phase"] == phase)]
            vals.append(int(row["n_subjects"].values[0]) if len(row) > 0 else 0)

        bars = ax.bar(
            x + i * bar_width, vals, bar_width,
            color=phase_colors[phase], edgecolor="white",
            linewidth=1.0, alpha=0.85, label=phase, zorder=2,
        )

        # Anzahl über Balken
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Referenzlinie: Gesamtzahl Subjects
    ax.axhline(y=n_total, color="black", linewidth=1.0, linestyle="--",
               alpha=0.5, zorder=1, label=f"Gesamt ({n_total} Probandinnen)")

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(exercises, fontsize=10)
    ax.set_ylabel("Anzahl Probandinnen mit bestem Trial", fontsize=11)
    ax.set_ylim(0, n_total + 1.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Verfügbarkeit der besten Trials pro Übung und Zyklusphase",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  SVG (Verfügb.)  : {out_path.name}")


def _create_peak_phase_plot(df_best: pd.DataFrame, out_path: Path):
    """
    Pro Übung: Jede Probandin als verbundene Linie über die 3 Phasen.
    Die Phase mit dem absolut besten Wert wird farbig hervorgehoben (großer Marker).
    Layout 2×2: CMJs oben, DJ + SQ unten. Häufigkeitstabelle unter jedem Subplot.
    """
    phase_order  = ["PER", "OVU", "LUT"]
    phase_colors = {"PER": "#C44462", "OVU": "#0EB55F", "LUT": "#FAD758"}
    phase_x      = {p: i for i, p in enumerate(phase_order)}

    exercise_order = [
        "CMJ bilateral", "CMJ einbeinig R",
        "DJ bilateral",  "SQ einbeinig R",
    ]
    exercise_order = [e for e in exercise_order if e in df_best["Übung"].values]
    n_ex = len(exercise_order)

    if n_ex <= 2:
        n_rows, n_cols = 1, n_ex
    else:
        n_rows, n_cols = 2, 2

    # Pro Übung: Plot-Zeile + Tabellen-Zeile → doppelte Anzahl Zeilen
    fig, axes = plt.subplots(
        n_rows * 2, n_cols, figsize=(5.5 * n_cols, 6 * n_rows),
        gridspec_kw={"height_ratios": [4, 1] * n_rows},
    )

    for idx, ex_name in enumerate(exercise_order):
        grid_row = (idx // n_cols) * 2   # Plot-Zeile (0 oder 2)
        grid_col = idx % n_cols           # Spalte (0 oder 1)

        ax       = axes[grid_row, grid_col]
        ax_table = axes[grid_row + 1, grid_col]

        sub = df_best[df_best["Übung"] == ex_name]
        subjects = sorted(sub["Subject"].unique())

        peak_phase_counts = {"PER": 0, "OVU": 0, "LUT": 0}
        higher_is_better = True

        for s_idx, subj in enumerate(subjects):
            subj_data = sub[sub["Subject"] == subj]

            vals = {}
            for phase in phase_order:
                row = subj_data[subj_data["Phase"] == phase]
                if not row.empty:
                    vals[phase] = float(row["Wert"].values[0])

            if not vals:
                continue

            if higher_is_better:
                best_phase = max(vals, key=vals.get)
            else:
                best_phase = min(vals, key=vals.get)
            peak_phase_counts[best_phase] += 1

            x_pts = [phase_x[p] for p in phase_order if p in vals]
            y_pts = [vals[p] for p in phase_order if p in vals]

            ax.plot(x_pts, y_pts,
                    color="#adb5bd", linewidth=1.0, alpha=0.6,
                    zorder=2)

            for phase, val in vals.items():
                is_best = (phase == best_phase)
                ax.scatter(
                    phase_x[phase], val,
                    color=phase_colors[phase],
                    s=120 if is_best else 30,
                    alpha=0.95 if is_best else 0.5,
                    edgecolors="black" if is_best else "white",
                    linewidth=1.5 if is_best else 0.5,
                    zorder=4 if is_best else 3,
                )

            best_val = vals[best_phase]
            ax.annotate(
                subj, xy=(phase_x[best_phase], best_val),
                fontsize=6.5, color="#333333", alpha=0.85,
                xytext=(6, 2), textcoords="offset points",
            )

        ax.set_xticks(range(len(phase_order)))
        ax.set_xticklabels(phase_order, fontsize=11)
        ax.set_title(ex_name, fontsize=12, fontweight="bold")
        unit = "Sprunghöhe [m]" if "CMJ" in ex_name or "DJ" in ex_name else "Max. Kniewinkel [°]"
        ax.set_ylabel(unit, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        # Häufigkeitstabelle
        ax_table.axis("off")
        table_data = [[peak_phase_counts[p] for p in phase_order]]
        tbl = ax_table.table(
            cellText=table_data,
            colLabels=phase_order,
            rowLabels=["Bestleistung\nin Phase"],
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.0, 1.8)

        for j, phase in enumerate(phase_order):
            tbl[0, j].set_facecolor(phase_colors[phase])
            tbl[0, j].set_text_props(color="white", fontweight="bold")
            tbl[1, j].set_text_props(fontweight="bold", fontsize=12)

    # Leere Subplots ausblenden
    for idx in range(n_ex, n_rows * n_cols):
        grid_row = (idx // n_cols) * 2
        grid_col = idx % n_cols
        axes[grid_row, grid_col].set_visible(False)
        axes[grid_row + 1, grid_col].set_visible(False)

    fig.suptitle(
        "Individuelle Bestleistung pro Probandin – in welcher Zyklusphase?\n"
        "(großer Punkt = absolute Bestleistung über alle 3 Phasen)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  SVG (Bestleist.): {out_path.name}")


if __name__ == "__main__":
    main()