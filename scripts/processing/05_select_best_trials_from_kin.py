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
import matplotlib.ticker as ticker


# ============================================================
# PFADE
# ============================================================
DATA_DIR   = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\03_preprocessed_emg_data")
OUTPUT_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\05_best_trials_group_and_individual")


# ============================================================
# EINSTELLUNGEN
# ============================================================
PHASES = ["01_PER", "02_OVU", "03_LUT"]
PHASE_LABELS = {
    "01_PER": "PER",
    "02_OVU": "OVU",
    "03_LUT": "LUT",
}

# Übung → (Ordner-Name, Seite, Selektionsmethode)
EXERCISE_CONFIGS = [
    {"exercise": "CMJ", "side": "BILATERAL", "method": "jumpheight",
     "label": "CMJ bilateral"},
    {"exercise": "CMJ", "side": "RIGHT",     "method": "jumpheight",
     "label": "CMJ einbeinig R"},
    {"exercise": "DJ",  "side": "BILATERAL", "method": "jumpheight",
     "label": "DJ bilateral"},
    {"exercise": "SQ",  "side": "RIGHT",     "method": "knee_angle",
     "label": "SQ einbeinig R"},
]


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

    # Spalte finden: 'Right Knee Angles' (ohne _Y / _Z Suffix)
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
                    trials_info.append(info)
                    all_records.append(info)

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
    _create_overview_pdf(df_best, df_group, OUTPUT_DIR / "beste_trials_uebersicht.pdf")

    # ── Plot: Balkendiagramm Gruppenmittelwerte ────────────────────────────
    _create_group_barplot(df_group, OUTPUT_DIR / "gruppenmittelwerte_plot.pdf")

    # ── Plot: Individuelle Werte pro Übung ─────────────────────────────────
    _create_individual_plot(df_best, OUTPUT_DIR / "individuelle_beste_trials_plot.pdf")

    print(f"\n{'='*70}")
    print("FERTIG")
    print(f"  Output-Ordner   : {OUTPUT_DIR}")
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
    """Balkendiagramm: Mittelwert ± SD pro Übung und Phase."""
    exercises = df_group["Übung"].unique()
    n_ex = len(exercises)

    fig, axes = plt.subplots(1, n_ex, figsize=(5 * n_ex, 5), sharey=False)
    if n_ex == 1:
        axes = [axes]

    phase_colors = {"PER": "#C44462", "OVU": "#0EB55F", "LUT": "#FAD758"}
    phase_order  = ["PER", "OVU", "LUT"]

    for ax, ex_name in zip(axes, exercises):
        sub = df_group[df_group["Übung"] == ex_name]
        # Sortierung nach phase_order erzwingen
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

        # Werte über Balken
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

    fig.suptitle("Gruppenmittelwerte der besten Trials", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PDF (Balken)    : {out_path.name}")


def _create_individual_plot(df_best: pd.DataFrame, out_path: Path):
    """Stripplot: individuelle Werte der besten Trials pro Übung × Phase."""
    exercises = df_best["Übung"].unique()
    n_ex = len(exercises)

    fig, axes = plt.subplots(1, n_ex, figsize=(5 * n_ex, 5), sharey=False)
    if n_ex == 1:
        axes = [axes]

    phase_colors = {"PER": "#C44462", "OVU": "#0EB55F", "LUT": "#FAD758"}
    phase_order  = ["PER", "OVU", "LUT"]

    for ax, ex_name in zip(axes, exercises):
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

    fig.suptitle("Individuelle beste Trials (pro Probandin × Phase)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  PDF (Stripplot) : {out_path.name}")


if __name__ == "__main__":
    main()