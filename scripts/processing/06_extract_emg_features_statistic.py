"""
06_extract_emg_features_statistic.py
===========================
Extrahiert EMG-Kennwerte (mean RMS, peak RMS) aus den verarbeiteten
EMG-Daten (_processed.csv) für die besten Trials (aus 05_select_best_trials).

Pro Probandin × Phase × Übung × Muskel werden berechnet:
  - mean_rms: Mittlere Amplitude über den gesamten Bewegungszyklus (% BL)
  - peak_rms: Maximale Amplitude im Bewegungszyklus (% BL)
  - peak_pct: Zeitpunkt des Peaks (% des Bewegungszyklus)

Für CMJ und DJ zusätzlich phasenspezifisch:
  - CMJ: Landungsphase (event_landing → event_end_jump)
  - DJ:  Bodenkontaktphase (event_landing1 → event_take_off)
         Zweite Landung (event_landing2 → event_end_jump)

Output:
  - emg_features.csv: Kennwert-Tabelle für die Statistik (ANOVA)
  - emg_features_overview.pdf: Visualisierung der Kennwerte
  - Reiter in Pipeline_Reports.xlsx
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import sys
sys.path.insert(0, str(Path(r'C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG')))

from scripts.utils.config import EXERCISE_MAP, PHASE_COLORS, MUSCLE_NAMES, PHASE_COLORS, PHASE_HATCHES

# ============================================================
# PFADE
# ============================================================
PROCESSED_DIR  = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\04_processed_emg_data")
PREPROCESSED_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\03_preprocessed_emg_data")
BEST_TRIALS_CSV = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\05_best_trials_group_and_individual\beste_trials.csv")
OUTPUT_DIR     = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\06_emg_features")
REPORT_PATH    = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\Pipeline_Reports.xlsx")


# ============================================================
# EINSTELLUNGEN
# ============================================================
PHASES = ["01_PER", "02_OVU", "03_LUT"]
PHASE_LABELS = {"01_PER": "PER", "02_OVU": "OVU", "03_LUT": "LUT"}
PHASE_REVERSE = {v: k for k, v in PHASE_LABELS.items()}
 
SIDE = "R"
 
 
# ============================================================
# HILFSFUNKTIONEN
# ============================================================
 
def load_best_trials(csv_path: Path) -> pd.DataFrame:
    """Laedt die beste-Trials-Tabelle aus Skript 05."""
    df = pd.read_csv(csv_path)
    # Erwartete Spalten: Subject, Übung, Phase, Bester Trial, Wert, Einheit
    return df
 
 
def find_processed_file(subject: str, phase_key: str, exercise_folder: str,
                        side_folder: str, trial_name: str) -> Path | None:
    """
    Findet die _processed.csv fuer einen bestimmten Trial.
    trial_name kommt aus 05 (z.B. 'CMJ_02'), Dateiname ist '<trial>_processed.csv'.
    """
    trial_dir = PROCESSED_DIR / subject / phase_key / exercise_folder / side_folder
    if not trial_dir.exists():
        return None
 
    # Exakter Match
    exact = trial_dir / f"{trial_name}_processed.csv"
    if exact.exists():
        return exact
 
    # Fallback: Suche nach Teilstring
    for f in trial_dir.glob("*_processed.csv"):
        if trial_name in f.stem:
            return f
 
    return None
 
 
def find_kin_file(subject: str, phase_key: str, exercise_folder: str,
                  side_folder: str, trial_name: str) -> Path | None:
    """Findet die _kin.csv fuer Event-Zeitpunkte."""
    trial_dir = PREPROCESSED_DIR / subject / phase_key / exercise_folder / side_folder
    if not trial_dir.exists():
        return None
 
    exact = trial_dir / f"{trial_name}_kin.csv"
    if exact.exists():
        return exact
 
    for f in trial_dir.glob("*_kin.csv"):
        if trial_name in f.stem:
            return f
 
    return None
 
 
def get_landing_pct_range(kin_path: Path, exercise_label: str) -> dict[str, tuple[float, float]]:
    """
    Berechnet die prozentualen Zeitpunkte der Landungsphasen
    relativ zur Gesamtdauer des Trials.
 
    Gibt ein dict mit Phasen-Bezeichnungen und (start_pct, end_pct) zurueck.
    """
    try:
        kin = pd.read_csv(kin_path, nrows=1, low_memory=False)
    except Exception:
        return {}
 
    # Gesamtdauer aus time_s der KIN-Datei (oder EMG-Events)
    # Die Events sind absolute Zeiten; die Dauer ist end - start
    result = {}
 
    if "CMJ" in exercise_label:
        cols_needed = ["event_start_s", "event_landing_s", "event_end_jump_s"]
        if not all(c in kin.columns for c in cols_needed):
            return {}
 
        t_start   = float(kin["event_start_s"].iloc[0])
        t_landing = float(kin["event_landing_s"].iloc[0])
        t_end     = float(kin["event_end_jump_s"].iloc[0])
        total     = t_end - t_start
 
        if total <= 0:
            return {}
 
        landing_start_pct = ((t_landing - t_start) / total) * 100
        landing_end_pct   = 100.0
 
        result["Landung"] = (landing_start_pct, landing_end_pct)
 
    elif "DJ" in exercise_label:
        cols_needed = ["event_landing1_s", "event_take_off_s",
                       "event_landing2_s", "event_end_jump_s"]
        if not all(c in kin.columns for c in cols_needed):
            return {}
 
        t_land1   = float(kin["event_landing1_s"].iloc[0])
        t_takeoff = float(kin["event_take_off_s"].iloc[0])
        t_land2   = float(kin["event_landing2_s"].iloc[0])
        t_end     = float(kin["event_end_jump_s"].iloc[0])
        total     = t_end - t_land1
 
        if total <= 0:
            return {}
 
        contact_start_pct = 0.0
        contact_end_pct   = ((t_takeoff - t_land1) / total) * 100
        land2_start_pct   = ((t_land2 - t_land1) / total) * 100
        land2_end_pct     = 100.0
 
        result["Bodenkontakt"] = (contact_start_pct, contact_end_pct)
        result["Landung2"]     = (land2_start_pct, land2_end_pct)
 
    return result
 
 
def extract_features_from_curve(pct: np.ndarray, signal: np.ndarray,
                                pct_start: float = 0.0,
                                pct_end: float = 100.0) -> dict:
    """
    Extrahiert mean_rms und peak_rms aus einem Abschnitt der Kurve.
    """
    mask = (pct >= pct_start) & (pct <= pct_end)
    segment = signal[mask]
 
    if len(segment) == 0:
        return {"mean_rms": np.nan, "peak_rms": np.nan, "peak_pct": np.nan}
 
    peak_idx = np.argmax(segment)
    pct_segment = pct[mask]
 
    return {
        "mean_rms": float(np.mean(segment)),
        "peak_rms": float(np.max(segment)),
        "peak_pct": float(pct_segment[peak_idx]),
    }
 
 
# ============================================================
# HAUPTFUNKTION
# ============================================================
 
def main():
    print("=" * 70)
    print("06_extract_emg_features_statistic.py  –  EMG-Kennwerte extrahieren")
    print("=" * 70)
 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
    # 1) Beste Trials laden
    if not BEST_TRIALS_CSV.exists():
        print(f"[FEHLER] Beste-Trials-CSV nicht gefunden:\n  {BEST_TRIALS_CSV}")
        return
 
    df_best = load_best_trials(BEST_TRIALS_CSV)
    print(f"\nBeste Trials geladen: {len(df_best)} Eintraege")
 
    # 2) Features extrahieren
    all_features = []
    n_ok, n_skip = 0, 0
 
    for _, row in df_best.iterrows():
        subject    = row["Subject"]
        label      = row["Übung"]
        phase_short = row["Phase"]
        trial_name = row["Bester Trial"]
 
        phase_key = PHASE_REVERSE.get(phase_short)
        if phase_key is None:
            print(f"  [SKIP] Unbekannte Phase: {phase_short}")
            n_skip += 1
            continue
 
        if label not in EXERCISE_MAP:
            print(f"  [SKIP] Unbekannte Übung: {label}")
            n_skip += 1
            continue
 
        ex_folder, side_folder = EXERCISE_MAP[label]
 
        # Processed EMG laden
        proc_path = find_processed_file(
            subject, phase_key, ex_folder, side_folder, trial_name
        )
        if proc_path is None:
            print(f"  [SKIP] {subject} {phase_short} {label} {trial_name} "
                  f"– keine processed-Datei")
            n_skip += 1
            continue
 
        try:
            df_proc = pd.read_csv(proc_path, low_memory=False)
        except Exception as e:
            print(f"  [FEHLER] {proc_path.name}: {e}")
            n_skip += 1
            continue
 
        if "pct" not in df_proc.columns:
            print(f"  [SKIP] {proc_path.name} – keine pct-Spalte")
            n_skip += 1
            continue
 
        pct = df_proc["pct"].values
 
        # Landungsphasen ermitteln (CMJ/DJ)
        landing_phases = {}
        if "CMJ" in label or "DJ" in label:
            kin_path = find_kin_file(
                subject, phase_key, ex_folder, side_folder, trial_name
            )
            if kin_path is not None:
                landing_phases = get_landing_pct_range(kin_path, label)
 
        # Features pro Muskel extrahieren
        for muscle in MUSCLE_NAMES:
            col = f"{SIDE}_{muscle}"
            if col not in df_proc.columns:
                continue
 
            signal = df_proc[col].values
 
            # Gesamter Bewegungszyklus
            feat_total = extract_features_from_curve(pct, signal)
            all_features.append({
                "Subject":   subject,
                "Phase":     phase_short,
                "Uebung":    label,
                "Muskel":    muscle,
                "Abschnitt": "Gesamt",
                "mean_rms":  feat_total["mean_rms"],
                "peak_rms":  feat_total["peak_rms"],
                "peak_pct":  feat_total["peak_pct"],
            })
 
            # Landungsphasen (falls vorhanden)
            for phase_name, (pct_s, pct_e) in landing_phases.items():
                feat_phase = extract_features_from_curve(
                    pct, signal, pct_s, pct_e
                )
                all_features.append({
                    "Subject":   subject,
                    "Phase":     phase_short,
                    "Uebung":    label,
                    "Muskel":    muscle,
                    "Abschnitt": phase_name,
                    "mean_rms":  feat_phase["mean_rms"],
                    "peak_rms":  feat_phase["peak_rms"],
                    "peak_pct":  feat_phase["peak_pct"],
                })
 
        n_ok += 1
 
    print(f"\n  Erfolgreich: {n_ok} Trials")
    print(f"  Uebersprungen: {n_skip} Trials")
 
    if not all_features:
        print("[FEHLER] Keine Features extrahiert.")
        return
 
    # 3) DataFrame aufbauen
    df_feat = pd.DataFrame(all_features)
 
    # Sortieren
    df_feat = df_feat.sort_values(
        ["Subject", "Uebung", "Phase", "Muskel", "Abschnitt"]
    ).reset_index(drop=True)
 
    # 4) CSV speichern
    csv_out = OUTPUT_DIR / "emg_features_statistic.csv"
    df_feat.to_csv(csv_out, index=False)
    print(f"\n  CSV gespeichert: {csv_out}")
 
    # 5) Pipeline-Report
    _save_to_pipeline_report({
        "06_EMG_Features": df_feat,
    })
    print(f"  Pipeline-Report aktualisiert: {REPORT_PATH.name}")
 
    # 6) Visualisierung
    _create_overview_plot(df_feat, OUTPUT_DIR / "emg_features_statistic_overview.svg")
    _create_responder_plot(df_feat, OUTPUT_DIR / "emg_features_responder.svg")
 
    # 7) Zusammenfassung: Gruppenmittelwerte
    print(f"\n{'='*70}")
    print("GRUPPENMITTELWERTE (mean RMS, Gesamt-Bewegungszyklus)")
    print(f"{'='*70}")
    df_gesamt = df_feat[df_feat["Abschnitt"] == "Gesamt"]
    grp = df_gesamt.groupby(["Uebung", "Phase", "Muskel"])["mean_rms"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    for _, r in grp.iterrows():
        print(f"  {r['Uebung']:22s} | {r['Phase']:4s} | {r['Muskel']:25s} | "
              f"{r['mean']:6.1f} ± {r['std']:5.1f} %BL  (n={int(r['count'])})")
 
    # 8) SPSS Export: Wide Format pro Abschnitt
    _export_spss_wide(df_feat)
 
    # 9) Peak-Zeitpunkt: Plot + Excel-Reiter
    _create_peak_pct_plot(df_feat, OUTPUT_DIR / "peak_zeitpunkt_pro_muskel.svg")
    _export_peak_pct_to_report(df_feat)
    
    print(f"\n{'='*70}")
    print("FERTIG")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}")
 
# ============================================================
# SPSS-EXPORT: Wide-Format
# ============================================================
 
def _export_spss_wide(df: pd.DataFrame):
    """
    Exportiert die EMG-Features im Wide-Format fuer SPSS.
 
    SPSS braucht fuer die Messwiederholungs-ANOVA:
      - Eine Zeile pro Probandin (Subject)
      - Die drei Zyklusphasen als separate Spalten
 
    Pro Abschnitt (Gesamt, Landung, Bodenkontakt, Landung2) und
    pro Kennwert (mean_rms, peak_rms) wird eine eigene CSV erzeugt.
 
    Dateinamen:
      spss_Gesamt_mean_rms.csv
      spss_Gesamt_peak_rms.csv
      spss_Landung_mean_rms.csv   (nur CMJ)
      ...
 
    Spaltenformat:
      Subject | Uebung | Muskel | PER | OVU | LUT
    """
    spss_dir = OUTPUT_DIR / "spss_export"
    spss_dir.mkdir(parents=True, exist_ok=True)
 
    abschnitte = df["Abschnitt"].unique()
    kennwerte  = ["mean_rms", "peak_rms"]
 
    n_files = 0
 
    for abschnitt in abschnitte:
        df_sub = df[df["Abschnitt"] == abschnitt].copy()
 
        for kennwert in kennwerte:
            # Pivot: Zeilen = Subject × Uebung × Muskel, Spalten = Phase
            try:
                df_wide = df_sub.pivot_table(
                    index=["Subject", "Uebung", "Muskel"],
                    columns="Phase",
                    values=kennwert,
                    aggfunc="first",
                ).reset_index()
            except Exception as e:
                print(f"  [WARNUNG] Pivot fehlgeschlagen fuer "
                      f"{abschnitt}/{kennwert}: {e}")
                continue
 
            # Spaltenreihenfolge sicherstellen: PER, OVU, LUT
            phase_cols = [p for p in ["PER", "OVU", "LUT"] if p in df_wide.columns]
            df_wide = df_wide[["Subject", "Uebung", "Muskel"] + phase_cols]
 
            # Spaltennamen SPSS-freundlich umbenennen
            df_wide = df_wide.rename(columns={
                p: f"{kennwert}_{p}" for p in phase_cols
            })
 
            # Sortieren
            df_wide = df_wide.sort_values(
                ["Uebung", "Muskel", "Subject"]
            ).reset_index(drop=True)
 
            # Speichern
            filename = f"spss_{abschnitt}_{kennwert}.csv"
            out_path = spss_dir / filename
            df_wide.to_csv(out_path, index=False, sep=";", decimal=",")
            n_files += 1
 
    print(f"\n  SPSS-Export: {n_files} Dateien in {spss_dir}")
    print(f"  Format: Semikolon-getrennt, Dezimalkomma")
 
# ============================================================
# VISUALISIERUNG
# ============================================================
 
def _create_overview_plot(df: pd.DataFrame, out_path: Path):
    """
    Balkendiagramm: mean RMS pro Übung × Muskel × Phase.
    Nur Abschnitt 'Gesamt'.
    """
    df_gesamt = df[df["Abschnitt"] == "Gesamt"].copy()
    exercises = sorted(df_gesamt["Uebung"].unique())
    n_ex = len(exercises)
 
    fig, axes = plt.subplots(n_ex, 1, figsize=(14, 4 * n_ex), sharex=False)
    if n_ex == 1:
        axes = [axes]
 
    phase_order = ["PER", "OVU", "LUT"]
    bar_width = 0.25
    x_offsets = {p: i * bar_width for i, p in enumerate(phase_order)}
 
    for ax, ex_name in zip(axes, exercises):
        sub = df_gesamt[df_gesamt["Uebung"] == ex_name]
        muscles = MUSCLE_NAMES
 
        x = np.arange(len(muscles))
 
        for phase in phase_order:
            phase_data = sub[sub["Phase"] == phase]
            means, sds = [], []
            for m in muscles:
                vals = phase_data[phase_data["Muskel"] == m]["mean_rms"]
                means.append(vals.mean() if len(vals) > 0 else 0)
                sds.append(vals.std() if len(vals) > 1 else 0)
 
            bars = ax.bar(
                x + x_offsets[phase], means, bar_width,
                yerr=sds, capsize=3,
                color=PHASE_COLORS[phase], alpha=0.85,
                label=phase, edgecolor="black", linewidth=0.5,
                hatch=PHASE_HATCHES.get(phase, ""),
            )
 
        ax.set_title(ex_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean RMS (% Baseline)", fontsize=10)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(muscles, fontsize=8, rotation=15, ha="right")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
 
    fig.suptitle(
        "EMG-Aktivierung: Gruppenmittelwerte der besten Trials (mean RMS)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot gespeichert: {out_path.name}")
 
 
def _create_responder_plot(df: pd.DataFrame, out_path: Path):
    """
    Spaghetti-Plot: Individuelle Verläufe pro Muskel × Übung über die 3 Phasen.
    Zeigt Responder (hohe Variabilität) vs. Non-Responder (stabil).
    """
    df_gesamt = df[df["Abschnitt"] == "Gesamt"].copy()
    exercises = sorted(df_gesamt["Uebung"].unique())
    muscles = MUSCLE_NAMES
    n_ex = len(exercises)
    n_mus = len(muscles)
 
    phase_order = ["PER", "OVU", "LUT"]
 
    fig, axes = plt.subplots(
        n_mus, n_ex, figsize=(5 * n_ex, 3 * n_mus),
        sharex=True, sharey=False,
    )
    if n_ex == 1:
        axes = axes.reshape(-1, 1)
 
    for col_idx, ex_name in enumerate(exercises):
        for row_idx, muscle in enumerate(muscles):
            ax = axes[row_idx, col_idx]
            sub = df_gesamt[
                (df_gesamt["Uebung"] == ex_name) &
                (df_gesamt["Muskel"] == muscle)
            ]
 
            subjects = sorted(sub["Subject"].unique())
            cv_values = []
 
            for s_idx, subj in enumerate(subjects):
                subj_data = sub[sub["Subject"] == subj]
                vals = []
                for phase in phase_order:
                    v = subj_data[subj_data["Phase"] == phase]["mean_rms"].values
                    vals.append(v[0] if len(v) > 0 else np.nan)
 
                vals_arr = np.array(vals)
                valid = vals_arr[~np.isnan(vals_arr)]
 
                if len(valid) >= 2:
                    cv = (np.std(valid, ddof=1) / np.mean(valid)) * 100
                else:
                    cv = 0.0
                cv_values.append(cv)
 
                # Farbe: rot wenn CV > 10%, grau wenn stabil
                color = "#e63946" if cv > 10 else "#adb5bd"
                alpha = 0.9 if cv > 10 else 0.4
                lw = 1.8 if cv > 10 else 0.8
 
                ax.plot(
                    range(3), vals,
                    color=color, alpha=alpha, linewidth=lw,
                    marker="o", markersize=3,
                )
 
                # Label nur für Responder
                if cv > 10:
                    ax.annotate(
                        subj, xy=(2, vals[-1]),
                        fontsize=6, color=color, alpha=0.8,
                        xytext=(5, 0), textcoords="offset points",
                    )
 
            # Subplot-Formatierung
            ax.set_xticks(range(3))
            ax.set_xticklabels(phase_order, fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
 
            if col_idx == 0:
                ax.set_ylabel(muscle, fontsize=8)
            if row_idx == 0:
                ax.set_title(ex_name, fontsize=10, fontweight="bold")
 
            # CV-Info
            n_resp = sum(1 for cv in cv_values if cv > 15)
            ax.text(
                0.98, 0.95,
                f"{n_resp}/{len(cv_values)} Resp.",
                transform=ax.transAxes, fontsize=7,
                ha="right", va="top", color="#555",
            )
 
    fig.suptitle(
        "Individuelle EMG-Verläufe über den Zyklus (rot = Responder, CV > 15 %)",
        fontsize=13, fontweight="bold",
    )
    fig.text(0.005, 0.5, "Mean RMS (% Baseline)",
             va="center", rotation="vertical", fontsize=10)
    plt.tight_layout(rect=[0.02, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot gespeichert: {out_path.name}")
 
 
# ============================================================
# PEAK-ZEITPUNKT: Plot + Excel
# ============================================================
 
def _create_peak_pct_plot(df: pd.DataFrame, out_path: Path):
    """
    Stripplot: Zeitpunkt der maximalen Aktivierung (peak_pct)
    während der Landungsphasen, pro Muskel × Übung × Phase.
 
    CMJ: Abschnitt "Landung"
    DJ:  Abschnitte "Bodenkontakt" und "Landung2" (je ein eigener Plot)
    SQ:  nur Abschnitt "Gesamt" (keine Landung)
 
    Layout: 5 Zeilen (Muskeln) × n Spalten (Übungen).
    Jede Probandin als Punkt, verbunden über die 3 Phasen.
    """
    phase_order = ["PER", "OVU", "LUT"]
 
    # Welche Abschnitte pro Übungstyp plotten?
    # Tupel: (Abschnitt-Filter, Titel-Suffix, Dateiname-Suffix)
    landing_configs = {
        "CMJ": [("Landung", "Landungsphase", "landung")],
        "DJ":  [("Bodenkontakt", "Bodenkontaktphase", "bodenkontakt"),
                ("Landung2", "2. Landungsphase", "landung2")],
    }
 
    # Alle Übungen + Abschnitte sammeln die geplottet werden sollen
    plot_jobs = []
    for uebung in sorted(df["Uebung"].unique()):
        matched = False
        for key, configs in landing_configs.items():
            if key in uebung:
                for abschnitt, titel_suffix, datei_suffix in configs:
                    sub = df[(df["Uebung"] == uebung) & (df["Abschnitt"] == abschnitt)]
                    if not sub.empty:
                        plot_jobs.append((uebung, abschnitt, titel_suffix, datei_suffix))
                matched = True
                break
        if not matched:
            # SQ etc.: Gesamt
            sub = df[(df["Uebung"] == uebung) & (df["Abschnitt"] == "Gesamt")]
            if not sub.empty:
                plot_jobs.append((uebung, "Gesamt", "Gesamter Zyklus", "gesamt"))
 
    if not plot_jobs:
        return
 
    # Gruppieren nach Abschnitt-Typ für gemeinsame Plots
    from collections import defaultdict
    groups = defaultdict(list)
    for uebung, abschnitt, titel_suffix, datei_suffix in plot_jobs:
        groups[datei_suffix].append((uebung, abschnitt, titel_suffix))
 
    muscles = MUSCLE_NAMES
    n_mus = len(muscles)
 
    for datei_suffix, job_list in groups.items():
        n_ex = len(job_list)
 
        fig, axes = plt.subplots(
            n_mus, n_ex, figsize=(5 * max(n_ex, 2), 3 * n_mus),
            sharex=True, sharey=True,
            squeeze=False,
        )
 
        for col_idx, (uebung, abschnitt, titel_suffix) in enumerate(job_list):
            sub_all = df[(df["Uebung"] == uebung) & (df["Abschnitt"] == abschnitt)]
 
            for row_idx, muscle in enumerate(muscles):
                ax = axes[row_idx, col_idx]
                sub = sub_all[sub_all["Muskel"] == muscle]
 
                subjects = sorted(sub["Subject"].unique())
 
                for subj in subjects:
                    subj_data = sub[sub["Subject"] == subj]
                    vals = []
                    for phase in phase_order:
                        v = subj_data[subj_data["Phase"] == phase]["peak_pct"].values
                        vals.append(v[0] if len(v) > 0 else np.nan)
 
                    # Verbindungslinie
                    x_pts = [i for i, v in enumerate(vals) if not np.isnan(v)]
                    y_pts = [vals[i] for i in x_pts]
                    ax.plot(x_pts, y_pts,
                            color="#adb5bd", linewidth=0.8, alpha=0.5, zorder=2)
 
                    # Punkte pro Phase
                    for p_idx, (phase, val) in enumerate(zip(phase_order, vals)):
                        if np.isnan(val):
                            continue
                        ax.scatter(
                            p_idx, val,
                            color=PHASE_COLORS[phase],
                            s=35, alpha=0.75,
                            edgecolors="white", linewidth=0.5,
                            zorder=3,
                        )
 
                # Mittelwert als horizontale Linie
                for p_idx, phase in enumerate(phase_order):
                    phase_vals = sub[sub["Phase"] == phase]["peak_pct"].dropna()
                    if len(phase_vals) > 0:
                        mean_val = phase_vals.mean()
                        ax.hlines(mean_val, p_idx - 0.3, p_idx + 0.3,
                                  color=PHASE_COLORS[phase],
                                  linewidth=2.5, zorder=4)
 
                ax.set_xticks(range(3))
                ax.set_xticklabels(phase_order, fontsize=8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(axis="y", alpha=0.2)
 
                if col_idx == 0:
                    ax.set_ylabel(muscle, fontsize=8)
                if row_idx == 0:
                    ax.set_title(f"{uebung}\n({titel_suffix})",
                                 fontsize=10, fontweight="bold")
 
        # Gemeinsamer Titel
        abschnitt_label = job_list[0][2]
        fig.suptitle(
            f"Peak-Zeitpunkt während {abschnitt_label} [% Bewegungszyklus]\n"
            f"(jeder Punkt = eine Probandin, Linie = Mittelwert)",
            fontsize=13, fontweight="bold",
        )
        fig.text(0.005, 0.5, "Peak-Zeitpunkt [% Bewegungszyklus]",
                 va="center", rotation="vertical", fontsize=10)
        plt.tight_layout(rect=[0.02, 0, 1, 0.94])
 
        suffix = datei_suffix
        plot_path = out_path.parent / f"peak_zeitpunkt_{suffix}.svg"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot gespeichert: {plot_path.name}")
 
 
def _export_peak_pct_to_report(df: pd.DataFrame):
    """
    Exportiert den Zeitpunkt der maximalen Aktivierung (peak_pct)
    als Excel-Reiter im Pipeline-Report.
 
    Separate Reiter für:
      - Gesamt (alle Übungen)
      - Landung (CMJ)
      - Bodenkontakt (DJ)
      - Landung2 (DJ)
 
    Wide-Format: Subject × Übung × Muskel | PER | OVU | LUT
    """
    sheets = {}
 
    for abschnitt in df["Abschnitt"].unique():
        df_sub = df[df["Abschnitt"] == abschnitt].copy()
        if df_sub.empty or df_sub["peak_pct"].isna().all():
            continue
 
        try:
            df_wide = df_sub.pivot_table(
                index=["Subject", "Uebung", "Muskel"],
                columns="Phase",
                values="peak_pct",
                aggfunc="first",
            ).reset_index()
        except Exception:
            continue
 
        phase_cols = [p for p in ["PER", "OVU", "LUT"] if p in df_wide.columns]
        df_wide = df_wide[["Subject", "Uebung", "Muskel"] + phase_cols]
        df_wide = df_wide.rename(columns={
            p: f"peak_pct_{p}" for p in phase_cols
        })
        df_wide = df_wide.sort_values(
            ["Uebung", "Muskel", "Subject"]
        ).reset_index(drop=True)
 
        sheet_name = f"06_PeakZeit_{abschnitt}"
        # Excel-Reitername max 31 Zeichen
        sheets[sheet_name[:31]] = df_wide
 
    if sheets:
        _save_to_pipeline_report(sheets)
        for name in sheets:
            print(f"  Pipeline-Report: Reiter '{name}' hinzugefügt")
 
 
# ============================================================
# PIPELINE-REPORT
# ============================================================
 
def _save_to_pipeline_report(sheets: dict[str, pd.DataFrame]):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists():
        from openpyxl import load_workbook
        with pd.ExcelWriter(REPORT_PATH, engine="openpyxl", mode="a",
                            if_sheet_exists="replace") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
    else:
        with pd.ExcelWriter(REPORT_PATH, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
 
 
if __name__ == "__main__":
    main()