"""
08_statistical_analysis.py
===========================
Führt die komplette statistische Auswertung der EMG-Features durch.
Ersetzt die manuelle SPSS-Analyse.
 
Für jede Kombination aus Übung × Muskel × Abschnitt × Kennwert:
  1. Deskriptive Statistik (M ± SD pro Phase)
  2. Shapiro-Wilk-Test auf Normalverteilung (pro Phase)
  3a. Bei Normalverteilung: Repeated-measures ANOVA (pingouin)
  3b. Bei Verletzung: Friedman-Test
 
Output:
  - statistische_ergebnisse.csv
  - statistische_ergebnisse.xlsx (formatiert wie SPSS-Vorlage)
  - latex_tabellen.tex (fertige LaTeX-Tabellen)
  - Reiter in Pipeline_Reports.xlsx
 
Voraussetzung:
  pip install pingouin
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# PFADE
# ============================================================
FEATURES_CSV = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\06_emg_features\emg_features_statistic.csv")
OUTPUT_DIR   = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\outputs\statistics")
#REPORT_PATH  = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\Pipeline_Reports.xlsx")

# ============================================================
# EINSTELLUNGEN
# ============================================================
ALPHA = 0.05
PHASE_ORDER = ["PER", "OVU", "LUT"]
KENNWERTE = ["mean_rms", "peak_rms"]
 
 
def interpret_eta2(eta2):
    if pd.isna(eta2):
        return ""
    if eta2 >= 0.14:
        return "gross"
    if eta2 >= 0.06:
        return "mittel"
    if eta2 >= 0.01:
        return "klein"
    return "kein"
 
 
def fmt_msd(m, sd):
    """Formatiert M ± SD mit 3 Nachkommastellen, Dezimalkomma."""
    if pd.isna(m) or pd.isna(sd):
        return "--"
    return f"{m:.3f} ± {sd:.3f}".replace(".", ",")
 
 
def fmt_p(p):
    """Formatiert p-Wert mit 3 Nachkommastellen."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0,001"
    return f"{p:.3f}".replace(".", ",")
 
 
def fmt_num(val, decimals=3):
    """Formatiert eine Zahl mit Dezimalkomma."""
    if pd.isna(val):
        return ""
    return f"{val:.{decimals}f}".replace(".", ",")
 
 
# ============================================================
# STATISTISCHE ANALYSE PRO KOMBINATION
# ============================================================
 
def analyse_combination(df_long: pd.DataFrame, kennwert: str) -> dict:
    result = {}
 
    # ── Deskriptive Statistik ──
    for phase in PHASE_ORDER:
        vals = df_long[df_long["Phase"] == phase][kennwert].values
        if len(vals) > 0:
            result[f"M_{phase}"] = np.mean(vals)
            result[f"SD_{phase}"] = np.std(vals, ddof=1)
            result[f"n_{phase}"] = len(vals)
        else:
            result[f"M_{phase}"] = np.nan
            result[f"SD_{phase}"] = np.nan
            result[f"n_{phase}"] = 0
 
    # ── Vollständige Fälle prüfen ──
    subjects_per_phase = {
        p: set(df_long[df_long["Phase"] == p]["Subject"].values)
        for p in PHASE_ORDER
    }
    complete_subjects = (
        subjects_per_phase["PER"]
        & subjects_per_phase["OVU"]
        & subjects_per_phase["LUT"]
    )
 
    if len(complete_subjects) < 3:
        result["Test"] = "nicht durchführbar"
        result["Anmerkungen"] = (
            f"Nur {len(complete_subjects)} Probandinnen "
            f"mit Daten in allen 3 Phasen"
        )
        return result
 
    df_complete = df_long[df_long["Subject"].isin(complete_subjects)].copy()
    result["n_komplett"] = len(complete_subjects)
 
    # ── Prüfen ob Varianz vorhanden (ANOVA braucht Varianz) ──
    for phase in PHASE_ORDER:
        vals = df_complete[df_complete["Phase"] == phase][kennwert].values
        if np.std(vals, ddof=1) == 0:
            result["Test"] = "nicht durchführbar"
            result["Anmerkungen"] = f"Keine Varianz in Phase {phase}"
            return result
 
    # ── Shapiro-Wilk ──
    normalverteilt = True
    sw_details = []
    for phase in PHASE_ORDER:
        vals = df_complete[df_complete["Phase"] == phase][kennwert].values
        if len(vals) >= 3:
            stat_sw, p_sw = stats.shapiro(vals)
            result[f"SW_stat_{phase}"] = stat_sw
            result[f"SW_p_{phase}"] = p_sw
            sw_details.append(f"{phase}: {fmt_p(p_sw)}")
            if p_sw < ALPHA:
                normalverteilt = False
        else:
            result[f"SW_stat_{phase}"] = np.nan
            result[f"SW_p_{phase}"] = np.nan
            sw_details.append(f"{phase}: n/a")
 
    result["Normalverteilung"] = "ja" if normalverteilt else "nein"
    result["SW_detail"] = "\n".join(sw_details)
 
    # ── Statistische Tests ──
    if normalverteilt:
        result["Test"] = "ANOVA"
        try:
            # Sicherstellen dass Phase als kategorisch sortiert ist
            df_complete["Phase"] = pd.Categorical(
                df_complete["Phase"], categories=PHASE_ORDER, ordered=True
            )
            df_complete = df_complete.sort_values(
                ["Subject", "Phase"]
            ).reset_index(drop=True)
 
            aov = pg.rm_anova(
                data=df_complete,
                dv=kennwert,
                within="Phase",
                subject="Subject",
                correction=True,
                detailed=True,
            )
 
            result["F"] = aov["F"].iloc[0]
            result["df"] = f"{aov['DF'].iloc[0]:.0f}"
            if len(aov) > 1:
                result["df"] += f", {aov['DF'].iloc[1]:.0f}"
            # pingouin >= 0.6: p_unc, ng2; ältere: p-unc, np2
            p_col = "p_unc" if "p_unc" in aov.columns else "p-unc"
            result["p_wert"] = aov[p_col].iloc[0]
 
            # Partielles Eta-Quadrat: η²p = SS_phase / (SS_phase + SS_error)
            # ng2 ist generalized eta², wir brauchen partial
            if "np2" in aov.columns:
                result["eta2_p"] = aov["np2"].iloc[0]
            elif "SS" in aov.columns and len(aov) > 1:
                ss_phase = aov["SS"].iloc[0]
                ss_error = aov["SS"].iloc[1]
                result["eta2_p"] = (
                    ss_phase / (ss_phase + ss_error)
                    if (ss_phase + ss_error) > 0 else np.nan
                )
            else:
                # Fallback auf ng2
                eta_col = "ng2" if "ng2" in aov.columns else "np2"
                result["eta2_p"] = aov[eta_col].iloc[0]
 
            result["eta2_interpretation"] = interpret_eta2(result["eta2_p"])
 
            # Sphärizität
            try:
                sph = pg.sphericity(
                    data=df_complete,
                    dv=kennwert,
                    within="Phase",
                    subject="Subject",
                )
                if isinstance(sph, tuple) and len(sph) >= 3:
                    result["Mauchly_p"] = sph[2]
                    sph_verletzt = sph[2] < ALPHA
                else:
                    result["Mauchly_p"] = np.nan
                    sph_verletzt = False
            except Exception:
                result["Mauchly_p"] = np.nan
                sph_verletzt = False
 
            result["Sphaerizitaet_verletzt"] = (
                "ja" if sph_verletzt else "nein"
            )
 
            if sph_verletzt:
                result["GG_Korrektur"] = "ja"
                # pingouin >= 0.6: p_GG_corr; ältere: p-GG-corr
                gg_col = next(
                    (c for c in aov.columns if "GG" in c.upper() and "p" in c.lower()),
                    None,
                )
                if gg_col:
                    result["p_wert"] = aov[gg_col].iloc[0]
                else:
                    # GG manuell berechnen: F bleibt gleich, nur df korrigiert
                    eps = aov["eps"].iloc[0] if "eps" in aov.columns else 1.0
                    from scipy import stats as sp_stats
                    df1_corr = result["F"] and aov["DF"].iloc[0] * eps
                    df2_corr = aov["DF"].iloc[1] * eps if len(aov) > 1 else 10
                    result["p_wert"] = 1 - sp_stats.f.cdf(
                        result["F"], df1_corr, df2_corr
                    )
            else:
                result["GG_Korrektur"] = "nein"
 
            # Teststatistik-String
            result["Teststatistik"] = fmt_num(result["F"])
 
        except Exception as e:
            # Fallback auf Friedman wenn ANOVA fehlschlägt
            print(f"    [INFO] ANOVA fehlgeschlagen ({e}), "
                  f"verwende Friedman als Fallback")
            result["Test"] = "Friedman"
            result["Anmerkungen"] = f"ANOVA fehlgeschlagen: {e}. Friedman als Fallback."
            normalverteilt = False  # Friedman-Block unten ausführen
 
    if not normalverteilt and result.get("Test") != "nicht durchführbar":
        if result.get("Test") != "Friedman":
            result["Test"] = "Friedman"
 
        vals_per_phase = []
        for phase in PHASE_ORDER:
            phase_vals = (
                df_complete[df_complete["Phase"] == phase]
                .sort_values("Subject")[kennwert]
                .values
            )
            vals_per_phase.append(phase_vals)
 
        try:
            stat_fr, p_fr = stats.friedmanchisquare(*vals_per_phase)
            result["Chi2"] = stat_fr
            result["Teststatistik"] = fmt_num(stat_fr)
            result["df"] = str(len(PHASE_ORDER) - 1)
            result["p_wert"] = p_fr
            result["Signifikant"] = "ja" if p_fr < ALPHA else "nein"
 
            # Kendalls W
            n = len(vals_per_phase[0])
            k = len(PHASE_ORDER)
            result["Kendalls_W"] = (
                stat_fr / (n * (k - 1)) if n * (k - 1) > 0 else np.nan
            )
 
            result["Mauchly_p"] = np.nan
            result["Sphaerizitaet_verletzt"] = ""
            result["GG_Korrektur"] = ""
            result["eta2_p"] = np.nan
            result["eta2_interpretation"] = ""
 
        except Exception as e:
            result["Test"] = "Friedman fehlgeschlagen"
            result["Anmerkungen"] = str(e)
            return result
 
    # ── Signifikanz setzen (falls noch nicht gesetzt) ──
    if "Signifikant" not in result:
        p = result.get("p_wert", np.nan)
        result["Signifikant"] = (
            "ja" if (not pd.isna(p) and p < ALPHA) else "nein"
        )
 
    # ── Post-hoc ──
    if result.get("Signifikant") == "ja":
        if result["Test"] == "ANOVA":
            try:
                posthoc = pg.pairwise_tests(
                    data=df_complete,
                    dv=kennwert,
                    within="Phase",
                    subject="Subject",
                    padjust="bonf",
                )
                for _, row in posthoc.iterrows():
                    a, b = row["A"], row["B"]
                    # pingouin >= 0.6: p_corr; ältere: p-corr
                    pcorr_col = "p_corr" if "p_corr" in row.index else "p-corr"
                    result[f"posthoc_{a}_{b}_p"] = row[pcorr_col]
            except Exception:
                pass
        else:
            pairs = [("PER", "OVU"), ("PER", "LUT"), ("OVU", "LUT")]
            for p1, p2 in pairs:
                v1 = (
                    df_complete[df_complete["Phase"] == p1]
                    .sort_values("Subject")[kennwert].values
                )
                v2 = (
                    df_complete[df_complete["Phase"] == p2]
                    .sort_values("Subject")[kennwert].values
                )
                try:
                    _, p_w = stats.wilcoxon(v1, v2)
                    result[f"posthoc_{p1}_{p2}_p"] = min(p_w * 3, 1.0)
                except Exception:
                    result[f"posthoc_{p1}_{p2}_p"] = np.nan
 
    return result
 
 
# ============================================================
# HAUPTFUNKTION
# ============================================================
 
def main():
    print("=" * 70)
    print("08_statistical_analysis.py  –  Statistische Auswertung")
    print("=" * 70)
 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
    if not FEATURES_CSV.exists():
        print(f"[FEHLER] Datei nicht gefunden:\n  {FEATURES_CSV}")
        return
 
    df = pd.read_csv(FEATURES_CSV)
    print(f"\nFeatures geladen: {len(df)} Zeilen")
    print(f"  Subjects:   {sorted(df['Subject'].unique())}")
 
    # ── Alle Kombinationen durchrechnen ──
    all_results = []
    n_anova, n_friedman, n_sig = 0, 0, 0
 
    combinations = (
        df.groupby(["Uebung", "Muskel", "Abschnitt"])
        .size().reset_index()
    )
    total = len(combinations) * len(KENNWERTE)
    print(f"\n{total} Analysen werden durchgeführt...\n")
 
    for _, combo in combinations.iterrows():
        uebung    = combo["Uebung"]
        muskel    = combo["Muskel"]
        abschnitt = combo["Abschnitt"]
 
        mask = (
            (df["Uebung"] == uebung)
            & (df["Muskel"] == muskel)
            & (df["Abschnitt"] == abschnitt)
        )
        df_sub = df[mask].copy()
 
        for kennwert in KENNWERTE:
            result = analyse_combination(df_sub, kennwert)
            result["Uebung"] = uebung
            result["Muskel"] = muskel
            result["Abschnitt"] = abschnitt
            result["Kennwert"] = kennwert
            all_results.append(result)
 
            test = result.get("Test", "")
            if test == "ANOVA":
                n_anova += 1
            elif test == "Friedman":
                n_friedman += 1
            if result.get("Signifikant") == "ja":
                n_sig += 1
 
            sig = " ***" if result.get("Signifikant") == "ja" else ""
            p_val = result.get("p_wert", np.nan)
            p_str = f"p={p_val:.4f}" if not pd.isna(p_val) else "p=n/a"
            print(
                f"  {uebung:22s} | {muskel:25s} | {abschnitt:12s} | "
                f"{kennwert:8s} | {test:12s} | {p_str}{sig}"
            )
 
    df_results = pd.DataFrame(all_results)
 
    # ── CSV speichern ──
    csv_out = OUTPUT_DIR / "statistische_ergebnisse.csv"
    df_results.to_csv(csv_out, index=False)
    print(f"\n  CSV: {csv_out}")
 
    # ── Formatierte Excel (wie SPSS-Vorlage) ──
    _save_formatted_excel(df_results, OUTPUT_DIR / "statistische_ergebnisse.xlsx")
 
    # ── Pipeline-Report ──
    #_save_to_pipeline_report({"08_Statistik_Ergebnisse": df_results})
 
    # ── LaTeX-Tabellen ──
    _generate_latex_tables(df_results, OUTPUT_DIR / "latex_tabellen.tex")
 
    # ── Zusammenfassung ──
    print(f"\n{'='*70}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*70}")
    print(f"  Analysen:              {len(df_results)}")
    print(f"  rm-ANOVA:              {n_anova}")
    print(f"  Friedman:              {n_friedman}")
    print(f"  Signifikant:           {n_sig}")
 
    if n_sig > 0:
        print(f"\n  Signifikante Ergebnisse:")
        for _, r in df_results[df_results["Signifikant"] == "ja"].iterrows():
            print(f"    {r['Uebung']} | {r['Muskel']} | "
                  f"{r['Abschnitt']} | {r['Kennwert']} | p={r['p_wert']:.4f}")
    else:
        print(f"\n  Kein signifikanter Effekt gefunden.")
 
    anova_rows = df_results[df_results["Test"] == "ANOVA"]
    if not anova_rows.empty and "eta2_p" in anova_rows.columns:
        eta2_vals = anova_rows["eta2_p"].dropna()
        if not eta2_vals.empty:
            print(f"\n  Effektgrössen (η²p):")
            print(f"    Median: {eta2_vals.median():.3f}")
            print(f"    Max:    {eta2_vals.max():.3f}")
            print(f"    Gross:  {(eta2_vals >= 0.14).sum()}")
            print(f"    Mittel: {((eta2_vals >= 0.06) & (eta2_vals < 0.14)).sum()}")
            print(f"    Klein:  {((eta2_vals >= 0.01) & (eta2_vals < 0.06)).sum()}")
 
    print(f"\n{'='*70}")
    print(f"FERTIG – Output: {OUTPUT_DIR}")
    print(f"{'='*70}")
 
 
# ============================================================
# EXCEL-AUSGABE IM SPSS-VORLAGE-FORMAT
# ============================================================
 
def _save_formatted_excel(df: pd.DataFrame, out_path: Path):
    """
    Speichert die Ergebnisse im gleichen Format wie die SPSS-Vorlage:
    - Titel und Untertitel oben
    - Abschnitts-Überschriften (Gesamt – mean_rms, etc.)
    - Header-Zeile mit den 20 Spalten
    - Datenzeilen mit formatierten Werten
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
 
    wb = Workbook()
    ws = wb.active
    ws.title = "Statistik_Ergebnisse"
 
    # ── Styles ──
    header_fill = PatternFill('solid', fgColor='2F5496')
    header_font = Font(bold=True, color='FFFFFF', name='Arial', size=10)
    section_fill = PatternFill('solid', fgColor='D6E4F0')
    section_font = Font(bold=True, name='Arial', size=10, color='2F5496')
    normal_font = Font(name='Arial', size=10)
    sig_fill = PatternFill('solid', fgColor='E2EFDA')
    center = Alignment(horizontal='center', vertical='top', wrap_text=True)
    left = Alignment(horizontal='left', vertical='top', wrap_text=True)
    thin_border = Border(
        left=Side(style='thin', color='B4C6E7'),
        right=Side(style='thin', color='B4C6E7'),
        top=Side(style='thin', color='B4C6E7'),
        bottom=Side(style='thin', color='B4C6E7'),
    )
    zebra = PatternFill('solid', fgColor='F2F7FB')
 
    # ── Spalten-Header ──
    headers = [
        "Übung", "Muskel", "Abschnitt", "Kennwert",
        "M ± SD\nPER", "M ± SD\nOVU", "M ± SD\nLUT",
        "Normalvert.\n(Shapiro-Wilk)",
        "Test\n(ANOVA/Friedman)",
        "Teststatistik\n(F / χ²)",
        "df",
        "p-Wert",
        "η²p\n(nur ANOVA)",
        "Sphärizität\n(Mauchly p)",
        "GG-Korrektur\nangewendet?",
        "Signifikant?\n(p < 0,05)",
        "Post-hoc\nPER↔OVU (p)",
        "Post-hoc\nPER↔LUT (p)",
        "Post-hoc\nOVU↔LUT (p)",
        "Anmerkungen",
    ]
    col_widths = [20, 22, 16, 12, 16, 16, 16, 18, 16, 14, 8, 12, 14, 14, 14, 14, 14, 14, 14, 30]
 
    # ── Titel ──
    ws.merge_cells('A1:T1')
    ws['A1'].value = "Statistische Ergebnisse – EMG-Kennwerte über die drei Zyklusphasen"
    ws['A1'].font = Font(bold=True, name='Arial', size=14, color='2F5496')
 
    ws.merge_cells('A2:T2')
    ws['A2'].value = ("repeated-measures ANOVA bzw. Friedman-Test | "
                      "Post-hoc: Bonferroni-korrigierte Paarvergleiche | α = 0,05")
    ws['A2'].font = Font(name='Arial', size=9, color='666666')
 
    ws.merge_cells('A3:T3')
    ws['A3'].value = "auf die 3 Nachkommastellen gerundet"
    ws['A3'].font = Font(name='Arial', size=9, color='666666')
 
    # ── Spaltenbreiten ──
    for c, w in enumerate(col_widths, 1):
        from openpyxl.utils import get_column_letter
        ws.column_dimensions[get_column_letter(c)].width = w
 
    # ── Abschnitte und Daten schreiben ──
    # Gruppierung: Abschnitt × Kennwert
    abschnitt_kennwert_order = [
        ("Gesamt", "mean_rms"),
        ("Gesamt", "peak_rms"),
        ("Landung", "mean_rms"),
        ("Landung", "peak_rms"),
        ("Bodenkontakt", "mean_rms"),
        ("Bodenkontakt", "peak_rms"),
        ("Landung2", "mean_rms"),
        ("Landung2", "peak_rms"),
    ]
 
    # Labels für Abschnitts-Überschriften
    abschnitt_labels = {
        ("Gesamt", "mean_rms"): "Gesamt – mean_rms",
        ("Gesamt", "peak_rms"): "Gesamt – peak_rms",
        ("Landung", "mean_rms"): "Landung (CMJ) – mean_rms",
        ("Landung", "peak_rms"): "Landung (CMJ) – peak_rms",
        ("Bodenkontakt", "mean_rms"): "Bodenkontakt (DJ) – mean_rms",
        ("Bodenkontakt", "peak_rms"): "Bodenkontakt (DJ) – peak_rms",
        ("Landung2", "mean_rms"): "Landung2 (DJ) – mean_rms",
        ("Landung2", "peak_rms"): "Landung2 (DJ) – peak_rms",
    }
 
    current_row = 4
 
    for abschnitt, kennwert in abschnitt_kennwert_order:
        sub = df[
            (df["Abschnitt"] == abschnitt)
            & (df["Kennwert"] == kennwert)
        ]
        if sub.empty:
            continue
 
        # Abschnitts-Überschrift
        ws.merge_cells(
            start_row=current_row, start_column=1,
            end_row=current_row, end_column=len(headers),
        )
        cell = ws.cell(
            row=current_row, column=1,
            value=abschnitt_labels.get(
                (abschnitt, kennwert), f"{abschnitt} – {kennwert}"
            ),
        )
        cell.font = section_font
        cell.fill = section_fill
        current_row += 1
 
        # Header
        for c, h in enumerate(headers, 1):
            cell = ws.cell(row=current_row, column=c, value=h)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center
            cell.border = thin_border
        current_row += 1
 
        # Datenzeilen
        sub_sorted = sub.sort_values(["Uebung", "Muskel"])
        for row_idx, (_, r) in enumerate(sub_sorted.iterrows()):
            values = [
                r.get("Uebung", ""),
                r.get("Muskel", ""),
                r.get("Abschnitt", ""),
                r.get("Kennwert", ""),
                fmt_msd(r.get("M_PER"), r.get("SD_PER")),
                fmt_msd(r.get("M_OVU"), r.get("SD_OVU")),
                fmt_msd(r.get("M_LUT"), r.get("SD_LUT")),
                r.get("SW_detail", ""),
                r.get("Test", ""),
                r.get("Teststatistik", ""),
                r.get("df", ""),
                fmt_p(r.get("p_wert")),
                fmt_num(r.get("eta2_p")) if not pd.isna(r.get("eta2_p", np.nan)) else "",
                fmt_p(r.get("Mauchly_p")) if not pd.isna(r.get("Mauchly_p", np.nan)) else "",
                r.get("GG_Korrektur", ""),
                r.get("Signifikant", ""),
                fmt_p(r.get("posthoc_PER_OVU_p", r.get("posthoc_PER_vs_OVU_p"))),
                fmt_p(r.get("posthoc_PER_LUT_p", r.get("posthoc_PER_vs_LUT_p"))),
                fmt_p(r.get("posthoc_OVU_LUT_p", r.get("posthoc_OVU_vs_LUT_p"))),
                r.get("Anmerkungen", ""),
            ]
 
            for c, val in enumerate(values, 1):
                cell = ws.cell(row=current_row, column=c, value=val)
                cell.font = normal_font
                cell.alignment = center if c > 4 else left
                cell.border = thin_border
 
                # Zebrastreifen
                if row_idx % 2 == 1:
                    cell.fill = zebra
 
            # Signifikante Zeilen grün
            if r.get("Signifikant") == "ja":
                for c in range(1, len(headers) + 1):
                    ws.cell(row=current_row, column=c).fill = sig_fill
 
            ws.row_dimensions[current_row].height = 45
            current_row += 1
 
        # Leerzeile nach jedem Abschnitt
        current_row += 1
 
    ws.freeze_panes = 'A5'
    wb.save(out_path)
    print(f"  Excel: {out_path.name}")
 
 
# ============================================================
# LATEX-TABELLEN
# ============================================================
 
def _generate_latex_tables(df: pd.DataFrame, out_path: Path):
    lines = [
        "% Automatisch generiert von 08_statistical_analysis.py",
        "",
    ]
 
    for (uebung, abschnitt), grp in df.groupby(["Uebung", "Abschnitt"]):
        label_safe = (
            uebung.replace(" ", "_").replace(".", "")
            + "_" + abschnitt
        )
        caption = (
            f"EMG-Kennwerte – {uebung} ({abschnitt})"
        )
 
        lines.append(r"\begin{table}[H]")
        lines.append(r"  \centering")
        lines.append(f"  \\caption{{{caption}}}")
        lines.append(f"  \\label{{tab:{label_safe}}}")
        lines.append(r"  \small")
        lines.append(r"  \begin{tabular}{l l c c c l c c}")
        lines.append(r"    \hline")
        lines.append(
            r"    \textbf{Muskel} & \textbf{Kennwert} & "
            r"\textbf{PER} & \textbf{OVU} & \textbf{LUT} & "
            r"\textbf{Test} & \textbf{p} & "
            r"\textbf{$\eta^2_p$} \\"
        )
        lines.append(r"    \hline")
 
        for muskel in grp["Muskel"].unique():
            first_row = True
            for kennwert in KENNWERTE:
                row = grp[
                    (grp["Muskel"] == muskel)
                    & (grp["Kennwert"] == kennwert)
                ]
                if row.empty:
                    continue
                r = row.iloc[0]
 
                muskel_str = (
                    f"M.~{muskel.lower()}" if first_row else ""
                )
                first_row = False
 
                def lfmt(phase):
                    m = r.get(f"M_{phase}", np.nan)
                    sd = r.get(f"SD_{phase}", np.nan)
                    if pd.isna(m):
                        return "--"
                    return f"{m:.1f} $\\pm$ {sd:.1f}"
 
                test_str = r.get("Test", "")
                if r.get("GG_Korrektur") == "ja":
                    test_str += "$^\\text{GG}$"
 
                p_val = r.get("p_wert", np.nan)
                if pd.isna(p_val):
                    p_str = "--"
                elif p_val < 0.001:
                    p_str = "$<$\\,0{,}001"
                else:
                    p_str = f"{p_val:.3f}".replace(".", "{,}")
 
                eta2 = r.get("eta2_p", np.nan)
                eta2_str = (
                    f"{eta2:.3f}".replace(".", "{,}")
                    if not pd.isna(eta2) else "--"
                )
 
                kw = "mean" if kennwert == "mean_rms" else "peak"
 
                lines.append(
                    f"    {muskel_str} & {kw} & "
                    f"{lfmt('PER')} & {lfmt('OVU')} & {lfmt('LUT')} & "
                    f"{test_str} & {p_str} & {eta2_str} \\\\"
                )
            lines.append(r"    \hline")
 
        lines.append(r"  \end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")
 
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  LaTeX: {out_path.name}")
 
 
# ============================================================
# PIPELINE-REPORT
# ============================================================
 
"""def _save_to_pipeline_report(sheets: dict[str, pd.DataFrame]):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists():
        from openpyxl import load_workbook
        with pd.ExcelWriter(
            REPORT_PATH, engine="openpyxl", mode="a",
            if_sheet_exists="replace",
        ) as writer:
            for name, dframe in sheets.items():
                dframe.to_excel(writer, sheet_name=name, index=False)
    else:
        with pd.ExcelWriter(REPORT_PATH, engine="openpyxl") as writer:
            for name, dframe in sheets.items():
                dframe.to_excel(writer, sheet_name=name, index=False)
    print(f"  Pipeline-Report: {REPORT_PATH.name}")"""
 
 
if __name__ == "__main__":
    main()