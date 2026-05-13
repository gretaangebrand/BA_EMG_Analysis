"""
05c_kinematic_control_table.py
================================
Erstellt eine formatierte LaTeX-Tabelle für die "Kinematische Äquivalenz".
Zeigt Gruppenmittelwert ± SD für die kinematischen Parameter 
(Sprunghöhe bzw. Kniewinkel) pro Übung über die drei Zyklusphasen.

Input:  gruppenmittelwerte_gesamte_uebung.csv (erzeugt von 05_select_best_trials.py)
Output: kinematik_kontroll_tabelle.tex
"""

from pathlib import Path
import pandas as pd
import sys

# ============================================================
# PFADE
# ============================================================
GROUP_MEANS_CSV = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\05_best_trials_group_and_individual\gruppenmittelwerte_gesamte_uebung.csv")
OUTPUT_DIR      = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\outputs\figures\best_trials_group_individual")

# ============================================================
# EINSTELLUNGEN
# ============================================================
PHASE_ORDER = ["PER", "OVU", "LUT"]
EXERCISE_ORDER = [
    "SQ bilateral", 
    "SQ einbeinig R", 
    "CMJ bilateral", 
    "CMJ einbeinig R", 
    "DJ bilateral"
]

def fmt_val(mean, sd, ex_name):
    """
    Formatiert M ± SD ins deutsche Format (Komma statt Punkt).
    Sprünge (m) -> 2 Nachkommastellen.
    Squats (°) -> 0 Nachkommastellen.
    """
    if pd.isna(mean) or pd.isna(sd):
        return "--"
    
    if "CMJ" in ex_name or "DJ" in ex_name:
        # 2 Nachkommastellen für Meter
        m_str = f"{mean:.2f}".replace(".", "{,}")
        sd_str = f"{sd:.2f}".replace(".", "{,}")
    else:
        # 0 Nachkommastellen für Grad (Kniewinkel)
        m_str = f"{mean:.0f}"
        sd_str = f"{sd:.0f}"
        
    return f"{m_str} $\\pm$ {sd_str}"

def get_parameter_name(ex_name):
    """Gibt den passenden Parameter und die Einheit zurück."""
    if "CMJ" in ex_name or "DJ" in ex_name:
        return "Sprunghöhe [m]"
    else:
        # Skript 05 liest bei Squats den Kniewinkel aus
        return "max. Kniewinkel [$^\\circ$]"

# ============================================================
# HAUPTPROGRAMM
# ============================================================
def main():
    print("=" * 70)
    print("05c – Erstelle LaTeX-Tabelle für Kinematische Äquivalenz")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not GROUP_MEANS_CSV.exists():
        print(f"[FEHLER] Datei nicht gefunden:\n  {GROUP_MEANS_CSV}")
        print("Bitte zuerst 05_select_best_trials.py ausführen!")
        return

    # Daten laden
    df = pd.read_csv(GROUP_MEANS_CSV)
    
    # ── Tabelle aufbauen ──
    # Wir brauchen ein Dictionary, um die Tabelle zeilenweise zu füllen
    table_data = {}
    
    for ex in EXERCISE_ORDER:
        sub = df[df["Übung"] == ex]
        if sub.empty:
            continue
            
        row_data = {"Übung": ex, "Parameter": get_parameter_name(ex)}
        
        for phase in PHASE_ORDER:
            phase_data = sub[sub["Phase"] == phase]
            if not phase_data.empty:
                m = phase_data["Mittelwert"].values[0]
                sd = phase_data["SD"].values[0]
                row_data[phase] = fmt_val(m, sd, ex)
            else:
                row_data[phase] = "--"
                
        table_data[ex] = row_data

    # ── LaTeX-Code generieren ──
    lines = [
        r"% Automatisch generiert von 05c_kinematic_control_table.py",
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption{Kinematische Parameter der untersuchten Übungen über die drei Zyklusphasen. "
        r"Angezeigt werden die Gruppenmittelwerte $\pm$ Standardabweichung der jeweils besten individuellen Trials. "
        r"Es zeigen sich deskriptiv keine relevanten Abweichungen, was auf eine stabile Bewegungsausführung hindeutet.}",
        r"  \label{tab:kinematische_aequivalenz}",
        r"  \small",
        r"  \begin{tabular}{l l c c c}",
        r"    \toprule",
        r"    \textbf{Übung} & \textbf{Parameter} & \textbf{PER} & \textbf{OVU} & \textbf{LUT} \\",
        r"    \midrule"
    ]

    for ex in EXERCISE_ORDER:
        if ex in table_data:
            d = table_data[ex]
            # Übungsnamen etwas schöner machen
            ex_clean = d['Übung'].replace(" bilateral", " bil.").replace(" einbeinig R", " einb. R")
            line = f"    {ex_clean} & {d['Parameter']} & {d['PER']} & {d['OVU']} & {d['LUT']} \\\\"
            lines.append(line)

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}"
    ])

    # Datei speichern
    out_path = OUTPUT_DIR / "kinematik_kontroll_tabelle.tex"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    
    print(f"✅ LaTeX-Tabelle erfolgreich erstellt:")
    print(f"  {out_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()