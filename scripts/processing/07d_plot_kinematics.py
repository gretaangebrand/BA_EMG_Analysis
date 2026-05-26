import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import interp1d

# =============================================================================
# 07d_plot_kinematics.py
# Ziel: Erstellt 5 separate SVG-Plots für die kinematischen Gruppenmittelwerte.
# Anpassungen: Individuelle Linienstile, keine Plot-Titel, keine Legenden-Titel.
# =============================================================================

# --- 1. Konfiguration & Pfade ---
INVENTORY_CSV = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\03_preprocessed_emg_data\trial_inventar_rohdaten.csv")
KIN_DATA_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\03_preprocessed_emg_data")
SAVE_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\outputs\figures\plots_kinematik")
SAVE_DIR.mkdir(parents=True, exist_ok=True) # Ordner erstellen, falls nicht vorhanden

# Definition der sauberen Namen für die Legende
PHASE_MAP = {
    "01_PER": "PER",
    "02_OVU": "OVU",
    "03_LUT": "LUT"
}

PHASE_COLORS = {
    "PER": "#D55E00",   # Vermillion (Rot)
    "OVU": "#0072B2",   # Blau
    "LUT": "#CC79A7",   # Pink/Lila
}


# Linienstile als Dash-Tupel (Strichlänge, Lückenlänge)
PHASE_LINESTYLES = {
    "PER": "",               # durchgezogen (leerer String = Solid)
    "OVU": (4, 2),           # gestrichelt (Dashed)
    "LUT": (4, 2, 1, 2)      # Strich-Punkt (Dash-Dot)
}

# --- Konfiguration der 5 Übungen ---
PLOT_CONFIG = {
    ("SQ", "BILATERAL"): {
        "variable": "Right Knee Angles",
        "y_label": "Kniewinkel (rechts) in °",
        "cut_phase": "full",
        "start_event": None
    },
    ("SQ", "RIGHT"): {
        "variable": "Right Knee Angles",
        "y_label": "Kniewinkel (rechts) in °",
        "cut_phase": "full",
        "start_event": None
    },
    ("CMJ", "BILATERAL"): {
        "variable": "vGRF_Total", 
        "y_label": "vBRK (gesamt) in KG",
        "cut_phase": "landing",
        "start_event": "event_landing_s"
    },
    ("CMJ", "RIGHT"): {
        "variable": "Right GRF_Z", 
        "y_label": "vBRK (rechts) in KG",
        "cut_phase": "landing",
        "start_event": "event_landing_s"
    },
    ("DJ", "BILATERAL"): {
        "variable": "vGRF_Total",
        "y_label": "vBRK (gesamt) in KG",
        "cut_phase": "landing",
        "start_event": "event_landing2_s"
    }
}

# --- 2. Hilfsfunktion: Zeitnormalisierung (0-100%) ---
def time_normalize_kinematics(df, kin_col, n_points=101):
    t_orig = df['time_s'].values
    if len(t_orig) < 2: return None 
    
    t_norm = np.linspace(t_orig[0], t_orig[-1], n_points)
    pct_axis = np.linspace(0, 100, n_points)
    
    f_interp = interp1d(t_orig, df[kin_col].values, kind='cubic', fill_value='extrapolate')
    
    out_df = pd.DataFrame({
        'pct': pct_axis,
        'kin_value': f_interp(t_norm)
    })
    return out_df

# --- 3. Hauptfunktion: Daten verarbeiten & Plotten ---
def process_and_plot():
    inv = pd.read_csv(INVENTORY_CSV)
    inv = inv[inv['kin_gespeichert'] == True]
    
    for (exercise, side), config in PLOT_CONFIG.items():
        print(f"\nVerarbeite Übung: {exercise} {side}...")
        
        var_name = config["variable"]
        y_label = config["y_label"]
        start_event = config["start_event"]
        
        all_norm_data = []
        df_trials = inv[(inv['exercise'] == exercise) & (inv['side'] == side)]
        
        for _, row in df_trials.iterrows():
            file_path = KIN_DATA_DIR / row['subject_id'] / row['phase'] / row['exercise'] / row['side'] / f"{row['trial_name']}_kin.csv"
            if not file_path.exists(): continue
                
            df_kin = pd.read_csv(file_path)
            
            # Spezialfall vGRF_Total
            if var_name == "vGRF_Total":
                if 'Left GRF_Z' in df_kin.columns and 'Right GRF_Z' in df_kin.columns:
                    df_kin['vGRF_Total'] = df_kin['Left GRF_Z'] + df_kin['Right GRF_Z']
                else:
                    continue
            elif var_name not in df_kin.columns:
                continue
            
            # ZUSCHNEIDEN (CUT)
            df_cut = df_kin.copy()
            if config["cut_phase"] != "full":
                if start_event not in df_kin.columns or pd.isna(df_kin[start_event].iloc[0]):
                    continue
                t_start = df_kin[start_event].iloc[0]
                df_cut = df_cut[df_cut['time_s'] >= t_start]
                
                end_event = "event_end_jump_s"
                if end_event in df_cut.columns and not pd.isna(df_cut[end_event].iloc[0]):
                    t_end = df_cut[end_event].iloc[0]
                    df_cut = df_cut[df_cut['time_s'] <= t_end]
            
            # NORMALISIEREN
            df_norm = time_normalize_kinematics(df_cut, var_name)
            if df_norm is not None:
                # Hier passiert die Umbenennung für die Legende:
                df_norm['phase'] = PHASE_MAP.get(row['phase'], row['phase'])
                df_norm['subject_id'] = row['subject_id']
                all_norm_data.append(df_norm)
                
        if not all_norm_data: continue
            
        df_master = pd.concat(all_norm_data, ignore_index=True)
        
        # --- 4. Plot erstellen ---
        plt.figure(figsize=(10, 6))
        
        # Seaborn Lineplot mit Linienstilen und Farben
        ax = sns.lineplot(
            data=df_master, 
            x='pct', 
            y='kin_value', 
            hue='phase', 
            style='phase',                 # Aktiviert die Unterscheidung nach Linienstil
            palette=PHASE_COLORS,
            dashes=PHASE_LINESTYLES,       # Weist die definierten Linienstile zu
            errorbar='sd',
            linewidth=2.5
        )
        
        # Achsenbeschriftungen (KEIN Titel)
        x_label = "Bewegungszyklus in %" if config["cut_phase"] == "full" else "Landephase in %"
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.xlim(0, 100)
        
        # Legende anpassen (KEIN Titel)
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles=handles, labels=labels, title=None, loc='best', frameon=True)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # Speichern als SVG im Zielordner
        safe_name = f"kinematik_{exercise}_{side}_{var_name}".replace(" ", "_")
        save_path = SAVE_DIR / f"{safe_name}.svg"
        
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"  -> Plot gespeichert: {save_path}")
        
        plt.close()

if __name__ == "__main__":
    process_and_plot()