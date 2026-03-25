import pandas as pd
from pathlib import Path
import re

# ============================================================
# EINSTELLUNGEN & PFADE
# ============================================================
# Der Ordner mit den ROhen, großen CSV-Dateien aus Vicon
RAW_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\anonymized_csv_data")

# Der Pfad zu der neuen Metadaten-Tabelle (BITTE ANPASSEN, falls sie woanders liegt)
METADATA_CSV = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\c3d_metadata_export.csv")

# Hier wird die Liste für deine Betreuerin gespeichert
REPORT_PATH = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\Missing_C3D_Files_For_Supervisor.xlsx")

SUBJECT_MAP = {
    "K_P01": "S01", "A_P02": "S02", "A_P05": "S03", "D_P06": "S04",
    "P_P07": "S05", "B_P09": "S06", "P_P10": "S07", "Batzner_P01": "S08",
    "Lorenz_P02": "S09", "Feik_P03": "S10", "Platzer_P04": "S11"
}

TARGET_EXERCISES = [
    'Counter-movement jump session',
    'Drop jump session', 
    'Drop jump session_2',
    'Squatting session', 
    'Squatting session_3'
]

def map_dates_to_phases(df_meta):
    """Ordnet die chronologischen Vicon-Daten den Phasen (01_PER, etc.) zu."""
    mapping = {}
    phases = ['01_PER', '02_OVU', '03_LUT']
    
    for sub in df_meta['Subject'].dropna().unique():
        sub_dates = df_meta[df_meta['Subject'] == sub]['SESSION_DATE'].unique()
        # Datum chronologisch sortieren
        sorted_dates = sorted(sub_dates, key=lambda d: pd.to_datetime(d, format='%d.%m.%Y'))
        
        for i, date_str in enumerate(sorted_dates):
            if i < 3:
                mapping[(sub, date_str)] = phases[i]
                
    return mapping

def get_exported_trials():
    """Liest die exportierten Trials inklusive ihrer Phase (01_PER) aus."""
    exported = set()
    if not RAW_DIR.exists(): return exported

    for raw_csv in RAW_DIR.rglob("*.csv"):
        try:
            sub_id = raw_csv.name.split('_')[0]
            
            # Suche nach der Phase im Dateinamen (z.B. 01_PER)
            phase_match = re.search(r'(0[1-3]_[A-Z]{3})', raw_csv.name)
            phase = phase_match.group(1) if phase_match else "UNKNOWN"
            
            df_head = pd.read_csv(raw_csv, header=None, nrows=1, low_memory=False)
            for val in df_head.iloc[0].dropna().unique():
                val_str = str(val).strip()
                if val_str.upper() not in ["NAN", "ITEM", "TIME", "FRAMES", "SUBFRAMES"]:
                    trial_stem = Path(val_str).stem.upper()
                    
                    # JETZT speichern wir: (Subjekt, Phase, Trialname)
                    exported.add((sub_id, phase, trial_stem))
                    
        except Exception:
            pass
            
    return exported

def generate_supervisor_report():
    print("Starte phasen-genauen Abgleich...")
    df_meta = pd.read_csv(METADATA_CSV, sep=';')
    
    # 1. Mappen & Filtern
    df_meta['Subject'] = df_meta['PARTICIPANT'].map(SUBJECT_MAP)
    df_meta = df_meta.dropna(subset=['Subject'])
    df_meta = df_meta[df_meta['EXERCISE'].isin(TARGET_EXERCISES)]
    
    # 2. Daten den Phasen zuordnen
    date_to_phase = map_dates_to_phases(df_meta)
    
    # 3. Exportierte Trials holen
    exported_trials = get_exported_trials()
    
    missing_files = []
    
    # 4. Checken!
    for _, row in df_meta.iterrows():
        sub_id = row['Subject']
        date_str = row['SESSION_DATE']
        expected_phase = date_to_phase.get((sub_id, date_str), "UNKNOWN_PHASE")
        
        c3d_filename = str(row['C3D_FILENAME']).strip()
        trial_stem = Path(c3d_filename).stem.upper()
        
        # Abfrage: Fehlt diese Datei GANZ GENAU in dieser Phase?
        if (sub_id, expected_phase, trial_stem) not in exported_trials:
            missing_files.append({
                "Zugeordnete Phase": expected_phase,
                "Anonyme_ID": sub_id,
                "Vicon_Participant": row['PARTICIPANT'],
                "Datum_der_Messung": date_str,
                "Vicon_Ordner (Session)": row['EXERCISE'],
                "FEHLENDE_DATEI": c3d_filename
            })

    df_missing = pd.DataFrame(missing_files)
    
    if df_missing.empty:
        print("\n[INFO] Es fehlen keine Dateien.")
        return

    # 5. Excel für Betreuerin erstellen
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Sortieren nach Phase, dann Proband, dann Übung
    df_missing = df_missing.sort_values(['Zugeordnete Phase', 'Vicon_Participant', 'Vicon_Ordner (Session)'])
    
    try:
        with pd.ExcelWriter(REPORT_PATH, engine='openpyxl') as writer:
            df_missing.to_excel(writer, sheet_name='Fehlende_C3D_Dateien', index=False)
            
        print(f"\n[OK] FERTIG! Es fehlen exakt {len(df_missing)} Dateien.")
        print("Diese Liste kannst du jetzt deiner Betreuerin schicken.")
        
    except PermissionError:
        print("\n[!!!] Bitte schließe die Excel-Datei, falls sie offen ist!")

if __name__ == "__main__":
    generate_supervisor_report()