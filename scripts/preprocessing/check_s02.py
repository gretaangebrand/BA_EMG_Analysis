import pandas as pd
from pathlib import Path

# Dein Pfad zu den Originaldaten
RAW_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\anonymized_csv_data\03_LUT\CMJ")

def check_raw_cmj_file():
    # Wir suchen alle Dateien für S02 in der Phase 03_LUT
    files = list(RAW_DIR.glob("S02_03_LUT*.csv"))
    
    # Filtere nach CMJ oder Counter-Movement
    cmj_files = [f for f in files if "CMJ" in f.name.upper() or "COUNTER" in f.name.upper()]
    
    if not cmj_files:
        print("FEHLER: Konnte die CMJ-Rohdatei für S02 (03_LUT) nicht finden!")
        return
        
    raw_file = cmj_files[0]
    print(f"\nUntersuche Original-Datei: {raw_file.name}")
    print("=" * 60)
    
    # Wir lesen NUR die allererste Zeile (Level 0), wo die Trial-Namen stehen
    try:
        df = pd.read_csv(raw_file, header=None, nrows=1, low_memory=False)
        
        # Wir holen alle Werte aus der ersten Zeile, ignorieren leere (NaN) und filtern Duplikate
        all_values = df.iloc[0].dropna().unique().tolist()
        
        print("Folgende Trials stehen GANZ OBEN in der Rohdatei (Level 0):")
        found_trials = 0
        
        # Typische "Müll"-Wörter, die wir ignorieren
        ignore_words = ["NAN", "ITEM", "TIME", "FRAMES", "SUBFRAMES", "SUBJECT", "CONTEXT"]
        
        for val in all_values:
            val_str = str(val).strip()
            if val_str.upper() not in ignore_words:
                print(f" -> {val_str}")
                found_trials += 1
                
        print("-" * 60)
        print(f"Insgesamt gefundene Trial-Blöcke: {found_trials}")
        
    except Exception as e:
        print(f"Fehler beim Lesen der Datei: {e}")

if __name__ == "__main__":
    check_raw_cmj_file()