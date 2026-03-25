import pandas as pd
from pathlib import Path
import re

# ============================================================
# EINSTELLUNGEN
# ============================================================
DATA_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\preprocessed_emg_data")
RAW_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\anonymized_csv_data")
REPORT_PATH = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\preprocessed_emg_data\processing_summary.xlsx")

def extract_all_custom_text(raw_path):
    """
    Sammelt JEDEN Text aus den ersten 10 Zeilen, der kein Vicon-Standard und keine Zahl ist.
    """
    if not raw_path.exists():
        return {}

    # Die Blacklist: Diese Begriffe sind völlig normal und werden ignoriert
    IGNORE_WORDS = {
        "NAN", "ANALOG", "EMG_RAW", "VOLTS", "V", "SECONDS", "S", "HZ", 
        "FRAMES", "SUBFRAMES", "ITEM", "UNITS", "LABEL", "DATA", "TIME",
        "ORIGINAL", "EMPTY", "NONE", "CH", "CHANNEL"
    }

    extracted_info = {}
    try:
        # Lese die ersten 10 Zeilen (Metadaten-Bereich)
        df_head = pd.read_csv(raw_path, header=None, nrows=10, low_memory=False)
        
        for col in df_head.columns:
            col_data = df_head[col].astype(str).tolist()
            
            # Zeile 0 ist meist der Pfad oder Trial-Name (z.B. "Squatting 1")
            trial_id_raw = col_data[0]
            trial_name = Path(trial_id_raw).stem 
            
            custom_texts = []
            for cell in col_data:
                val = str(cell).strip().upper()
                
                # 1. Leere Zellen ignorieren
                if val == "NAN" or val == "": 
                    continue
                
                # 2. Reine Zahlen ignorieren (auch Kommazahlen oder negative Zahlen)
                if re.match(r'^-?\d+(\.\d+)?$', val): 
                    continue
                
                # 3. Standard-Vicon-Wörter ignorieren
                if val in IGNORE_WORDS: 
                    continue
                
                # 4. Den eigentlichen Trial-Namen ignorieren (kennen wir schon)
                if val == trial_name.upper() or val == trial_id_raw.upper(): 
                    continue

                # Wenn es ein Dateipfad ist, machen wir ihn kürzer, damit die Excel lesbar bleibt
                if "\\" in val or "/" in val:
                    val = Path(val).name

                custom_texts.append(val)
            
            # Duplikate entfernen, aber Reihenfolge beibehalten
            if custom_texts:
                unique_texts = list(dict.fromkeys(custom_texts))
                extracted_info[trial_name] = " | ".join(unique_texts)
            else:
                extracted_info[trial_name] = "Nur Standard-Werte"
                
    except Exception as e:
        print(f"  [INFO] Fehler beim Scannen von {raw_path.name}: {e}")
        
    return extracted_info

def generate_total_scan_audit():
    if not DATA_DIR.exists():
        print(f"[FEHLER] Ordner {DATA_DIR} nicht gefunden!")
        return

    audit_data = []
    all_files = list(DATA_DIR.rglob("*.csv"))
    raw_cache = {}

    print(f"Starte ALLES-SCANNER auf {len(all_files)} Dateien...")

    for file in all_files:
        parts = file.relative_to(DATA_DIR).parts
        if len(parts) >= 3:
            sub, ph, ex = parts[0], parts[1], parts[2]
            raw_filename = f"{sub}_{ph}_{ex}.csv"
            raw_path = RAW_DIR / raw_filename
            
            # Datei nur einmal einlesen und cachen
            if raw_filename not in raw_cache:
                raw_cache[raw_filename] = extract_all_custom_text(raw_path)
            
            my_trial_info = "Trial in Original-CSV nicht gefunden"
            current_raw_info = raw_cache[raw_filename]
            
            # Ordne die gefundenen Texte unserer Datei zu (z.B. SQ_L_01 -> Squatting Left 1)
            file_stem_upper = file.stem.upper()
            f_num = "".join(filter(str.isdigit, file.stem)) # Holt die Zahl '01'
            
            for v_name, content in current_raw_info.items():
                v_name_upper = v_name.upper()
                v_num = "".join(filter(str.isdigit, v_name))
                
                # Prüfe, ob es die gleiche Übung (z.B. CMJ) und die gleiche Nummer ist
                if v_num == f_num or int(v_num or 0) == int(f_num or 0):
                    # Unterscheide zwischen Left, Right und Bilateral
                    is_left = "_L_" in file_stem_upper
                    is_right = "_R_" in file_stem_upper
                    v_is_left = "LEFT" in v_name_upper
                    v_is_right = "RIGHT" in v_name_upper
                    
                    if (is_left == v_is_left) and (is_right == v_is_right):
                        my_trial_info = content
                        break
            
            audit_data.append({
                "Subject": sub,
                "Phase": ph,
                "Exercise": ex,
                "File": file.name,
                "Gefundener_Zusatztext": my_trial_info
            })

    df = pd.DataFrame(audit_data)
    
    # Pivot-Tabelle
    summary_table = df.pivot_table(index=["Subject", "Phase"], columns="Exercise", values="File", aggfunc="count", fill_value=0)
    summary_table["TOTAL_TRIALS"] = summary_table.sum(axis=1)

    # Excel Export
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(REPORT_PATH, engine='openpyxl') as writer:
            summary_table.to_excel(writer, sheet_name='Statistik_Übersicht')
            df.sort_values(["Subject", "Phase", "Exercise"]).to_excel(writer, sheet_name='Alles_Scanner_Details', index=False)
            
            # NEU: Wir drucken den KOMPLETTEN Inhalt der Original-Dateien in ein drittes Blatt
            raw_data_list = []
            for datei, inhalte in raw_cache.items():
                for trial_name, text in inhalte.items():
                    raw_data_list.append({"Original_CSV": datei, "Trial_Spalte": trial_name, "Gefundener_Text": text})
            
            pd.DataFrame(raw_data_list).to_excel(writer, sheet_name='RAW_Original_Inhalte', index=False)
    except PermissionError:
        print("\n[!!!] FEHLER: Excel-Datei ist noch offen. Bitte schließen!")

if __name__ == "__main__":
    generate_total_scan_audit()