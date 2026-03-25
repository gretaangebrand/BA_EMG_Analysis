import pandas as pd
from pathlib import Path
import re

# ============================================================
# EINSTELLUNGEN
# ============================================================
DATA_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\preprocessed_emg_data")
RAW_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\anonymized_csv_data")
REPORT_PATH = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\anonymized_csv_data\preprocessing_summary.xlsx")


# Die Blacklist für den Scanner: Wörter, die völlig normal sind und ignoriert werden
IGNORE_WORDS = {
    "NAN", "ANALOG", "EMG_RAW", "VOLTS", "V", "SECONDS", "S", "HZ", 
    "FRAMES", "SUBFRAMES", "ITEM", "UNITS", "LABEL", "DATA", "TIME",
    "ORIGINAL", "EMPTY", "NONE", "CH", "CHANNEL", "MV"
}

def scan_all_raw_files():
    """
    Geht direkt in den RAW_DIR und liest aus jeder Originaldatei die ersten 10 Zeilen.
    Sucht nach manuellen Kommentaren (z.B. "BAD", "FAILED" oder anderen Texten).
    """
    raw_data_list = []
    
    if not RAW_DIR.exists():
        print(f"[FEHLER] RAW Ordner nicht gefunden: {RAW_DIR}")
        return pd.DataFrame()

    all_raw_files = list(RAW_DIR.rglob("*.csv"))
    print(f"Scanne {len(all_raw_files)} ORIGINAL-Dateien auf Kommentare...")

    for raw_path in all_raw_files:
        try:
            # Lese die ersten 10 Zeilen der Original-Datei
            df_head = pd.read_csv(raw_path, header=None, nrows=10, low_memory=False)
            
            for col in df_head.columns:
                # Hole alle Werte dieser Spalte als Liste
                col_data = df_head[col].tolist()
                
                # Der Trial-Name steht meist ganz oben in Zeile 0
                trial_id_raw = str(col_data[0]).strip()
                trial_name = Path(trial_id_raw).stem 
                
                # Wir überspringen leere Spalten oder reine Zeit-Spalten
                if trial_name.upper() in ["NAN", "TIME", "FRAMES"]:
                    continue
                
                custom_texts = []
                for cell in col_data:
                    # GANZ WICHTIG (Der Bugfix): Zwinge den Wert zu einem String!
                    val = str(cell).strip().upper()
                    
                    # 1. Leere Werte ignorieren
                    if val == "NAN" or val == "": 
                        continue
                    
                    # 2. Reine Zahlen ignorieren (auch Kommazahlen)
                    if re.match(r'^-?\d+(\.\d+)?$', val): 
                        continue
                        
                    # 3. Standard-Begriffe ignorieren
                    if val in IGNORE_WORDS: 
                        continue
                        
                    # 4. Den eigentlichen Spaltennamen ignorieren
                    if val == trial_name.upper() or val == trial_id_raw.upper(): 
                        continue

                    # Wenn es ein Dateipfad ist, kürze ihn, damit die Excel lesbar bleibt
                    if "\\" in val or "/" in val: 
                        val = Path(val).name
                        
                    custom_texts.append(val)
                
                # Nur speichern, wenn wir was gefunden haben, das nicht Standard ist
                if custom_texts:
                    # Duplikate entfernen
                    unique_texts = " | ".join(list(dict.fromkeys(custom_texts)))
                    raw_data_list.append({
                        "Original_Datei": raw_path.name,
                        "Trial_Spalte": trial_name,
                        "Gefundener_Zusatztext": unique_texts
                    })
                    
        except Exception as e:
            print(f"  [WARNUNG] Konnte {raw_path.name} nicht analysieren: {e}")

    return pd.DataFrame(raw_data_list)


def scan_processed_files():
    """
    Scant den Ordner mit den fertig verarbeiteten Trials und erstellt die Statistik.
    """
    audit_data = []
    if not DATA_DIR.exists():
        print(f"[FEHLER] Processed Ordner nicht gefunden: {DATA_DIR}")
        return pd.DataFrame(), pd.DataFrame()

    all_processed = list(DATA_DIR.rglob("*.csv"))
    print(f"Zähle {len(all_processed)} VERARBEITETE Dateien...")
    
    for file in all_processed:
        parts = file.relative_to(DATA_DIR).parts
        if len(parts) >= 3:
            audit_data.append({
                "Subject": parts[0],
                "Phase": parts[1],
                "Exercise": parts[2],
                "Gespeicherte_Datei": file.name
            })

    df = pd.DataFrame(audit_data)
    if df.empty:
        return df, pd.DataFrame()
        
    # Erstelle die Pivot-Tabelle (Übersicht)
    summary_table = df.pivot_table(index=["Subject", "Phase"], columns="Exercise", values="Gespeicherte_Datei", aggfunc="count", fill_value=0)
    summary_table["TOTAL_TRIALS"] = summary_table.sum(axis=1)
    
    return df, summary_table


if __name__ == "__main__":
    print("Starte Bulletproof-Scan...\n" + "="*40)
    
    # 1. Hole alle Kommentare aus den Originaldateien
    df_raw_comments = scan_all_raw_files()
    
    # 2. Hole die Statistik der verarbeiteten Dateien
    df_processed, df_summary = scan_processed_files()
    
    # 3. Speichere alles sauber in Excel ab
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(REPORT_PATH, engine='openpyxl') as writer:
            
            # Tab 1: Statistik
            if not df_summary.empty:
                df_summary.to_excel(writer, sheet_name='Statistik_Übersicht')
                
            # Tab 2: Alle verarbeiteten Dateien
            if not df_processed.empty:
                df_processed.sort_values(["Subject", "Phase", "Exercise"]).to_excel(writer, sheet_name='Verarbeitete_Dateien', index=False)
            
            # Tab 3: Alle gefundenen Kommentare aus den Original-CSVs
            if not df_raw_comments.empty:
                df_raw_comments.sort_values(["Original_Datei", "Trial_Spalte"]).to_excel(writer, sheet_name='RAW_Inhalte_Komplett', index=False)
            else:
                pd.DataFrame({"Info": ["Keine abweichenden Metadaten in den Original-CSVs gefunden."]}).to_excel(writer, sheet_name='RAW_Inhalte_Komplett', index=False)
                
        print(f"\n[OK] Excel erfolgreich erstellt: {REPORT_PATH}")
        
    except PermissionError:
        print("\n[!!!] FEHLER: Die Excel-Datei ist noch offen. Bitte in Excel schließen und Skript neu starten!")