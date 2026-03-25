import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ============================================================
# EINSTELLUNGEN
# ============================================================
DATA_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\preprocessed_emg_data")
PLOT_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\plots\raw_signals")

# TEST-MODUS: Wenn True, wird nur S01 geplottet (zum Ausprobieren, ob die Bilder gut aussehen)
# Wenn False, werden ALLE 630 Dateien geplottet (kann ein paar Minuten dauern!)
TEST_MODE = True  

def plot_and_save_trial(csv_path):
    """Liest eine CSV und speichert einen Plot mit allen Kanälen als PNG."""
    df = pd.read_csv(csv_path)
    
    # Finde alle Muskel-Kanäle (alles außer 'time_s')
    emg_channels = [c for c in df.columns if c != "time_s"]
    num_channels = len(emg_channels)
    
    if num_channels == 0:
        return False

    # Dynamische Bildhöhe: Je mehr Muskeln, desto höher das Bild
    fig_height = max(6, num_channels * 1.5)
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, fig_height), sharex=True)
    
    # Falls es nur einen Kanal gibt, machen wir axes zu einer Liste, damit der Code gleich bleibt
    if num_channels == 1:
        axes = [axes]

    # Jeden Kanal in einen eigenen kleinen Plot (Subplot) zeichnen
    for i, col in enumerate(emg_channels):
        ax = axes[i]
        ax.plot(df["time_s"], df[col], color='#1f77b4', linewidth=0.6)
        
        # Titel des Subplots = Name des Muskels
        ax.set_title(col, fontsize=9, loc='right', pad=2)
        ax.set_ylabel("Amp (V)", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Grid für bessere Lesbarkeit
        ax.grid(True, linestyle='--', alpha=0.5)

    # Gemeinsame X-Achse beschriften
    axes[-1].set_xlabel("Zeit (s)", fontsize=10)
    
    # Haupttitel des Bildes (z.B. S01 - 01_PER - CMJ - CMJ_01)
    trial_name = csv_path.stem
    subject = csv_path.parts[-5]
    phase = csv_path.parts[-4]
    exercise = csv_path.parts[-3]
    fig.suptitle(f"Rohdaten: {subject} | {phase} | {exercise} | {trial_name}", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Speicherpfad zusammenbauen (gleiche Ordnerstruktur wie bei den Daten)
    rel_path = csv_path.relative_to(DATA_DIR)
    save_path = PLOT_DIR / rel_path.parent / f"{trial_name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Bild speichern und Speicher wieder freigeben (SEHR WICHTIG bei 630 Bildern!)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    
    return True

def run_batch_plotter():
    if not DATA_DIR.exists():
        print(f"[FEHLER] Daten-Ordner nicht gefunden: {DATA_DIR}")
        return

    all_files = list(DATA_DIR.rglob("*.csv"))
    
    if TEST_MODE:
        # Nur Dateien von S01 filtern
        all_files = [f for f in all_files if "S01" in f.parts]
        print(f"TEST-MODUS AKTIV: Zeichne nur {len(all_files)} Dateien für S01...")
    else:
        print(f"Zeichne {len(all_files)} Dateien. Das kann ein paar Minuten dauern...")

    count = 0
    for i, file_path in enumerate(all_files):
        # Optional: Überspringen, falls das Bild schon existiert (spart Zeit beim Neustart)
        rel_path = file_path.relative_to(DATA_DIR)
        save_path = PLOT_DIR / rel_path.parent / f"{file_path.stem}.png"
        
        if save_path.exists():
            continue # Bild gibt es schon, gehe zum nächsten
            
        if plot_and_save_trial(file_path):
            count += 1
            if count % 10 == 0:
                print(f"  ... {count} Bilder generiert ({i+1}/{len(all_files)})")

    print(f"\n[OK] FERTIG! {count} neue Plots gespeichert unter:\n{PLOT_DIR}")

if __name__ == "__main__":
    run_batch_plotter()