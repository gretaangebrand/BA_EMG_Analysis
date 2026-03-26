import pandas as pd
import matplotlib.pyplot as plt

def plot_preprocessed_data(csv_path):
    print("Lese vorverarbeitete Daten ein...")
    
    # Daten einlesen (einfaches Einlesen reicht, da die Datei gut formatiert ist)
    df = pd.read_csv(csv_path)
    
    # Variablen definieren
    muscles_to_plot = [
        'L_Biceps Femoris',
        'L_Gastrocnemius medial',
        'L_Gluteus Medius',
        'L_Semitendinosus',
        'L_Vastus Lateralis'
    ]
    
    kinematics_to_plot = [
        'Left Hip Angles',
        'Left Knee Angles',
        'Left Ankle Angles'
    ]
    
    # Überprüfen, ob die Zeit-Spalte 'time_s' vorhanden ist
    if 'time_s' not in df.columns:
        print("Fehler: Konnte die Zeitachse ('time_s') nicht finden.")
        return

    # Figure mit 2 Subplots erstellen
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # --- PLOT 1: Alle 5 Muskeln ---
    for muscle in muscles_to_plot:
        if muscle in df.columns:
            # Wir plotten über die Zeit ('time_s')
            axes[0].plot(df['time_s'], df[muscle], label=muscle.replace("L_", ""), linewidth=1.5)
        else:
            print(f"Warnung: Muskel '{muscle}' nicht gefunden.")
            
    axes[0].set_title("EMG Daten - Alle 5 Muskeln (Linkes Bein, Preprocessed)", fontsize=14)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Zeit (s)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    # --- PLOT 2: Kinematik ---
    for joint in kinematics_to_plot:
        if joint in df.columns:
            axes[1].plot(df['time_s'], df[joint], label=f"{joint} (Sagittal)", linewidth=2)
        else:
            print(f"Warnung: Gelenkwinkel '{joint}' nicht gefunden.")
            
    axes[1].set_title("Kinematik Daten - Gelenkwinkel Sagittalebene", fontsize=14)
    axes[1].set_ylabel("Winkel (°)")
    axes[1].set_xlabel("Zeit (s)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    # Layout anpassen und anzeigen
    plt.tight_layout()
    plt.show()

# Dein Pfad zur vorverarbeiteten Datei
path = r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\preprocessed_emg_data\S07\02_OVU\DJ\BILATERAL\DJ_01.csv"

# Funktion aufrufen
plot_preprocessed_data(path)