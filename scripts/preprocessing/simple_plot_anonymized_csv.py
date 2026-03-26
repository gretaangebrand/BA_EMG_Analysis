import pandas as pd
import matplotlib.pyplot as plt

def plot_all_muscles_and_kinematics(csv_path):
    print("Lese Daten ein...")
    
    # 1. Header lesen
    with open(csv_path, "r", encoding="utf-8") as f:
        file_names = f.readline().strip().split(",")  
        var_names = f.readline().strip().split(",")   
        f.readline()  # var_types (überspringen)
        f.readline()  # var_groups (überspringen)
        coords = f.readline().strip().split(",")      
        
    # 2. Daten einlesen (ab Zeile 6)
    df = pd.read_csv(csv_path, skiprows=5, header=None)
    
    # 3. Variablen definieren, die wir plotten wollen
    muscles_to_plot = [
        'R_Biceps Femoris',
        'R_Gastrocnemius medial',
        'R_Gluteus Medius',
        'R_Semitendinosus',
        'R_Vastus Lateralis'
    ]
    
    # Wir nehmen hier die Gelenkwinkel für das rechts Bein als Zusatz-Plot
    kinematics_to_plot = [
        'Right Hip Angles',
        'Right Knee Angles',
        'Right Ankle Angles'
    ]
    
    muscle_cols = {}
    kinematic_cols = {}
    
    # 4. Spaltenindizes suchen (für den ersten Sprung in der Datei, X-Achse)
    for i in range(len(var_names)):
        var = var_names[i]
        coord = coords[i]
        
        # Muskeln speichern (nur die erste gefundene X-Spalte)
        if var in muscles_to_plot and coord == "X" and var not in muscle_cols:
            muscle_cols[var] = i
            
        # Kinematik speichern (nur die erste gefundene X-Spalte)
        if var in kinematics_to_plot and coord == "X" and var not in kinematic_cols:
            kinematic_cols[var] = i

    # 5. Figure mit 2 Subplots erstellen (etwas größer für bessere Lesbarkeit)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # --- PLOT 1: Alle 5 Muskeln ---
    # Wir iterieren durch unser Dictionary und plotten jeden Muskel
    for muscle, col_idx in muscle_cols.items():
        emg_data = df.iloc[:, col_idx].dropna()
        axes[0].plot(emg_data, label=muscle.replace("R_", ""), linewidth=1.5)
        
    axes[0].set_title("EMG Daten - Alle 5 Muskeln (Rechtes Bein, Sprung 1)", fontsize=14)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Frames")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    # Legende rechts neben den Plot setzen, damit sie keine Daten verdeckt
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    # --- PLOT 2: Kinematik ---
    for joint, col_idx in kinematic_cols.items():
        kin_data = df.iloc[:, col_idx].dropna()
        axes[1].plot(kin_data, label=f"{joint} (X-Achse)", linewidth=2)
        
    axes[1].set_title("Kinematik Daten - verschiedene Gelenkwinkel (Sprung 1)", fontsize=14)
    axes[1].set_ylabel("Winkel (°)")
    axes[1].set_xlabel("Frames (Kinematik Abtastrate)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    # Layout anpassen (wichtig, damit die Legenden nicht abgeschnitten werden)
    plt.tight_layout()
    plt.show()

# Dein Pfad
path = r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\anonymized_csv_data\01_PER\CMJ\S07_01_PER_CMJ.csv"

# Funktion aufrufen
plot_all_muscles_and_kinematics(path)