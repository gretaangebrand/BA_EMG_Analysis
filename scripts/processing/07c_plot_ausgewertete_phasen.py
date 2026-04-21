import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Zielordner
output_path = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\outputs\figures\plot_ausgewertete_phasen")

# Ordner erstellen, falls er noch nicht existiert
output_path.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 5))

exercises = ['Squat', 'DJ', 'CMJ']
y_pos = np.arange(len(exercises))
height = 0.4

# Farben exakt aus EVENT_CONFIG
c_start_end = '#888888'
c_takeoff = '#0072B2'
c_landing = '#D55E00'

highlight_color = '#D55E00' 
flight_color = '#f0f0f0'
gray_color = '#e0e0e0'

# Abstrakte Zeitpunkte (0 bis 1) für die Blöcke
cmj_t = [0.0, 0.4, 0.7, 1.0]
dj_t = [0.0, 0.2, 0.5, 0.8, 1.0]
sq_t = [0.0, 1.0]

# --- CMJ ---
y = 2
ax.barh(y, cmj_t[1]-cmj_t[0], left=cmj_t[0], height=height, color=gray_color, edgecolor='black')
ax.barh(y, cmj_t[2]-cmj_t[1], left=cmj_t[1], height=height, color=flight_color, edgecolor='black', hatch='//')
ax.barh(y, cmj_t[3]-cmj_t[2], left=cmj_t[2], height=height, color=highlight_color, edgecolor='black')

# CMJ Event Linien & Text
ax.plot([cmj_t[0], cmj_t[0]], [y-height/2, y+height/2], color=c_start_end, linestyle=':', lw=2)
ax.text(cmj_t[0], y+height/2 + 0.05, 'Start', ha='center', va='bottom', fontsize=9, color=c_start_end, fontweight='bold')

ax.plot([cmj_t[1], cmj_t[1]], [y-height/2, y+height/2], color=c_takeoff, linestyle='-', lw=2)
ax.text(cmj_t[1], y+height/2 + 0.05, 'Take-off', ha='center', va='bottom', fontsize=9, color=c_takeoff, fontweight='bold')

ax.plot([cmj_t[2], cmj_t[2]], [y-height/2, y+height/2], color=c_landing, linestyle='-', lw=2)
ax.text(cmj_t[2], y+height/2 + 0.05, 'Landing', ha='center', va='bottom', fontsize=9, color=c_landing, fontweight='bold')

ax.plot([cmj_t[3], cmj_t[3]], [y-height/2, y+height/2], color=c_start_end, linestyle=':', lw=2)
ax.text(cmj_t[3], y+height/2 + 0.05, 'End', ha='center', va='bottom', fontsize=9, color=c_start_end, fontweight='bold')

ax.text(cmj_t[2] + (cmj_t[3]-cmj_t[2])/2, y, 'Auswertung\n(Landung)', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# --- DJ ---
y = 1
ax.barh(y, dj_t[1]-dj_t[0], left=dj_t[0], height=height, color=flight_color, edgecolor='black', hatch='//')
ax.barh(y, dj_t[2]-dj_t[1], left=dj_t[1], height=height, color=gray_color, edgecolor='black')
ax.barh(y, dj_t[3]-dj_t[2], left=dj_t[2], height=height, color=flight_color, edgecolor='black', hatch='//')
ax.barh(y, dj_t[4]-dj_t[3], left=dj_t[3], height=height, color=highlight_color, edgecolor='black')

# DJ Event Linien & Text
ax.plot([dj_t[1], dj_t[1]], [y-height/2, y+height/2], color=c_landing, linestyle='-', lw=2)
ax.text(dj_t[1], y+height/2 + 0.05, 'Landing 1', ha='center', va='bottom', fontsize=9, color=c_landing, fontweight='bold')

ax.plot([dj_t[2], dj_t[2]], [y-height/2, y+height/2], color=c_takeoff, linestyle='-', lw=2)
ax.text(dj_t[2], y+height/2 + 0.05, 'Take-off', ha='center', va='bottom', fontsize=9, color=c_takeoff, fontweight='bold')

ax.plot([dj_t[3], dj_t[3]], [y-height/2, y+height/2], color=c_landing, linestyle='--', lw=2)
ax.text(dj_t[3], y+height/2 + 0.05, 'Landing 2', ha='center', va='bottom', fontsize=9, color=c_landing, fontweight='bold')

ax.plot([dj_t[4], dj_t[4]], [y-height/2, y+height/2], color=c_start_end, linestyle=':', lw=2)
ax.text(dj_t[4], y+height/2 + 0.05, 'End', ha='center', va='bottom', fontsize=9, color=c_start_end, fontweight='bold')

ax.text(dj_t[3] + (dj_t[4]-dj_t[3])/2, y, 'Auswertung\n(Landung 2)', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# --- Squat ---
y = 0
ax.barh(y, sq_t[1]-sq_t[0], left=sq_t[0], height=height, color=highlight_color, edgecolor='black')
ax.text(0.5, y, 'Auswertung', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# --- Formatierung ---
ax.set_yticks(y_pos)
ax.set_yticklabels(exercises, fontsize=11, fontweight='bold')
ax.set_xticks([])
ax.set_xlim(-0.05, 1.1)
ax.set_ylim(-0.5, 2.7)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.title('Übersicht der ausgewerteten Bewegungsabschnitte und Event-Markierungen', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

# Grafik speichern
plt.savefig(output_path / 'Auswertungsphasen_Events_Uebersicht.svg', format='svg', bbox_inches='tight')