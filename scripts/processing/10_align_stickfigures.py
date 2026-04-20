"""
align_stickfigures.py
---------------------
Angleichung aller Strichmaennchen auf eine einheitliche Leinwand,
sodass alle Figuren gleich gross erscheinen und die Bodenlinie immer
an derselben Stelle liegt.

Vorgehen:
  1. Fuer jede Figur die Bounding-Box des nicht-transparenten Inhalts bestimmen.
  2. Die hoechste Figur (mit dem groessten BBox-Height) definiert die
     Leinwand-Hoehe -> alle anderen werden auf diese Hoehe gebracht.
  3. Jede Figur wird so ausgerichtet, dass die Unterkante ihrer
     Bounding-Box mit der Unterkante der Leinwand uebereinstimmt
     (Bodenlinie unten = Boden).
  4. Horizontal wird mittig platziert.

Die Skalierung der Figuren bleibt unveraendert - nur der transparente
Rand wird angepasst.
"""

from pathlib import Path
from PIL import Image
import numpy as np

# Input: die bereits transparent gemachten PNGs
SOURCE_DIR = Path("/mnt/user-data/outputs/stickfigures")
# Output: angeglichene Versionen
OUTPUT_DIR = Path("/mnt/user-data/outputs/stickfigures_aligned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_content_bbox(img_array):
    """Gibt (top, bottom, left, right) der nicht-transparenten Region zurueck."""
    alpha = img_array[:, :, 3]
    non_transparent = alpha > 10
    if not non_transparent.any():
        return None
    rows = np.where(non_transparent.any(axis=1))[0]
    cols = np.where(non_transparent.any(axis=0))[0]
    return rows[0], rows[-1], cols[0], cols[-1]


def main():
    # 1. Analyse: alle Bilder laden und BBox-Hoehen sammeln
    images = {}
    max_height = 0
    max_width = 0

    for png in sorted(SOURCE_DIR.glob("*.png")):
        img = np.array(Image.open(png).convert("RGBA"))
        bbox = get_content_bbox(img)
        if bbox is None:
            print(f"  [WARNUNG] {png.name}: komplett transparent, uebersprungen")
            continue
        top, bottom, left, right = bbox
        height = bottom - top + 1
        width  = right  - left + 1
        images[png.name] = (img, bbox)
        max_height = max(max_height, height)
        max_width  = max(max_width,  width)

    # 2. Leinwand-Groesse mit etwas Rand
    canvas_h = max_height + 20   # 10 px Rand oben und unten
    canvas_w = max_width  + 20   # 10 px Rand links und rechts

    print(f"\nEinheitliche Leinwand: {canvas_w} x {canvas_h} px")
    print(f"Groesste Figur: {max_width} x {max_height} px\n")

    # 3. Jedes Bild auf einheitliche Leinwand setzen (Boden unten)
    for name, (img, bbox) in images.items():
        top, bottom, left, right = bbox
        content = img[top:bottom+1, left:right+1]   # nur Inhalt ausschneiden

        # neue Leinwand: komplett transparent
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

        # Position: horizontal mittig, vertikal Bodenlinie bei canvas_h - 10
        h, w = content.shape[:2]
        x_offset = (canvas_w - w) // 2
        y_offset = canvas_h - 10 - h   # Unterkante 10 px ueber Canvas-Unterkante

        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = content

        out_path = OUTPUT_DIR / name
        Image.fromarray(canvas).save(out_path, "PNG")
        print(f"  [OK] {name}  ({w}x{h} -> {canvas_w}x{canvas_h})")

    print(f"\nFertig. Angeglichene Dateien: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
