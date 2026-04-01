from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PFAD ZUM PREPROCESSED-ORDNER
# ============================================================

DATA_DIR = Path(r"C:\Users\Greta\OneDrive\Desktop\MCI\3-SS2026\BA\BA_Daten_EMG\data\preprocessed_emg_data")


# ============================================================
# EINSTELLUNGEN
# ============================================================

PHASES = ["01_PER", "02_OVU", "03_LUT"]


# ============================================================
# ALLE SUBJECTS UND DATEIEN PRÜFEN
# ============================================================

def check_emg_file(filepath: Path) -> dict:
    """Prüft eine einzelne _emg.csv Datei und gibt eine Zusammenfassung zurück."""
    df = pd.read_csv(filepath, low_memory=False)

    # Spaltentypen klassifizieren
    event_cols  = [c for c in df.columns if c.startswith("event_")]
    scalar_cols = [c for c in df.columns if c != "time_s"
                   and not c.startswith("event_")
                   and not c.startswith("L_")
                   and not c.startswith("R_")]
    emg_cols    = [c for c in df.columns if c.startswith("L_") or c.startswith("R_")]
    l_cols      = [c for c in emg_cols if c.startswith("L_")]
    r_cols      = [c for c in emg_cols if c.startswith("R_")]

    # Muskelnamen extrahieren
    muscles_l = sorted(set(c.replace("L_", "") for c in l_cols))
    muscles_r = sorted(set(c.replace("R_", "") for c in r_cols))

    # Events lesen (nur Zeile 0)
    events = {}
    for c in event_cols:
        val = pd.to_numeric(df[c].iloc[0], errors="coerce")
        if pd.notna(val):
            events[c] = val

    # Scalars lesen (nur Zeile 0)
    scalars = {}
    for c in scalar_cols:
        val = pd.to_numeric(df[c].iloc[0], errors="coerce")
        if pd.notna(val):
            scalars[c] = val

    # Sampling-Rate schätzen
    if "time_s" in df.columns and len(df) > 1:
        dt = df["time_s"].diff().median()
        fs = round(1.0 / dt) if dt > 0 else 0
    else:
        fs = 0

    # EMG-Amplituden prüfen (Wertebereich → Einheit OK?)
    emg_max = {}
    for c in emg_cols:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals) > 0:
            emg_max[c] = vals.abs().max()

    return {
        "file": filepath.name,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "fs_hz": fs,
        "time_range": (df["time_s"].iloc[0], df["time_s"].iloc[-1]) if "time_s" in df.columns else (0, 0),
        "events": events,
        "scalars": scalars,
        "emg_cols": emg_cols,
        "muscles_l": muscles_l,
        "muscles_r": muscles_r,
        "emg_max": emg_max,
    }


def check_kin_file(filepath: Path) -> dict:
    """Prüft eine einzelne _kin.csv Datei und gibt eine Zusammenfassung zurück."""
    df = pd.read_csv(filepath, low_memory=False)

    event_cols  = [c for c in df.columns if c.startswith("event_")]
    kin_cols    = [c for c in df.columns if c != "time_s"
                  and not c.startswith("event_")
                  and not c.startswith("L_") and not c.startswith("R_")]
    # Kinematik-Spalten mit L/R Prefix
    lr_cols     = [c for c in df.columns if (c.startswith("Left ") or c.startswith("Right "))]
    all_kin     = kin_cols + lr_cols

    # Sampling-Rate
    if "time_s" in df.columns and len(df) > 1:
        dt = df["time_s"].diff().median()
        fs = round(1.0 / dt) if dt > 0 else 0
    else:
        fs = 0

    return {
        "file": filepath.name,
        "n_rows": len(df),
        "fs_hz": fs,
        "n_kin_cols": len(all_kin),
        "kin_cols_sample": all_kin[:10],
    }


# ============================================================
# HAUPTFUNKTION
# ============================================================

def main():
    print("=" * 70)
    print("01_b_check_csv_structure.py  –  Preprocessed Dateien prüfen")
    print("=" * 70)

    if not DATA_DIR.exists():
        print(f"[FEHLER] DATA_DIR nicht gefunden:\n  {DATA_DIR}")
        return

    subjects = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    print(f"\nGefundene Subjects: {len(subjects)}")
    print(f"  {', '.join(subjects)}\n")

    total_emg = 0
    total_kin = 0
    issues    = []

    for subject in subjects:
        print(f"\n{'─'*70}")
        print(f"{subject}")
        print(f"{'─'*70}")

        subject_dir = DATA_DIR / subject

        for phase in PHASES:
            phase_dir = subject_dir / phase
            if not phase_dir.exists():
                print(f"  {phase}: Ordner fehlt")
                continue

            # Alle Übungsordner durchgehen
            for ex_dir in sorted(phase_dir.iterdir()):
                if not ex_dir.is_dir():
                    continue
                for side_dir in sorted(ex_dir.iterdir()):
                    if not side_dir.is_dir():
                        continue

                    exercise = ex_dir.name
                    side     = side_dir.name

                    emg_files = sorted(side_dir.glob("*_emg.csv"))
                    kin_files = sorted(side_dir.glob("*_kin.csv"))

                    if not emg_files and not kin_files:
                        continue

                    print(f"\n  {phase} / {exercise} / {side}:")
                    print(f"    EMG-Dateien: {len(emg_files)}  |  KIN-Dateien: {len(kin_files)}")

                    # EMG-Dateien prüfen
                    for ef in emg_files:
                        total_emg += 1
                        info = check_emg_file(ef)

                        t0, t1 = info["time_range"]
                        duration = t1 - t0

                        print(f"\n    📄 {info['file']}")
                        print(f"       Zeilen: {info['n_rows']:,}  |  "
                              f"fs: {info['fs_hz']} Hz  |  "
                              f"Dauer: {duration:.3f} s  |  "
                              f"t: {t0:.3f}–{t1:.3f} s")
                        print(f"       EMG-Kanäle: {len(info['emg_cols'])}  "
                              f"(L: {len(info['muscles_l'])}, R: {len(info['muscles_r'])})")
                        print(f"       Muskeln R: {', '.join(info['muscles_r'])}")

                        # Events
                        if info["events"]:
                            evt_str = ", ".join(f"{k.replace('event_','').replace('_s','')}="
                                                f"{v:.3f}s" for k, v in info["events"].items())
                            print(f"       Events: {evt_str}")
                        else:
                            print(f"       Events: keine")

                        # Scalars
                        if info["scalars"]:
                            sc_str = ", ".join(f"{k}={v:.4f}" for k, v in info["scalars"].items())
                            print(f"       Scalars: {sc_str}")

                        # Einheiten-Check: EMG sollte im µV-Bereich sein (1–500 µV typisch)
                        for col, maxval in info["emg_max"].items():
                            if maxval < 0.01:
                                issues.append(f"{subject}/{phase}/{exercise}/{side}/{info['file']}: "
                                              f"{col} max={maxval:.6f} → evtl. noch in Volt?")
                            elif maxval > 10000:
                                issues.append(f"{subject}/{phase}/{exercise}/{side}/{info['file']}: "
                                              f"{col} max={maxval:.1f} → ungewöhnlich hoch")

                    # KIN-Dateien prüfen (kurzfassung)
                    for kf in kin_files:
                        total_kin += 1
                        kinfo = check_kin_file(kf)
                        print(f"\n    📄 {kinfo['file']}")
                        print(f"       Zeilen: {kinfo['n_rows']:,}  |  "
                              f"fs: {kinfo['fs_hz']} Hz  |  "
                              f"Kin-Spalten: {kinfo['n_kin_cols']}")
                        if kinfo["kin_cols_sample"]:
                            print(f"       Beispiele: {', '.join(kinfo['kin_cols_sample'][:5])}")

    # ── Zusammenfassung ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*70}")
    print(f"  Subjects           : {len(subjects)}")
    print(f"  EMG-Dateien geprüft: {total_emg}")
    print(f"  KIN-Dateien geprüft: {total_kin}")

    if issues:
        print(f"\n  ⚠️  MÖGLICHE PROBLEME ({len(issues)}):")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print(f"\n  ✅ Keine Auffälligkeiten gefunden.")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()