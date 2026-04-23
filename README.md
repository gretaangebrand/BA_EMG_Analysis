# BA_EMG_Analysis ⚡️

**Code-Repository zur Bachelorarbeit von Greta Angebrand**

Dieses Repository enthält den vollständigen Code zur Verarbeitung, Analyse und Visualisierung von Elektromyographie (EMG)-Daten im Rahmen meiner Bachelorarbeit zum Thema: *"Menstruationszyklus und neuromuskuläre Aktivierung: Eine EMG-Analyse ausgewählter Bein- und Hüftmuskeln während Return-to-Sport Übungen"*.

## 📖 Projektübersicht

Ziel dieses Projekts ist die systematische Auswertung von Roh-EMG-Signalen. Der Code umfasst die gesamte Pipeline von der Datenaufbereitung über die Signalfilterung bis hin zur Extraktion relevanter Merkmale (Features) und der statistischen Auswertung.

### Kernfunktionen (Pipeline)
* **Datenimport:** Einlesen der Rohdaten aus `[.csv / .txt]`-Dateien.
* **Vorverarbeitung & Filterung:** Bereinigung der Signale von Artefakten.
* **Signalverarbeitung:** Gleichrichtung und Glättung.
* **Merkmalsextraktion:** Berechnung von relevanten Parametern.
* **Visualisierung:** Erstellung von Plots zur Darstellung der Rohsignale bzw. der verarbeiteten Signale.

---

## 📂 Projektstruktur

```text
BA_EMG_Analysis/
│
├── data/               # Ordner mit Datensätzen (aus Datenschutzgründen nicht in Git vorhanden)
├── figures/            # Stickfigures für Visualisierungen
├── sripts/             # Quellcode / Hilfsfunktionen
├── outputs/            # Generierte Plots und Tabellen
├── requirements.txt    # Liste der benötigten Pakete/Bibliotheken
└── README.md           # Diese Datei