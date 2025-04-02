
# 3DMM-Fitting

Das 3DMM-Fitting muss nach dem _preprocess_-Skript und vor dem _Mesh Head_-Training durchgeführt werden.
Es werden zuerst 2D-Landmarks in sämtlichen Frames erkannt und danach ein 3DMM-Model daran gefittet.

Es muss zuerst eine korrekte Conda-Environment aufgesetzt werden. Hierzu bitte den Instruktionen des [ursprünglichen Repositories](https://github.com/YuelangX/Multiview-3DMM-Fitting) folgen.

Konfigurationen können in Datei `config/VCI.yaml` vorgenommen werden.
Das Dataset muss in `VCI/preprocessing_output` vorliegen.
Visualisierungen (falls aktiviert) werden in `VCI/preprocessing_masks` gespeichert.

Unterstützte 3DMM-Modelle sind **BFM**, **FaceVerse** und **FLAME**, wenngleich **FLAME** stets Fehlermeldungen geworfen hat und ggf. Anpassungen notwendig sind.

Das Skript `vci_detect_and_fit.sh [Trainings-GPU-Nummer]` führt basierend auf der Konfig-Datei sämtliche Trainings durch.


## Referencen

Der Code wurde von [3DMM-Fitting](https://github.com/YuelangX/Multiview-3DMM-Fitting) übernommen.
