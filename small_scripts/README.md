
# Helfer-Skripte


## NeRSemble_to_VCI

Das Skript wurde verwendet, um _NeRSemble_-Daten inklusiver Kameraparameter in das Format der _VCI_-Daten inklusive `calibration_dome.json` umzuwandeln.

Die _NeRSemble_-Videos müssen in `input/` liegen und die Kamera-Parameter in `camera_params/`.
Nach der Transformierung liegen die `calibration_dome.json` in `camera_params/` und die Bilder in `output/`.

Die `gha`-Environment ist ausreichend, um das Skript auszuführen.


## Visualizer

In diesem Ordner befinden sich einige Skripte, welche verwendet wurden, um Teile des Projektes zu visualisieren.
Um die Skripte zu verwenden, muss eine Environment anhand der dort gegebenen `requirements.txt`-Datei aufgesetzt werden.

Grundsätzlich müssen alle Input-Daten in `input/` zur Verfügung gestellt werden und die Visualisierungen werden in `output/` gespeichert.

Sämtliche Skripte nutzen die `argparse`-Bibliothek, um Argumente zu übergeben, sodass nähere Erläuterungen durch Angabe von `--help` erhalten werden können.

