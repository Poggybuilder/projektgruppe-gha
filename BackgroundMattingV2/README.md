
# Entfernung des Hintergrundes

Das Background Matting muss nach dem _preprocess_-Skript und vor dem _Mesh Head_-Training durchgeführt werden.
Mithilfe der in `background` gegebenen Hintergrundbilder werden Masken für sämtliche Bilder in `images` erstellt und dort abgespeichert.

Als _conda_-Environment kann die `gha`-Environment aus dem _Gaussian Head Avatar_-Repo verwendet werden.
Vor Verwendung des Projektes muss [pytorch_resnet101.pth](https://drive.google.com/file/d/1zysR-jW6jydA2zkWfevxD1JpQHglKG1_/view?usp=drive_link) heruntergeladen und in den `model/`-Ordner gelegt werden.

Zur Entfernung der Hintergründe kann das `remove_background_vci.py`-Skript verwendet werden. Innerhalb des Skripts können die gewünschten Kameras und der Ort des Datasets angepasst werden.


## Referencen

Der Code wurde von [Background Matting V2](https://github.com/PeterL1n/BackgroundMattingV2) übernommen.
