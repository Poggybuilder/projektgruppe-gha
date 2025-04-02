
# Projektgruppe zu Gaussian Head Avatars

## Gaussian Heads trainieren

Um das Projekt zu verwenden und einen Gaussian Head zu erstellen, müssen folgende Schritte durchgeführt werden:

1. `gha`-_conda_-Environment aufsetzen.
2. Die VCI-Daten in die richtige Dataset-Struktur überführen
3. Hintergründe entfernen
4. 3DMM-Fitting (benötigt andere Environment)
5. Mesh Head trainieren (hier scheiterte das Projekt immer)
6. Gaussian Head trainieren

Grundsätzlich wird immer davon ausgegangen, dass die verwendeten Datasets im jeweiligen `VCI/`-Ordner liegen. Es wird empfohlen, die darin liegenden Ordner durch Symlinks zu ersetzen, welche an die tatsächliche Speicherstelle verweisen.


### `gha`-Environment aufsetzen

Die folgenden Schritte müssen im `gaussian-head-avatar`-Ordner ausgeführt werden.

```bash
# Environment erstellen
conda env create -f environment.yaml

# Richtige CUDA-Version verwenden (Installationsort kann ggf. variieren)
conda env config vars set CUDA_HOME=/usr/local/cuda-11.8/

# Environment neustarten
conda deactivate
conda activate gha

# Pytorch3D und Kaolin installieren
pip install --no-index --no-cache-dir pytorch3d -f
https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1120/download.html
pip install kaolin==0.13.0 -f
https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.0_cu113.html
```

Es müssen zusätzlich Submodules aus dem [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)-Projekt importiert werden, um die Gaussians rendern zu können.
Hierfür muss das Projekt geklont und die Datei `submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h` so modifiziert werden, dass `NUM_CHANNELS 3` zu `NUM_CHANNELS 32` wird (sonst sind keine Multi-Channel-Farben möglich).

```bash
# Im GaussianSplatting-Ordner
# Nach Modifikation von submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```



### Vorverarbeitung der Daten

Aus dem `gaussian-head-avatar/`-Ordner kann das `preprocess/preprocess_vci.py`-Skript aufgerufen werden.
Dieses nimmt sämtliche Daten aus `VCI/preprocessing_input`, strukturiert sie um und speichert sie daraufhin in `VCI/preprocessing_output`.
Im Skript können außerdem einige Konfigurationen angepasst werden.


### Hintergründe entfernen

Die Hintergründe werden mit [Background Matting V2](https://github.com/PeterL1n/BackgroundMattingV2) entfernt.

Anweisungen dazu sind in der `README.md` im `BackgroundMattingV2`-Ordner zu finden.


### 3DMM Fitting

Das Fitting der 3DMM-Modelle geschieht durch [Multiview 3DMM Fitting](https://github.com/YuelangX/Multiview-3DMM-Fitting).

Anweisungen dazu sind in der `README.md` im `Multiview-3DMM-Fitting`-Ordner zu finden.


### Mesh Head und Gaussian Head trainieren

Beide Trainings sollten aus dem `gaussian-head-avatar/`-Ordner gestartet werden.

**Mesh Head** Training:
```bash
python train_meshhead.py --config config/train_meshhead_VCI.yaml
```

**Gaussian Head** Training:
```bash
python train_gaussianhead.py --config config/train_gaussianhead_VCI.yaml
```

Die Konfigurationsdateien liegen alle in `config/` und das Dataset in `VCI/preprocessing_output`.


## Reenactment

Der _Reenactment_-Schritt ähnelt dem _Gaussian Head_-Training sehr. 

In der Konfigurationsdatei `config/reenactment_VCI.yaml` können Gaussianhead, Source Actor, Kamerapfad, etc. angegeben werden.
Danach kann das Rendering der Szene gestartet werden:
```bash
python reenactment.py --config config/reenactment_VCI.yaml
```



## Helfer-Skripte

Im Verlauf der Projektgruppe waren einige selbstgeschriebene Skripte unabdingbar. Diese finden sich in `small_scripts/` wieder.

Erklärungen zu diesen sind in der README.md in `small_scripts/` zu finden.


## Referencen

Der Code wurde zu großen Teilen von [Gaussian Head Avatar](https://yuelangx.github.io/gaussianheadavatar/) übernommen.

