#! /bin/sh

# Export GPU
export CUDA_VISIBLE_DEVICES=$1
echo ">>>> CUDA_VISIBLE_DEVICES = \"$CUDA_VISIBLE_DEVICES\""


# Detect all Landmarks
echo "==== detect_landmarks.py"
python detect_landmarks.py --config config/VCI.yaml


# Fit the 3DMM-Model
echo "==== fitting.py"
PYOPENGL_PLATFORM=egl python fitting.py --config config/VCI.yaml

echo "==== DONE"
