# YOLOv12 + Intel RealSense Real-Time Object Detection

Real-time object detection with depth estimation using YOLOv12 and Intel RealSense D400 series cameras on Ubuntu 22.04 / ROS2 Humble.

## Features
- Real-time detection using YOLOv12n (turbo)
- Per-object depth reading from aligned RealSense depth stream
- Side-by-side color + depth colormap display
- Screenshot capture with `s` key
- Box color shifts green → red with distance

## Requirements
- Ubuntu 22.04
- CUDA 12.1 compatible GPU
- Intel RealSense D400 series camera
- Python 3.11

## Installation

```bash
# Create venv
python3.11 -m venv ~/yolov12_env
source ~/yolov12_env/bin/activate

# Install YOLOv12
git clone https://github.com/sunsmarterjie/yolov12 ~/yolov12
PYTHONPATH="" pip install -e ~/yolov12

# Install dependencies
PYTHONPATH="" pip install -r requirements.txt

# Download model weights
wget https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12n.pt -O ~/yolov12n.pt
```

## Usage

```bash
PYTHONPATH="" PYTHONNOUSERSITE=1 ~/yolov12_env/bin/python detect.py
```

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot |

## Controls
- Edit `detect.py` top config block to change model, confidence, resolution, or switch `cuda:0` → `cpu`

## Acknowledgements
- [YOLOv12](https://github.com/sunsmarterjie/yolov12) by Yunjie Tian et al.
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
