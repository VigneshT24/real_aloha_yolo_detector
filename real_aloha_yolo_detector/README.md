# Robotics & Explainable AI: Real-Time Object Detection & Spatial Referencing using Yolo for ALOHA Teleoperation System

Real-time object detection pipeline for the ALOHA bimanual robotic system, combining YOLOv12, Intel RealSense depth sensing, ArUco marker-based spatial referencing, and a stateful object-memory system.

## Features 
* YOLOv12 object detection with GPU acceleration via CUDA
* Intel RealSense D400 depth integration: aligned depth plus color streams with depth colormap overlay
* ArUco marker detection for spatial referencing and pose estimation (X/Y/Z axes)
* Object-memory (MEM lock) system: persists detected object locations through classifier dropout and bounding-box flickering
* Depth-based lock invalidation: automatically removes memory locks when an object physically moves
* Configurable detection zone: restricts YOLO inference to a defined pixel region
* Duplicate detection filtering: suppresses overlapping detections of the same object using grid-cell deduplication
* CLAHE contrast enhancement: improves detection accuracy under varying lighting conditions

## Hardware Requirements
* Intel RealSense D400 series depth camera
* NVIDIA GPU with CUDA 12.1+ support
* Ubuntu 22.04
* Python 3.11

## Installation
```bash
# Clone the repo
git clone https://github.com/your-username/real_aloha_yolo_detector.git
cd real_aloha_yolo_detector

# Create conda environment
conda create -n fusionvision python=3.11
conda activate fusionvision

# Install dependencies
pip install -r requirements.txt

# Clone and install YOLOv12
git clone https://github.com/sunsmarterjie/yolov12 yolov12
pip install -e yolov12

# Download YOLOv12x weights
wget https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12x.pt -O ~/yolov12x.pt
```

## Usage
```bash
conda activate fusionvision
python detect.py
```

| Key | Action |
| --- | ------ |
| 'q' |  QUIT  |
| 's' | SAVE SCREENSHOT |

## Configuration
At the top of ```detect.py```:
| Parameter | Default | Description |
| --- | ------ | ------------------ |
| ```MODEL_PATH``` |  ```~/yolov12x.pt```  | Path to YOLO weights |
| ```CONFIDENCE``` | ```0.25``` | Detection confidence threshold |
| ```DEVICE``` | ```cuda:0``` | 	Inference device |
| ```WIDTH/HEIGHT/FPS``` | ```640/480/30``` | 	RealSense stream configuration |
| ```LOCK_IMMEDIATELY``` | ```['cell_phone']``` | Classes to memory-lock instantly on first detection |

## Object Memory System 
The MEM lock system solves the bounding-box flickering problem common in real-time YOLO inference.

How it works:
1. **Detection** - YOLO detects an object and updates its position in memory every frame
2. **Locking** - after an object is stationary for ```hold_seconds``` (default 1.5s), its location is locked in memory. Classes in ```LOCK_IMMEDIATELY``` are locked on first detection
3. **Persistence** - once locked, the object's position is drawn every frame regardless of whether YOLO currently detects it (e.g., a phone with its screen off)
4. **Invalidation** - if the depth value at the remembered location changes by more than ```threshold``` metres (default 0.05m), the lock is removed, indicating the object has physically moved
5. **Stale cleanup** - objects not seen by YOLO for more than ```timeout_seconds``` (default 5s) are removed from memory

Visual Indicators:
| Color | Meaning |
| --- | ------ |
| Green to Red | Live YOLO detection (color shifts with depth) |
| Blue/Orange | MEM Locked Object |
| ```[MEM]``` label | Object currently visible AND locked |
| ```[LOCKED]``` label | Object in memory but not currently detected by YOLO |

## ArUco Marker Integration
The pipeline detects ArUco markers (DICT_4X4_1000) and uses them as spatial reference points.
* Draws detected marker outlines and IDs on the frame
* Estimates 6DoF pose (```rvec```, ```tvec```) using ```cv2.solvePnP``` with the camera intrinsics
* Draws X (red), Y (green), Z (blue) axes on each detected marker
* Marker size assumed: 5cm x 5cm. Update ```obj_points``` in ```detect.py``` if your markers differ

Camera intrinsics (update if using a different RealSense unit):
```python
K = np.array([[426.1166343, 0, 315.2250461],
              [0, 425.9295924, 205.3090416],
              [0, 0, 1.0]])
```

## Detection Zone
Only objects whose bounding box center falls within the defined pixel region are processed:
```python
# top-left: (113, 87), bottom-right: (593, 403)
if not (113 <= cx <= 593 and 87 <= cy <= 403):
    continue
```
Adjust these coordinates in ```detect.py``` to match your camera's field of view and workspace.

## Acknowledgements
* YOLOv12 by sunsmarterjie
* Ultralytics
* Intel RealSense SDK
* Akhil Joshi - original YOLOv12 RealSense detector
