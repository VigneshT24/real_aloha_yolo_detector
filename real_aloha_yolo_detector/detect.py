#!/usr/bin/env python3
"""
YOLOv12 and Intel RealSense: real-time object detection with depth
Usage: python detect.py
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
from ultralytics import YOLO

# configuration
MODEL_PATH = "" # ENTER YOUR YOLO MODEL PATH
CONFIDENCE   = 0.25
DEVICE       = "cuda:0"   # or cpu
WIDTH, HEIGHT, FPS = 640, 480, 30

# camera intrinsics
K = np.array([[426.1166343, 0, 315.2250461],
              [0, 425.9295924, 205.3090416],
              [0, 0, 1.0]])
DIST = np.zeros((5, 1))

LOCK_IMMEDIATELY = ["cell phone"]

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.adaptiveThreshWinSizeStep = 10
aruco_params.minMarkerPerimeterRate = 0.03
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX


class ObjectMemory:
    def __init__(self, hold_seconds=1.5, move_threshold=30):
        self.hold_seconds = hold_seconds
        self.move_threshold = move_threshold  # pixels
        self.memory = {}  # label: {cx, cy, depth, first_seen, locked, last_seen}

    def update(self, label, cx, cy, depth):
        now = time.time()
        key = label

        if key in self.memory:
            prev = self.memory[key]
            dist = ((cx - prev['cx']) ** 2 + (cy - prev['cy']) ** 2) ** 0.5

            if dist > self.move_threshold:
                # if object moved, then reset timer
                self.memory[key] = {
                    'cx': cx, 'cy': cy, 'depth': depth,
                    'first_seen': now, 'locked': False,
                    'last_seen': now
                }
            else:
                # if object stationary, check if held long enough
                if not prev['locked'] and (now - prev['first_seen']) >= self.hold_seconds:
                    self.memory[key]['locked'] = True
                
                # position smoothing
                self.memory[key]['cx'] = int(0.8 * prev['cx'] + 0.2 * cx)
                self.memory[key]['cy'] = int(0.8 * prev['cy'] + 0.2 * cy)
                self.memory[key]['depth'] = depth
                self.memory[key]['last_seen'] = now
        else:
            self.memory[key] = {
                'cx': cx, 'cy': cy, 'depth': depth,
                'first_seen': now, 'locked': False,
                'last_seen': now
            }

        return self.memory[key]

    def check_depth_changed(self, depth_image, label, threshold=0.05):
        """Check if depth at remembered location has changed significantly (in meters)."""
        if label not in self.memory or not self.memory[label]['locked']:
            return False

        cx = self.memory[label]['cx']
        cy = self.memory[label]['cy']
        stored_depth = self.memory[label]['depth']

        # sample current depth at remembered location
        current_depth = depth_image[cy, cx] * 0.001

        if current_depth == 0:
            return False  # no valid depth reading

        # if depth changed significantly, remove lock
        if abs(current_depth - stored_depth) > threshold:
            del self.memory[label]
            return True

        return False

    def get_locked(self):
        """Return all locked (remembered) objects."""
        return {k: v for k, v in self.memory.items() if v['locked']}

    def cleanup_stale(self, timeout_seconds=5.0):
        """Remove objects not detected for longer than timeout_seconds."""
        now = time.time()
        to_delete = [k for k, v in self.memory.items() if 'last_seen' in v and (now - v['last_seen']) > timeout_seconds]
        for key in to_delete:
            del self.memory[key]


def get_depth_at(depth_frame, x, y, sample=5):
    """Average depth over a small patch to reduce noise (in meters)."""
    depths = []
    for dx in range(-sample, sample + 1):
        for dy in range(-sample, sample + 1):
            d = depth_frame.get_distance(
                max(0, min(x + dx, WIDTH - 1)),
                max(0, min(y + dy, HEIGHT - 1))
            )
            if d > 0:
                depths.append(d)
    return round(sum(depths) / len(depths), 3) if depths else 0.0


def main():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    print("Model ready.\n")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device("") # ENTER YOUR INTEL REALSENSE CAMERA SERIAL NUMBER
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    profile = pipeline.start(config)

    # align depth to color frame
    align = rs.align(rs.stream.color)

    # depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale:.4f} m/unit")
    print("Press 'q' to quit, 's' to save screenshot.\n")

    obj_memory = ObjectMemory(hold_seconds=1.5, move_threshold=30)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # draw detection zone first
            cv2.rectangle(color_image, (113, 87), (593, 403), (255, 255, 255), 1)
            cv2.putText(color_image, "Detection Zone", (113, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # ArUco marker detection & pose estimation
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_enhanced = clahe.apply(gray)

            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            corners, ids, _ = detector.detectMarkers(gray_enhanced)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                obj_points = np.array([[-0.025,  0.025, 0],
                                       [ 0.025,  0.025, 0],
                                       [ 0.025, -0.025, 0],
                                       [-0.025, -0.025, 0]], dtype=np.float32)

                for i in range(len(ids)):
                    mid = ids[i][0]
                    cx = int(np.mean(corners[i][0][:, 0]))
                    cy = int(np.mean(corners[i][0][:, 1]))
                    obj_memory.update(f"aruco_{mid}", cx, cy, 0.0)

                    retval, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0], K, DIST)
                    cv2.drawFrameAxes(color_image, K, DIST, rvec, tvec, 0.05)

            # CLAHE enhancement for YOLO
            lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            color_image_enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

            # YOLO inference
            results = model(color_image_enhanced, conf=CONFIDENCE, verbose=False, augment=True)[0]
            num_detections = len(results.boxes)

            # grid deduplication
            seen_locations = {}
            for box in results.boxes:
                cx = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                cy = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                conf = float(box.conf[0])
                key = (cx // 50, cy // 50)
                if key not in seen_locations or conf > seen_locations[key][1]:
                    seen_locations[key] = (box, conf)

            filtered_boxes = [v[0] for v in seen_locations.values()]
            locked_items = set()

            for box in filtered_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # enforce detection zone
                if not (113 <= cx <= 593 and 87 <= cy <= 403):
                    continue

                depth_m = get_depth_at(depth_frame, cx, cy)
                state = obj_memory.update(label, cx, cy, depth_m)

                if label in LOCK_IMMEDIATELY and not state['locked']:
                    obj_memory.memory[label]['locked'] = True

                ratio = min(depth_m / 4.0, 1.0)
                color = (0, int(255 * (1 - ratio)), int(255 * ratio))

                if state['locked']:
                    # draw visible and locked object
                    mcx, mcy = state['cx'], state['cy']
                    cv2.circle(color_image, (mcx, mcy), 8, (255, 100, 0), -1)
                    cv2.putText(color_image, f"[MEM] {label} {depth_m:.2f}m",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 100, 0), 2)
                    locked_items.add(label)
                else:
                    # draw unlocked live object
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(color_image, f"{label} {conf:.2f} | {depth_m:.2f}m",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    cv2.circle(color_image, (cx, cy), 4, color, -1)

            # check for depth invalidation & cleanup
            for lbl in list(obj_memory.get_locked().keys()):
                obj_memory.check_depth_changed(depth_image, lbl, threshold=0.05)
            obj_memory.cleanup_stale(timeout_seconds=5.0)

            # draw memory-only objects
            for lbl, state in obj_memory.get_locked().items():
                if lbl not in locked_items and not lbl.startswith("aruco_"):
                    mcx, mcy = state['cx'], state['cy']
                    cv2.circle(color_image, (mcx, mcy), 10, (255, 100, 0), 2)
                    cv2.putText(color_image, f"[LOCKED] {lbl} {state['depth']:.2f}m",
                                (mcx - 40, mcy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

            # render depth colormap display
            depth_clipped = np.clip(depth_image, 0, 4000)
            depth_normalized = cv2.convertScaleAbs(depth_clipped, alpha=255.0 / 4000.0)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            display = np.hstack((color_image, depth_colormap))

            cv2.putText(display, f"Objects: {num_detections}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Device: {DEVICE}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            cv2.imshow("YOLOv12 + RealSense  [q=quit  s=save]", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"capture_{cv2.getTickCount()}.png"
                cv2.imwrite(fname, display)
                print(f"Saved: {fname}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("done.")


if __name__ == "__main__":
    main()
