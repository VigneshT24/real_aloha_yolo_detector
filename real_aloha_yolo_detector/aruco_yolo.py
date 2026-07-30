import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

model = YOLO("yolo11m.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_device("021222070323")
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

aruco_dicts = [
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_4X4_100,
    cv2.aruco.DICT_5X5_50,
    cv2.aruco.DICT_6X6_50,
]

def detect_aruco(gray, image, depth_image):
    all_corners = []
    for dict_type in aruco_dicts:
        dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            all_corners.extend(corners)
            cv2.aruco.drawDetectedMarkers(image, corners)

    board_center = None
    board_bbox = None

    if all_corners:
        centers = []
        all_points = []
        for corner in all_corners:
            cx = int(corner[0][:, 0].mean())
            cy = int(corner[0][:, 1].mean())
            centers.append((cx, cy))
            all_points.extend(corner[0].tolist())

        board_center = (
            int(np.mean([c[0] for c in centers])),
            int(np.mean([c[1] for c in centers]))
        )

        all_points = np.array(all_points)
        x_min = int(all_points[:, 0].min())
        x_max = int(all_points[:, 0].max())
        y_min = int(all_points[:, 1].min())
        y_max = int(all_points[:, 1].max())
        board_bbox = (x_min, y_min, x_max, y_max)

        cv2.circle(image, board_center, 8, (0, 0, 255), -1)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, "Board Center",
                   (board_center[0] + 10, board_center[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        depth_val = depth_image[board_center[1], board_center[0]] * 0.001
        cv2.putText(image, f"Depth: {depth_val:.2f}m",
                   (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return board_center, board_bbox

print("[INFO] YOLO + ArUco detection started. Press Q to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        # image = apply_canny_overlay(image)
        depth_image = np.asanyarray(depth_frame.get_data())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ArUco detection
        board_center, board_bbox = detect_aruco(gray, image, depth_image)

        # enhance contrast (fast)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        image_enhanced = cv2.merge([l, a, b])
        image_enhanced = cv2.cvtColor(image_enhanced, cv2.COLOR_LAB2BGR)

        # fast but accurate settings
        results = model.track(image_enhanced, verbose=False, conf=0.2, 
                      iou=0.45, imgsz=1280, persist=True, 
                      tracker="botsort.yaml")
        annotated = results[0].plot()

        # show object position relative to board
        if board_center is not None:
            for box in results[0].boxes:
                bx = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                by = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                rel_x = bx - board_center[0]
                rel_y = by - board_center[1]
                obj_depth = depth_image[by, bx] * 0.001
                label = results[0].names[int(box.cls)]
                cv2.putText(annotated,
                           f"{label} rel:({rel_x},{rel_y}) {obj_depth:.2f}m",
                           (bx, by - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # draw ArUco overlays on annotated frame too
        if board_center:
            cv2.circle(annotated, board_center, 8, (0, 0, 255), -1)
            if board_bbox:
                x_min, y_min, x_max, y_max = board_bbox
                cv2.rectangle(annotated, (x_min, y_min),
                             (x_max, y_max), (255, 0, 0), 2)

        cv2.imshow("YOLO + ArUco Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Done.")