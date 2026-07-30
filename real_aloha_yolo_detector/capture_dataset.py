import cv2
import pyrealsense2 as rs
import numpy as np
import os

# Create folder to save images
save_dir = "dataset/images"
os.makedirs(save_dir, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_device("021222070323")
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

count = 0
print("[INFO] Press SPACE to capture image, Q to quit")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        cv2.imshow("Capture Dataset", image)

        key = cv2.waitKey(1)
        if key == ord(' '):
            filename = f"{save_dir}/image_{count:04d}.jpg"
            cv2.imwrite(filename, image)
            print(f"Saved: {filename}")
            count += 1
        elif key == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"[INFO] Captured {count} images")