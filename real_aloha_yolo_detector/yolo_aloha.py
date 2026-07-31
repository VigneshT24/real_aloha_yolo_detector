from ultralytics import YOLOWorld
import cv2
import pyrealsense2 as rs
import numpy

model = YOLOWorld("yolov8x-worldv2.pt")
model.set_classes(["cell phone", "water bottle", "scissors", "coffee cup"])

pipeline = rs.pipeline()
config = rs.config()

config.enable_device("") # ENTER YOUR INTEL REALSENSE CAMERA SERIAL NUMBER


config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

print("[INFO] RealSense camera started. Please press Q to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = numpy.asanyarray(color_frame.get_data())
        depth_image = numpy.asanyarray(depth_frame.get_data())

        results = model(color_image, verbose=False, conf=0.40)

        annotated_frames = results[0].plot()

        cv2.imshow("ALOHA YOLO Detection", annotated_frames)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Camera stopped.")