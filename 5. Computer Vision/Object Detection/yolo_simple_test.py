from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.predict(source=0, show=True, stream=True, imgsz=256, vid_stride=3, device="mps")  # Use device="mps" for Mac with Apple Silicon