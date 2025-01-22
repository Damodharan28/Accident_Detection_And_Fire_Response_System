from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")

model.track(source='testing.mp4', show=True)