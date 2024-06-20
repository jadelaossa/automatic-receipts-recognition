from ultralytics import YOLO


# load pre-trained model
model = YOLO("yolov8n.pt")

# train the model
results = model.train(
    data="config.yaml",
    epochs=300,
    imgsz=640
)
