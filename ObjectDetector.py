from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="C:/python_work/AI/DWG2CAD/TrainingData/NumsDataset.v5i.yolov8/data.yaml", epochs=500)
