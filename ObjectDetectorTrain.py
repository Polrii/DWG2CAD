from ultralytics import YOLO


# Training from the start
"""
model = YOLO("yolov8n.pt")  # Load pre-trained YOLO model
model.train(data="C:/python_work/AI/DWG2CAD/TrainingData/NumsDataset.v6i.yolov8/data.yaml", epochs=1000000, save_period=100, patience=0)
"""


# Training from last checkpoint
""""""
model = YOLO("C:/python_work/AI/DWG2CAD/runs/detect/train4/weights/last.pt")  # Load last checkpoint
model.train(resume=True)
""""""
