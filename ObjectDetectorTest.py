from ultralytics import YOLO

# Load the trained model
model = YOLO("C:/python_work/AI/DWG2CAD/runs/detect/train2/weights/best.pt")  # Or "last.pt" for the latest state


results = model("C:/python_work/AI/DWG2CAD/TrainingData/NumsDataset.v5i.yolov8/test/images/24-01-01_drawing_jpg.rf.1524bae88e98b40ba874f3371cd2f9f6.jpg")  # Run object detection

# Show the detected numbers
results[0].show()  # Displays the image with bounding boxes
