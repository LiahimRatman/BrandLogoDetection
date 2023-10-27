from roboflow import Roboflow
from ultralytics import YOLO


# load Roboflow dataset
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("mikhail-korotkov").project("brand-logos-f5red")
dataset = project.version(11).download("yolov8")

# Load a model
model = YOLO('yolov8x.yaml')  # build a new model from scratch
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='data.yaml', epochs=100, verbose=True)
