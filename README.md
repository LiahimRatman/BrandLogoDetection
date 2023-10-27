# Nike Logo Detection

<p float="left">
  <img src="https://github.com/LiahimRatman/BrandLogoDetection/blob/main/nike.gif" width="400" />
  <img src="https://github.com/LiahimRatman/BrandLogoDetection/blob/main/nike2.gif" width="400" /> 
</p>

This repository is an example of YOLOv8-based Brand Logo Detection model trained on Nike logo images.It contains of a Nike logo detection library as well as a dockerized API for photo detection.

# Dataset

Dataset with Nike logos consists with Instagram posts of official Nike accounts or posts with Nike hashtags. Collected data was annotated manually using Roboflow. The final dataset for training can be found here:
https://universe.roboflow.com/mikhail-korotkov/brand-logos-f5red

# Train 
You can train similar model by using train.py from "nike_detector.nike_detector.service.train" 

# Installation
To use Nike logo detection you need to install it using pip:
pip install nike_detection==0.0.18

# Detection
To run detection on your image:

from PIL import Image
from nike_detector.nike_detector.service.service import init_nike_model, predict

#initialize model
model = init_nike_model()
#load image
im = Image.open("your_image_name.jpg")
#run detection
results = predict(im)

# Track
Similar for tracking:

from nike_detector.nike_detector.service.service import init_nike_model, track

video = 'you_video_name.mp4'
#initialize model
model = init_nike_model()
#run tracking
results = track(video, False, False)

# Set up image detection API in Docker
You can also run this model in a Docker container. It is a basic version of FastApi with one endpoint "vectorize_batch" for detection. 
You can easily build this image using:run container using%
docker build -f Dockerfile -t you_image_name:v1.0.0 .

and run it with:
docker run -p 8021:8021 you_image_name:v1.0.0

