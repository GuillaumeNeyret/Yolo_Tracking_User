import cv2
import numpy as np 
import cam_manager

# Load YOLO model
net = cv2.dnn.readNet("assets/yolov3.weights", "assets/yolov3.cfg")

camera = cam_manager.Camera(index=0)
camera.init_cam()

# Get image dimensions
(height, width) = (camera.height,camera.width)

while True:
    camera.read()
    # Define the neural network input
    blob = cv2.dnn.blobFromImage(camera.frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

