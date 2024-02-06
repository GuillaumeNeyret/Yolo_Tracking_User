from ultralytics import YOLO
import cv2
import math 

import cam_manager
import time

USER_MIN_SIZE = 500

# start webcam
camera = cam_manager.Camera(index=1)
camera.init_cam()


# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


prev_frame_time = 0
image = None

while True:
    camera.read()
    results = model(camera.frame, stream=True, conf=0.8, verbose=False)
    print("-------------------NEW FRAME -----------------")
    # print(model.predictor)
    # print("----------- NEW FRAME -----------------")
    # print("Results Type")
    # print(type(results))
    print("Results")
    print(results)
    i = 0
    User_Max = (0,0)
    Presence = False

    # coordinates
    for r in results:
        boxes = r.boxes
        print(f"R{i} : ",r)
        i+=1
        # print("boxes :", boxes)
        for box in boxes:

            # class name
            cls = int(box.cls[0])

            if cls != 0 : 
                image = None
                continue

            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            # put box in camc:\Users\neyre\Desktop\RENO\RENO-MIMIC\reno-mimic\python-vision\user_track.py
            cv2.rectangle(camera.frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            # confidence = math.ceil((box.conf[0]*100))/100
            # # print("Confidence --->",confidence)
                
            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            # cv2.putText(camera.frame, classNames[cls], org, font, fontScale, color, thickness)

            x,y,w,h = box.xywh[0]

            if (w,h)>User_Max and  w>USER_MIN_SIZE and h >USER_MIN_SIZE:
                User_Max = (w,h)
                ROI = camera.frame[int(y1*0.95):int(y2*1.05), int(x1*0.95):int(x2*1.05)]
                Presence = True
                cv2.putText(camera.frame, "USER DETECTER", [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 4)

    if Presence:
            camera.frame = ROI


    # FPS Display
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(camera.frame, str(fps), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('DISPLAY',cv2.cvtColor(camera.frame, cv2.COLOR_RGB2BGR))


    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()