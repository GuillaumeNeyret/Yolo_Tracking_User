import numpy, cv2
from ultralytics import YOLO
from typing import Tuple

class User_Track():
    """
    User Track Class

    Proprieties: 
    - model : Yolo model.
    - height (int) : Height of the camera. 
    - index (int) : Index of the camera device.
    - frame (np array): Last image captured by the cam (IN RGB)
    - cap (cv2.VideoCapture): Opencv VideoCapture object
    - presence (bool) : True if User detected
    """

    def __init__(self,model : str = "yolo-Weights/yolov8n.pt", limit : Tuple[int, int] = (500,500)):
        """
        Initializes and creates User Track object.

        Args:
        - model (str): Yolo model path.


        Returns:
        - Nothing.

        """
        self.model = YOLO(model)
        self.presence = False
        self.limit = limit

    def detect(self, image, stream = True, conf = 0.9, verbose = False, crop = True , draw = True):

        results = self.model(image, stream=stream, conf=conf, verbose=verbose)
        User_Max = (0,0)
        self.presence = False
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # class name
                cls = int(box.cls[0])

                if cls != 0 : 
                    continue

                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                _,_,w,h = box.xywh[0]
                if draw:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)

                if (w,h)>User_Max and  w>self.limit[0] and h >self.limit[1]:
                    User_Max = (w,h)
                    User_coord = (x1,y1,x2,y2)
                    ROI = image[int(y1*0.95):int(y2*1.05), int(x1*0.95):int(x2*1.05)]
                    self.presence = True

        if crop and  self.presence:
            x1,y1,x2,y2 = User_coord 
            return image[int(y1*0.95):int(y2*1.05), int(x1*0.95):int(x2*1.05)]
        
        return image