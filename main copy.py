import cv2

from user_track import User_Track

import cam_manager
import time

USER_MIN_SIZE = (500,500)

user_tracking = User_Track("yolo-Weights/yolov8n.pt", limit=USER_MIN_SIZE)

# start webcam
camera = cam_manager.Camera(index=1)
camera.init_cam()

prev_frame_time = 0
image = None

while True:
    camera.read()
    camera.frame = user_tracking.detect(camera.frame, stream=True, conf=0.8, verbose=False)

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