import cv2

from user_track import User_Track

import cam_manager
import time

USER_MIN_SIZE = (500,500)

user_tracking = User_Track("yolo-Weights/yolov8n.pt", limit=USER_MIN_SIZE)

# classNames = user_tracking.model.names
# print(f"TYPE model name  {type(classNames)}")
# print(f"model name  {classNames}")

# specific_object = ['person','chair','mouse']

# for obj in specific_object:
#     if obj not in classNames.values():
#         print(f"The object '{obj}' can be detected because not in pre-trained model.")

# start webcam
camera = cam_manager.Camera(index=0)
camera.init_cam()

prev_frame_time = 0
image = None

while True:
    camera.read()
    camera.frame = user_tracking.detect_user(camera.frame, stream=True, conf=0.8, verbose=False,crop=True, draw = True)
    # camera.frame = user_tracking.detect_all(camera.frame, stream=True, conf=0.5, verbose=False, specific_object=['chair','mouse'])

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