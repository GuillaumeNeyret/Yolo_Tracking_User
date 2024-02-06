import numpy, cv2

class Camera():
    """
    Camera Class

    Proprieties: 
    - width (int): Width of the camera image.
    - height (int) : Height of the camera. 
    - index (int) : Index of the camera device.
    - frame (np array): Last image captured by the cam (IN RGB)
    - cap (cv2.VideoCapture): Opencv VideoCapture object
    - Opened (bool) : Camera status
    """

    def __init__(self,width : int = 1920 , height :int = 1080,index : int= 0):
        """
        Initializes and creates Camera object.

        Args:
        - width (int): Width of the camera image. Default is 1920.
        - height (int) : Height of the camera. Default is 1080.
        - index (int) : Index of the camera device. Default is 0.

        Returns:
        - Nothing.

        Example:
        CameraDevice = Camera( width = 3840, height = 2160, index = 0)
        """
        print('Cam init ...')
        self.width = width
        self.height = height
        self.index = index
        self.frame = numpy.zeros((height,width,3))
        self.cap = None
        self.Opened = False

    def init_cam(self):
        """
        Set the Camera (cv2.VideoCapture) resolution and opened the Camera.

        Returns:
        - Nothing.

        Example:
        CameraDevice.init_cam()
        """
        self.cap = cv2.VideoCapture(self.index,cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.Opened = True   

    def read(self):
        """
        Open the video stream and Update the camera frame.
        Rq : The FRAME IN RGB !
        
        Returns:
        - Nothing.

        Example:
        CameraDevice.read()
        """
        ret,frame = self.cap.read()
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None

    def release(self):
        """
        Release the capture object.

        Returns:
        - Nothing.

        Example:
        CameraDevice.release()
        """
        self.cap.release()
        self.Opened = False

    def crop(self, crop_width : int,crop_height : int, center_x : int, center_y : int):
        """
        Crop the camera frame to (crop_width, crop_height) object.
        Center of this image is 

        Returns:
        - Nothing.

        Example:
        CameraDevice.read()
        """

        if center_y - (crop_height)//2 <0 or center_y + (crop_height)//2 > self.height or center_x- (crop_width)//2<0 or center_y + (crop_width)//2 > self.width or \
            center_x < 0 or center_y < 0 :
            raise ValueError("Crop dimension and/or center position incompatibles with camera resolution") 
    
        self.frame = self.frame[center_y - (crop_height)//2 : center_y + (crop_height)//2, center_x- (crop_width)//2 : center_x + (crop_width)//2]
