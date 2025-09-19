#Open a new thread that handles a camera stream every time a new camera stream is needed 
#Uses code from: https://stackoverflow.com/questions/55099413/python-opencv-streaming-from-camera-multithreading-timestamps/55131226
#           and  https://stackoverflow.com/questions/58592291/how-to-capture-multiple-camera-streams-with-opencv

from threading import Thread
import cv2
import time 
from enum import Enum 
class cameraStreamWidget(object):
    def __init__(self, src, frameName):
        #inits object/Widget
        self.capture = cv2.VideoCapture(src)
        self.frameName = frameName 
        #Start the thread to begin reading frames from the video/camera stream
        self.thread = Thread(target=self.update,args=())
        self.thread.daemon = True
        self.thread.start() 

    def update(self):
        #Read the next frame from the stream in a different thread 
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(0.01) #Stops the above from being called too many times

    def show_frame(self):
        #Displays the current frame 
        cv2.imshow(self.frameName, self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def returnFrame(self):
        ret, sFrame = self.capture.retrieve()
        return sFrame
    
    def closeStream(self):
        self.capture.release()

class CameraNotation(Enum):
    CENTER = 0
    RIGHT = 1
    LEFT =  2

if __name__ == '__main__':
    camera_stream_widget = cameraStreamWidget()
    while True:
        try:
            camera_stream_widget.show_frame()
        except AttributeError:
            pass
