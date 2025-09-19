from cameraWidget import cameraStreamWidget
from reading import *

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
def main():
    # p1 = multiprocessing.Process(target=writeToCSV, args=())
    # p2 = multiprocessing.Process(target=test, args=())
    # # writeToCSV()
    # # test()
    # p2.start()
    # #p2.start()
    # p2.join
    #p2.join
    # cameras = []
    # cameras.append(cameraStreamWidget("/dev/video2", "One"))
    # cameras.append(cameraStreamWidget((gstreamer_pipeline(flip_method=0, sensor_id=0)), "Two"))
    # cameras.append(cameraStreamWidget((gstreamer_pipeline(flip_method=0, sensor_id=1)), "Three"))
    # try: 
    #     while True:
    #         try:
    #             for cam in cameras:
    #                 cam.show_frame()
    #             if cv2.waitKey(1) == ord('q'):#diplays the image  a set amount of time 
    #                 break
    #         except AttributeError:
    #             pass
    # except KeyboardInterrupt:
    #     pass
    #cameras.append(cameraStreamWidget())
    processEachFrame()
    # #print("Completed")
    # for cam in cameras:
    #     cam.closeStream() 
    print("Done")
main()

