
#Takes the data points taken from reading.py (functions likely called from reading.py itself) and adapts it into Justin's orginal code 

#using selfDrive.py as a point of reference
import logging.config
import time
import numpy as np
import cv2
import serial
import matplotlib.pyplot as plt
import Jetson.GPIO as GPIO  # Change this if you use a different library
from reading import *
import multiprocessing
from laneMemory import *
import sharedFunctions as sf
import logging 
import logging.config
from datetime import datetime
from input import keyboardListener
from distance_model import load_distance_models

class PIDController:
    #Ctrl + C  & Ctrl + V
    def __init__(self, kp, ki, kd,integral_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.last_error = 0
        self.integral_limit = integral_limit
    
    def update(self, error, dt):
        self.integral += error * dt
        # Clamp the integral to prevent windup
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        derivative = (error - self.last_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return output
    
def angleToDutyCycle(angle):
    return (angle / 180.0) * 10 + 2.5

def main(logger):
    #
    selfDrvieAdapt(logger)

def send_data(command):
    #sends data to serial port
    ser.write(f"{command}\n".encode())
    ser.flush()

def sendAngle(pwm, angle):
    #changes the angle the CAV is steering at
    pwm.ChangeDutyCycle(angle)


def drive(memory, midX, laneCenter, previousCommand,pid, frame_rate, commandQueue, angleQueue):
    previousCommand = command
    error = midX - laneCenter
    steering_adjustment = pid.update(error, 0.1/frame_rate)
    angle = 90 + (steering_adjustment * (-0.5)) 
    if memory.leftExist or memory.rightExist:
        command = 'F'
        print("Forward Sent")
    else:
        command = 'S'
        print("stop Sent")
    ##Creating processes 
    if(previousCommand != command): #to handle buffer
        commandQueue.put(command)
    clip_angle = max(20, min(160, angle))
    if 20 <= clip_angle <= 160: #change 30 and 160 to 20 and 160 respectively
        duty_cycle = angleToDutyCycle(clip_angle)
        print(f'duty cycle: {duty_cycle}, clipped angle: {clip_angle}')
    elif 0 <= clip_angle < 20: #new addition, covering cases that were generalised into else, hope this helps
        duty_cycle = angleToDutyCycle(20)  #left
        print("HARD LEFT -- Duty Cycle: {duty_cycle}  Clip Angle: {clip_angle}")
    elif 160 <= clip_angle <= 180: #right
        duty_cycle = angleToDutyCycle(160)
        print("HARD RIGHT-- Duty Cycle: {duty_cycle}  Clip Angle: {clip_angle}")
    else:
        duty_cycle = angleToDutyCycle(90.01)
    angleQueue.put(duty_cycle)

    return previousCommand


def commandSender(commandQueue):
    #https://stackoverflow.com/questions/29571671/basic-multiprocessing-with-while-loop
    while True:
        newVal = commandQueue.get() #Block until something is plaves on the queue
        if newVal == "END": #Terminate Queue on this condition
            break 
        send_data(newVal)
    return 
def angleSender(angleQueue, pwm):
    #https://stackoverflow.com/questions/29571671/basic-multiprocessing-with-while-loop
    while True:
        newVal = angleQueue.get() #Block until something is plaves on the queue
        if newVal == "END": #Terminate Queue on this condition
            break 
        sendAngle(pwm, newVal)
    return 
 
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
def selfDrvieAdapt(logger):
    #Define PID Controller 
    pid = PIDController(kp = 0.3, ki = 0.2, kd = 0.0002, integral_limit = 100)
    #
    GPIO.setwarnings(False)
    servoPin = 33
    GPIO.setmode(GPIO.BOARD)  # Use physical pin numberintrimmed.webmg
    GPIO.setup(servoPin, GPIO.OUT)
    pwm = GPIO.PWM(servoPin, 50)
    pwm.start(0) #Intialisation with 0% duty cycle 
    frame_rate = 30
    #Testing reading prediction output 
    #Takes the base line from writetoCSV and adapts selfDive.py over it 
    print("Starting...")
    snapString = 'NULL'
    cameras = []
    #init all streams 
    cameras.append(cameraStreamWidget("/dev/video2", "One"))
    cameras.append(cameraStreamWidget((gstreamer_pipeline(flip_method=0, sensor_id=0)), "Two"))
    cameras.append(cameraStreamWidget((gstreamer_pipeline(flip_method=0, sensor_id=1)), "Three"))
    model_name='/home/jetson/CAV-objectDetection/bestJul25.pt' #manual replace with our current model here 
    command = "s0\n"
    previousCommand = 0
    laneState = lc.laneController()
    #load model
    model = torch.hub.load('/home/jetson/CAV-objectDetection/yolov5', 'custom', source='local', path = model_name, force_reload = True)
    distance_predictor = load_distance_models()
    logger.info("Loaded TorchHub Model")
    ###### Multiprocessing Shenagans  -- https://stackoverflow.com/questions/29571671/basic-multiprocessing-with-while-loop
    #Create a manager
    manager = multiprocessing.Manager()
    #Data strcutres
    commandQueue = manager.Queue()
    angleQueue = manager.Queue()
    p1 = multiprocessing.Process(target=commandSender, args=(commandQueue, ))
    p2 = multiprocessing.Process(target=angleSender, args=(angleQueue, pwm, ))
    p1.start()
    p2.start()
    
    #videoPath = "/dev/video0"
    #videoPath = "http://172.25.0.46:9001/camera.cgi" #remoting via vpn 
    
    firstFrame = True
    #Opening with openCV
    #capture = cv2.VideoCapture(videoPath)
    frame_count = 0
    leftLane = []
    rightLane = []
    detections = 0
    #capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    #Processing each frame
    condition = True 
    # keyboard = keyboardListener()
    # keyboard.initKeyboard() 
    try:
        while condition:
            # ret, frame = capture.retrieve()
            # if not ret: 
            #     break #bad practice to have a break here, this however is the only remaining line from when I used chatgpt as a point of reference
            tO = time.time()
            frame = cameras[0].returnFrame()
            if firstFrame:
                midX = int((frame.shape[1])/2)
                firstFrame = False
                laneCenter = midX
                scale = sf.calcScale(midX)
                newMemory = laneMemory(False,False,[],[])
            
            #Convert each frame into RBG
            print("State: ", laneState.getState())
            rFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rFrame)
            #results.print() #prints to terminal (optional)
            #results.save() #saves the image to an exp file (optional)
            #results.xyxy[0]  # im redictions (tensor) 
        
            df = pd.DataFrame(results.pandas().xyxy[0].sort_values("ymin")) #df = Data Frame, sorts x values left to right (not a perfect solution)
            df = df.reset_index() # make sure indexes pair with number of rows
            df.iterrows()
            polygonList, signList = sf.usingCSVData(df)
            counts, xedges, yedges = np.histogram2d(sf.convertToYList(polygonList), sf.convertToXList(polygonList), bins=540)
            counts_img = cv2.normalize(counts, None, 0, 255, cv2.NORM_MINMAX)
            counts_img = counts_img.astype(np.uint8)
            cv2.imshow('counts', counts_img)
            # Hough Line Transform
            lines = cv2.HoughLinesP(counts_img, 1, np.pi/180, 68, minLineLength=10, maxLineGap=250)
            if distance_predictor:
                results = distance_predictor.predict_batch(signList)
                print(f"Distance Predictions:  {results}")
            #print(lines)

            #if lines:
            #    for line in lines: 
            #        x1, y1, x2, y2 = line[0]
            #        cv2.line(rFrame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            #cv2.imshow('lines', rFrame)
            
            laneCenter, newMemory, commandFloat = laneState.proccess(frame, scale, model, df, midX, laneCenter, newMemory, cameras)
            if cv2.waitKey(1) == ord('q'):#diplays the image  a set amount of time 
                break
            frame_count += 1
            if frame_count > 10:
                
                error = midX - laneCenter
                steering_adjustment = pid.update(error, 0.1/frame_rate)
                angle = 90 + (steering_adjustment * (-0.5)) 
                if newMemory.leftExist or newMemory.rightExist:
                    #range is 0 - 100
                    command = "S" + str(commandFloat) + "\n"
                    print("Forward Sent - ", command)
                else:
                    command = "S0\n"
                    commandFloat = 0 
                    print("Stop Sent - 0")
                ##Creating processes 
                if(round(previousCommand, 1) != round(commandFloat,1)): #to handle buffer
                    logger.info(f"Sent {command} to ardino")
                    commandQueue.put(command)
                previousCommand = commandFloat  
                clip_angle = max(20, min(160, angle))
                if 20 <= clip_angle <= 160: #change 30 and 160 to 20 and 160 respectively
                    duty_cycle = angleToDutyCycle(clip_angle)
                    print(f'duty cycle: {duty_cycle}, clipped angle: {clip_angle}')
                elif 0 <= clip_angle < 20: #new addition, covering cases that were generalised into else, hope this helps
                    duty_cycle = angleToDutyCycle(20)  #left
                    print("HARD LEFT -- Duty Cycle: {duty_cycle}  Clip Angle: {clip_angle}")
                elif 160 <= clip_angle <= 180: #right
                    duty_cycle = angleToDutyCycle(160)
                    print("HARD RIGHT-- Duty Cycle: {duty_cycle}  Clip Angle: {clip_angle}")
                else:
                    duty_cycle = angleToDutyCycle(90.01)
                angleQueue.put(duty_cycle)   
            #Handling user input
            # userInput = keyboard.getLastKey()
            # if (userInput == 'q') : #exit condition
            #     print("User entered termination condition") 
            #     condition = False
            t2 = time.time()
            dt = tO - t2
            print(f"Time Elasped: {dt}")
    except Exception as e: #neccesary to ensure cameras are turned off properly otherwise the CAV will need to be reset
        print("Immediate stop of function: ", e)
        logger.error("Immediate stop of function: ", e)
    #Close and release
    # angle = 160
    # clip_angle = max(20, min(160, angle))
    # if 20 <= clip_angle <= 160: #change 30 and 160 to 20 and 160 respectively
    #     duty_cycle = angleToDutyCycle(clip_angle)
    #     print(f'duty cycle: {duty_cycle}, clipped angle: {clip_angle}')
    # else:
    #     duty_cycle = angleToDutyCycle(90.01)
    # angleQueue.put(duty_cycle)
    # time.sleep(5)
    commandQueue.put("END")
    angleQueue.put("END")
    p1.join()
    logger.info("P1 ENDED") 
    p2.join()
    logger.info("P2 ENDED") 
    send_data("S0\n")
    logger.info("Sent S0") 
    #capture.release()
    for cam in cameras: 
        cam.closeStream() 
        logger.info("Cam closed stream") 
    cv2.destroyAllWindows()
    pwm.stop() 
    logger.info("PWM Stopped") 
    GPIO.cleanup()
    logger.info("GPIO cleaned up") 
    # print("Press 'q' to end.")
    # keyboard.endKeyboard()
    # logger.info("Closed Keyboard")
    return 0

#creating and configure a logger
currentTime = datetime.now()
formatedTime = currentTime.strftime("%Y-%m-%d_%H:%M:%S")
logging.basicConfig(filename="cav_lanekeeping_"+formatedTime+".log",
                    format='%(asctime)s %(message)s',
                    filemode='a')

#Creating a logging object
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger()

# Set the threshold of logger to DEBUG
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    # Open serial port
    try:
        ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        logger.info("Connected to serial port /dev/ttyACM0")
        GPIO.setwarnings(False)
        time.sleep(2)  # wait for the serial connection to initialize
    except Exception as e:
        print("Could not open serial port: ", e)
        logger.error("Could not open serial port: ", e)
        ser = None  # Ensure ser is defined even if the port couldn't be opened
    # Check CUDA availability
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        print("CUDA not available - the program requires a GPU with CUDA.")
        logger.error("CUDA not available - the program requires a GPU with CUDA.")
        exit()  # Exit if CUDA is not available
    print("Serial port is connected and GPU is available")
    logger.info("Serial port is connected and GPU is available")
    time.sleep(1)
    main(logger)
    print("end")
    # Close serial if open
    if ser is not None:
        ser.close()
