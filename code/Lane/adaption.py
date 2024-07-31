#You guys are gonna hate me for my naming conventions
#Takes the data points taken from reading.py (functions likely called from reading.py itself) and adapts it into Justin's orginal code 

#using selfDrive.py as a point of reference
import time
import numpy as np
import cv2
import serial
import matplotlib.pyplot as plt
import Jetson.GPIO as GPIO  # Change this if you use a different library
from reading import *

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
    #undertmined as to what it does yet
    return (angle / 180.0) * 10 + 2.5

def main():
    #
    selfDrvieAdapt()#follows what is done in reading.py

def send_data(command):
    ser.write(f"{command}\n".encode())
    ser.flush()

def selfDrvieAdapt():
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
    print("writing")
    snapString = 'NULL'
    model_name='lb2OO07.pt' #manual replace with our current model here 
    command = 's'
    #load model
    model = torch.hub.load('yolov5', 'custom', source='local', path = model_name, force_reload = True)
    
    videoPath = "0"
    #videoPath = "http://172.25.0.46:9001/camera.cgi" #remoting via vpn 
    outputDir = 'outputFrames'
    firstFrame = True
    #Opening with openCV
    capture = cv2.VideoCapture(videoPath)
    frame_count = 0
    leftLane = []
    rightLane = []
    #Processing each frame
    while capture.isOpened():
        #used chatgpt as a reference
        ret, frame = capture.read()
        if firstFrame:
            midX = int((frame.shape[1])/2)
            firstFrame = False
            laneCenter = midX
            scale = calcScale(midX)
        if not ret:
            break
        #Convert each frame into RBG
        rFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rFrame)
        results.print() #prints to terminal (optional)
        #results.save() #saves the image to an exp file (optional)
        #results.xyxy[0]  # im redictions (tensor) 
    
        df = pd.DataFrame(results.pandas().xyxy[0].sort_values("ymin")) #df = Data Frame, sorts x values left to right (not a perfect solution)
        df = df.reset_index() # make sure indexes pair with number of rows
        df.iterrows()
        polygonList = usingCSVData(df)
        polygonList = sortByDist(polygonList, scale) #is necessary
        margin = marginOfError(scale, laneCenter, midX)
        leftLane, rightLane = splitLaneByImg(polygonList, margin, scale)
        #print(polygonList, leftLane, rightLane)
        #print(scale)
        leftExist, rightExist, leftLane, rightLane = doesLeftOrRightExist(leftLane, rightLane, scale)
        laneCenter = findLaneCenter(leftLane, rightLane, 1000 * scale, midX, leftExist, rightExist, laneCenter)
        #print(laneCenter)
        newFrame = overlayimage(scale, leftLane, rightLane, laneCenter, frame)
        cv2.imshow("Final", newFrame)
        if cv2.waitKey(1) == ord('q'):#diplays the image for a set amount of time 
            break

        if frame_count % 7 == 0:
            
            if frame_count > 10:
                error = midX - laneCenter
                steering_adjustment = pid.update(error, 0.1/frame_rate)
                angle = 90 + (steering_adjustment * (-0.5)) 
                if leftExist or rightExist:
                    command = 'F'
                    print("Forward Sent")
                    send_data(command)
                else:
                    command = 'S'
                    print("stop Sent")
                    send_data(command)
                clip_angle = max(30, min(160, angle))
                if 30 <= clip_angle <= 160:
                    duty_cycle = angleToDutyCycle(clip_angle)
                    print(f'duty cycle: {duty_cycle}, clipped angle: {clip_angle}')
                    pwm.ChangeDutyCycle(duty_cycle)
                else:
                    duty_cycle = angleToDutyCycle(90.01)
                    pwm.ChangeDutyCycle(duty_cycle)
        frame_count += 1

    #Close
    capture.release()
    cv2.destroyAllWindows()
    pwm.stop()
    GPIO.cleanup()
    return 0


if __name__ == "__main__":
    # Open serial port
    try:
        ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        GPIO.setwarnings(False)
        time.sleep(2)  # wait for the serial connection to initialize
    except Exception as e:
        print(f"Could not open serial port: {e}")
        ser = None  # Ensure ser is defined even if the port couldn't be opened
    # Check CUDA availability
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        print("CUDA not available - the program requires a GPU with CUDA.")
        exit()  # Exit if CUDA is not available
    print("Serial port is connected and GPU is available")
    time.sleep(1)
    main()
    print("end")
    # Close serial if open
    if ser is not None:
        ser.close()
