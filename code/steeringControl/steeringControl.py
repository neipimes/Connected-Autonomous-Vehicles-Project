import Jetson.GPIO as GPIO

class Steering:
    def __init__(self, servoPin=33, servoPinFreq=50, lowerBound=20, upperBound=160):
        self.servoPin = servoPin
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.servoPin, GPIO.OUT)
        self.servoPWM = GPIO.PWM(self.servoPin, servoPinFreq)  # 50Hz frequency
        self.servoPWM.start(0)  # Neutral position

    def angleToDutyCycle(angle):
        return (angle / 180.0) * 10 + 2.5

    def setDutyCycle(self, dutyCycle):
        self.servoPWM.ChangeDutyCycle(dutyCycle)

    def setSteeringAngle(self, angle):
        clip_angle = max(self.lowerBound, min(self.upperBound, angle))
        dutyCycle = self.angleToDutyCycle(clip_angle)
        self.setDutyCycle(dutyCycle)

    def close(self):
        self.servoPWM.stop()
        GPIO.cleanup()