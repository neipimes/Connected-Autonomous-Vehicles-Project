
import cv2
import pandas as pd
import sharedFunctions as sf
from laneMemory import laneMemory
from cameraWidget import * 
from cavErrors import *
import speed as sp 
#Enters in after a time limit in oneLaneState
#the aim is to reposition the CAV into seeing two lanes 



class correctionState:
    #initalise lane state 
    def __init__(self, laneState):
        self.laneState = laneState
        self.presistentMemory = laneMemory(False, False,[],[])
        self.idx = 0
        self.curStream = 0
        self.othStream = 0
        self.speed = "S13\n"
        # self.left = left #Left Lane exists: Boolean
        # self.right = right #Right Lane exists: Boolean
        # #Ideally one one should ever be true 

    def assignPresistentMemory(self, newMem):
        self.presistentMemory = newMem

    #change state to two lane state when both lanes are able to be detected
    def changeStateTwoLane(self):
        print("State changed to two lanes")
        self.idx = 0
        self.laneState.state =  self.laneState.twolanestate
        
    def changeStateTurning(self):
        print("Now entering turning state")
        self.idx = 0
        self.laneState.state = self.laneState.turningstate

    def getState(self):
        return "Correction State"
    def getSpeed(self):
        return self.speed
    #an unique proccess that continues to turn for a bit, but if it goes too long enter a search functionality
    def proccess(self, frame, scale, model, df, midX, laneCenter, newMemory, cameras):
        if self.idx == 0: 
            #First entered state 
            self.assignPresistentMemory(newMemory)
            if self.presistentMemory.leftExist == True: 
                self.curStream = CameraNotation.RIGHT.value 
                self.othStream = CameraNotation.LEFT.value
                print("Assigned Right Cam")
            else: 
                self.curStream = CameraNotation.LEFT.value
                self.othStream = CameraNotation.RIGHT.value
                print("Assigned Left Cam")
            self.idx = 1
        
        #CHECK MAIN CAMERA
        #OTHERWISE CHECK OTHER CAMERA
        polygonList, signList = sf.usingCSVData(df)
        margin = sf.marginOfError(scale, laneCenter, midX) #For if the centre of the lane is left or right favoured
        leftLane, rightLane = sf.splitLaneByImg(polygonList, margin, scale) #easiest way to split the list 
        newMemory = sf.doesLeftOrRightExist(leftLane, rightLane, scale, newMemory)
        laneCenter = sf.findLaneCenter(newMemory.leftLane, newMemory.rightLane, 900 * scale, midX, laneCenter)
        #Check main camera 
        
        if newMemory.leftExist == True and newMemory.rightExist == True:
            self.changeStateTwoLane() 
        elif (laneCenter <= 2*frame.shape[1]/8 or laneCenter >= 6*frame.shape[1]/8): #switches over after 15 detections and if the laneCenter is defined in the center of the screen 
            #makes sure turning state is correctly defined 
            leftLane, rightLane = self.defineList(leftLane + rightLane)
            newMemory = laneMemory(self.presistentMemory.leftExist, self.presistentMemory.rightExist, leftLane, rightLane)
            self.changeStateTurning()
            self.idx = 0
        else:
            nFrame = cameras[self.curStream].returnFrame() 
            if nFrame is not None: #if it exists 
                rFrame = cv2.cvtColor(nFrame, cv2.COLOR_BGR2RGB)

                results = model(rFrame)
                df2 = pd.DataFrame(results.pandas().xyxy[0].sort_values("ymin")) #df = Data Frame, sorts x values left to right (not a perfect solution)
                df2 = df2.reset_index() # make sure indexes pair with number of rows
                df2.iterrows()
                polygonList2 = sf.usingCSVData(df2)
                
                leftLane, rightLane = self.defineList(leftLane + rightLane)
                newMemory = laneMemory(self.presistentMemory.leftExist, self.presistentMemory.rightExist, leftLane, rightLane)
                print("LL: ", newMemory.leftExist, "RL: ", newMemory.rightExist)
                #if there are enough detections then turn normally otherwise turn the opposite direction
                if len(polygonList2) > 3: # enough detections
                    if self.presistentMemory.rightExist == True:
                        laneCenter = laneCenter/2 #maxed out left
                    else:
                        laneCenter = laneCenter + laneCenter/2 #maxed out right
                else: 
                    if self.presistentMemory.leftExist == True:
                        laneCenter = laneCenter - frame.shape[1]/4 #half left
                    else:
                        laneCenter = laneCenter + frame.shape[1]/4 #half right
                    #swap cameras 
                    self.swapStreams()
                sideFrame = sf.overlaySideImage(polygonList2, nFrame)
                cv2.imshow("side_cam", sideFrame)
            else: #raise an error message 
                CameraStreamError("Camera Stream is null")#raise error
        command = sp.calc_speed(newMemory.leftLane, newMemory.rightLane, scale)
        newFrame = sf.overlayimage(scale, newMemory.leftLane, newMemory.rightLane, laneCenter, frame)
        
        cv2.imshow("final", newFrame)
        print("CS INDEX", self.idx, "PRESISTANT ", self.presistentMemory.leftExist, " ", self.presistentMemory.rightExist)
        return laneCenter, newMemory,command

    def defineList(self, polygonList):
        leftLane = []
        rightLane = []
        if self.presistentMemory.leftExist == True:
            leftLane = polygonList
        elif self.presistentMemory.rightExist == True:
            rightLane = polygonList
        return leftLane, rightLane
    
    def swapStreams(self):
        dummy = self.othStream
        self.othStream = self.curStream
        self.curStream = dummy

  
    
def openSideStream(camera_stream):
    #opens the side camera stream as needed 
    print("opening side camera")
    capture = cv2.VideoCapture(camera_stream, cv2.CAP_GSTREAMER)
    #raise error 
    if not capture.isOpened():
        raise ValueError(f"Failed to open camera stream: {camera_stream}")
    return capture


    
