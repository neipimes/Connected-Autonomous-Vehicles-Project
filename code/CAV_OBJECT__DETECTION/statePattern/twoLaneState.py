import cv2
import pandas as pd
import sharedFunctions as sf
import speed as sp 

class twoLaneState:
    #Init State
    def __init__(self, laneState):
        self.laneState = laneState
        # self.left = left #Left Lane exists: Boolean
        # self.right = right #Right Lane exists: Boolean
        #BOTH SHOULD BE TRUE 
        self.speed = "S15\n"

    #Change state when only one lane is being detected
    def changeState(self):
        print("State changed to one lane")
        self.laneState.state =  self.laneState.correctionstate
    
    def changeStateTurning(self):
        print("Now entering turning state")
        self.idx = 0
        self.laneState.state = self.laneState.turningstate

    def getState(self):
        return "Two Lane State"
    def getSpeed(self):
        return self.speed
    #Follows the original process 
    def proccess(self, frame, scale, model, df, midX, laneCenter, newMemory, cameras):
        oldLaneCenter = laneCenter
        polygonList, signList = sf.usingCSVData(df)
        polygonList = sf.sortByDist(polygonList, scale) #Gets rid of outliers
        margin = sf.marginOfError(scale, laneCenter, midX) #For if the centre of the lane is left or right favoured
        leftLane, rightLane = sf.splitLaneByImg(polygonList, margin, scale) #easiest way to split the list 
        newMemory = sf.doesLeftOrRightExist(leftLane, rightLane, scale, newMemory)
        laneCenter = sf.findLaneCenter(newMemory.leftLane, newMemory.rightLane, 900 * scale, midX, laneCenter)
        command = sp.calc_speed(newMemory.leftLane, newMemory.rightLane, scale)
        newFrame = sf.overlayimage(scale, newMemory.leftLane, newMemory.rightLane, laneCenter, frame)
        cv2.imshow("final", newFrame)
        if newMemory.leftExist == False or newMemory.rightExist == False:
            if (laneCenter <= 2*frame.shape[1]/8 or laneCenter >= 6*frame.shape[1]/8):
                self.changeStateTurning()
            else:
                self.changeState()
        return laneCenter, newMemory, command
    

    def _checkCenterAccuracy(laneCenter, oldLaneCenter):
        
        #confirm that the center point detected is accurate with previous lane centers and the state it is
        
        #Check one frame data 
        
        #How will we confirm that the lane center is good ? 
        """
        Lane Center should be:
            - in between the bounds of the lane detection ? 
                - both lanes should be detected 
            - Within the lane center of the previous detection ?  
        """
        #What should we do if lane center detected is rejected?? 
        
        pass

    def betterSort(self, leftLane, rightLane):
        #iterate through list and ensure correct placement 
        looping = True 
        
        idx = 0
        thisList = []
        
        totalList = (leftLane + rightLane)
        thisList.append(totalList[0])
        x = totalList[idx]
        while idx < len(totalList)  - 1: 
            idx = idx+ 1 
            y = totalList[idx] 
            if sf.getDist(x,y) < 40:
                thisList.append(y)
                if y in rightLane:
                    rightLane.remove(y)
                x = y
        return thisList, rightLane


