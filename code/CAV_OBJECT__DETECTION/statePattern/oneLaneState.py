import cv2
import pandas as pd
import sharedFunctions as sf
from laneMemory import laneMemory
import speed as sp
"""
USED TO CORRECT LANE DETECTION, WANTS TO REENTER TWO LANE STATE
"""
class oneLaneState:
    #initalise lane state 
    def __init__(self, laneState):
        self.laneState = laneState
        self.presistentMemory = laneMemory(False, False,[],[])
        self.idx = 0
        self.speed = "S14\n"
        # self.left = left #Left Lane exists: Boolean
        # self.right = right #Right Lane exists: Boolean
        # #Ideally one one should ever be true 
    
    def assignPresistentMemory(self, newMem):
        self.presistentMemory = newMem

    #change state to two lane state when both lanes are able to be detected
    def changeStateTwoLane(self):
        print("State changed to two lanes")
        self.idx = 0
        #self.assignPresistentMemory(laneMemory(False,False,[],[]))
        self.laneState.state =  self.laneState.twolanestate
    
    def changeStateCorrection(self):
        print("State changed to correction state")
        self.idx = 0
        #self.assignPresistentMemory(laneMemory(False,False,[],[]))
        self.laneState.state = self.laneState.correctionstate

    def changeStateTurning(self):
        print("Now entering turning state")
        self.idx = 0
        self.laneState.state = self.laneState.turningstate

    def getState(self):
        return 1
    
    def getSpeed(self):
        return self.speed
    
    #an unique proccess that continues to turn for a bit, but if it goes too long enter a search functionality
    def proccess(self, frame, scale, model, df, midX, laneCenter, newMemory, cameras):
        if self.idx == 0: 
            #First entered state 
            print("ENTERED ONE LANE REASSIGNMENT")
            self.assignPresistentMemory(laneMemory(False,False,[],[]))
            self.idx = 1
            self.assignPresistentMemory(newMemory)
        polygonList, signList = sf.usingCSVData(df)
        margin = sf.marginOfError(scale, laneCenter, midX) #For if the centre of the lane is left or right favoured
        #TODO FIXERROR HERE
        #print("pl", polygonList, "margin", margin, "sclae", scale)
        leftLane, rightLane = sf.splitLaneByImg(polygonList, margin, scale) #easiest way to split the list 
        newMemory = sf.doesLeftOrRightExist(leftLane, rightLane, scale, newMemory)
        if newMemory.leftExist == True and newMemory.rightExist == True: #two lane exit
            self.changeStateTwoLane() 
        elif (laneCenter <= 2*frame.shape[1]/8 or laneCenter >= 6*frame.shape[1]/8): #switches over after 15 detections and if the laneCenter is defined in the center of the screen 
            #makes sure turning state is correctly defined 
            leftLane, rightLane = self.defineList(leftLane + rightLane)
            newMemory = laneMemory(self.presistentMemory.leftExist, self.presistentMemory.rightExist, leftLane, rightLane)
            self.changeStateTurning()
            self.idx = 0
        else:
            leftLane, rightLane = self.defineList(leftLane + rightLane)
            #print("LL: ", newMemory.leftExist, leftLane, "RL: ", newMemory.rightExist, rightLane)
            newMemory = laneMemory(self.presistentMemory.leftExist, self.presistentMemory.rightExist, leftLane, rightLane)
            self.idx = self.idx + 1
        laneCenter = sf.findLaneCenter(newMemory.leftLane, newMemory.rightLane, 900 * scale, midX, laneCenter)
        command = sp.calc_speed(newMemory.leftLane, newMemory.rightLane, scale)
        newFrame = sf.overlayimage(scale, newMemory.leftLane, newMemory.rightLane, laneCenter, frame) 
        rightFrame = cameras[1].returnFrame()  # one = right, 2 = left
        leftFrame = cameras[2].returnFrame()
        if (rightFrame is not None and leftFrame is not None) : #if it exists 
            rPL = sf.getPolygonList(rightFrame, model) 
            lPL = sf.getPolygonList(leftFrame, model)
            laneCenter = compareRightCamAndLeftCam(rPL, lPL, laneCenter, frame.shape[1])
            rightFrame = sf.overlaySideImage(rPL, rightFrame)
            leftFrame = sf.overlaySideImage(lPL, leftFrame)
            cv2.imshow("right_cam", rightFrame)
            cv2.imshow("left_cam", leftFrame)
        cv2.imshow("final", newFrame)
        #print("OLS INDEX ", self.idx, "PRESISTANT ", self.presistentMemory.leftExist, " ", self.presistentMemory.rightExist)
        return laneCenter, newMemory, command

    
    def defineList(self, polygonList):
        leftLane = []
        rightLane = []
        if self.presistentMemory.leftExist == True:
            leftLane = polygonList
        elif self.presistentMemory.rightExist == True:
            rightLane = polygonList
        return leftLane, rightLane


def compareRightCamAndLeftCam(rPL, lPL, lc, frameWidth):
    # compares the polygon list of both right and left cameras, and uses it to judge where the CAV is in relation to the road   
    # adds/removes 30 pixels to the lane center in order to help rebalance 
    # rPL = right Polygon List
    # lPL = left Polygon List 
    # lc  = lane Center 
    # IF X AVG OF rPL ~= lPl (moe of 75 pix) then we are in the centre of the frame 
    rAvg = getXAvg(rPL)
    lAvg = getXAvg(lPL)
    lAvg = frameWidth - lAvg #swap it over
    if(rAvg - 75 >= lAvg): #heavily left favoured, so we move right
        lc = lc + 100
    elif(rAvg + 75 <= lAvg): #heavily right favoured, so we move left
        lc = lc - 100
    else: #equal/neglible 
        pass #do nothing 
    return lc
        
    


def getXAvg(list):
    #given a list of coordinates get x average of all coordinates 
    x_coord = [coordinates[0] for coordinates in list]
    if x_coord is None or len(list) < 5 or len(x_coord) == 0:
        return 0   #Guard Condition TODO: ERR handle
    midX = sum(x_coord)/len(x_coord)
    return midX 