#A state to very handle merging in a very basic manner 
#Will need to use game theory to decide how the CAV will merge into the lane if there are other vehicles 
import laneMemory 
import sharedFunctions as sf
import speed as sp 
import cv2
#HOW WILL WE KNOW WHEN TO MERGE 
#HOW WILL WE KNOW THAT WE HAVE MERGERD?

class mergeState: 
    def __init__(self, laneState):
        self.laneState = laneState
        self.presistentMemory = laneMemory(False, False,[],[])
        self.idx = 0
        self.speed = "S15\n"
        self.mergeLeft = False 
        self.mergeRight = False

    def getState(self):
        if self.mergeLeft:
            return 5
        elif self.mergeRight:
            return 6 
        else:
            return -2 #error state 
    
    def getSpeed(self):
        return self.speed
    
    def changeState(self):
        print("Now entering turning state")
        self.idx = 0
        self.laneState.state = self.laneState.turningstate
    

    #an unique proccess that continues to turn for a bit, but if it goes too long enter a search functionality
    def proccess(self, frame, scale, model, df, midX, laneCenter, newMemory, cameras):
        if self.idx == 0: 
            #First entered state 
            self.assignPresistentMemory(laneMemory(False,False,[],[]))
            self.idx = 1
            self.assignPresistentMemory(newMemory)
        print("PM L:", self.presistentMemory.leftExist, " PM R: ", self.presistentMemory.rightExist)
        polygonList, signList = sf.usingCSVData(df)
        margin = sf.marginOfError(scale, laneCenter, midX) #For if the centre of the lane is left or right favoured
        leftLane, rightLane = sf.splitLaneByImg(polygonList, margin, scale) #easiest way to split the list 
        newMemory = sf.doesLeftOrRightExist(leftLane, rightLane, scale, newMemory)
        
        if newMemory.leftExist == True and newMemory.rightExist == True: #two lane exit
            self.idx == 0
            self.changeStateTwoLane() 
        else:
            leftLane, rightLane = self.defineList(leftLane + rightLane)
            newMemory = laneMemory(self.presistentMemory.leftExist, self.presistentMemory.rightExist, leftLane, rightLane)
        laneCenter = sf.findLaneCenter(newMemory.leftLane, newMemory.rightLane, 900 * scale, midX, laneCenter)
        command = sp.calc_speed(newMemory.leftLane, newMemory.rightLane, scale)
        newFrame = sf.overlayimage(scale, newMemory.leftLane, newMemory.rightLane, laneCenter, frame)
        
        cv2.imshow("final", newFrame)
        return laneCenter, newMemory, command