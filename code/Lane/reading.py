#Reads a source (has to be manually defined (a possible optimisation/QoL)) passes it through YoloV5, pulls the data from YoloV5 to be used elsewhere
#Left lane = -ve gradient
#Right Lane = +ve gradient
# 0,0 top left
import torch
import cv2
import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use("QtAgg")
#from matplotlib import pyplot as ply
from shapely.geometry import Polygon
#import geopandas as gpd
#from adaption import *
from laneFitting import *
import math

def writeToFile(snapString):
    #Call to write to a file  
    #unused 
    file = open("coords.txt", "w") 
    file.writelines(snapString)
    file.close() 

def getCord(min, max):
    #midpoint formulae 
    return (min + max)/2


def usingCSVData(dataFrame):
    #xCentre = [] 
    #yCentre = [] 
    polygonL = [] #list of coordinates, bad naming convention I know
    for index, row in dataFrame.iterrows():
        #row names ,xmin,ymin,xmax,ymax,confidence,class,name
        if(row["confidence"] >= 0.2):
            #Gets the midpoint of xmin and xmax, and ymin and ymax, appending it to the list polygonL as a list of coordinates
            xMid = getCord(row["xmin"], row["xmax"])
            yMid = getCord(row["ymin"], row["ymax"])
            #xCentre.append((xMid)) #unused could potentally be used to sort data-points further 
            #yCentre.append((yMid)) #unused
            polygonL.append((float(xMid), float(yMid)))
    return polygonL

def sortClockWise(cordList):
    #https://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order
    return 0

def overlayimage(scale, leftLane, rightLane, laneCenter, image):
    #takes list and turns it into polygon
    alpha = 0.5 #transparency for overlay
    #print(len(leftLane + rightLane))
    if(len(leftLane + rightLane) >= 4):
        #Enough Coordinates to make a polygon 
        
        leftLane.reverse()
        polygonTo = Polygon(rightLane + leftLane) #converts to a polygon, hence polygonTo (horrible naming convention i know)
        leftLane.reverse()
        #FOLLOWING CODE USES:
        #https://stackoverflow.com/questions/13574751/overlay-polygon-on-top-of-image-in-python 
        int_coords = lambda x: np.array(x).round().astype(np.int32)
        exterior = [int_coords(polygonTo.exterior.coords)]
        overlay = image.copy()
        cv2.fillPoly(overlay, exterior, color=(150, 255, 0)) #RGB - fills polygon with the colour
        if len(leftLane) >= 1:
            i = 0
            for line in leftLane:
                if i == 0:
                    i =+ 1
                    x0, y0 = line
                else: 
                    x2, y2 = line 
                    cv2.line(overlay, ((int)(x0), (int)(y0)), ((int)(x2), (int)(y2)), (255, 0, 0), 10) #draws lines 
                    i =+ 1
                    x0 = x2
                    y0 = y2
                
        if len(rightLane) >= 1:
            i = 0
            for line in rightLane:
                if i == 0:
                    i =+ 1
                else: 
    
                    x2, y2 = line 
                    cv2.line(overlay, ((int)(x0), (int)(y0)), ((int)(x2), (int)(y2)), (0, 0, 255), 10) #draws lines 
                    i =+ 1
                x0, y0 = line
        y = (int)(800*scale)
        cv2.circle(overlay, ((int)(laneCenter), (y)), 10, (125, 125, 0), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image) #overlaays the image with the polygon
    return image

def calcScale(num):
    #calculates scale using midx 960
    scaled = (num / 960) 
    return scaled

def doesLeftOrRightExist(leftLane, rightLane, scale):
    #use lane gradients to determine if a lane exists
    #helps for defining centre and turns
    #Left lane = -ve gradient
    #Right Lane = +ve gradient
    leftExist = False
    rightExist = False 

    if len(leftLane) > 1:
        gradLeft = lineOfBest(leftLane)
    else: 
        gradLeft = 0

    if len(rightLane) > 1: 
        gradRight = lineOfBest(rightLane)
    else:
        gradRight = 0

    if gradLeft < 0: #-ve
        leftExist = True
    if gradRight > 0: #+ve
        rightExist = True
    #Then dictate whether or not it exists by xmax and ymax of both lines and distance

    #Empties list if one lane exists and the other doesn't
    if rightExist and not leftExist:
        if leftLane:
            #WORKS FINE
            max = len(leftLane)
            # x0, y0 = leftLane[0]
            # x2, y2 = leftLane[max -1]
            # xL = getCord(x0, x2)
            # yL = getCord(y0, y2)
            # max = len(rightLane)
            # x0, y0 = rightLane[0]
            # x2, y2 = rightLane[max -1]
            # xR = getCord(x0, x2)
            # yR = getCord(y0, y2)
            dist = getDist(leftLane[0],rightLane[len(rightLane) -1])
            if dist > (500*scale):
                leftExist = True
            else:
                while leftLane:
                    rightLane.append(leftLane.pop())
        else:   
            while leftLane:
                rightLane.append(leftLane.pop())
    
    if leftExist and not rightExist:
        #print("le and no re")
        if rightLane:
            #print("if right lane")
            #A BIT FUNKY
            max = len(leftLane)
            # x0, y0 = leftLane[0]
            # x2, y2 = leftLane[max - 1]
            # xL = getCord(x0, x2)
            # yL = getCord(y0, y2)
            # max = len(rightLane)
            # x0, y0 = rightLane[0]
            # x2, y2 = rightLane[max - 1]
            # xR = getCord(x0, x2)
            # yR = getCord(y0, y2)
            dist = getDist(leftLane[max -1],rightLane[0])
            if dist > (500*scale):
                rightExist = True
                #print("Right exist entered")
            else:
                while rightLane:
                    leftLane.append(rightLane.pop())
                #print("UNO ", rightLane)   
        else:
            while rightLane:
                leftLane.append(rightLane.pop())
            #print("DOS ", rightLane)

    return leftExist, rightExist, leftLane, rightLane

def sortByDist(givenList,scale):
    #CORDS IN (X, Y) FORMAT
    #using https://codereview.stackexchange.com/questions/224704/grouping-sorted-coordinates-based-on-proximity-to-each-other
    groupList = []
    while givenList: #not empty
        farPoints = []
        ref = givenList.pop()
        groupList.append(ref)
        for point in givenList:
            d = getDist(ref, point)
            if d < (70*scale): #change distance param her3
                groupList.append(point)
            else: 
                farPoints.append(point)
        givenList = farPoints

    return groupList

def splitLaneByImg(coordList, midX, scale):
    leftLane = []
    rightLane = []
    for point in coordList:
        x, y = point 
        if x < midX and y > (500*scale + 50):
            leftLane.append(point)
        elif x >= midX and y > (500*scale + 50) : #300 when using https/webcam --- 500 with video
            rightLane.append(point)
    return leftLane, rightLane

def getDist(ref, point):
    #using https://codereview.stackexchange.com/questions/224704/grouping-sorted-coordinates-based-on-proximity-to-each-other
    x1, y1 = ref
    x2, y2 = point
    return math.hypot(x2 - x1, y2 - y1)   #H^2 = A^2 + B^2
       

def lineOfBest(coordList):
    #finds the line of best fit and returns it 
    xList = []
    yList = []
    for line in coordList:
        x0, y0 = line
        xList.append(x0)
        yList.append(y0)
    x = np.array(xList)
    y = np.array(yList)
    #find line of best fit

    a, b = np.polyfit(x, y, 1)
    #add points to plot
    plt.scatter(x, y)
    #add line of best fit to plot
    plt.plot(x, a*x+b)
    return a 

def convertToXList(list):
    xList = []
    if(len(list) > 0):
        for line in list:
            x, y = line 
            xList.append(x)
    return xList

def findLaneCenter(leftLane, rightLane, laneWidth, midX, leftExist, rightExist, lastLaneCenter):
    #Justin's code adapted
    #finds lane center 
    laneCenter = midX

    #need to conver right lane and left lane to JUST x coordinates
    if rightLane:
        medianRightX = np.mean(convertToXList(rightLane))
    if leftLane:
        medianLeftX = np.mean(convertToXList(leftLane))

    if leftLane and rightLane:
        laneCenter = (medianRightX + medianLeftX)/2
    elif rightLane:
        laneCenter = medianRightX - (laneWidth/2)
    elif leftLane:
        laneCenter = medianLeftX + (laneWidth/2)
    else:
        laneCenter = lastLaneCenter
    return laneCenter

def marginOfError(scale, laneCenter, midX):
    if laneCenter > midX:
        margin = midX + (scale * 200)
    elif laneCenter < midX:
        margin = midX - (scale * 200)
    else:
        margin = midX
    return margin

def writeToCSV():
    #Testing reading prediction output 
    
    print("writing")
    snapString = 'NULL'
    model_name='lb2OO07.pt' #manual replace with our current model here 

    #load model
    model = torch.hub.load('yolov5', 'custom', source='local', path = model_name, force_reload = True)
    
    #videoPath = "Videos/Cropped-0.mp4"
    videoPath = "http://172.25.0.46:9001/camera.cgi" #remoting via vpn 
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
        polygonList = sortByDist(polygonList, scale)#is neccessary
        margin = marginOfError(scale, laneCenter, midX)
        leftLane, rightLane = splitLaneByImg(polygonList, margin, scale)
        #print(polygonList, leftLane, rightLane)
        #print(scale)
        leftExist, rightExist, leftLane, rightLane = doesLeftOrRightExist(leftLane, rightLane, scale)
        #print("Left: ", leftExist, "  ", leftLane, "\nRight: ", rightExist, "  ", rightLane)
        laneCenter = findLaneCenter(leftLane, rightLane, 1000 * scale, midX, leftExist, rightExist, laneCenter)
        #print(laneCenter)
        newFrame = overlayimage(scale, leftLane, rightLane, laneCenter, frame)
        cv2.imshow("Final", newFrame)
        if cv2.waitKey(1) == ord('q'):#diplays the image for a set amount of time 
            break
        frame_count += 1

    #Close
    capture.release()
    cv2.destroyAllWindows()
    return 0

def selfDriveAdaptBETAV0P02BV():
    #Takes the base line from writetoCSV and adapts selfDive.py over it 
    #lane width with Params
    lineWidth = 550 #400
    #freme details 
    fps = 15 #framerate
    prevLaneCenter = 512
    #NO VIDEO IMPORTED
    #skipped to line 107

    #defining variables
    laneCenterX = 512 #Constant
    M = None #no clue
    M_inv = None #no clue
    firstFrame = True
    leftLaneDetected = True
    rightLaneDetected = True
    optimalWhite = None #if it is image proccessing does not matter
    command = 's'

    model_name='lb2OO07.pt' #manual replace with our current model here 

    #load model
    model = torch.hub.load('yolov5', 'custom', source='local', path = model_name, force_reload = True)
    
    #videoPath = "Videos/Cropped-0.mp4"
    videoPath = "http://172.25.0.46:9001/camera.cgi" #remoting via vpn 
    outputDir = 'outputFrames'

    #Opening with openCV
    capture = cv2.VideoCapture(videoPath)
    frame_count = 0

    #Processing each frame
    while capture.isOpened():
        #used chatgpt as a reference
        ret, frame = capture.read()
        if not ret:
            break
        if firstFrame:
            #for purpose of not crashing when trying to recall undefined data 
            midX = int((frame.shape[1])/2)
            laneCenter = midX
            firstFrame = False
        #Convert each frame into RBG
        rFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rFrame)
        results.print() #prints to terminal (optional)


        df = pd.DataFrame(results.pandas().xyxy[0].sort_values("xmin")) #df = Data Frame, sorts x values left to right (not a perfect solution)
        df = df.reset_index() # make sure indexes pair with number of rows
        df.iterrows()
        polygonList = usingCSVData(df)
        leftLane, rightLane = splitLaneByImg(polygonList, midX)
        print(polygonList, leftLane, rightLane)
        
        leftLaneDetected, rightLaneDetected, leftLane, rightLane = doesLeftOrRightExist(leftLane, rightLane)
        if frame_count%7 ==0:
            #so every 7 frame
            lastLeftCond = leftLaneDetected
            lastRightCond = rightLaneDetected
            polygonList = sortByDist(polygonList)
            if leftLaneDetected:
                # for inde in range(1, len(ploty)):
                #     None\
                None
        
        
        newFrame = overlayimage(polygonList, leftLane, rightLane, frame)
        cv2.imshow("Final", newFrame)
        if cv2.waitKey(1) == ord('q'):#diplays the image for a set amount of time 
            break
        frame_count += 1
        
       
           

    #Close
    capture.release()
    cv2.destroyAllWindows()
    
    return 0