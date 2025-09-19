#used to share functionally and declutter reading.py 
import cv2
import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use("QtAgg")
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
#import geopandas as gpd
#from adaption import *
#from laneFitting import *
import math
from laneMemory import laneMemory
from lanes import *
from scipy.spatial import distance

 
def usingCSVData(dataFrame):
        #xCentre = [] 
        #yCentre = [] 
        polygonL = [] #list of coordinates, bad naming convention I know
        signList = []
        for index, row in dataFrame.iterrows():
            #row names ,xmin,ymin,xmax,ymax,confidence,class,name
            if(row["confidence"] >= 0.2 and row["class"] == 0):
                #Gets the midpoint of xmin and xmax, and ymin and ymax, appending it to the list polygonL as a list of coordinates
                xMid = getCord(row["xmin"], row["xmax"])
                yMid = getCord(row["ymin"], row["ymax"])
                polygonL.append((float(xMid), float(yMid)))
            elif(row["class"] != 0):
                # class, midx, midy, width, height
                newList = [row["class"],getCord(row["xmin"], row["xmax"]), getCord(row["ymin"], row["ymax"]), (row["xmax"]- row["xmin"]), (row["ymax"]- row["ymin"])]
                signList.append(newList)
        return polygonL, signList
        
def getCord(min, max):
    #midpoint formulae 
    return (min + max)/2
    
    
def calcScale(num):
    #calculates scale using midx 960
    scaled = (num / 960) 
    return scaled

def minimum(matrix):
    #return the minimum sized number in a array
    min = 9999999999999999 # +ve infinity 
    for array in matrix: #O(n^2) Very bad 
        for x in array:
            if (x < min):
                min = x
    return min 

def sortByDistance(array):
    thisList = []
    idx = 0 
    if len(array) <= 1 :
        return []
    
    thisList.append(array[0])
    x = array[idx]
    while idx < len(array)  - 1: 
        idx = idx+ 1 
        y = array[idx] 
        i, j = y
        if getDist(x,y) < 40:
            thisList.append(y)
            x = y
    #DEBUG print(thisList)
    return thisList


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
        #CODE DUPLICATION BELOW
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
        y = (int)(400*scale)
        cv2.circle(overlay, ((int)(laneCenter), (y)), 10, (125, 125, 0), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image) #overlaays the image with the polygon
    # overlay = image.copy()
    # y = (int)(400*scale)
    # cv2.circle(overlay, ((int)(laneCenter), (y)), 10, (125, 125, 0), -1)
    # cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image) #overlaays the image with the polygon
    return image
    

def overlaySideImage(list, image):
    #display every point for side cameras and the x avg
    alpha = 0.5 #transparency for overlay
    overlay = image.copy()
    for x, y in list: 
        cv2.circle(overlay, (int(x), int(y)), 2, (255, 125, 0), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image) #overlays the image with the polygon
    # cv2.circle(overlay, ((int)(laneCenter), (240)), 2, (0, 125, 255), -1)
    # cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image) #overlays the image with the polygon
    return image

def doesLeftOrRightExist(leftLane, rightLane, scale, oldMemory):
    #use lane gradients to determine if a lane exists
    #helps for defining centre and turns
    #Left lane = -ve gradient
    #Right Lane = +ve gradient

    dist = 0
    leftExist = False
    rightExist = False 
    #Validating that left and right lanes exist and are correctly defined
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
    
    if len(leftLane) >= 1 and len(rightLane) >= 1: #if 0 then line of best calculation will crash when no lines are detected
        #check distance using cdist
        # matrix = distance.cdist(leftLane, rightLane, metric='euclidean')
        # min = minimum(matrix) 
        #we check distance to ensure that the lanes are apporiately separated 
        #this distance checking grabs the minimum distance between all points of both lanes
        #it works but if there are many many points in the definition it will run gradually slower as it needs to sort through
        #what is effectively a 2d array
        # if (min < 390 * scale):
        if oldMemory.leftExist == True and oldMemory.rightExist == False and 0 >= lineOfBest(leftLane + rightLane): #turning right 
            leftExist = True
            rightExist = False
            leftLane.extend(rightLane)
            rightLane.clear() 
        elif oldMemory.rightExist == True and oldMemory.leftExist == False and 0 < lineOfBest(leftLane + rightLane) : #turning left
            rightExist = True
            leftExist = False
            rightLane.extend(leftLane)
            leftLane.clear() 
    newMemory = laneMemory(leftExist, rightExist, leftLane, rightLane)
    #DEBUG print("LE ", leftExist, "\nRE ", rightExist, "\nLL: ", leftLane, "\nRR: ",rightLane, "\ndist ", dist, "\ngradLeft ", gradLeft, "\ngradRight ", gradRight)
    return newMemory


def sortByDist(givenList,scale):
    #SORTS OUT THE POINTS TO DISCARD OF OUTLIERS 
    #CORDS IN (X, Y) FORMAT
    #using https://codereview.stackexchange.com/questions/224704/grouping-sorted-coordinates-based-on-proximity-to-each-other
    groupList = []
    while givenList: #not empty
        farPoints = []
        ref = givenList.pop()
        groupList.append(ref)
        for point in givenList:
            d = getDist(ref, point)
            if d < (600*scale): #change distance param here 
                #more specifically this says if the point is less than 70 pixels from the last point the append it to the list 
                groupList.append(point)
            else: 
                farPoints.append(point)
        givenList = farPoints

    return groupList

def splitLaneByImg(coordList, midX, scale):
    leftLane = []
    rightLane = []
    #DEBUGGING STUFF
    #print("Overall gradient: ", lineOfBest(coordList))
    if coordList is None: #Guard conditon One
        return leftLane, rightLane 
    #define midx as the average (mean) of the coordlist x coordinates 
    # x_coord = [coordinates[0] for coordinates in coordList]
    # if x_coord is None or len(x_coord) == 0:
    #     return leftLane, rightLane #Guard Condition 2 
    # midX = sum(x_coord)/len(x_coord)
    for point in coordList:
        x, y = point 
        if x < midX and y > (900*scale): #TOP LEFT IS 0,0 and bottom right is +ve, +ve
            leftLane.append(point)
        elif x >= midX and y > (900*scale) : 
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
    # #add points to plot
    # plt.scatter(x, y)
    # #add line of best fit to plot
    # plt.plot(x, a*x+b)
    return a 

def convertToXList(list):
    xList = []
    if(len(list) > 0):
        for line in list:
            x, y = line 
            xList.append(x)
    return xList

def convertToYList(list):
    yList = []
    if(len(list) > 0):
        for line in list:
            x, y = line 
            yList.append(y)
    return yList

def findLaneCenter(leftLane, rightLane, laneWidth, midX, lastLaneCenter):
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
        margin = midX + (scale * 100)
    elif laneCenter < midX:
        margin = midX - (scale * 100)
    else:
        margin = midX
    return margin


def getPolygonList(frame, model):
    #to reduce code dupe 
    #process a frame and returns a list of points 
    nFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(nFrame)
    df = pd.DataFrame(results.pandas().xyxy[0].sort_values("ymin")) #df = Data Frame, sorts x values left to right (not a perfect solution)
    df = df.reset_index() # make sure indexes pair with number of rows
    df.iterrows()
    polygonList = usingCSVData(df) 
    polygonList = [coordinates for coordinates in polygonList if coordinates[1] > 300] #filtering the list
    return polygonList