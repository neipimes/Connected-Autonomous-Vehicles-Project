#Reads a source (has to be manually defined (a possible optimisation/QoL)) passes it through YoloV5, pulls the data from YoloV5 to be used elsewhere
#Left lane = -ve gradient
#Right Lane = +ve gradient
# 0,0 top left 
import torch
import cv2
from gstreamerPipeline import gstreamer_pipeline 
import pandas as pd
import numpy as np
from cameraWidget import cameraStreamWidget
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
from statePattern import laneController as lc
import sharedFunctions as sf
from cavErrors import * 
from input import keyboardListener
def writeToFile(snapString):
    #Call to write to a file  
    #unused 
    file = open("coords.txt", "w") 
    file.writelines(snapString)
    file.close() 

def getSignData(dataFrame):
    polygonL = [] #list of coordinates
    for index, row in dataFrame.iterrows():
        #row names ,xmin,ymin,xmax,ymax,confidence,class,name
        if(row["confidence"] >= 0.2 and row["class"] != 0): #ebery single class except 
            #Gets the midpoint of xmin and xmax, and ymin and ymax, appending it to the list polygonL as a list of coordinates
            xMid = sf.getCord(row["xmin"], row["xmax"])
            yMid = sf.getCord(row["ymin"], row["ymax"])
            polygonL.append((float(xMid), float(yMid)))
    
    return polygonL


def signDetails(image, list):
    alpha = 0.5 
    overlay = image.copy()
    #DEBUG print(list)
    for element in list: 
        #DEBUG print(element)
        x,y,z = element
        cv2.circle(overlay, (x,y), 10, (0, 125, 125), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image) #overlaays the image with the polygon
    return image


def openStream(name):
    #open the stream and return it
    print("writing")
    model_name='/home/raf/local/cuda/bin/lb2OO07.pt'
    #load model
    model = torch.hub.load('/home/raf/local/cuda/bin/yolov5', 'custom', source='local', path = model_name, force_reload = True)
    firstFrame = True
    #Opening with openCV
    capture = cv2.VideoCapture(name)
    return capture, model 

def proccess(frame, scale, model, midX, laneCenter, newMemory, displayName): 
    #frame = frame[(int)(2*frame.shape[0]/5): frame.shape[0], 0:frame.shape[1]] #y, x
    rFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rFrame)
    #DEBUG results.print() 

    df = pd.DataFrame(results.pandas().xyxy[0].sort_values("ymin")) #df = Data Frame, sorts x values left to right (not a perfect solution)
    df = df.reset_index() # make sure indexes pair with number of rows
    df.iterrows()
    polygonList = sf.usingCSVData(df)
    polygonList = sf.sortByDist(polygonList, scale) #Gets rid of outliers
    margin = sf.marginOfError(scale, laneCenter, midX) #For if the centre of the lane is left or right favoured
    leftLane, rightLane = sf.splitLaneByImg(polygonList, margin, scale) #easiest way to split the list 
    # leftLane = sortByDistance(leftLane)
    # rightLane = sortByDistance(rightLane)
    #leftLane, rightLane = sortByDistance(polygonList)
    newMemory = sf.doesLeftOrRightExist(leftLane, rightLane, scale, newMemory)
    #print("Left: ", leftExist, "  ", leftLane, "\nRight: ", rightExist, "  ", rightLane)
    laneCenter = sf.findLaneCenter(newMemory.leftLane, newMemory.rightLane, 1000 * scale, midX,  laneCenter)
    #print(laneCenter)
    newFrame = sf.overlayimage(scale, newMemory.leftLane, newMemory.rightLane, laneCenter, frame)
    cv2.imshow(displayName, newFrame)
    return laneCenter, newMemory

def signDetect(frame, model):
    rFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(rFrame)
    #DEBUG results.print() 
    df = pd.DataFrame(results.pandas().xyxy[0].sort_values("ymin")) #df = Data Frame, sorts x values left to right (not a perfect solution)
    df = df.reset_index() # make sure indexes pair with number of rows
    df.iterrows()
    polygonList = getSignData(df)
    if len(polygonList) >0 : #not empty
        frame = signDetails(polygonList, frame)
    cv2.imshow("Signs", frame)
    
def convertBird(frame):
    #take a region of an image and convert it to birdseye view
    #CREDIT: Nikolasent -- Bird's Eye View Transfromation 
    #https://nikolasent.github.io/opencv/2017/05/07/Bird's-Eye-View-Transformation.html
    if frame.all() == None: #guard condition
        return 
    
    imageH = frame.shape[0]
    imageW = frame.shape[1]
    #DEBUG print("imageH ", imageH)
    #ISSUE WITH DETECTION IS HERE 
    src = np.float32([[0, imageH], [(1207/1080) * imageH, imageH], [0,0], [imageW, 0]]) # source image
    dst = np.float32([[(469/1080) * imageH, imageH], [(1207/1080) * imageH, imageH], [0,0], [imageW, 0]]) # roi
    M = cv2.getPerspectiveTransform(src,dst) #transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src) #inverse transformation
    #SLICE THAT IMAGE 
    img = frame[(int)(450/1080*imageH):((int)(450/1080*imageH)+imageH), 0:imageW] # apply np slicing for ROI crop
    warpedImg = cv2.warpPerspective(img, M,(imageW,imageH)) #image warping
    nwIm = warpedImg #cv2.cvtColor(warpedImg, cv2.COLOR)
    # cv2.imshow("BE",nwIm)
    return nwIm

def processEachFrame():
    #BREAKING DOWN writeToCSV()
    cameras = []
    #init all streams 
    cameras.append(cameraStreamWidget("/home/raf/local/cuda/bin/vivs/vid.webm", "One"))
    cameras.append(cameraStreamWidget("/home/raf/local/cuda/bin/vivs/vid.webm", "Two"))
    cameras.append(cameraStreamWidget("/home/raf/local/cuda/bin/vivs/vid.webm", "Three"))
    model_name='/home/raf/local/cuda/bin/bestJul25.pt'
    #load model
    model = torch.hub.load('/home/raf/local/cuda/bin/yolov5', 'custom', source='local', path = model_name, force_reload = True)
    firstFrame = True 
    frame_count = 0
    leftLane = []
    rightLane = []
    laneState = lc.laneController() 
    #Processing each frame
    condition = True
    keyboard = keyboardListener()
    keyboard.initKeyboard() 
    try:
        while condition:
            frame = cameras[0].returnFrame()
            if firstFrame:
                midX = int((frame.shape[1])/2)
                firstFrame = False
                laneCenter = midX
                scale = sf.calcScale(midX)
                newMemory = laneMemory(False, False, [], [])
                detections = 0
            #if not ret:
            #     break
            ### ###
            oldMemory = newMemory
            detections += 1 #used for lane weighting 
            if frame is None: 
                pass
                # raise CameraStreamError("Camera Stream is null")#raise error
            else:
                rFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(rFrame)
                df = pd.DataFrame(results.pandas().xyxy[0].sort_values("ymin")) #df = Data Frame, sorts x values left to right (not a perfect solution)
                df = df.reset_index() # make sure indexes pair with number of rows
                df.iterrows()
                frame_count += 1
                # gray = cv2.cvtColor(rFrame,50, 200) -- Canny edge 
                # edges = cv2.Canny(gray, 50, 200)
                # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 68, minLineLength=1, maxLineGap=100)
                # for line in lines: 
                #     x1, y1, x2, y2 = line[0]
                #     cv2.line(rFrame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                # cv2.imshow('lines', rFrame)
                polygonList = sf.usingCSVData(df)
                counts, xedges, yedges = np.histogram2d(sf.convertToYList(polygonList), sf.convertToXList(polygonList), bins=540)
                # plt.scatter(sf.convertToXList(polygonList), sf.convertToYList(polygonList))
                # plt.show()
                # plt.imshow(counts)
                # plt.show
                #cv2.imshow('lines', counts)
                # counts = np.array(counts)
                            
                # Normalize to 0-255 and convert to uint8 image
                counts_img = cv2.normalize(counts, None, 0, 255, cv2.NORM_MINMAX)
                counts_img = counts_img.astype(np.uint8)
                cv2.imshow('counts', counts_img)
                # Hough Line Transform
                lines = cv2.HoughLinesP(counts_img, 1, np.pi/180, 68, minLineLength=10, maxLineGap=250)
                print(lines)
                if lines:
                    for line in lines: 
                        x1, y1, x2, y2 = line[0]
                        cv2.line(rFrame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.imshow('lines', rFrame)
                laneCenter, newMemory, command = laneState.proccess(frame, scale, model, df, midX, laneCenter, newMemory, cameras)
                print("Current State: ", laneState.getState())    
                print(command)     
                if cv2.waitKey(1) == ord('q'):#diplays the image for a set amount of time 
                    break
                frame_count += 1
                if(detections >= 3): 
                        newMemory = laneMemory(oldMemory.leftExist, oldMemory.rightExist, [], [])
                        detections = 0
                ### ### ### ### ### ### ### ### ###
                userInput = keyboard.getLastKey()
                print("last key", userInput)

                if (userInput == 'q') : #exit condition
                    print("User entered termination condition") 
                    condition = False
    except KeyboardInterrupt:
        pass 
    # except Exception as e: #neccesary to ensure cameras are turned off properly otherwise the CAV will need to be reset
    #     print(f"Immediate stop of function: {e}")
    #Close
    #capture.release()
    for cam in cameras: 
        cam.closeStream() 
    keyboard.endKeyboard()
    cv2.destroyAllWindows()
    
