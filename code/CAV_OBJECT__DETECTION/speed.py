#File Used to handle all functions related to the speed of the CAV
#Speed must be sent in the form of "S[value]\n"
#Max of S100, Min of S-100
#Negaives will reverse, S0 is stop
# 0,0 top left 
import sharedFunctions as sf
#CONSTANTS 
MAX_SPEED = 15 #straight aways with high vision - might be too quick 
MIN_SPEED = 13 #turning - very slow
dx = 0 
change = 0.2 

def calc_speed(leftLane, rightLane, scale):
    command = 0
    if not leftLane and not rightLane :
        return command
    rightList = []
    leftList = []
    #determine a speed based on how far up the highest point is and whether or not both lanes are detected 
    if rightLane:
        rightList = sf.convertToYList(rightLane)
    if leftLane:
        leftList = sf.convertToYList(leftLane)

    ymax = min(rightList + leftList)
    #print(f"ymax: {ymax} scaled height: {(scale*1080*1.5)}")
    if leftLane and rightLane:
        yfactor = ((scale*1080*1.5) - ymax)/(scale*1080*1.5) #0,0 y increases down, therefore y = size - ymax 
    else:
        yfactor = ((scale*1080*1.5) - ymax)/(scale*1080*3)

    speed = MIN_SPEED + ((MAX_SPEED - MIN_SPEED)*yfactor)
    if speed > MAX_SPEED:
        speed = MAX_SPEED
    elif speed < MIN_SPEED: 
        speed = MIN_SPEED
    command = speed
    return command