import cv2
import numpy as np
import matplotlib.pyplot as ply
#CANNY EDGE DETECTION ATTEMPT 
def canny(image): 
    #PART 2 - grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #RGB to greyscale
    
    #PART 3 - Applying Gausian Blur 
    blur = cv2.GaussianBlur(gray, (5,5), 0) #applies a 5 x 5 gausian blur 

    #PART 4 - Canny (Gives us the gradient image)
    canny = cv2.Canny(blur, 50, 150) #maps pixels above 150 threshold, and above 50 if and only if connected to strong gradient
    return canny

#PART 5 - identifying lanes
def region_of_intrest(image):
    height = image.shape[0] #rows (y axis)
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)] #TODO: FIXME: 
        ]) #creates a pologons over the lane, using a matrix of matrices 
    mask = np.zeros_like(image) #creates a black mask over the image
    cv2.fillPoly(mask, polygons, 255) #Area bounded by the polygon will be white 
    #PART 6 - Bitwise &
    masked_image = cv2.bitwise_and(image, mask) #bit wise & comparion - shows us the lane we are looking at
    return masked_image

#PART 8 - dispalying the liens 
def display_Lines(image, lines):
    line_image = np.zeros_like(image) #array of zeros size of image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            print(x1," ", x2," ", y1, " ", y2)
            h, w, _ = image.shape
            if 0 <= x1 < w and 0 <= x2 < w and 0 <= y1 <= h and 0 <= y2 <= h:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10) # I WANT TO CRY
            else:
                print(f"Skipping invalid line: {x1, y1, x2, y2}")  # Debugging
    return line_image

#PART 9
def make_coordinates(image, line_parameters):
    print("lp", line_parameters)
    #check for Nan
    if np.isnan(line_parameters).any():
        return np.array([0, 0, 0, 0])

    print("pass")
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    print("mc ", y1, y2, x1, x2)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = [] #coords of the lines on the left
    right_fit = [] #coords of the lines on the right
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1) #fits a polynomial within the points 
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            print("lf, s, i", slope, intercept)
            left_fit.append((slope, intercept))
        else:
            print("rf, s, i", slope, intercept)
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    print("lfa", left_fit_average)
    right_fit_average = np.average(right_fit, axis=0)
    print("rfa", right_fit_average)
    left_line = make_coordinates(image, left_fit_average)
    print("ll", left_line)
    right_line = make_coordinates(image, right_fit_average)
    print("rl", right_line)
    return np.array([left_line, right_line])
#read and load image - MAIN
# #PART 1
def processCED(input):
    #PROCESS CANNY EDGE DETECTUIB 
    image = np.array(input) #read image and return it as a numpy array                                                    
    #PART 2 - Greyscale Conversion
    lane_image = np.copy(image) #making changes to lane image
    #PART 5 - identifying lanes 
    canny_image = canny(lane_image)
    #PART 6 - bitwise & 
    cropped_image = region_of_intrest(canny_image)
    #PART 8 - Hough transformation
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) #Used for lane detection, in radians  , image a b c - threshold (votes) d - placeholder array, e - min line length, f max line gap
    #PART 9
    print("lines: ", lines)
    averaged_lines = average_slope_intercept(lane_image, lines)
    print(averaged_lines)
    #P8 cont.
    line_image = display_Lines(lane_image, averaged_lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) #overlaying using weighted sum
    #PART 1
    cv2.imshow('result', combo_image) #render the image in a tab called 'result'

#PART 10
# cap = cv2.VideoCapture("Videos/Cropped-0.mp4")
# while(cap.isOpened()):
#     _, frame = cap.read()
#     #PART 5 - identifying lanes 
#     canny_image = canny(frame)
#     #PART 6 - bitwise & 
#     cropped_image = region_of_intrest(canny_image)
#     #PART 8 - Hough transformation
#     lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) #Used for lane detection, in radians  , image a b c - threshold (votes) d - placeholder array, e - min line length, f max line gap
#     #PART 9
#     averaged_lines = average_slope_intercept(frame, lines)
#     #P8 cont.
#     line_image = display_Lines(frame, averaged_lines)
#     combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) #overlaying using weighted sum
#     #PART 1
#     cv2.imshow('result', combo_image) #render the image in a tab called 'result'
#     if cv2.waitKey(1) == ord('q'):#diplays the image for a set amount of time 
#         break
# cap.release()
# cv2.destroyAllWindows()