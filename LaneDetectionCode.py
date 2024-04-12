import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial
import time
import struct
import line_profiler
import Jetson.GPIO as GPIO  # Change this if you use a different library


class PIDController:
    #@profile
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.last_error = 0
    #@profile
    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return output
        #if frame_count % int(frame_rate / 3) == 0:
         #   print(f"P: {pid.kp}, I: {pid.ki}, D: {pid.kd}, previous_error: {self.previous_error}, error: {error}")

class KalmanFilter:
    #@profile
    def __init__(self, process_variance, measurement_variance, estimated_measurement_variance):
        self.process_variance = process_variance # Process variance ('q') - how much we trust our dynamic model
        self.measurement_variance = measurement_variance # Measurement variance ('r') - how much we trust the measurement
        self.estimated_measurement_variance = estimated_measurement_variance # Estimated measurement variance ('p') - initial estimate of measurement variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0
    #@profile
    def update(self, measurement):
        # Prediction update
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # Measurement update
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate

def send_data(command):
    #combined_data = (system_flag << 8) | data
    #bytes_to_send = struct.pack('>H', combined_data)
    #ser.write(bytes_to_send)
    #ser.write(command.encode())
    ser.write(f"{command}\n".encode())
    ser.flush()

def read_data():
    while True:
        line = ser.readline()
        if line:
            print("Received:", line.decode().strip())


def make_coord(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def canny(image):
    # Check if the image is already a single channel image
    if len(image.shape) == 2 or image.shape[2] == 1:
        gray = image  # Image is already grayscale
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Continue with your existing canny edge detection..
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    low_threshold = 60  # Lower this to increase sensitivity
    high_threshold = 300  # Increase this to reduce noise
    canny = cv2.Canny(blur, low_threshold, high_threshold)
    return canny

def clamp_coordinates(line):
    # Clamp each coordinate to the respective limits
    x1, y1, x2, y2 = line
    x1 = max(0, min(1024, x1))
    y1 = max(0, min(768, y1))
    x2 = max(0, min(1024, x2))
    y2 = max(0, min(768, y2))
    return [x1, y1, x2, y2]

def region_of_int(image):
    height, width = image.shape[0], image.shape[1]
    # Define points for the polygon
    polygons = np.array([
        [(0, height),
         (0, int(height * 0.55)),
         (int(width * 0.3), int(height * 0.5)),
         (int(width * 0.7), int(height * 0.5)),
         (width, int(height * 0.55)),
         (width, height)]
    ], dtype=np.int32)

    # Corrected to create a single-channel (grayscale) mask
    mask = np.zeros((height, width), dtype=np.uint8)  # Note change here
    cv2.fillPoly(mask, polygons, 255)
    return mask

def plot_histogram(section_histograms, out_img, window_img):
    # Blending the images (ensure out_img and window_img have compatible sizes and channels)
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.figure()
    plt.imshow(result)
    plt.title('Sliding Windows')
    plt.show(block=False)
    
    # Create a figure for the histograms
    plt.figure(figsize=(10, 6))
    
    # Number of histograms
    num_histograms = len(section_histograms)
    plot_colors = ["orange" , "purple", "green"]
    # Create a subplot for each histogram
    for i, histogram in enumerate(section_histograms):
        ax = plt.subplot(num_histograms, 1, num_histograms - i)
        ax.plot(histogram, color = plot_colors[i])
        ax.set_title(f'Section {i+1} Histogram')
        ax.set_xlabel('Pixel Position')
        ax.set_ylabel('Counts')
    
    plt.tight_layout()  # Adjust layout to not overlap
    plt.show(block=False)

def filter_by_cluster_size(good_inds_x, cluster_threshold=50):
    if len(good_inds_x) == 0:
        return np.array([])

    # Sort the x positions
    sorted_x = np.sort(good_inds_x)
    
    # Find differences between consecutive x positions
    diffs = np.diff(sorted_x)
    
    # Identify large gaps indicating potential cluster boundaries
    gap_indices = np.where(diffs > cluster_threshold)[0]
    
    # If no significant gaps, consider the entire set a valid cluster
    if len(gap_indices) == 0:
        return sorted_x

    # If gaps found, identify the largest cluster
    start_idx = 0
    max_cluster_size = 0
    max_cluster = np.array([])
    for end_idx in gap_indices:
        cluster = sorted_x[start_idx:end_idx + 1]
        if len(cluster) > max_cluster_size:
            max_cluster = cluster
            max_cluster_size = len(cluster)
        start_idx = end_idx + 1
    # Check the last cluster
    last_cluster = sorted_x[start_idx:]
    if len(last_cluster) > max_cluster_size:
        max_cluster = last_cluster
    
    return max_cluster


def find_lane_pixels(binary_warped, last_left_cond, last_right_cond):
    height, width = binary_warped.shape[:2]
    # Parameters for windowing
    nwindows = 9  # Total windows
    windows_per_section = 3  # Windows per section
    nsections = nwindows // windows_per_section  # Should be 3 sections
    window_height = height // nwindows  # Height of each window
    print(f'last_left_cond:{last_left_cond}, last_right_cond:{last_right_cond}')
    left_lane_detected, right_lane_detected = last_left_cond, last_right_cond 
    minpix = 20  # Min pixels to recenter window
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    section_histograms = []
    last_left_y = None
    last_right_y = None
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    midpoint = width//2
    
    # Step through each section
    for section in range(nsections):
        # Calculate the vertical bounds of the current section
        section_y_high = height - (section * height // nsections)
        section_y_low = height - ((section + 1) * height // nsections)
        left_lane_found =  windows_per_section
        right_lane_found = windows_per_section
        # print(f'top of section:{section} right_lane_found:{right_lane_found}, left_lane_found:{left_lane_found}')
        # # Calculate histogram for the current section
        histogram = np.sum(binary_warped[section_y_low:section_y_high, :], axis=0)
        section_histograms.append(histogram)
        if left_lane_detected and right_lane_detected: margin = 100  # Window margin
        else: margin = 130
        # print(f'last_left_cond:{last_left_cond}')
        # print(f'last_right_cond:{last_right_cond}')
        # if right_lane_detected and section == 0:
        if section == 0:
            rightx_base = int(np.argmax(histogram[midpoint:]) + midpoint)
            
            if rightx_base <= midpoint + 50:
                #
                rightx_base = int(np.argmax(histogram))
                rightx_current = rightx_base 
                # #print(f'Took argmax total last_right condition rightx_base:{rightx_base}')
            else:
                #print(f'Took argmax midpoint last_right condition rightx_base:{rightx_base}')
                # rightx_base = rightx_base + 25 
                rightx_current = rightx_base 
        if section == 0:
        # if left_lane_detected and section == 0:
            leftx_base = int(np.argmax(histogram[:midpoint]))
            
            if leftx_base > midpoint - 50 :
                # #print(f'inside last_left condition leftx_base:{leftx_base}')
                leftx_base = int(np.argmax(histogram))
                leftx_current = leftx_base
                #print(f'Took argmax total last_left condition leftx_base:{leftx_base}')
            else:
                #print(f'Took argmax midpoint last_left condition leftx_base:{leftx_base}')
                # leftx_base = leftx_base - 25
                leftx_current = leftx_base
        # Split the section into windows
        #print(f'\n\n =================== section:{section} ===================\n')
        for window in range(windows_per_section):
            #print(f'\n =========== window:{window} ===========')
            leftx_current = max(min(leftx_current, 1023), 1)
            rightx_current = max(min(rightx_current, 1023), 1)
            win_y_high = int(section_y_low + ((windows_per_section - window) * window_height)) # This ensures the first window starts from the bottom
            win_y_low = int(win_y_high - window_height)
            ##print(f"window:{window}, win_y_high:{win_y_high}, win_y_low:{win_y_low}")
            win_xleft_low = int(leftx_current - margin)
            win_xleft_high = int(leftx_current + margin)
            win_xright_low = int(rightx_current - margin)
            win_xright_high = int(rightx_current + margin)
            win_xleft_low = max(min(win_xleft_low, 1023), 1)
            win_xleft_high = max(min(win_xleft_high, 1023), 1)
            win_xright_low = max(min(win_xright_low, 1023), 1)
            win_xright_high = max(min(win_xright_high, 1023), 51)
            # Draw the windows on the visualization image
            cv2.rectangle(window_img, (win_xleft_low, win_y_low), 
                          (win_xleft_high, win_y_high), (50, 50, 255), 3)
            cv2.rectangle(window_img, (win_xright_low, win_y_low), 
                          (win_xright_high, win_y_high), (255, 20, 20), 3)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # #print(f'good_left_inds:{good_left_inds}')
            # #print(f'good_right_inds:{good_right_inds}')
            # #print(f'top window:{window}, top rightx_current:{rightx_current}, top leftx_current:{leftx_current}\n')
            # Append these indices to the lists
            filtered_good_left_inds_x = filter_by_cluster_size(good_left_inds, cluster_threshold=50)
            
            
            # lane chasing function
            #print(f'len(good_left_inds):{len(good_left_inds)}')
            if len(good_left_inds) > minpix:
                left_lane_found = left_lane_found - 1
                good_x_left = nonzerox[good_left_inds]  # Extract x positions for good left indices
                current_min_left_y = np.min(nonzeroy[good_left_inds])
                #print(f'left_lane_found:{left_lane_found}')
                left_lane_inds.append(good_left_inds)
                if last_left_y is None or abs(last_left_y - current_min_left_y) <= window_height*3:
                    # Update the last known position only if within one window_height
                    last_left_y = current_min_left_y
                    # if left_lane_detected:
                    #     left_lane_inds.append(good_left_inds)
                    #     # Proceed with updating left lane detection
                if window == 0:
                    leftx_current = int(np.min(good_x_left)) + 25 # Use mean x position for the first window
                    #print(f'window:{window},: leftx_mean::{leftx_current}')
                    ##print(f'inside 1st window leftx_current:{leftx_current} left_lane_found:{left_lane_found}')
                elif window == 1:
                    leftx_current = int(np.min(good_x_left)) + 25 # Use max x position for the second window
                    #print(f'inside window:{window} leftx_current:{leftx_current} left_lane_found:{left_lane_found}')
                    ##print(f'window:{window},: leftx_max::{leftx_current}')
                elif window ==2:
                    leftx_current = int(np.max(good_x_left)) + 25 # Use max x position for the second window
                    #print(f'inside window:{window}, section:{section} leftx_current:{leftx_current} left_lane_found:{left_lane_found}')
                    if left_lane_found < 1 and section <= 1:
                        left_lane_detected = True
                        #print(f'left_lane_detected:{left_lane_detected}')     
            else:
                leftx_current = leftx_base
                if  left_lane_found == 3 and section == 0:
                    #print(f'len(good_left_inds):{len(good_left_inds)}')
                    left_lane_detected = False
                    
            #print(f'len(good_right_inds):{len(good_right_inds)}')
            if len(good_right_inds) > minpix:
                right_lane_found = right_lane_found - 1
                good_x_right = nonzerox[good_right_inds]  # Extract x positions for good right indices
                current_min_right_y = np.min(nonzeroy[good_right_inds])
                ##print(f'right_lane_found:{right_lane_found}')
                right_lane_inds.append(good_right_inds)
                # if last_right_y is None or abs(last_right_y - current_min_right_y) <= window_height*3:
                #     # Update the last known position only if within one window_height
                #     last_right_y = current_min_right_y
                #     if right_lane_detected:
                #         right_lane_inds.append(good_right_inds)
                if window == 0:
                    rightx_current = int(np.min(good_x_right)) - 25 # Use mean x position for the first window
                    #print(f'inside window:{window} rightx_current:{rightx_current} right_lane_found:{right_lane_found}')
                    ##print(f'window:{window},: rightx_mean::{rightx_current}\n')
                elif window == 1:
                    rightx_current = int(np.min(good_x_right)) - 25 # Use min x position for the second window
                    #print(f'inside window:{window} rightx_current:{rightx_current} right_lane_found:{right_lane_found}')
                    ##print(f'window:{window},: rightx_min::{rightx_current}\n')
                elif window ==2:
                    rightx_current = int(np.min(good_x_right)) - 25# Use min x position for the second window
                    #print(f'inside window:{window}, section:{section}, rightx_current:{rightx_current} right_lane_found:{right_lane_found}')
                    if right_lane_found < 1 and section <= 1:
                        right_lane_detected = True
                        #print(f'right_lane_detected:{right_lane_detected}')
                    
            else:
                rightx_current = rightx_base
                if  right_lane_found == 3 and section == 0:
                    #print(f'len(good_right_inds):{len(good_right_inds)}')
                    right_lane_detected = False

    #print(f'bottom left_lane_detected:{left_lane_detected}, right_lane_detected:{right_lane_detected}')
    # Concatenate the arrays of indices and extract left and right line pixel positions
    # plt.close('all')
    # plot_histogram(section_histograms, out_img, window_img)
    if left_lane_detected and left_lane_inds:
        left_lane_inds = np.concatenate(left_lane_inds)
        # filtered_left_lane_inds = filter_by_density(nonzerox, nonzeroy, left_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
    else:
        lefty = np.array([], np.int_)
        leftx = np.array([], np.int_)
    if right_lane_detected and right_lane_inds:
        right_lane_inds = np.concatenate(right_lane_inds)
        # filtered_right_lane_inds = filter_by_density(nonzerox, nonzeroy, right_lane_inds)
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    else:
        rightx = np.array([], np.int_)
        righty = np.array([], np.int_)
    # print("Left lane pixels (x, y):", list(zip(leftx, lefty)))
    # print("Right lane pixels (x, y):", list(zip(rightx, righty)))
    return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, left_lane_detected, right_lane_detected


def filter_by_density(nonzerox, nonzeroy, good_inds):
    # Initialize filtered lists
    density_threshold=20
    proximity=70
    filtered_inds = []
    
    # Iterate through good indices
    for idx in good_inds:
        # Calculate the distance to all other points
        distances = np.sqrt((nonzerox[idx] - nonzerox[good_inds])**2 + (nonzeroy[idx] - nonzeroy[good_inds])**2)
        
        # Count how many points are within the specified proximity
        density = np.sum(distances < proximity)
        
        # If density exceeds the threshold, keep the index
        if density > density_threshold:
            filtered_inds.append(idx)
    
    return np.array((filtered_inds),np.int_)


def create_lane_overlay(binary_warped_shape, left_fitx, right_fitx, ploty, left_lane_detected, right_lane_detected, lane_width_pixels=700):
    # Create an empty image to draw the lanes
    warp_zero = np.zeros((binary_warped_shape[0], binary_warped_shape[1])).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Proceed only if there are lanes detected
    
    if left_lane_detected:
        # Define points for constructing the polygon for filling between lanes
        if len(right_fitx) != len(ploty):
            right_fitx = np.linspace(np.min(right_fitx), np.max(right_fitx), len(ploty))
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))]) if left_lane_detected else None
    
    if right_lane_detected:
        if len(right_fitx) != len(ploty):
            right_fitx = np.linspace(np.min(right_fitx), np.max(right_fitx), len(ploty))
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]) if right_lane_detected else None

    # Create combined points array for the polygon based on detected lanes
    if left_lane_detected and right_lane_detected:
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    elif left_lane_detected:  # Only left lane detected, simulate right lane
        simulated_right_fitx = left_fitx + lane_width_pixels
        pts_simulated_right = np.array([np.flipud(np.transpose(np.vstack([simulated_right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_simulated_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    elif right_lane_detected:  # Only right lane detected, simulate left lane
        simulated_left_fitx = right_fitx - lane_width_pixels
        pts_simulated_left = np.array([np.transpose(np.vstack([simulated_left_fitx, ploty]))])
        pts = np.hstack((pts_simulated_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    return color_warp

def sample_polynomial(coefficients, y_start, y_end, step=20):
    # Create an array of y-values from y_start to y_end, stepping by 'step'
    y_values = np.arange(y_start, y_end, step)
    # Evaluate the polynomial at each of these y-values
    x_values = np.polyval(coefficients, y_values)
    return x_values, y_values

def validate_lane_origins(leftx, lefty, rightx, righty, frame_width, frame_height, last_left_cond, last_right_cond, left_lane_detected, right_lane_detected):
    #determine if last lane detected is held
    if last_left_cond or left_lane_detected: valid_left_lane = True
    else: valid_left_lane = False
    if last_right_cond or right_lane_detected: valid_right_lane = True
    else: valid_right_lane = False
    ##print(f"valid_left_lane:{valid_left_lane}, valid_right_lane:{valid_right_lane}")
    
    middle_x = frame_width // 2
    height_threshold = int(frame_height * 0.5)
    
    # Calculate average x positions
    left_ave_pos = np.average(leftx) if leftx.size else None
    right_ave_pos = np.average(rightx) if rightx.size else None
    
    # Define a minimum acceptable distance between lane averages (this could be adjusted)
    min_distance = frame_width * 0.1  # Example: 10% of the frame width

    # #print(f"left_ave_pos:{left_ave_pos} right_ave_pos:{right_ave_pos}")
    
    # Initial validation based on position and height
    if leftx.size and lefty.size:
        if np.any(lefty >= height_threshold) and np.min(leftx[lefty >= height_threshold]) < middle_x and last_left_cond:
            valid_left_lane = True

    if rightx.size and righty.size:
        if np.any(righty >= height_threshold) and np.max(rightx[righty >= height_threshold]) > middle_x and last_right_cond:
            valid_right_lane = True
    #print(f"valid_left_lane:{valid_left_lane}, valid_right_lane:{valid_right_lane}")
    
    if np.any(np.in1d(leftx, rightx)):
        left_overlap_count = np.sum(np.in1d(leftx, rightx))
        right_overlap_count = np.sum(np.in1d(rightx, leftx))
        total_left_pixels = len(leftx)
        total_right_pixels = len(rightx)
        #print(f' abs(left_overlap_count:{left_overlap_count} - right_overlap_count:{right_overlap_count}):{abs(left_overlap_count - right_overlap_count)}')
        if abs(left_overlap_count - right_overlap_count) >= 10:
            if total_left_pixels > total_right_pixels:
                valid_right_lane = False
            elif total_right_pixels >= total_left_pixels:
                valid_left_lane = False

    return valid_left_lane, valid_right_lane


def fit_polynomial(binary_warped, last_left_cond, last_right_cond):
    leftx, lefty, rightx, righty, _, _ ,left_lane_detected,right_lane_detected = find_lane_pixels(binary_warped, last_left_cond, last_right_cond)
    # Initialize default polynomials in case of no lane pixels detected
    frame_width, frame_height = binary_warped.shape[1], binary_warped.shape[0]
    valid_left_lane, valid_right_lane = validate_lane_origins(leftx, lefty, rightx, righty, frame_width, frame_height, last_left_cond, last_right_cond, left_lane_detected, right_lane_detected)
    left_lane_detected = left_lane_detected and valid_left_lane
    right_lane_detected = right_lane_detected and valid_right_lane
    #print(f'LEFT DETECTED----:{left_lane_detected}, RIGHT DETECTED ++++:{right_lane_detected}')
    left_fit = np.array([0, 0, 0])
    right_fit = np.array([0, 0, 0])
    ##print(f'binary_warped.shape[0] - 1:{binary_warped.shape[0] - 1}, binary_warped.shape[0]:{binary_warped.shape[1]-1}')
    ploty_left = np.linspace(0, frame_width - 1, frame_height)
    ploty_right = np.linspace(0, frame_width - 1, frame_height)
    #ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = np.zeros_like(ploty_left)  # Default if fitting fails or no lane found
    right_fitx = np.zeros_like(ploty_right)
    
    left_fitx = np.clip(left_fitx, 0, frame_width - 1)
    right_fitx = np.clip(right_fitx, 0, frame_width - 1) # Default if fitting fails or no lane found
    
    #Left Lane fit 
    if left_lane_detected:
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            #left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            left_fitx, ploty_left = sample_polynomial(left_fit, 0, frame_height, step=20)
            # Ensure that the fitted x-values do not go beyond the image width 
        except:
            print("FAILED to fit left line!")
            #left_fitx = np.zeros_like(ploty)  # Placeholder if fitting failed
    else:
        left_fit = np.array([0, 0, 0])
        ##print("NO LEFT LANE FOUND")   
        #right Lane fit 
    if right_lane_detected:
        try:
            right_fit = np.polyfit(righty, rightx, 2)
            right_fitx, ploty_right = sample_polynomial(right_fit, 0, frame_height, step=20)
            #right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
            # Ensure that the fitted x-values do not go beyond the image width
        except:
            print("FAILED to fit right line!")
            #right_fitx = np.zeros_like(ploty)  # Placeholder if fitting failed
    else:
        ##print("NO RIGHT LANE FOUND")
        right_fit = np.array([0, 0, 0])
    
    right_fitx = np.clip(right_fitx, 0, frame_width - 1)
    left_fitx = np.clip(left_fitx, 0, frame_width - 1)
    """
    plt.figure()
    plt.imshow(binary_warped, cmap='gray')
    if left_lane_detected:
        plt.plot(left_fitx, ploty_left, color='green', label='Left Lane Line')
    if right_lane_detected:
        plt.plot(right_fitx, ploty_right, color='red', label='Right Lane Line')
    plt.title('Lane Lines with Polynomial Fit')
    plt.xlim(0, frame_width)
    plt.ylim(frame_height, 0)
    plt.legend()

    plt.show(block=False)
    """
    #print(f'LEFT DETECTED----:{left_lane_detected}, RIGHT DETECTED ++++:{right_lane_detected}')
    ploty = ploty_left if left_lane_detected else ploty_right
    return left_fit, right_fit, left_fitx, right_fitx, ploty, left_lane_detected, right_lane_detected 


def map_value(x, in_min, in_max, out_min, out_max):
    # Map a value from one range to another
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def improved_lane_center(corrected_lines, lane_width_pixels, middle_x, left_lane_detected, right_lane_detected, last_lane_center):
    # Initialize lane_center with the desired_center_position as default
    lane_center = middle_x
    median_right_lane_x = np.mean(corrected_lines[1])
    median_left_lane_x = np.mean(corrected_lines[0])
    #print(f'median_right_lane_x:{median_right_lane_x}, median_left_lane_x:{median_left_lane_x}')
    # Check if both lanes are detected
    #print(f"corrected Lines:{corrected_lines}")
    if right_lane_detected and left_lane_detected:
        #print("inside both")trimmed.webm
        lane_center = (median_right_lane_x + median_left_lane_x) / 2
    elif right_lane_detected:
        #lane_center = (median_right_lane_x - lane_width_pixels) // 2
        
        lane_center = median_right_lane_x - (lane_width_pixels / 2)

        
        #lane_center = lane_center - (1024-median_right_lane_x//3)
        #print("inside right")
    elif left_lane_detected:
        #print("inside left")
        #lane_center = (median_left_lane_x + lane_width_pixels) // 2
        
        lane_center = median_left_lane_x + (lane_width_pixels / 2)

        
        #lane_center = lane_center + (median_left_lane_x//3)
    # Uncomment the following line if you want to handle cases where neither lane is detected
    else:
        lane_center = last_lane_center
    return lane_center
 
def angle_to_duty_cycle(angle):
    return (angle / 180.0) * 10 + 2.5

def warp_image(image):
    height, width = image.shape[:2]
    # Define calibration box in source (original) and destination (desired or warped) coordinates
    src = np.float32([
        [width * 0.25, height * 0.5],
        [width * 0.75, height * 0.5],
        [width * 0.05, height],
        [width * 0.95, height]
    ])
    dst = np.float32([
        [width * 0.15, 0],
        [width * 0.85, 0],
        [width * 0.15, height],
        [width * 0.85, height]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped, M


def filter_grey_road(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    base_threshold = 128
    brightness = hsv[:, :, 2].mean()  # Average brightness of the value channel
    #brightness_factor = brightness / 255
    #extra dark
    if brightness <= 30:
        base_h_lower_g, base_h_upper_g = 0, 179
        base_s_lower_g, base_s_upper_g = 0, 255
        base_v_lower_g, base_v_upper_g = 0, 40
    elif 30 < brightness < 50:
        base_h_lower_g, base_h_upper_g = 0, 179
        base_s_lower_g, base_s_upper_g = 0, 110
        base_v_lower_g, base_v_upper_g = 10, 142
    else:
        base_h_lower_g, base_h_upper_g = 0, 179
        base_s_lower_g, base_s_upper_g = 0, 91
        base_v_lower_g, base_v_upper_g = 74, 200
    lower_grey = np.array([base_h_lower_g, base_s_lower_g, base_v_lower_g])
    upper_grey = np.array([base_h_upper_g, base_s_upper_g, base_v_upper_g])
    print(f'brightness:{brightness}')
    # lower_grey = np.array([0, 0, 40])  # Light grey
    # upper_grey = np.array([160, 50, 200])  # Dark grey
    mask = cv2.inRange(hsv, lower_grey, upper_grey)
    return mask, brightness

def get_road_contour_mask(mask):
    # Detect contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(mask)
    if contours:
        # Assuming the largest contour is the road
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contours = sorted_contours[:2]  # Get the first two largest contours
        # largest_contour = max(contours, key=cv2.contourArea)
        # Draw and dilate the contour to slightly expand it
        cv2.drawContours(contour_mask, largest_contours, -1, (255), thickness=cv2.FILLED)
        dilated_contour_mask = cv2.dilate(contour_mask, np.ones((5, 5), np.uint8), iterations=1)
    return dilated_contour_mask

def filter_white_lanes(image, brightness):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #extra dark
    if brightness <= 30:
        base_h_lower_w, base_h_upper_w = 34, 153
        base_s_lower_w, base_s_upper_w = 0, 118
        base_v_lower_w, base_v_upper_w = 50, 255 
    elif 30 < brightness < 50:
    #dark conds
        base_h_lower_w, base_h_upper_w = 67, 125
        base_s_lower_w, base_s_upper_w = 0, 70
        base_v_lower_w, base_v_upper_w = 200, 255
    else: 
        #bright
        base_h_lower_w, base_h_upper_w = 30, 130
        base_s_lower_w, base_s_upper_w = 0, 45
        base_v_lower_w, base_v_upper_w = 215, 255

    lower_white = np.array([base_h_lower_w, base_s_lower_w, base_v_lower_w])
    upper_white = np.array([base_h_upper_w, base_s_upper_w, base_v_upper_w])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    # Define area thresholds
    min_area = 1000  # Minimum area to be considered as a lane
    max_area = 130000  # Maximum area to exclude large non-lane patches
    for contour in contours:
        area = cv2.contourArea(contour)
        #print(f'area:{area}')
        if min_area < area < max_area:
            cv2.drawContours(filtered_mask, [contour], -1, (255), thickness=cv2.FILLED)
    #cv2.imshow("White Mask", mask)
    #cv2.imshow("White after filtered_mask", filtered_mask)
    emphasized_mask = emphasize_lower_frame(filtered_mask, image.shape[0], image.shape[1])
    #cv2.imshow("emphasized White Mask", emphasized_mask)
    # Dilate the emphasized mask to connect fragmented markings
    dilated_mask = cv2.dilate(emphasized_mask, np.ones((5, 5), np.uint8), iterations=2)
    #cv2.imshow("dilated_mask", dilated_mask)
    
    return dilated_mask



# Example usage
# Assuming 'frame' is your input BGR image from the video capture
# dilated_mask = filter_white_lanes(frame, base_threshold, frame_brightness)


def emphasize_lower_frame(mask, height, width, factor=1.5):
    """ Apply a gradient emphasis to the lower part of the frame. """
    # Create a vertical gradient that increases from top to bottom
    emphasis = np.linspace(1, factor, height).reshape(-1, 1)
    # Repeat the emphasis across all columns
    full_emphasis = np.repeat(emphasis, width, axis=1)
    # Convert the emphasis to the same data type as the mask
    full_emphasis = np.uint8(full_emphasis * 255 / factor)  # Scale emphasis to the range of 0-255
    # Apply the gradient to the mask
    emphasized_mask = cv2.multiply(mask, full_emphasis, scale=1/255)  # Scale back down after multiplication
    return emphasized_mask


def main():

    steeringMax = 160
    steeringMin = 20
    # Define PID controller and video processing parameters
    pid = PIDController(kp=0.2, ki=0.2, kd=0.0001)
    kf = KalmanFilter(process_variance=1, measurement_variance=3, estimated_measurement_variance=1)
    GPIO.setwarnings(False)
    servo_pin = 33
    GPIO.setmode(GPIO.BOARD)  # Use physical pin numberintrimmed.webmg
    GPIO.setup(servo_pin, GPIO.OUT)
    pwm = GPIO.PWM(servo_pin, 50)
    pwm.start(0)  # Initialization with 0% duty cycle
    #lane width Parameters`
    lane_width_pixels = 550 #400
    #frame details
    frame_rate = 15  
    frame_count = 0
    last_lane_center = 512
    #video import
    videoPath ="/dev/video0" #"trimmed2803.webm"  
    capture = cv2.VideoCapture(videoPath)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #print(f"frames: width{frame_width}, =====  height {frame_height}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,1024)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT,768)
    middle_x = 512
    M = None
    M_inv = None
    first_frame = True
    capture.set(cv2.CAP_PROP_FPS, frame_rate)
    left_lane_detected = True
    right_lane_detected = True
    optimal_white = None
    command = 'S'
    while capture.isOpened():
        ret, frame = capture.read()
        #frame = cv2.resize(frame, (1024, 768))
        #print(ret)
        if ret == True:
            if first_frame:
                roi_mask = region_of_int(frame)
                roi_mask = roi_mask.astype(np.uint8)
                roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
                road_mask, brightness = filter_grey_road(roi_frame)
                #cv2.imshow("road_mask GREY", road_mask)
                
                road_contour_mask = get_road_contour_mask(road_mask)
                white_mask_dilated = filter_white_lanes(roi_frame, brightness)
                lanes_on_road_mask = cv2.bitwise_and(road_contour_mask, white_mask_dilated)
                lanes_on_road = cv2.bitwise_and(roi_frame, roi_frame, mask=lanes_on_road_mask)
                c1 = canny(lanes_on_road_mask)
                
                warped_image, M = warp_image(c1)
                M_inv = np.linalg.inv(M)
                middle_x = int((frame.shape[1])/2)
                first_frame = False
            elif frame_count %5 == 0:
                print(f'frame:{frame_count}')
                last_left_cond = left_lane_detected
                last_right_cond = right_lane_detected
                #print(f'last_right_cond:{last_right_cond}, right_lane_detected:{right_lane_detected}')
                #print(f'last_left_cond:{last_left_cond}, left_lane_detected:{left_lane_detected}')
                
                #nnprint(f'frame[0] ===== {frame.shape[0]}, frame.shape[1]:==={frame.shape[1]} ')
                roi_mask = region_of_int(frame)
                roi_mask = roi_mask.astype(np.uint8)
                roi_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
                road_mask, brightness = filter_grey_road(roi_frame)
                #cv2.imshow("road_mask GREY", road_mask)
                
                road_contour_mask = get_road_contour_mask(road_mask)
                white_mask_dilated = filter_white_lanes(roi_frame, brightness)
                lanes_on_road_mask = cv2.bitwise_and(road_contour_mask, white_mask_dilated)
                lanes_on_road = cv2.bitwise_and(roi_frame, roi_frame, mask=lanes_on_road_mask)
                c1 = canny(lanes_on_road_mask)
                
                warped_image, M = warp_image(c1)
                M_inv = np.linalg.inv(M)
                left_fit, right_fit, left_fitx, right_fitx, ploty, left_lane_detected, right_lane_detected = fit_polynomial(warped_image, last_left_cond, last_right_cond)
                corrected_lines = (left_fitx, right_fitx)
                #lane_center_x = lane_center(corrected_lanes, lane_width_pixels, middle_x)
                color_warp= create_lane_overlay(warped_image.shape, left_fitx, right_fitx, ploty, left_lane_detected, right_lane_detected, lane_width_pixels)
                #print(f'left_lane_main:{left_lane_detected}, right_lane_main:{right_lane_detected}')
                # Corrected usage
                if left_lane_detected:
                    for index in range(1, len(ploty)):
                        if index < len(left_fitx):
                            #print(f"Left lane detected: {left_lane_detected}")
                            cv2.line(color_warp, (int(left_fitx[index - 1]), int(ploty[index - 1])),
                                    (int(left_fitx[index]), int(ploty[index])), (255, 0, 0), 10)

                if right_lane_detected:
                    for index in range(1, len(ploty)):  # Ensure 'index' is defined as part of a new loop
                        if index < len(right_fitx):
                            cv2.line(color_warp, (int(right_fitx[index - 1]), int(ploty[index - 1])),
                                    (int(right_fitx[index]), int(ploty[index])), (0, 0, 255), 10)
                #look at the bottom of the image
                y_eval = int(np.max(ploty)-200)
                lane_center_raw = improved_lane_center(corrected_lines, lane_width_pixels, middle_x, left_lane_detected, right_lane_detected,last_lane_center)
                lane_center_bounded = max(min(lane_center_raw, 1023), 1)
                last_lane_center = lane_center_bounded
                #print(f'lane center before KF:{lane_center_raw}')
                lane_center_x = lane_center_bounded    #int(kf.update(lane_center_bounded))
                print(f'lane center x:{lane_center_x}')
                #calculate vehicle center
                newwarp = cv2.warpPerspective(color_warp, M_inv, (frame.shape[1], frame.shape[0]))
                result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
                cv2.circle(result, (int(lane_center_x), y_eval), 10, (0, 0, 255), -1)  # Lane center
                cv2.circle(result, (middle_x, y_eval), 10, (0, 0, 0), -1)  # Vehicle center
                
                # cv2.imshow("white_mask_dilated", white_mask_dilated)
                # cv2.imshow("canny", c1)
                # cv2.imshow("road_contour_mask", road_contour_mask)
                # cv2.imshow("lanes_on_road_mask", lanes_on_road_mask)
                # cv2.imshow("lanes_on_road", lanes_on_road)
                cv2.imshow('Lane Detection Result', result)
                # cv2.waitKey(0)
                # if cv2.waitKey(0) & 0xFF == ord('n'):
                #     cv2.destroyAllWindows()
                    # cv2.destroyWindow('white_mask_dilated')
                    # cv2.destroyWindow('Lane Detection Result')
                    # cv2.destroyWindow('road_contour_mask')
                    # cv2.destroyWindow('lanes_on_road')
                    # cv2.destroyWindow('lanes_on_road_mask')
                    # cv2.destroyWindow('canny')

                
                if frame_count > 10:
                    error = middle_x - lane_center_x
                    #error = vehicle_center_x - lane_center_x
                    #print(f"error before PID: {error}")
                    steering_adjustment = pid.update(error, 0.1/frame_rate)
                    #print(f'After PID')
                    #steering_angle = np.clip(steering_adjustment, 35, 155)
                    #print(f"steering_angle after clip: {steering_angle}")
                    #angle = float(steering_adjustment)
                    angle = 160 - (float(steering_adjustment) + 102.4)*(130/(102.4 + 102.4))
                    if left_lane_detected or right_lane_detected:
                        command = 'F'
                        print("Forward Sent")
                        send_data(command)
                    else:
                        command = 'S'
                        print("stop Sent")
                        send_data(command)
                    
                    #angle = calculate_steering_angle(steering_adjustment)
                    clip_angle = max(30, min(160, angle))
                    if 30 <= clip_angle <= 160:
                        duty_cycle = angle_to_duty_cycle(clip_angle)
                        print(f'duty cycle: {duty_cycle}, clipped angle: {clip_angle}')
                        pwm.ChangeDutyCycle(duty_cycle)
                    else:
                        duty_cycle = angle_to_duty_cycle(90.01)
                        pwm.ChangeDutyCycle(duty_cycle)
                #time.sleep(0.01)
                newwarp = cv2.warpPerspective(color_warp, M_inv, (frame.shape[1], frame.shape[0])) 

                ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("User Exited Program")
                    break  # Break the loop if 'q' is pressed
        else:
            print('No Image input, Exiting Program')
            pwm.ChangeDutyCycle(90.0)
            command = 'S'
            print("stop Sent")
            send_data(command)
            break
        frame_count += 1

    capture.release()
    cv2.destroyAllWindows()
    pwm.stop()
    GPIO.cleanup()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    # Open serial port
    try:
        ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        GPIO.setwarnings(False)
        time.sleep(2)  # wait for the serial connection to initialize
    except Exception as e:
        print(f"Could not open serial port: {e}")
        ser = None  # Ensure ser is defined even if the port couldn't be opened

    # Check CUDA availability
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        print("CUDA not available - the program requires a GPU with CUDA.")
        exit()  # Exit if CUDA is not available
    print("Serial port is connected and GPU is available")
    time.sleep(1)
    main()
    print("test")

    # Close serial if open
    if ser is not None:
        ser.close()
