import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

cal_file='cal.npz'
mtx=[]
dist=[]


if os.path.exists(cal_file):  
    print('Calibration file exists. Load file...')
    cal_data=np.load(cal_file)
    mtx=cal_data['mtx']
    dist=cal_data['dist']
else:
    print('Calibration not exist')
    exit(-1)

def save_img(img,filename='./output_image/undefined.jpg',binary=False):
    if binary==True:
        img=img*255
    cv2.imwrite(filename,img)


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  

left_lines=[]
right_lines=[]
frame_num=0
last_left_line=Line()
last_right_line=Line()
last_n_left_fitx=[]
last_n_right_fitx=[]

def average_n_fit(fit,new_fit,detection,n=5):
    if len(fit) >= n:
        fit.pop()
        if detection == True:
            fit.insert(0,new_fit)
        return fit,np.mean(np.array(fit), axis=0)
    else:
        if detection == True:
            fit.insert(0,new_fit)
        return fit, np.mean(fit, axis=0)



def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

def threshhold_binary(img, s_thresh=(130, 240), sx_thresh=(20, 100), verbose=False):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.zeros_like(s_binary)
    color_binary[(s_binary==1) | (sxbinary==1)]=1
    if verbose==True:
        plt.imshow(s_channel)
        plt.show()
        color_img = np.dstack(( np.zeros_like(s_binary), s_binary, sxbinary)) * 255
        plt.imshow(color_img)
        plt.show()
    return color_binary

def warper(img):
    img_size=(img.shape[1], img.shape[0])
    src=np.float32([[278,675], [602,445],  [681,445],  [1041,675]])
    dst=np.float32([[320,720], [320,0],   [960,0],    [960, 720]])

    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    
    '''
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.plot(src[:2,0],src[:2,1],'r--', lw=2)
    ax1.plot(src[-2:,0],src[-2:,1],'b--', lw=2)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(warp)
    ax2.plot(dst[:2,0],dst[:2,1],'r--', lw=2)
    ax2.plot(dst[-2:,0],dst[-2:,1],'b--', lw=2)
    ax2.set_title('Warp Image', fontsize=30)
    
    #plt.savefig('.\\output_images\\warp2.png')
    plt.show()
    '''
    return warp

def unwarper(img):
    img_size=(img.shape[1], img.shape[0])
    src=np.float32([[278,675], [602,445],  [681,445],  [1041,675]])
    dst=np.float32([[320,720], [320,0],   [960,0],    [960, 720]])

    M = cv2.getPerspectiveTransform(dst, src)
    warp = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    
    '''
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.plot(src[:2,0],src[:2,1],'r--', lw=2)
    ax1.plot(src[-2:,0],src[-2:,1],'b--', lw=2)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(warp)
    ax2.plot(dst[:2,0],dst[:2,1],'r--', lw=2)
    ax2.plot(dst[-2:,0],dst[-2:,1],'b--', lw=2)
    ax2.set_title('Warp Image', fontsize=30)
    
    #plt.savefig('.\\output_images\\warp2.png')
    plt.show()
    '''
    return warp

def hist(img):
    # TO-DO: Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]

    # TO-DO: Sum across image pixels vertically - make sure to set `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)

    return histogram

def find_lane_pixels(binary_warped, verbose=False):
    global last_left_line
    global last_right_line

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    if verbose==True:
        print('hist')
        plt.plot(histogram)
        plt.show()

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)

    #print(last_left_line.line_base_pos, np.argmax(histogram[:midpoint]))
    if last_left_line.detected == True:
        #print('center in last line')
        leftx_base = last_left_line.line_base_pos
    else:
        leftx_base = np.argmax(histogram[:midpoint])

    if last_right_line.detected == True:
        #print('center in last line')
        rightx_base = last_right_line.line_base_pos
    else:
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin_left = 100
    margin_right = 100
    # Set minimum number of pixels found to recenter window
    minpix = 200

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    leftx_lastoffset = 0
    rightx_current = rightx_base
    rightx_lastoffset = 0

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = int(leftx_current - margin_left)
        win_xleft_high = int(leftx_current + margin_left)
        win_xright_low = int(rightx_current - margin_right)
        win_xright_high = int(rightx_current + margin_right)
        
        if verbose==True:
        # Draw the windows on the visualization image
            print(win_xleft_low,win_y_low,win_xleft_high,win_y_high)
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > 0:
            u=np.mean(nonzerox[good_left_inds])
            s=np.std(nonzerox[good_left_inds])

            if verbose==True:
                print('Left Window:#'+str(window))
                print(len(good_left_inds), u, s)
            
            # Remove outliers
            outliers=[]
            for ind in good_left_inds:
                if nonzerox[ind]<u-2*s or nonzerox[ind]>u+2*s:
                    outliers.append(ind)
            good_left_inds = np.array([e for e in good_left_inds if e not in outliers])
        else:
            if verbose==True:
                print('No pix found for left lane in window #'+str(window))


        if len(good_left_inds) > minpix:
            u=np.mean(nonzerox[good_left_inds])
            s=np.std(nonzerox[good_left_inds])

            if s < 30:
                leftx_lastoffset = np.int(u) - leftx_current
                if(leftx_lastoffset > 50):
                    leftx_lastoffset=50
                elif(leftx_lastoffset < -50):
                    leftx_lastoffset=-50
                left_lane_inds.append(good_left_inds)
                margin_left = 100
            else:
                margin_left = 150
        else:
            margin_left = 150    
        leftx_current += leftx_lastoffset


        if len(good_right_inds) > 0:
            u=np.mean(nonzerox[good_right_inds])
            s=np.std(nonzerox[good_right_inds])

            if verbose==True:  
                print('Right Window:#'+str(window))
                print(len(good_right_inds), np.mean(nonzerox[good_right_inds]), np.std(nonzerox[good_right_inds]))

            # Remove outliers
            outliers=[]
            for ind in good_right_inds:
                if nonzerox[ind]<u-2*s or nonzerox[ind]>u+2*s:
                    outliers.append(ind)
            good_right_inds = np.array([e for e in good_right_inds if e not in outliers])
        else: 
            if verbose==True:
                print('No pix found for right lane in window #'+str(window))

        if len(good_right_inds) > minpix:
            u=np.mean(nonzerox[good_right_inds])
            s=np.std(nonzerox[good_right_inds])

            if s < 30:
                rightx_lastoffset = np.int(u) - rightx_current
                if(rightx_lastoffset > 50):
                    rightx_lastoffset=50
                elif(rightx_lastoffset < -50):
                    rightx_lastoffset=-50
                right_lane_inds.append(good_right_inds)
                margin_right = 100
            else:
                margin_right = 150
        else:
            margin_right = 150
        rightx_current += rightx_lastoffset

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

ym_per_pix = 50/720 # meters per pixel in y dimension
xm_per_pix = 3.7/640 # meters per pixel in x dimension

def sanity_check(left_fitx, right_fitx):
    global last_left_line
    global last_right_line
    lane_width=right_fitx-left_fitx
    u=np.mean(lane_width)
    s=np.std(lane_width)
    if u>500 and u<750 and s<60:
        return True
    else:
        return False

def fit_polynomial(binary_warped, verbose=False, return_type='img'):
    global last_left_line
    global last_right_line
    global last_n_left_fitx
    global last_n_right_fitx

    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped, verbose=verbose)

    if verbose==True:
        plt.imshow(out_img)
        plt.show()

    # Fit a second order polynomial to each using `np.polyfit`
    
    if return_type=='img':
        
        
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        # Generate x and y values for plotting
        
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            if verbose==True:
                print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty
        
        found_lane=False
        if sanity_check(left_fitx, right_fitx):
            last_left_line.detected = True
            last_right_line.detected = True
            last_left_line.line_base_pos = left_fitx[-1]
            last_right_line.line_base_pos = right_fitx[-1]
            last_n_left_fitx, left_fitx = average_n_fit(last_n_left_fitx,left_fitx,last_left_line.detected)
            last_n_right_fitx, right_fitx = average_n_fit(last_n_right_fitx,right_fitx,last_right_line.detected)

            ## Visualization ##
            # Colors in the left and right lane regions
            '''
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.imshow(out_img)
            plt.show()
            '''


            path_img = np.zeros_like(out_img)

            fitx=np.concatenate((left_fitx, right_fitx[::-1]), axis=None)
            ploty=np.concatenate((ploty,ploty[::-1]),  axis=None)
            pts = np.vstack((fitx,ploty)).astype(np.int32).T
            cv2.fillPoly(path_img,np.int32([pts]),(0,255,0))
            found_lane=True
            


        else:
            last_left_line.detected = False
            last_right_line.detected = False
            last_n_left_fitx, left_fitx = average_n_fit(last_n_left_fitx,left_fitx,last_left_line.detected)
            last_n_right_fitx, right_fitx = average_n_fit(last_n_right_fitx,right_fitx,last_right_line.detected)
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]
            path_img = np.zeros_like(out_img)

            fitx=np.concatenate((left_fitx, right_fitx[::-1]), axis=None)
            ploty=np.concatenate((ploty,ploty[::-1]),  axis=None)
            pts = np.vstack((fitx,ploty)).astype(np.int32).T
            cv2.fillPoly(path_img,np.int32([pts]),(0,255,0))

            found_lane=False
        return found_lane, path_img
        
    else:
        try:
            left_fit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
            right_fit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            if verbose==True:
                print('The function failed to fit a line!')

        

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

        return ploty, left_fit, right_fit

def measure_curvature(ploty, left_fit, right_fit):
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    return left_curverad, right_curverad

def measure_offset(ploty, left_fit, right_fit):
    y_eval = np.max(ploty)*ym_per_pix

    left_fitx = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_fitx = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]

    offset_pix=(right_fitx+left_fitx)/2-(640*xm_per_pix)
    return offset_pix


def draw_path(img, path):
    return cv2.addWeighted(img, 1, path, 0.3, 0.)

def add_text(img, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    cv2.putText(img, text, (position[0]+2, position[1]+2), font, 1., (0, 0, 0), 2)
    cv2.putText(img, text, position, font, 1., (255, 255, 255), 2)



def pipeline(img, verbose=False):
    global frame_num
    global last_left_line
    global last_right_line
    frame_num +=1
    undist=undistort(img)
    warp=warper(undist)
    
    binary_warped=threshhold_binary(warp, verbose=verbose)


    found_lane, fit_img=fit_polynomial(binary_warped, verbose=verbose)

    if found_lane==True:
        
        mapped_img=draw_path(img, unwarper(fit_img))
    
        ploty, left_fit, right_fit = fit_polynomial(binary_warped, return_type='poly', verbose=verbose)
    
        left_curverad, right_curverad = measure_curvature(ploty, left_fit, right_fit)
        r_curvature = int((left_curverad + right_curverad)/2)
        veh_offset=measure_offset(ploty, left_fit, right_fit)
        
        add_text(mapped_img, 'Radius of curvature:'+str(r_curvature)+'m', (50, 50))
        add_text(mapped_img, 'Offset:'+str(round(veh_offset,2))+'m', (50, 100))

        last_left_line.radius_of_curvature = left_curverad
        '''
        cv2.imshow('img', cv2.cvtColor(mapped_img, cv2.COLOR_RGB2BGR))
        #cv2.imwrite('./output_images/example.jpg',cv2.cvtColor(mapped_img,cv2.COLOR_RGB2BGR))
        cv2.waitKey()
        '''

        return mapped_img
    else:
        cv2.imwrite('./unrec_img/'+str(frame_num)+'.jpg',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
        mapped_img=draw_path(img, unwarper(fit_img))
    
        ploty, left_fit, right_fit = fit_polynomial(binary_warped, return_type='poly')
    
        left_curverad, right_curverad = measure_curvature(ploty, left_fit, right_fit)
        r_curvature = int((left_curverad + right_curverad)/2)
        veh_offset=measure_offset(ploty, left_fit, right_fit)
        
        add_text(mapped_img, 'Radius of curvature:'+str(r_curvature)+'m', (50, 50))
        add_text(mapped_img, 'Offset:'+str(round(veh_offset,2))+'m', (50, 100))

        return mapped_img

''' 
unrec_img=glob.glob('./unrec_img/*.jpg')
for i in unrec_img[:20]:
    img=mpimg.imread(i)
    cv2.imshow('img', cv2.cvtColor(pipeline(img, verbose=True), cv2.COLOR_RGB2BGR))
    cv2.waitKey()




'''

from moviepy.editor import VideoFileClip

project_output = './output_project.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip = VideoFileClip("./project_video.mp4")
white_clip = clip.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(project_output, audio=False)


