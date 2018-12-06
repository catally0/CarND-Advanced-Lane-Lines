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

def save_img(img,filename='./output_image/undefined.jpg'):
    cv2.imwrite(filename,img)

def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

def threshhold_binary(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
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
    plt.imshow(color_binary, cmap='gray')
    plt.show()
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary



test_img=glob.glob('./test_images/test*.jpg')
i=1
for fname in test_img:
    img = cv2.imread(fname) 
    undist_img = undistort(img)
    binary_img=threshhold_binary(undist_img)
    #save_img(binary_img,'./output_images/'+str(i)+'.jpg')
    i+=1



