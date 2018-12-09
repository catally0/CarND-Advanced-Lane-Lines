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

    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary

def warper(img):
    img_size=(img.shape[1], img.shape[0])
    src=np.float32([[278,675], [602,445],  [681,445],  [1041,675]])
    dst=np.float32([[320,720], [320,0],   [960,0],    [960, 720]])

    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    
    ''''''
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.plot(src[:2,0],src[:2,1],'r--', lw=2)
    ax1.plot(src[-2:,0],src[-2:,1],'b--', lw=2)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(warp)
    ax2.plot(dst[:2,0],dst[:2,1],'r--', lw=2)
    ax2.plot(dst[-2:,0],dst[-2:,1],'b--', lw=2)
    ax2.set_title('Warp Image', fontsize=30)
    
    plt.savefig('.\\output_images\\warp2.png')
    plt.show()
    return warp

    
    
img=mpimg.imread('./test_images/straight_lines2.jpg')
img=undistort(img)
warper(img)


