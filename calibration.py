import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import os

cal_file='cal.npz'
file_list = glob.glob('camera_cal/calibration*.jpg')

if not os.path.exists(cal_file):
    print("Calibration file not found, start calibration...")
    # prepare object points
    nx = 9#TODO: enter the number of inside corners in x
    ny = 6#TODO: enter the number of inside corners in y

    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    for fname in file_list:
        print(fname)

        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret == True:
            # Draw and display the corners
            print('Found corners')
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    np.savez('cal', mtx=mtx, dist=dist)
else:
    print('Calibration file exists. Load file...')
    cal_data=np.load(cal_file)
    mtx=cal_data['mtx']
    dist=cal_data['dist']

i=0
for fname in file_list:
    img = cv2.imread(fname)
    #cv2.imshow('img', img)
    #cv2.waitKey(500)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    #cv2.imshow('img', dst)
    #cv2.waitKey(500)
    
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    dst=cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.savefig('camera_cal\\undist_'+str(i)+'.png')
    i+=1

