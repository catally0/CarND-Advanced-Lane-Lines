## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/undist_11.png "Undistorted11"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/binary_1.jpg "Binary Example"
[image4]: ./output_images/warp2.png "Warp Example"
[image5]: ./output_images/window.png "Window visual"
[image6]: ./output_images/example.jpg "Output"
[video1]: ./output_project.mp4 "Video"
[image7]: ./test_images/undist_test1.png "Undistort road image"
[image8]: ./output_images/fit_line.png "Fit visual"
[image9]: ./output_images/curve_fomula.png "Formula"
[image10]: ./output_images/histogram.png "Histogram"

---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `calibration.py`

First I check whether `cal.npz` exists. This is the file that I choose to store the calibration data. If the file doesn't exist, the calibraiton will start.

I started the calibration by defining the number of chessboard corners. By looking at the pictures in `camera_cal` folder, I can see that there're 6 rows and each row has 9 corners. Then I create a coordinate list for all these points, which looks like [[0,0,0],[1,0,0],...[0,1,0],[1,1,0]...,[8,5,0]]. Here I am assuming the chessboard is fixed on the (x,y) plane at z=0. This list will be used as object points to represent the the corner points found in actual calibration image.

Then I iterate the calibration images in camera_cal folder. For each image, I covert it to gray and use `cv2.findChessboardCorners` to search for corners. If the corners are found, I will add the object points and actual image points to a list called `objpoints` and `imgpoints`.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. Then I save the calibration result `mtx` and `dist` to `cal.npz`.

If `cal.npz` exists from beginning, I will load them and not to run the calibration again.

After getting the calibration data, I applied this distortion correction to the calibration image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

It's quite simple. Like I did in `calibration.py`, I firstly load the calibration data `mtx` and `dist` in `cal.npz`. If the calibration file doesn't exist, then exit.

Then I define a function that takes the loaded image as input and return undistot image by using `cv2.undistort` with parameter `mtx` and `dist`, which are loaded above.

So for the example image above, the undistort result looks like this

![alt text][image7]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in function `threshhold_binary` in `pipeline.py`).  Here's an example of my output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper` in `pipeline.py`. The `warper` function takes an image as input and returns the warped imange based on perspective transform calculation.

I manually picked 4 source points on the left and right lane in `./testimages/straight_lines2.jpg`. Providing the lane is straight, I can give the corresponding destination points for these 4 source points

| Source        | Destination   | Location | 
|:-------------:|:-------------:|:--------:|
| 278, 675      | 320, 720      | Bottom, Left|
| 602, 445      | 320, 0        | Upper, Left|
| 681, 445      | 960, 0        | Upper, Right|
| 1041, 675     | 960, 720      | Bottom, Right|

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

First I defined a Line class to tell the history lane finding result. If it's the first image or there is no lane found in the hisotry, it will start to search where the left and right line begins. By drawing a histogram for the bottom part of binary warped image, I can see the peak in the left and right part, which are the base that the line search should start with. Take the histogram below as an example, I will start to search the left line from around 380 and right from around 960.

![alt text][image10]

Otherwise, I will just use the previous line base position to start with.

- Starting from base position, I use 9 sliding windows respectively for left and right, which has width=100px and height=(img_height/9), to search the line pixels within the window. 

![alt text][image5]

- I will exclude the outliers first.  If the number of remaining pixels is less than `minpix`, I will say there're no enough evidence to tell whether the line exists or not
- Then if the remaining points are too much wide spreaded, I would say there's too much noise.
- If I see a lot of points with a small deviation on x. I would say they are line pixels
- Then I will combine all the founded line pixels for left and right respectively.
- With all the line pixels, I can fit a 2 degree polynomial by using `np.polyfit`.

![alt text][image8]

- Then I will check the distance between right and left lines, if it's too small or too wide, or varies too much, it means the result is not valid. And the lane is not actually detected.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in function `measure_curvature` in my code in `pipeline.py` based on the following formula.

![alt text][image9]

To convert the pixel result to real world unit, I estimate the lane width is 3.7m of around 640px and the longitude length of road is around 50 meters. So every pixel in x direction represents 3.7/640 meter, and every pixel in y direction 50/720 meter.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #407-#412 and #426-#429 in my code in `pipeline.py` in the function `fit_polynomial`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_project.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This code in pipeline.py is not a robust. It doesn't take left and right line as sepearate line to track, instead it tracks the lane. If there's only one line is detected, the pipeline would fail. Plus, if both line are in the left/right part of image, the pipeline will not be able to make any detection.

To make it more robust, I probably want to track the left and right line seperately and apply a confidence level to each line. In this case, I can start to search one of line based on the position of the other line with higher confidence level.