# Video_Point_Tracking_OpenCV
Track points in a video using the SIFT algorithm and OpenCV.

I recently read about scale-invariant feature extraction. Basically, from a image you get some keypoints using difference of Gaussian functions which eliminates unstable points **and** also you get *descriptors*. What they do is get a reference direction based on the direction and magnitude of the image gradient around each point to archive *inveriance to rotation*. Now you have information of pretty good stable points, its position, scale and rotation. (More info: https://www.cs.ubc.ca/~lowe/keypoints/)

For each frame you get the keypoints and the descriptors (using the OpenCV SIFT implementation) and store them. For the next frame you also get the keypoins and the descriptors and you compare them to the last frame's ones using the KNNMatch function. You filter them for more stability (by distance) and there you go, you just tracked some points in a video and got information about the relative angle and position of the camera. Now you can project a 3D model or whatever mentally insane idea you think about.


![](video_tracker.gif)


