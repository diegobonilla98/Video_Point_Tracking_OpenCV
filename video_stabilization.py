import cv2
from math import sin, cos, atan2, sqrt
import numpy as np

sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()
# video = cv2.VideoCapture('unstable_video.mp4')
video = cv2.VideoCapture(0)
M = np.float32([[1, 0, 0], [0, 1, 0]])
old_frame = None
old_kp = None
old_des = None
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    height, width = frame.shape[:2]
    frame = frame[70:height-70, 0:width]
    # frame = cv2.resize(frame, (int(width // 2), int(height // 2)), cv2.INTER_AREA)
    delta_pos = frame.copy()
    height, width = frame.shape[:2]
    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if old_frame is None:
        old_frame = new_frame
        old_kp, old_des = sift.detectAndCompute(old_frame, None)
        continue
    kp2, des2 = sift.detectAndCompute(new_frame, None)
    matches = bf.knnMatch(old_des, des2, k=2)
    old_new_image_kp = []
    for m, n in matches:
        if m.distance < 0.3 * n.distance:
            pt1 = old_kp[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            old_new_image_kp.append(pt1 + pt2)
    angles = 0
    magnitudes = 0
    total_num = len(old_new_image_kp)
    if total_num < 5:
        old_frame = new_frame
        old_kp, old_des = sift.detectAndCompute(old_frame, None)
        continue
    for coords in old_new_image_kp:
        (x1, y1, x2, y2) = [int(c) for c in coords]
        delta_y = y2-y1
        delta_x = x2-x1
        angles += atan2(delta_y, delta_x)
        magnitudes += sqrt(delta_y**2 + delta_x**2)
    mean_angle = angles / total_num
    mean_magnitude = magnitudes / total_num
    y_end = int(sin(mean_angle) * mean_magnitude)
    x_end = int(cos(mean_angle) * mean_magnitude)
    M[0][2] = -x_end
    M[1][2] = -y_end
    delta_pos = cv2.warpAffine(delta_pos, M, (width, height))
    centerX, centerY = int(height / 2), int(width / 2)
    radiusX, radiusY = int(40 * height / 100), int(40 * width / 100)
    minX, maxX = centerX - radiusX, centerX + radiusX
    minY, maxY = centerY - radiusY, centerY + radiusY
    delta_pos = delta_pos[minX:maxX, minY:maxY]
    delta_pos = cv2.resize(delta_pos, (width, height))
    cv2.imshow('Original', frame)
    cv2.imshow('Delta position', delta_pos)
    if cv2.waitKey(1) == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
