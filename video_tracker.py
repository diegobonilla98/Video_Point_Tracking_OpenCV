import cv2
from math import sin, cos, atan2, sqrt, degrees

sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()
video = cv2.VideoCapture('video_test.mp4')
old_frame = None
old_kp = None
old_des = None
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    height, width = frame.shape[:2]
    original = frame.copy()
    frame = cv2.resize(frame, (int(width * 1.5), int(height * 1.5)), cv2.INTER_AREA)
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
        if m.distance < 0.35 * n.distance:
            pt1 = old_kp[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            old_new_image_kp.append(pt1 + pt2)
    angles = 0
    magnitudes = 0
    total_num = len(old_new_image_kp)
    for coords in old_new_image_kp:
        (x1, y1, x2, y2) = [int(c) for c in coords]
        cv2.circle(frame, (x2, y2), 5, (0, 0, 255), 1)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        delta_y = y2-y1
        delta_x = x2-x1
        angles += atan2(delta_y, delta_x)
        magnitudes += sqrt(delta_y**2 + delta_x**2)
    mean_angle = angles / total_num
    mean_magnitude = magnitudes / total_num
    x_start = width//2
    y_start = height//2
    y_end = int(sin(mean_angle) * mean_magnitude + y_start)
    x_end = int(cos(mean_angle) * mean_magnitude + x_start)
    cv2.arrowedLine(delta_pos, (x_end, y_end), (x_start, y_start), (0, 0, 255), 5)
    cv2.putText(delta_pos, "Rel. Angle: " + str(round(degrees(mean_angle), 1)), (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(delta_pos, "Rel. Displacement: " + str(round(mean_magnitude, 1)), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imshow('Motion tracker', frame)
    cv2.imshow('Original', original)
    cv2.imshow('Delta position', delta_pos)
    if cv2.waitKey(1) == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
