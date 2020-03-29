import numpy as np
import cv2

def save_images(video):
    return cv2.imwrite("data/frame%d.png" % count, video)

def mouse_click(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        points.append([x, y])
    return points

def crop_images(video, points):
    image = cv2.imread("data/frame0.png")
    cropped_image = image[points[0][1]:points[1][1], points[0][0]:points[3][0]]
    return cropped_image

# Capture video from file
cap = cv2.VideoCapture('detectbuoy.avi')

count = 0
points = []
while True:

    _, frame = cap.read()
    # save_images(frame)
    count += 1

    cv2.namedWindow("frame", 1)
    cv2.setMouseCallback("frame", mouse_click)
    cv2.imshow('frame', frame)
    # print(points)
    # cropped_image = crop_images(frame,points)
    # cv2.imshow("cropped_image", cropped_image)
    if cv2.waitKey(60) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()