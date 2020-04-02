import cv2
import matplotlib.pyplot as plot
import numpy as np
import numpy.linalg as alg
import os

i = 0
mu = [0, 0, 0]
for i in range(118):
    image = cv2.imread("Red_DataSet/Crop_segmented/Train/frame" + str(i) + '.png')
    mu += np.mean(image, axis=(0, 1))
    # print(i)

# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# pixel = image[0]
# print(pixel[0])
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# histogram = cv2.calcHist([hsv], [0, 1], None, [75, 256], [0, 75, 90, 256])
# plot.plot(histogram, 'r')
# plot.show()
# cv2.imshow("image", image)
# for images in list

mean = mu/117
cv2.waitKey()
