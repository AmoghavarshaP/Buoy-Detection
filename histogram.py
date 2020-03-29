import cv2
import numpy as np
import matplotlib.pyplot as plot

if __name__ == '__main__':
    # Read image
    roi = cv2.imread("cropped_imagesY/frame0.png")
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    target = cv2.imread("data/frame170.png")
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    histogram = cv2.calcHist([hsv], [0, 1], None, [75, 256], [0, 75, 90, 256])

    # normalize histogram and apply backprojection
    cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt], [0, 1], histogram, [0, 75, 90, 256], 1)

    # Now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(dst, -1, disc, dst)

    # threshold and binary AND
    ret, thresh = cv2.threshold(dst, 50, 255, 0)
    thresh = cv2.merge((thresh, thresh, thresh))
    res = cv2.bitwise_and(target, thresh)

    # res = np.vstack((target, thresh, res))
    cv2.imshow('result', res)
    # cv2.imwrite('res.jpg', res)
    plot.show()
    # Display cropped image
    # cv2.imshow("Image", roi)
    cv2.imwrite("Results/segment_1dY.png", res )
    cv2.waitKey(0)


