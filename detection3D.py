import cv2
import numpy as np
import numpy.linalg as lin
import random
import scipy

def writer(vid):
    height, width, layers = vid.shape
    size = (width, height)
    cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc('M','P','E','G'), 30, size)

def calcGaussian(pixel, mean, variance):

    # matrix conversion
    mean = np.array(mean)
    pixel = np.array(pixel)
    variance = np.array(variance)

    # calculate gaussian
    den = (2*np.pi*np.linalg.det(variance))**(3/2)
    diff = pixel - mean
    N = (1 / den) * np.exp((-1 / 2) * np.matmul(np.matmul(diff, (np.linalg.inv(variance))), diff.T))

    return N
def probability(vid):
    mean = np.array([[192.18277387, 248.02716919, 160.15305901],
                     [118.34966997, 195.96161778, 119.86500038]])
    weight = np.array([[0.38780729], [0.61219271]])
    sigma = np.array([[[533.97805778, 628.53245143], [24.2028958, 618.59601136], [374.47564871, 380.58925531]],
                      [[24.2028958, 618.59601136], [34.46157895, 887.41899556], [-10.27712459, 362.22617415]],
                      [[374.47564871, 380.58925531], [-10.27712459, 362.22617415], [343.15834175, 262.078583]]])
    probVid = 0 * vid
    height, width, channel = vid.shape
    for i in range(height):
        for j in range(width):
            # if (i > 370) or (i < 200):
            #     vid[i,j,:] = [0, 0, 0]
            # print(vid[i,j,:])

            pixel = vid[i,j,:]
            # print(pixel)
            probVid[i, j, 0] = 25500 * (weight[0, 0] * calcGaussian(pixel, mean[0, :], sigma[:, :, 0]) + weight[1, 0] * calcGaussian(pixel, mean[1, :], sigma[:, :, 1]))
            # print(probVid)
            if probVid[i,j,2] < 242:
                probVid[i,j,:] = [0,0,0]
            # print(probVid)

    return probVid





# def printImage(vid):
#     height, width, channel = vid.shape
#     for i in range(height):
#         for j in range(width):
#             # print(img[i,j,2])
#             if vid[i,j,2] < 242:
#                 vid[i,j,:] = [0,0,0]
#
#
#     return vid

def circleDraw(img, cimg):
    red = [0, 0, 255]
    green = [0, 255, 0]
    blue = [255, 0, 0]


    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,180,
                                param1=100,param2=10,minRadius=13,maxRadius=30)
    # print(circles[0,19,:])
    # print(np.size(circles))
    circles = np.uint16(np.around(circles))
    # print(circles)
    print(circles[0,:])
    for i in circles[0,:]:
        if i[1] > 225 and i[1] < 400:
            cv2.circle(cimg,(i[0],i[1]),i[2],(red),2)
        # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(blue),3)
    return cimg

if __name__ == '__main__':
    img = cv2.imread('data_set/frame-10.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    cimg = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    print(img.shape[:])
    # result = circleDraw(blur, cimg)
    # cv2.imshow('detected circles',result)
    # black = printImage(img)

    video = cv2.VideoCapture("detectbuoy.avi")

    while True:
        ret, frame = video.read()
        if ret == 1:

            bframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cframe = cv2.GaussianBlur(bframe, (9,9), 0)
            cframe = cv2.cvtColor(bframe, cv2.COLOR_GRAY2BGR)
            # res = circleDraw(bframe, frame)
            # segmented = printImage(frame)
            # cv2.imshow("black", segmented)
            # cv2.imshow("frame", res)
            # writer(segmented)
            # vid = probability(frame)
            # cv2.imshow("black", vid)
            proVid = probability(frame)
            cv2.imshow("frame",proVid)

        if cv2.waitKey(30) and 0Xff == ord('q'):
            break
    cv2.destroyAllWindows()
    video.release()


