import cv2
import numpy as np
import numpy.linalg as lin
import random
import scipy
import os
import copy


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

def probabilityRed(vid):
    mean = np.array([[118.43316908, 183.83282332, 243.42817752], [140.93158785, 186.2959427, 225.12107122]])
    weight = np.array([[0.70447737], [0.29552263]])
    sigma = np.array([[[230.46876748, 488.78160736], [388.37900424, 622.98223896], [-75.70861234, 45.06529594]],
                      [[388.37900424, 622.98223896], [732.18245657, 1001.46283557], [-148.81014188, 172.54529128]],
                      [[-75.70861234, 45.06529594], [-148.81014188, 172.54529128], [49.1459872, 332.56742226]]])
    probVid = vid
    height, width, channel = vid.shape
    for i in range(height):
        for j in range(width):
            pixel = vid[i,j,:]
            # print(pixel)
            probVid[i, j, 0] = 25500 * (weight[0, 0] * calcGaussian(pixel, mean[0, :], sigma[:, :, 0]) + weight[1, 0] * calcGaussian(pixel, mean[1, :], sigma[:, :, 1]))
            if probVid[i,j,2] < 242:
                probVid[i,j,:] = [0,0,0]

    return probVid

def probabilityYellow(vid):
    mean = np.array([[147.5337616, 248.20747064, 237.80292652], [116.22233547, 218.95261565, 219.68868842]])
    weight = np.array([[0.65543956], [0.34456044]])
    sigma = np.array([[[932.48845401, 977.39506935], [-23.68820292, 301.16232229], [-48.74952313, 52.70243927]],
                      [[-23.68820292, 301.16232229], [7.89874624, 681.30543371], [3.76102091, 811.14484296]],
                      [[-48.74952313, 52.70243927], [3.76102091, 811.14484296], [12.15513611, 1231.3601203]]])

    probVid = vid
    height, width, channel = vid.shape
    for i in range(height):
        for j in range(width):
            pixel = vid[i,j,:]
            # print(pixel)
            probVid[i, j, 0] = 25500 * (weight[0, 0] * calcGaussian(pixel, mean[0, :], sigma[:, :, 0]) + weight[1, 0] * calcGaussian(pixel, mean[1, :], sigma[:, :, 1]))
            if probVid[i,j,2] < 242:
                probVid[i,j,:] = [0,0,0]

    return probVid

def probabilityGreen(vid):
    mean = np.array([[192.18277387, 248.02716919, 160.15305901],
                     [118.34966997, 195.96161778, 119.86500038]])
    weight = np.array([[0.38780729], [0.61219271]])
    sigma = np.array([[[533.97805778, 628.53245143], [24.2028958, 618.59601136], [374.47564871, 380.58925531]],
                      [[24.2028958, 618.59601136], [34.46157895, 887.41899556], [-10.27712459, 362.22617415]],
                      [[374.47564871, 380.58925531], [-10.27712459, 362.22617415], [343.15834175, 262.078583]]])
    probVid = vid
    height, width, channel = vid.shape
    for i in range(height):
        for j in range(width):
            pixel = vid[i,j,:]
            # print(pixel)
            probVid[i, j, 0] = 25500 * (weight[0, 0] * calcGaussian(pixel, mean[0, :], sigma[:, :, 0]) + weight[1, 0] * calcGaussian(pixel, mean[1, :], sigma[:, :, 1]))
            if probVid[i,j,2] < 242:
                probVid[i,j,:] = [0,0,0]

    return probVid



def printImageRed(vid):
    height, width, channel = vid.shape

    for x in range(height):
        for y in range(width):
            if vid[x,y,2] < 242:
                vid[x,y,:] = [0,0,0]


    return vid

def printImageGreen(vid):
    height, width, channel = vid.shape
    for i in range(height):
        for j in range(width):
            if vid[i,j,2] < 242:
                vid[i,j,:] = [0,0,0]


    return vid


def circleDrawRed(img, cimg):
    red = [0, 0, 255]
    green = [0, 255, 0]
    blue = [255, 0, 0]
    yellow = [0,255,255]
    height, width, layers = cimg.shape


    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,180,
                                param1=100,param2=10,minRadius=13,maxRadius=30)
    circles = np.uint16(np.around(circles))

    for x in range(height):
        for y in range(width):
            for i in circles[0,:]:
                    if cimg[i[1], i[0], 2] > 100 and cimg[i[1], i[0], 1] > 255:
                            if i[0] > 100:
                                cv2.circle(cimg,(i[0],i[1]),i[2],(red),2)
                                cv2.circle(cimg,(i[0],i[1]),2,(blue),3)

    return cimg

def circleDrawGreen(img, cimg):
    red = [0, 0, 255]
    green = [0, 255, 0]
    blue = [255, 0, 0]
    yellow = [0,255,255]
    height, width, layers = cimg.shape


    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,180,
                                param1=100,param2=10,minRadius=13,maxRadius=30)
    circles = np.uint16(np.around(circles))

    for x in range(height):
        for y in range(width):
            for i in circles[0,:]:
                if cimg[i[1], i[0], 1] > 200 or cimg[i[1], i[0], 2] > 200:
                    if i[0] > 300 and i[1] > 312:
                        cv2.circle(cimg,(i[0],i[1]),i[2],(green),2)
                        cv2.circle(cimg,(i[0],i[1]),2,(blue),3)

    return cimg

def circleDrawYellow(img, cimg):
    red = [0, 0, 255]
    green = [0, 255, 0]
    blue = [255, 0, 0]
    yellow = [0,255,255]
    height, width, layers = cimg.shape


    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,180,
                                param1=100,param2=10,minRadius=13,maxRadius=30)
    circles = np.uint16(np.around(circles))

    for x in range(height):
        for y in range(width):
            for i in circles[0,:]:
                if cimg[i[1], i[0], 1] > 200 or cimg[i[1], i[0], 2] > 250:
                    if i[0] < 450 and i[1] < 450:
                        cv2.circle(cimg,(i[0],i[1]),i[2],(yellow),2)
                        cv2.circle(cimg,(i[0],i[1]),2,(blue),3)

    return cimg

if __name__ == '__main__':


    video = cv2.VideoCapture("detectbuoy.avi")
    # image_array = []
    currentframe = 0

    while True:
        ret, frame = video.read()
        if ret == 1:

            bframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cframe = cv2.GaussianBlur(bframe, (9,9), 0)
            cframe = cv2.cvtColor(bframe, cv2.COLOR_GRAY2BGR)
            red_image = copy.deepcopy(frame)
            # cv2.imshow("frame", red_image)
            try:
                resRed = circleDrawRed(bframe, frame)
                resGreen = circleDrawGreen(bframe,frame)
                resYellow = circleDrawYellow(bframe, frame)
            except:
                pass
            # resYellow = circleDrawYellow(bframe, frame)
            resGreen = circleDrawGreen(bframe, frame)
            resRed = circleDrawRed(bframe, frame)
            segmented = printImageRed(red_image)
            # cv2.imshow('egmented', segmented)
            # bframe = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
            resRed = circleDrawRed(bframe, frame)
            cv2.imshow('frame', resRed)


           # cv2.imshow("black", segmented)
            # cv2.imshow("frame", resYellow)
            # writer()

            # frame


                # reading from frame

            # if True:
            #     # if video is still left continue creating images
            #     name = './DataR/rbuoydetect/frame' + str(currentframe) + '.png'
            #     cv2.imwrite(name, resRed)
            #
            #     # increasing counter so that it will
            #     # show how many frames are created
            #     currentframe += 1
            #     # print(currentframe)
            # else:
            #     break

            # Release all space and windows once done
            # cam.release()

            # vid = probability(frame)
            # cv2.imshow("black", vid)
            # proVid = probability(frame)
            # cv2.imshow("frame",proVid)

        if cv2.waitKey(30) and 0Xff == ord('q'):
            break
    cv2.destroyAllWindows()
    video.release()


