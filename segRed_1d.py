import matplotlib.pyplot as plot
import numpy as np
import scipy.stats as stats
import random
import cv2
import glob


def calcGaussian(pixel, mean, stddev):
    N = (1/((2*np.pi*np.linalg.det(stddev))**(1/2)))*math.exp((-1/2)*((pixel - mean).T) *(np.linalg.inv(stddev)) * (pixel - mean))
    return N

def createData(N):
    pixel = []
    for x in range(N):
        image = cv2.imread("Red_DataSet/Crop_segmented/Train/frame%d.png" %x)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i,j,2] > 50:
                    pixel = np.append([pixel], [image[i,j,2]])
                    print(pixel)
        # print(image.shape[0])

createData(118)
print('done')


