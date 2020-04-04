import matplotlib.pyplot as plot
import numpy as np
import scipy.stats as stats
import random
import cv2
import glob


def calcGaussian(pixel, mean, stddev):
    N = (1/((2*np.pi*np.linalg.det(stddev))**(1/2)))*math.exp((-1/2)*((pixel - mean).T) *(np.linalg.inv(stddev)) * (pixel - mean))
    return N

def createData(N, image):
    pixel = []
    for i in range()
    image = (cv2.imread(file) for file in glob.glob("Red_DataSet/Crop_segmented/Train/*.png"))


