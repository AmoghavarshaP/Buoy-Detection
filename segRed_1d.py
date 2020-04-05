import matplotlib.pyplot as plot
import numpy as np
import scipy.stats as stats
import random
import cv2
import math
import glob


# function to calculate gaussian
def calcGaussian(pixel, mean, variance):

    # matrix conversion
    mean = np.array(mean)
    pixel = np.array(pixel)
    variance = np.array(variance)

    # calculate gaussian
    N = (1/((2*np.pi*np.linalg.det(variance))**(3/2))) * math.exp((-1/2) * (pixel - mean) * (np.linalg.inv(variance)) * (pixel - mean).T)

    return N


# function to convert images into pixel data
def createData(N):
    pixel_b = []
    pixel_g = []
    pixel_r = []

    for x in range(N):
        image = cv2.imread("Red_DataSet/Crop_segmented/Train/frame%d.png" %x)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j, 0] or image[i, j, 1] or image[i, j, 2] > 50:
                    pixel_b = np.append([pixel_b], [image[i, j, 0]])
                    pixel_g = np.append([pixel_g], [image[i, j, 1]])
                    pixel_r = np.append([pixel_r], [image[i, j, 2]])

    pixel = np.vstack((pixel_b, pixel_g, pixel_r))
    pixel = pixel.T

    return pixel


# function to estimate values for classification
def estimateValue():

    # Posterior Probability
    postProb = np.ones((len(pixels), len(mean)))

    # Likelihood
    like = 0

    for i in range(len(pixels)):
        for j in range(len(mean)):
            postProb[i][j] = weight[j, :] * calcGaussian(pixels[i, :], mean[j, :], sigma[:, :, j])

    for i in range(len(pixels)):
        den = np.sum(postProb[i, :])
        for j in range(len(mean)):
            postProb /= den

    for i in range(len(pixels)):
        like += math.log(np.sum(postProb[i, :]))

    return postProb, like


if __name__ == '__main__':

    # Create pixel data from train images
    pixels = createData(118)

    # assume mean, co-variance, weight parameters
    mean = np.array([[50, 240, 200], [50, 240, 150]])
    sigma = np.ones((3, 3, 2))
    sigma[:, :, 0] = [[500, 0, 0], [0, 800, 0], [0, 0, 200]]
    sigma[:, :, 1] = [[100, 0, 0], [0, 500, 0], [0, 0, 600]]
    weight = np.array([[1/2], [1/2]])

    # log likelihood
    likelihood = -math.inf

    while True:
        temp_likelihood = likelihood

        # estimation step
        posteriorProb, likelihood = estimateValue()
        print(posteriorProb)

