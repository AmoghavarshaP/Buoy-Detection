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
    # mean = np.array(mean)
    # pixel = np.array(pixel)
    # variance = np.array(variance)

    # calculate gaussian
    N = (1 / ((2 * np.pi * np.linalg.det(variance)) ** (3 / 2))) * np.exp((-1 / 2) * np.matmul(np.matmul((pixel - mean), np.linalg.inv(variance)), (pixel - mean).T))

    return N


# function to convert images into pixel data
def createData(N):
    pixel_b = []
    pixel_g = []
    pixel_r = []

    for x in range(N):
        image = cv2.imread("Red_DataSet/Crop_segmented/Train/frame%d.png" % x)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j, 2] > 50:
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
            postProb[i][j] = 100000 * weight[j, :] * calcGaussian(pixels[i, :], mean[j, :], sigma[:, :, j])

    for i in range(len(pixels)):
        for j in range(len(mean)):
            postProb[i, j] /= np.sum(postProb[i, :])

    for i in range(len(pixels)):
        like += math.log(np.sum(postProb[i, :]))

    return postProb, like


# function to maximize posterior probability
def maximize():
    # new parameters
    new_mean = np.zeros((len(posteriorProb[0]), 3))
    new_sigma = np.zeros((3, 3, len(posteriorProb[0])))
    new_weight = np.zeros((len(posteriorProb[0]), 1))
    total_weight = 0

    # weight
    for j in range(len(posteriorProb[0])):
        total_weight = total_weight + np.sum(posteriorProb[:, j])

    for j in range(len(posteriorProb[0])):
        new_weight[j, 0] = np.sum(posteriorProb[:, j]) / total_weight

    # mean
    for k in range(3):
        for i in range(len(posteriorProb[0])):
            m = 0
            for j in range(len(posteriorProb)):
                m = m + pixels[j, k] * posteriorProb[j, i]
            new_mean[i, k] = m / np.sum(posteriorProb[:, i])

    # sigma
    for j in range(len(posteriorProb[0])):
        sigma_bb, sigma_rr, sigma_gg, sigma_bg, sigma_br, sigma_rg = 0, 0, 0, 0, 0, 0
        sum = np.sum(posteriorProb[:, j])
        for i in range(len(posteriorProb)):
            sigma_bb = sigma_bb + (posteriorProb[i, j] * (pixels[i, 0] - new_mean[j, 0]) ** 2) / sum
            sigma_gg = sigma_gg + (posteriorProb[i, j] * (pixels[i, 1] - new_mean[j, 1]) ** 2) / sum
            sigma_rr = sigma_rr + (posteriorProb[i, j] * (pixels[i, 2] - new_mean[j, 2]) ** 2) / sum
            sigma_bg = sigma_bg + (
                        posteriorProb[i, j] * (pixels[i, 0] - new_mean[j, 0]) * (pixels[i, 1] - new_mean[j, 1])) / sum
            sigma_br = sigma_br + (
                        posteriorProb[i, j] * (pixels[i, 0] - new_mean[j, 0]) * (pixels[i, 2] - new_mean[j, 2])) / sum
            sigma_rg = sigma_rg + (
                        posteriorProb[i, j] * (pixels[i, 2] - new_mean[j, 2]) * (pixels[i, 1] - new_mean[j, 1])) / sum

        new_sigma[:, :, j] = [[sigma_bb, sigma_bg, sigma_br], [sigma_bg, sigma_gg, sigma_rg],
                              [sigma_br, sigma_rg, sigma_rr]]

    return new_mean, new_sigma, new_weight


if __name__ == '__main__':

    # Create pixel data from train images
    pixels = createData(118)

    # assume mean, co-variance, weight parameters
    mean = np.array([[50, 240, 200], [50, 240, 150]])
    sigma = np.ones((3, 3, 2))
    sigma[:, :, 0] = [[500, 0, 0], [0, 800, 0], [0, 0, 200]]
    sigma[:, :, 1] = [[100, 0, 0], [0, 500, 0], [0, 0, 600]]
    weight = np.array([[1 / 2], [1 / 2]])

    # log likelihood
    likelihood = -math.inf

    # while True:
    for i in range(2):
        temp_likelihood = likelihood

        # estimation step
        posteriorProb, likelihood = estimateValue()

        # maximization step
        mean, sigma, weight = maximize()

        if likelihood < temp_likelihood:
            break

    print(mean, 'mean')
    print(sigma, 'sigma')
    print(weight, 'weight')

