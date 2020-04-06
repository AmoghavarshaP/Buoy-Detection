import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

def AverageHistogram(N,type,channel):
    total=0
    for i in range(N+1):
        if type==1:
            image=cv2.imread('Green/Crop_segmented/frame-%d.png' %i)
        if type==2:
            image=cv2.imread('Red/Crop_segmented/frame-%d.png' %i)
        if type==3:
            image=cv2.imread('Yellow/Crop_segmented/frame-%d.png' %i)
        histogram = cv2.calcHist([image],[channel],None,[256],[50,256])
        total=total+histogram
    average_histogram=total/(N+1)

    if channel==0:
        plt.plot(average_histogram,'-b',label="blue")
    elif channel==1:
        plt.plot(average_histogram,'-g',label="green")
    elif channel==2:
        plt.plot(average_histogram,'-r',label="red")

    plt.xlabel("Intensity")
    plt.legend()
    plt.ylabel("Number of pixels")
    return average_histogram


def PlottingCurve(sum,mean,sigma,channel):
    x = np.linspace(mean - 3*sigma, mean + 3*sigma, 100)
    plt.plot(x,sum*stats.norm.pdf(x, mean, sigma))
    if channel==1:
        plt.title("Gaussian-Green Buoy ")
    elif channel==2:
        plt.title("Gaussian-Red Buoy ")
    elif channel==3:
        plt.title("Gaussian-Yellow Buoy ")
    plt.xlabel("Intensity")
    plt.legend()
    plt.ylabel("Number of Pixels")

def Sigma(average,start,end):
    sum=0
    VarianceSum=0
    ComMean=0
    for i in range(start,end+1):
        sum=sum+average[i,0]
        ComMean=ComMean+i*average[i,0]
    mean=ComMean/sum

    for i in range(start,end+1):
        VarianceSum=VarianceSum+(i-mean)**2*average[i,0]
    sigma=np.sqrt(VarianceSum/(end-start))

    return sum,mean,sigma

def MeanVariance(Average,channel):
    gaussian=np.zeros((2,3))
    if channel==1:
        gaussian[0,0],gaussian[0,1],gaussian[0,2]=Sigma(Average,231,255)
        plt.plot(hist_GG,'-g',label="green")
        PlottingCurve(gaussian[0,0]*2.5,gaussian[0,1],gaussian[0,2],G_channel)
    if channel==2:
        plt.plot(hist_RR,'-r',label="red")
        gaussian[0,0],gaussian[0,1],gaussian[0,2]=Sigma(Average,236,255)
        PlottingCurve(gaussian[0,0]*2.5,gaussian[0,1],gaussian[0,2],R_channel)
    if channel==3:
        plt.plot(hist_YY,'-y',label="yellow")
        gaussian[0,0],gaussian[0,1],gaussian[0,2]=Sigma(Average,225,255)

        PlottingCurve(gaussian[0,0]*2.5,gaussian[0,1],gaussian[0,2],Y_channel)
    plt.grid()
    return gaussian

B_channel=0
G_channel=1
R_channel=2
Y_channel=3

hist_GB=AverageHistogram(20,G_channel,B_channel)
hist_GG=AverageHistogram(20,G_channel,G_channel)
hist_GR=AverageHistogram(20,G_channel,R_channel)
plt.title("Gaussian Histogram - Green Buoy")
plt.show()

Green_GMM=MeanVariance(hist_GG,G_channel)
plt.show()

hist_RB=AverageHistogram(30,R_channel,B_channel)
hist_RG=AverageHistogram(30,R_channel,G_channel)
hist_RR=AverageHistogram(30,R_channel,R_channel)
plt.title("Gaussian Histogram - Red Buoy")
plt.show()
Red_GMM=MeanVariance(hist_RR,R_channel)
plt.show()

hist_YB=AverageHistogram(30,Y_channel,B_channel)
hist_YG=AverageHistogram(30,Y_channel,G_channel)
hist_YR=AverageHistogram(30,Y_channel,R_channel)
plt.title("Gaussian Histogram - Yellow Buoy")
hist_YY=(hist_YG+hist_YR)/2
plt.show()
Red_GMM=MeanVariance(hist_YY,Y_channel)
plt.show()
