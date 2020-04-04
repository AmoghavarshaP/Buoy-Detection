import cv2
import numpy as np
import matplotlib.pyplot as plt


def mouse_click(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        points.append([x, y])


def histogram(roi, target):

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    histogram = cv2.calcHist([hsv], [0, 1], None, [75, 256], [0, 75, 90, 256])
    plt.plot(histogram, 'r')
    cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt], [0, 1], histogram, [0, 75, 90, 256], 1)

    # Now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(dst, -1, disc, dst)


    # threshold and binary AND
    ret, thresh = cv2.threshold(dst, 50, 255, 0)
    thresh = cv2.merge((thresh, thresh, thresh))
    res = cv2.bitwise_and(target, thresh)
    # plot.show()
    return res

def plt_histogram(image,title,mask=None):
        channel = cv2.split(image)
        color = ("b" , "g" ,"r")
        plt.figure()
        plt.title(title)
        plt.xlabel("Bins")
        plt.ylabel("Number of Pixels")
        for i,col in zip(channel,color):
            histogram = cv2.calcHist([i],[0],mask,[256],[0,256])
            plt.plot(histogram,color = col)
            plt.xlim([0,256])
        plt.show()
#The code below doesnot take into account the diferent channels,
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histogram1 = cv2.calcHist([image],[i],None,[256],[0,256])
#     plt.plot(histogram1,color = col)
#     plt.xlim([0,256])

if __name__ == '__main__':

    video = cv2.VideoCapture('detectbuoy.avi')
    ctr = 0

    while video:
        _, frame = video.read()

        # resize image to easily crop the ROI
        frame = cv2.resize(frame, (1280,720), interpolation = cv2.INTER_AREA)

        points = []
        cv2.namedWindow("frame", 1)
        cv2.setMouseCallback("frame", mouse_click)
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        # if 0xFF == ord('q'):
        #     break

        # Display the selected points as a polygon
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))
        # frame = cv2.polylines(frame, [points], True, (0, 255, 255))

        # crop image for dataset
        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        crop = frame[y:y + h, x:x + w].copy()
        cv2.imwrite('New/Crop/frame-'+str(ctr)+'.png', crop)
        frame = cv2.polylines(frame, [points], True, (0, 255, 255))
        cv2.imshow('cropped_image', crop)
        cv2.imshow('frame', frame)
        # cv2.waitKey(0)

        # masking the background
        pts = points - points.min(axis=0)
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        dst = cv2.bitwise_and(crop, crop, mask=mask)
        cv2.imwrite('New/Crop_Segmented/frame-'+str(ctr)+'.png', dst)
        cv2.imshow('dst', dst)
        # cv2.waitKey(0)

        # (4) add the white background
        bg = np.ones_like(crop, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=mask)
        dst2 = bg + dst
        result = histogram(dst2, frame)
        # cv2.imshow("image", dst2)
        cv2.imwrite('New/Segmented/frame-'+str(ctr)+'.png', result)
        # cv2.imshow("result", result)
        cv2.waitKey(0)
        # cv2.imshow('frame', frame)
        ctr += 1

        read= cv2.imread('C:/Users/acer/Desktop/Python/ENPM673-Project 3/Green/Crop_segmented/frame-0.png')
        smooth = cv2.GaussianBlur(read,(5,5),0)
        plt_histogram(smooth,"Unmasked")
        plt_histogram(dst,"Masked", mask=mask)
        # print("lala")

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

video.release()
cv2.destroyAllWindows()
