import glob
import cv2
img_array = []

# filenames = glob.glob("data_set/frame-*.png")
# filenames.sort()
# print(filenames)
# images = [cv2.imread(img) for img in filenames]

for i in range(200):
    image = cv2.imread('data_set/frame-' + str(i) + '.png')
    height, width, layers = image.shape
    # width = img.shape[0]
    # height = img.shape[1]
    size = (width, height)
    img_array.append(image)

out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

# cv2.imshow("video",out)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()