import numpy as np
import cv2
import sys

# compute the optimal threshold and colour center for a given section
def getThresholdAndCenters(img):

    # one long array and turn into floats
    imgarray = np.float32(img.flatten())

    # find centers
    # center = np.array([[0, 0, 0], [255, 255, 255]])
    centers = np.array([0, 255])

    # image histogram
    histogram = np.histogram(imgarray, 256, (0, 255))[0]
    normHistogram = histogram / histogram.sum()

    # init values
    threshold = 0
    histogramSum = (normHistogram * np.arange(256)).sum()
    sum0 = 0
    q0 = 0
    maxVariance = 0

    # calculate each between variance and note max
    for t in range(0, 256):
        q0 += normHistogram[t]
        q1 = 1 - q0
        if q0 == 0 or q1 == 0:
            continue
        sum0 += t * normHistogram[t]
        sum1 = histogramSum - sum0
        mean0 = sum0 / q0
        mean1 = sum1 / q1
        variance = q0 * q1 * (mean0 - mean1) ** 2

        if variance > maxVariance:
            maxVariance = variance
            threshold = t
            centers = [mean0, mean1]

    return threshold, centers

# split images colour space into 2 distinct colours
def binmeans(img):
    
    # size
    yRange, xRange = img.shape

    thresholds = np.zeros(img.shape)
    basethresholds = [[0, 0], [0, 0]]
    baseCenters = np.array([[[0, 255], [0, 255]], [[0, 255], [0, 255]]])
    basethresholds[0][0], baseCenters[0][0] = getThresholdAndCenters(img[:yRange//4, :xRange//4])
    basethresholds[0][1], baseCenters[0][1] = getThresholdAndCenters(img[:yRange//4, xRange//4*3:])
    basethresholds[1][0], baseCenters[1][0] = getThresholdAndCenters(img[yRange//4*3:, :xRange//4])
    basethresholds[1][1], baseCenters[1][1] = getThresholdAndCenters(img[yRange//4*3:, xRange//4*3:])

    # get max values for center colours
    centers = np.array([baseCenters[:,:,0].min(), baseCenters[:,:,1].max()])
    centers = np.array([0, 255])

    # interpolate thresholds into image sized array
    # get vertical gradients of 1/4 and 3/4 way strip (center of each quarter)
    leftCenterT = (basethresholds[0][0] + basethresholds[1][0]) / 2
    rightCenterT = (basethresholds[0][1] + basethresholds[1][1]) / 2
    leftGradient = (basethresholds[0][0] - basethresholds[1][0]) / -yRange
    rightGradient = (basethresholds[0][1] - basethresholds[1][1]) / -yRange
    leftTopT = leftCenterT - leftGradient * yRange / 2
    rightTopT = rightCenterT - rightGradient * yRange / 2

    # fill in the first and last vertical strip
    for y in range(yRange):
        thresholds[y, 0] = leftTopT + leftGradient * y
        thresholds[y, xRange-1] = rightTopT + rightGradient * y

    # fill in remaining treashold image
    centerTs = (thresholds[:, 0] + thresholds[:, xRange-1]) / 2
    gradients = (thresholds[:, 0] - thresholds[:, xRange-1]) / -xRange * 2
    leftTs = centerTs - gradients * xRange / 2
    for x in range(xRange):
        thresholds[:, x] = leftTs + gradients * x


    # one long array and turn into floats
    thresholds = thresholds.flatten()
    imgarray = np.float32(img.flatten())

    # determine labels
    label = np.uint8(np.zeros(imgarray.shape[0]))
    label[imgarray > thresholds] = 1

    # convert from label to corresponding rgb value
    centers = np.uint8(centers) # to int
    img2array = centers[label]

    # convert results back to img form
    img2 = img2array.reshape(grey.shape)

    return img2

img = cv2.imread(sys.argv[1])
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# show
cv2.imshow("img", grey)
binimg = binmeans(grey)
cv2.imshow("sudoku", binimg)
cv2.imwrite("output/binary.png", binimg)

cv2.waitKey(60000)