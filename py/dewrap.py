import sys
import numpy as np
import math
import cv2
from common import *
import inputs

# global variables
showProgress = True
saveProgress = False

# main program, processes each image file specified
# parameters: 	None
# returns:		None
def main():

	if len(sys.argv) <= 1:
		print('No images specified')
		return

	for file in sys.argv[1:]:
		processImageFile(file)

# processes each image file
# parameters: 	filename: String	file to be processed
# returns:		None
def processImageFile(filename):

	# read file and convert to grey scale
	imgOriginal = cv2.imread(filename)
	imgOriginalGrey = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

	# show
	cv2.imshow('imgOriginal', imgOriginal)

	# test all parameters
	test(imgOriginal)

	# smooth image
	# smoothed = cv2.GaussianBlur(imgOriginal, (3, 3), 1)
	# smoothed = cv2.GaussianBlur(imgOriginal, (15, 15), 5)

	# blur image
	# gaussianBlur(imgOriginal);

	# find the relevant corners
	# findCorners(smoothed)

	# find the relevant edges
	# findEdges(smoothed)

	# find edges and find corners from edges
	# findEdgesAndCorners(smoothed)
	
	# find lines from edges
	# findLines(smoothed)

	# wait for preview to finish
	# cv2.waitKey(10000)

# measure confidence in the results with the specified parameters
# parameters: 	imgOriginal: nparray	image to test
#				parameters: {}			parameters to apply 
# returns:		float: confidence
def measureConfidence(imgOriginal, parameters):



	return 0

def findCorners(img, parameters):

	# blur
	imgSmooth = smoothImg(img, **parameters)
	logImg(imgSmooth, 'smoothed')

	# dilate image
	imgDilated = dilateImg(imgSmooth, **parameters)

	# erode image
	imgEroded = erodeImg(imgDilated, **parameters)
	logImg(imgEroded, 'closed')
	
	erodedGrey = grayFlaotImg(imgEroded)

	# harris corner detection
	harrisIntensity = harrisCornerImg(erodedGrey, **parameters)
	harrisThreshold = harrisThresholdImg(harrisIntensity, **parameters);
	harrisCorners = paintPixelsImg(img, harrisThreshold, [0, 0, 255])
	logImg(harrisCorners, 'harris corners')

	# canny edges
	imgEdges = cannyEdgeImg(erodedGrey, **parameters)
	logImg(imgEdges, 'canny edges')

	# # probabilistic hough lines
	# imgProbLines = probabilisticHoughTransformImg(imgEdges, **state)

	# hough lines
	houghLines = houghTransformLines(imgEdges, **parameters)
	groupedLines = groupLines(houghLines, **parameters)
	imgLines = drawLinesImg(black(img), groupedLines, 255)
	imgLinesOverlay = drawLinesImg(img, groupedLines, (255, 0, 0))
	logImg(imgLinesOverlay, 'hough lines')
	logImg(imgLines, 'imgLines')

	# find 4 sided contours
	contours = contours4Img(imgLines)
	imgContours = darwContousImg(img, contours)
	logImg(imgContours, 'contours')



	return []

def logImg(img, name):
	if showProgress:
		cv2.imshow(name, img)

	if saveProgress:
		cv2.imwrite('output/'+name+".png", img)

def black(size):
	if type(size) == np.ndarray:
		return np.zeros(size.shape[:2], dtype=np.uint8)
	else:
		return np.zeros(size[:2], dtype=np.uint8)

def grayIntImg(img):
	return np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
	
def grayFlaotImg(img):
	return np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

def smoothImg(img, blur_kernel_size, blur_diviation, **args):

	imgSmooth = cv2.GaussianBlur(
		img, 
		(blur_kernel_size, blur_kernel_size), 
		blur_diviation
	)

	return imgSmooth

def dilateImg(img, dilation_size, **args):

	dilationKernel = cv2.getStructuringElement(
		cv2.MORPH_RECT, 
		(dilation_size, dilation_size)
	)

	imgDilated = cv2.dilate(img, dilationKernel)

	return imgDilated

def erodeImg(img, erode_size, **args):

	erotionKernel = cv2.getStructuringElement(
		cv2.MORPH_RECT, 
		(erode_size, erode_size)
	)

	imgEroded = cv2.erode(img, erotionKernel)

	return imgEroded

def harrisCornerImg(img, harris_block_size, hharris_ksize, harris_k, **args):

	intensity = cv2.cornerHarris(
		img, 
		harris_block_size, 
		hharris_ksize, 
		harris_k
	)

	intensity = intensity / intensity.max()

	return intensity

def nonMaximumSupressImg(img, **args):

	return img

def harrisThresholdImg(img, harris_threashold, **args):

	threshold = img > harris_threashold * img.max()

	return threshold

def paintPixelsImg(img, pixel, color = (0, 0, 255), **args):

	copy = img.copy()
	copy[pixel] = color
	return copy

def cannyEdgeImg(img, canny_lower, canny_upper, **args):
	
	imgEdges = cv2.Canny(
		np.uint8(img), 
		canny_lower, 
		canny_upper
	)

	return imgEdges

def probabilisticHoughTransformImg(img, hough_rho, hough_theta, hough_threshold, hough_srn, hough_stn, **args):
	
	imgProbLines = black(img)

	lines = cv2.HoughLinesP(
		img, 
		hough_rho, 
		hough_theta * np.pi / 180, 
		hough_threshold, 
		hough_srn, 
		hough_stn
	)

	if lines is not None:
		for i in range(0, len(lines)):
			line = lines[i][0]
			cv2.line(imgProbLines, (line[0], line[1]), (line[2], line[3]), 255, 1)

	return imgProbLines

def houghTransformLines(img, hough_rho, hough_theta, hough_threshold, **args):
	
	lines = cv2.HoughLines(
		img,
		hough_rho,
		hough_theta * np.pi / 180, 
		hough_threshold
	)

	if lines is None:
		return []

	lines = [(r, t) for [[r, t]] in lines]

	# convert all vector magnitues to positive
	for i in range(len(lines)):
		(r, t) = lines[i]

		if r < 0:
			lines[i] = (-r, (t + math.pi) % (2 * math.pi))

	return lines

def groupLines(lines, hough_group_threshold_rho, hough_group_threshold_theta, **args):

	groups = {(r, t):[] for (r, t) in lines}

	tr = hough_group_threshold_rho
	tt = hough_group_threshold_theta

	for index in groups:
		for line in lines:
			dr = index[0] - line[0]
			dt = index[1] - line[1]
			if dt < -math.pi:
				dt += 2 * math.pi
			if dt > math.pi:
				dt -= 2 * math.pi

			if index != line and -tr < dr < tr and -tt < dt < tt:
				groups[index].append(line)

	progress = True
	while progress:
		progress = False
		for index in groups:
			if groups[index] is not None:
				for item in groups[index][:]:

					if groups[index] is None:
						break

					if groups[item] is None:
						continue

					if len(groups[index]) == len(groups[item]):
						if index in groups[item]:
							groups[item].remove(index)
							progress = True
						elif item in groups[index]:
							groups[index].remove(item)
							progress = True

					elif len(groups[index]) > len(groups[item]):
						if index in groups[item]:
							groups[item].remove(index)
							progress = True
						elif len(groups[item]) > 0:
							groups[item].pop()
							progress = True

					elif len(groups[item]) > len(groups[index]):
						if item in groups[index]:
							groups[index].remove(item)
							progress = True
						elif len(groups[index]) > 0:
							groups[index].pop()
							progress = True

					if len(groups[item]) == 0:
						groups[item] = None

					if len(groups[index]) == 0:
						groups[index] = None
						break

	groupCenters = []

	for index in groups:
		if groups[index] is not None:
			groupCenters.append(index)

	return groupCenters

def drawLinesImg(img, lines, color = 255, **args):

	imgLines = img.copy()

	if lines is not None:
		for [rho, theta] in lines:
			# print(rho, theta)
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))

			cv2.line(imgLines, (x1, y1), (x2, y2), color, 1)

	return imgLines

def contours4Img(img, **args):

	contours4 = []

	img0, allContours, hierarchy = cv2.findContours(black(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	allContours = sorted(allContours, key = cv2.contourArea, reverse = True)
	for contour in allContours:

		# approximate the contour
		peri = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

		if len(approx) == 4:
			contours4.append(approx)

	return contours4

def darwContousImg(img, contours, color = (0, 255, 0), **args):
	
	imgContours = img.copy()

	cv2.drawContours(imgContours, contours, -1, color, 1)

	return imgContours

def findLineIntersectionsPoints(lines, **args):

	points = []

	for (pa, ta) in lines:
		for (pb, tb) in lines:
			xa = (math.sin(ta) * (pb - )) + ()


	return points



def test(img):

	def show(state):

		findCorners(img, state)


	def update(control, value, state):
		state[control.name] = value
		show(state)

	class var:
		def __init__(self, v, vmin, vmax, steps = 1):
			self.v = v
			self.min = vmin
			self.max = vmax
			self.steps = steps

	state = {}
	parameters = {
		'blur_kernel_size': 			var(v = 3, 		vmin = 1, vmax = 15, 	steps = 2		),
		'blur_diviation': 				var(v = 1, 		vmin = 1, vmax = 9						),
		'dilation_size': 				var(v = 6, 		vmin = 1, vmax = 50						),
		'erode_size': 					var(v = 6, 		vmin = 1, vmax = 50						),
		'harris_block_size': 			var(v = 25, 	vmin = 1, vmax = 31, 	steps = 2		),
		'hharris_ksize': 				var(v = 3, 		vmin = 1, vmax = 31,  	steps = 2		),
		'harris_k': 					var(v = 0.04, 	vmin = 0, vmax = 1, 	steps = 0.001	),
		'harris_threashold':			var(v = 0.03, 	vmin = 0, vmax = 1, 	steps = 0.001	),
		'canny_lower': 					var(v = 50, 	vmin = 1, vmax = 255					),
		'canny_upper': 					var(v = 200, 	vmin = 1, vmax = 255					),
		'hough_rho': 					var(v = 1, 		vmin = 1, vmax = 32						),
		'hough_theta': 					var(v = 1,	 	vmin = 1, vmax = 360					),
		'hough_threshold': 				var(v = 60, 	vmin = 1, vmax = 256					),
		'hough_srn': 					var(v = 0, 		vmin = 1, vmax = 255					),
		'hough_stn': 					var(v = 0, 		vmin = 1, vmax = 255					),
		'hough_group_threshold_rho': 	var(v = 10,		vmin = 0, vmax = 100					),
		'hough_group_threshold_theta': 	var(v = 0.1, 	vmin = 0, vmax = 0.5, 	steps = 0.01	),
		}


	app = App()
	ins = inputs.Inputs(state=state)

	for name in parameters:
		properties = parameters[name]
		state[name] = properties.v
		slider = inputs.InputSlider(name, properties.v, onUpdate=update, sMin=properties.min, sMax=properties.max, sSteps=properties.steps)
		ins.addInput(slider)

	show(state)
	ins.getFrames()
	app.show()

# run main program
if __name__ == "__main__":
	main()