import sys
import numpy as np
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




	return []

def logImg(img, name):
	if showProgress:
		cv2.imshow(name, img)

	if saveProgress:
		cv2.imwrite('output/'+name+".png", img)

def grayIntImg(img):
	return np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
	
def grayFlaotImg(img):
	return np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

def smoothImg(img, parameters, name = 'gaussian blured'):

	imgSmooth = cv2.GaussianBlur(
		img, 
		(parameters['blur kernel size'].v, parameters['blur kernel size'].v), 
		parameters['blur diviation'].v
	)

	logImg(imgSmooth, name)

	return imgSmooth

def dilateImg(img, parameters, name = 'dilated'):

	dilationKernel = cv2.getStructuringElement(
		cv2.MORPH_RECT, 
		(parameters['dilation size'].v, parameters['dilation size'].v)
	)

	imgDilated = cv2.dilate(img, dilationKernel)

	logImg(imgDilated, name)

	return imgDilated

def erodeImg(img, parameters, name = 'eroded'):

	erotionKernel = cv2.getStructuringElement(
		cv2.MORPH_RECT, 
		(parameters['erode size'].v, parameters['erode size'].v)
	)

	imgEroded = cv2.erode(img, erotionKernel)

	logImg(imgEroded, name)

	return imgEroded

def harrisCornerImg(img, parameters, name = 'harris intensity'):

	intensity = cv2.cornerHarris(
		img, 
		parameters['harris block size'].v, 
		parameters['harris ksize'].v, 
		parameters['harris k'].v
	)

	intensity = intensity / intensity.max()

	logImg(intensity, name)

	return intensity

def nonMaximumSupressImg(img, name = 'harris peaks'):

	logImg(img, name)

	return img

def harrisThresholdImg(img, parameters):

	threshold = img > parameters['harris threashold'].v * img.max()

	# logImg(threshold, 'harris threshold')

	return threshold

def paintPixelsImg(img, pixel, color = [0, 0, 255]):

	copy = img.copy()
	copy[pixel] = color
	return copy

def cannyEdgeImg(img, parameters, name = 'canny edges'):
	
	imgEdges = cv2.Canny(
		np.uint8(img), 
		parameters['canny lower'].v, 
		parameters['canny upper'].v
	)

	logImg(imgEdges, name)

	return imgEdges

def probabilisticHoughTransformImg(img, parameters, name = 'probabilistic hough lines'):
	
	imgProbLines = np.zeros(img.shape, dtype=np.uint8)

	lines = cv2.HoughLinesP(
		img, 
		parameters['hough rho'].v, 
		parameters['hough theta'].v * np.pi / 180, 
		parameters['hough threshold'].v, 
		parameters['hough srn'].v, 
		parameters['hough stn'].v)

	if lines is not None:
		for i in range(0, len(lines)):
			line = lines[i][0]
			cv2.line(imgProbLines, (line[0], line[1]), (line[2], line[3]), 255, 1)

	logImg(imgProbLines, name)

	return imgProbLines

def houghTransformImg(img, parameters, name = 'hough lines'):

	imgLines = np.zeros(img.shape, dtype=np.uint8)
	
	lines = cv2.HoughLines(
		img,
		parameters['hough rho'].v,
		parameters['hough theta'].v * np.pi / 180, 
		parameters['hough threshold'].v
	)

	if lines is not None:
		for [[rho, theta]] in lines:
			# print(rho, theta)
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))

			cv2.line(imgLines, (x1, y1), (x2, y2), 255, 1)

	logImg(imgLines, name)

	return imgLines

def contours4Img(img):

	contours4 = []

	img0, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key = cv2.contourArea, reverse = True)
	for c in contours:

		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		if len(approx) == 4:
			contours4.append(approx)
			# cv2.drawContours(imgContours, [approx], -1, (255, 0, 0), 1)

	return contours4

def darwContousImg(img, contours, color = (255, 0, 0), name = 'contour lines'):
	
	copy = img.copy()

	cv2.drawContours(copy, contours, -1, color, 1)

	logImg(copy, name)

	return copy

def test(img):

	def show(state):

		cv2.imwrite('output/original.png', img)

		# blur
		imgBlured = smoothImg(img, state)

		# dilate image
		imgDilated = dilateImg(imgBlured, state)

		# erode image
		imgEroded = erodeImg(imgDilated, state)
		
		erodedGrey = grayFlaotImg(imgEroded)

		# harris corner detection
		harrisIntensity = harrisCornerImg(erodedGrey, state)
		
		harrisPeaks = nonMaximumSupressImg(harrisIntensity)

		harrisThreshold = harrisThresholdImg(harrisIntensity, state);

		harrisCorners = paintPixelsImg(img, harrisThreshold, [0, 0, 255])
		logImg(harrisCorners, 'harris corners')
		
		# # dilate image
		# harrisCornersDilated = dilateImg(harrisIntensity, state, 'harris dilated')

		# # erode image
		# harrisCornersEroded = erodeImg(harrisCornersDilated, state, 'harris eroded')

		# canny edges
		imgEdges = cannyEdgeImg(erodedGrey, state)

		# probabilistic hough lines
		imgProbLines = probabilisticHoughTransformImg(imgEdges, state)

		# hough lines
		imgLines = houghTransformImg(imgEdges, state)

		# find 4 sided contours
		contours1 = contours4Img(imgProbLines)
		contours2 = contours4Img(imgLines)

		darwContousImg(img, contours1)
		darwContousImg(img, contours2)


	def update(control, value, state):
		state[control.name].v = value * state[control.name].scale
		show(state)

	class var:
		def __init__(self, v, vmin, vmax, steps = 1, scale = 1):
			self.v = v
			self.min = vmin
			self.max = vmax
			self.steps = steps
			self.scale = scale

	state = {
		'blur kernel size': 	var(v = 3, 		vmin = 1, vmax = 15, steps = 2		),
		'blur diviation': 		var(v = 1, 		vmin = 1, vmax = 9					),
		'dilation size': 		var(v = 6, 		vmin = 1, vmax = 50					),
		'erode size': 			var(v = 6, 		vmin = 1, vmax = 50					),
		'harris block size': 	var(v = 25, 	vmin = 1, vmax = 31, steps = 2		),
		'harris ksize': 		var(v = 3, 		vmin = 1, vmax = 31,  steps = 2		),
		'harris k': 			var(v = 0.04, 	vmin = 1, vmax = 1000, scale = 0.001),
		'harris threashold':	var(v = 0.03, 	vmin = 1, vmax = 1000, scale = 0.001),
		'canny lower': 			var(v = 50, 	vmin = 1, vmax = 255				),
		'canny upper': 			var(v = 200, 	vmin = 1, vmax = 255				),
		'hough rho': 			var(v = 1, 		vmin = 1, vmax = 32					),
		'hough theta': 			var(v = 1	, 	vmin = 1, vmax = 360				),
		'hough threshold': 		var(v = 60, 	vmin = 1, vmax = 256				),
		'hough srn': 			var(v = 0, 		vmin = 1, vmax = 255				),
		'hough stn': 			var(v = 0, 		vmin = 1, vmax = 255				),
		'experiment': 			var(v = 0, 		vmin = 1, vmax = 100				),
		}

	show(state)

	app = App()
	ins = inputs.Inputs(state=state)

	for name in state:
		properties = state[name]
		slider = inputs.InputSlider(name, properties.v / properties.scale, onUpdate=update, sMin=properties.min, sMax=properties.max, sSteps=properties.steps)
		ins.addInput(slider)

	ins.getFrames()
	app.show()



def gaussianBlur(imgOriginal):
	
	def show(state):
		imgGrey = cv2.cvtColor(np.uint8(imgOriginal), cv2.COLOR_BGR2GRAY)
		imgGrey = np.float32(imgGrey)
		imgColor = imgOriginal.copy()

		blured = cv2.GaussianBlur(imgOriginal, (state[0], state[0]), state[1])

		cv2.imshow('blured', blured)		

	def update1(control, value, state):
		state[0] = value
		show(state)	

	def update2(control, value, state):
		state[1] = value
		show(state)

	state = [3, 1]

	app = App()

	slider1 = inputs.InputSlider("size", 2, onUpdate=update1, sMin=1, sMax=50, sSteps=2)
	slider2 = inputs.InputSlider("sdv", 2, onUpdate=update2, sMin=1, sMax=50, sSteps=2)

	ins = inputs.Inputs(state=state)

	ins.addInput(slider1)
	ins.addInput(slider2)
	ins.getFrames()

	show(state)

	app.show()

# find corners in image and finds the four page corners
# parameters: 	imgGrey:			grey scale image
# returns:		( (x, y), ... ):	four corners that are the page
def findCorners(imgOriginal):

	# cv2.imshow('orb', imgOriginal)
	# img = np.copy(imgOriginal)
	# cv2.waitKey(10000)

	# # Initiate ORB detector
	# orb = cv2.ORB_create()
	# # find the keypoints with ORB
	# kp = orb.detect(img, None)
	# # compute the descriptors with ORB
	# kp, des = orb.compute(img, kp)
	# # draw only keypoints location,not size and orientation
	# img2 = cv2.drawKeypoints(img, kp, None, color = (0, 0, 255), flags = 5)
	# cv2.imshow('orb', img)

	def show(state):
		imgGrey = cv2.cvtColor(np.uint8(imgOriginal), cv2.COLOR_BGR2GRAY)
		imgGrey = np.float32(imgGrey)
		imgColor = imgOriginal.copy()

		dst = cv2.cornerHarris(imgGrey, state[0], state[1], state[2])

		#result is dilated for marking the corners, not important
		dst = cv2.dilate(dst, None)

		# Threshold for an optimal value, it may vary depending on the image.
		imgColor[dst > state[3] * dst.max()] = [0,0,255]

		cv2.imshow('corners', imgColor)		

	def update1(control, value, state):
		state[0] = value
		show(state)	

	def update2(control, value, state):
		state[1] = value
		show(state)	

	def update3(control, value, state):
		state[2] = value / 1000
		show(state)	

	def update4(control, value, state):
		state[3] = value / 1000
		show(state)

	state = [2, 5, 0.01, 0.01]

	app = App()

	slider1 = inputs.InputSlider("threshold1", 2, onUpdate=update1, sMin=1, sMax=50, sSteps=1)
	slider2 = inputs.InputSlider("threshold2", 5, onUpdate=update2, sMin=3, sMax=31, sSteps=2)
	slider3 = inputs.InputSlider("threshold3/1000", 10, onUpdate=update3, sMin=1, sMax=1000, sSteps=1)
	slider4 = inputs.InputSlider("threshold4/1000", 10, onUpdate=update4, sMin=1, sMax=1000, sSteps=1)

	ins = inputs.Inputs(state=state)

	ins.addInput(slider1)
	ins.addInput(slider2)
	ins.addInput(slider3)
	ins.addInput(slider4)
	ins.getFrames()

	app.show()

def findEdges(imgOriginal):

	def show(state):
		imgGrey = cv2.cvtColor(np.uint8(imgOriginal), cv2.COLOR_BGR2GRAY)

		edges = cv2.Canny(imgGrey, state[0], state[1])

		cv2.imshow('edges', edges)		

	def update1(control, value, state):
		state[0] = value
		show(state)	

	def update2(control, value, state):
		state[1] = value
		show(state)	

	state = [35, 275]

	app = App()

	slider1 = inputs.InputSlider("threshold1", 35, onUpdate=update1, sMin=0, sMax=500, sSteps=1)
	slider2 = inputs.InputSlider("threshold2", 275, onUpdate=update2, sMin=0, sMax=500, sSteps=1)

	ins = inputs.Inputs(state=state)

	ins.addInput(slider1)
	ins.addInput(slider2)
	ins.getFrames()

	show(state)

	app.show()

def findEdgesAndCorners(imgOriginal):

	def show(state):
		imgGrey = cv2.cvtColor(np.uint8(imgOriginal), cv2.COLOR_BGR2GRAY)
		imgColor = imgOriginal.copy()

		edges = cv2.Canny(imgGrey, state[0], state[1])
		edges = np.float32(edges)
		edgesColor = cv2.cvtColor(np.uint8(edges), cv2.COLOR_GRAY2BGR)

		dst = cv2.cornerHarris(edges, state[2], state[3], state[4])

		#result is dilated for marking the corners, not important
		dst = cv2.dilate(dst, None)

		# Threshold for an optimal value, it may vary depending on the image.
		edgesColor[dst > state[5] * dst.max()] = [0,0,255]

		cv2.imshow('edges and corners', edgesColor)

	def update1(control, value, state):
		state[0] = value
		show(state)	

	def update2(control, value, state):
		state[1] = value
		show(state)		

	def update3(control, value, state):
		state[2] = value
		show(state)	

	def update4(control, value, state):
		state[3] = value
		show(state)	

	def update5(control, value, state):
		state[4] = value / 1000
		show(state)	

	def update6(control, value, state):
		state[5] = value / 1000
		show(state)

	state = [35, 275, 2, 5, 0.01, 0.01]

	app = App()

	slider1 = inputs.InputSlider("threshold1", 35, onUpdate=update1, sMin=0, sMax=500, sSteps=1)
	slider2 = inputs.InputSlider("threshold2", 275, onUpdate=update2, sMin=0, sMax=500, sSteps=1)
	slider3 = inputs.InputSlider("threshold3", 2, onUpdate=update3, sMin=1, sMax=50, sSteps=1)
	slider4 = inputs.InputSlider("threshold4", 5, onUpdate=update4, sMin=3, sMax=31, sSteps=2)
	slider5 = inputs.InputSlider("threshold5/1000", 10, onUpdate=update5, sMin=1, sMax=1000, sSteps=1)
	slider6 = inputs.InputSlider("threshold6/1000", 10, onUpdate=update6, sMin=1, sMax=1000, sSteps=1)

	ins = inputs.Inputs(state=state)

	ins.addInput(slider1)
	ins.addInput(slider2)
	ins.addInput(slider3)
	ins.addInput(slider4)
	ins.addInput(slider5)
	ins.addInput(slider6)
	ins.getFrames()

	show(state)

	app.show()

def findLines(imgOriginal):

	def show(state):
		imgGrey = cv2.cvtColor(np.uint8(imgOriginal), cv2.COLOR_BGR2GRAY)

		edges = cv2.Canny(imgGrey, state[0], state[1])	

		edgesL = edges // 15
		lines = cv2.HoughLinesP(edges, 1, np.pi / 180, state[2], 100, state[3])
		if lines is not None:
			for i in range(0, len(lines)):
				line = lines[i][0]
				cv2.line(edgesL, (line[0], line[1]), (line[2], line[3]), 200, 1)
		cv2.imshow("edgesL", edgesL)

	def update1(control, value, state):
		state[0] = value
		show(state)	

	def update2(control, value, state):
		state[1] = value
		show(state)	

	def update3(control, value, state):
		state[2] = value
		show(state)	

	def update4(control, value, state):
		state[3] = value
		show(state)	

	state = [10, 100, 35, 275]

	app = App()

	slider1 = inputs.InputSlider("threshold1", 35, onUpdate=update1, sMin=0, sMax=500, sSteps=1)
	slider2 = inputs.InputSlider("threshold2", 275, onUpdate=update2, sMin=0, sMax=500, sSteps=1)
	slider3 = inputs.InputSlider("threshold3", 10, onUpdate=update3, sMin=0, sMax=100, sSteps=1)
	slider4 = inputs.InputSlider("threshold4", 100, onUpdate=update4, sMin=0, sMax=20, sSteps=1)

	ins = inputs.Inputs(state=state)

	ins.addInput(slider1)
	ins.addInput(slider2)
	ins.addInput(slider3)
	ins.addInput(slider4)
	ins.getFrames()

	show(state)

	app.show()

	cv2.waitKey(10000)

# run main program
if __name__ == "__main__":
	main()