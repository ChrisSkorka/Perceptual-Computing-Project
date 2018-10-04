import sys
import numpy as np
import cv2
from common import *
import inputs

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

def test(imgOriginal):

	def show(state):

		# blur
		imgBlured = cv2.GaussianBlur(
			imgOriginal, 
			(state['blur kernel size'].v, state['blur kernel size'].v), 
			state['blur diviation'].v)

		cv2.imshow('gaussian blured', imgBlured)	

		# dilate image
		dilationKernel = cv2.getStructuringElement(
			cv2.MORPH_RECT, 
			(state['dilation size'].v, state['dilation size'].v)
			)
		imgDilated = cv2.dilate(imgBlured, dilationKernel)

		cv2.imshow('dilation', imgDilated)	

		# harris corner detection
		bluredGrey = np.float32(cv2.cvtColor(np.uint8(imgBlured), cv2.COLOR_BGR2GRAY))
		dilatedGrey = np.float32(cv2.cvtColor(np.uint8(imgDilated), cv2.COLOR_BGR2GRAY))

		dst = cv2.cornerHarris(
			dilatedGrey, 
			state['harris block size'].v, 
			state['harris ksize'].v, 
			state['harris k'].v)
	
		dst = cv2.dilate(dst, None) # dilate

		imgCorners = imgBlured.copy()
		imgCorners[dst > state['harris threashold'].v * dst.max()] = [0,0,255] # threshold
		cv2.imshow('harris corners', imgCorners)

		# canny edges
		imgEdges = cv2.Canny(
			np.uint8(dilatedGrey), 
			state['canny lower'].v, 
			state['canny upper'].v)

		cv2.imshow('canny edges', imgEdges)	

		# hough lines
		imgLines = imgEdges // 10
		lines = cv2.HoughLinesP(
			imgEdges, 
			state['hough rho'].v, 
			state['hough theta'].v * np.pi / 180, 
			state['hough threshold'].v, 
			state['hough srn'].v, 
			state['hough stn'].v)
		if lines is not None:
			for i in range(0, len(lines)):
				line = lines[i][0]
				cv2.line(imgLines, (line[0], line[1]), (line[2], line[3]), 200, 1)
		cv2.imshow("hough lines", imgLines)


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
		'dilation size': 		var(v = 9, 		vmin = 1, vmax = 50					),
		'harris block size': 	var(v = 1, 		vmin = 1, vmax = 31, steps = 2		),
		'harris ksize': 		var(v = 3, 		vmin = 1, vmax = 31,  steps = 2		),
		'harris k': 			var(v = 40, 	vmin = 1, vmax = 1000, scale = 0.001),
		'harris threashold':	var(v = 10, 	vmin = 1, vmax = 1000, scale = 0.001),
		'canny lower': 			var(v = 100, 	vmin = 1, vmax = 255				),
		'canny upper': 			var(v = 200, 	vmin = 1, vmax = 255				),
		'hough rho': 			var(v = 1, 		vmin = 1, vmax = 32					),
		'hough theta': 			var(v = 1	, 	vmin = 1, vmax = 360				),
		'hough threshold': 		var(v = 100, 	vmin = 1, vmax = 256				),
		'hough srn': 			var(v = 0, 		vmin = 1, vmax = 255				),
		'hough stn': 			var(v = 0, 		vmin = 1, vmax = 255				),
		}

	show(state)

	app = App()
	ins = inputs.Inputs(state=state)

	for name in state:
		properties = state[name]
		slider = inputs.InputSlider(name, properties.v, onUpdate=update, sMin=properties.min, sMax=properties.max, sSteps=properties.steps)
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