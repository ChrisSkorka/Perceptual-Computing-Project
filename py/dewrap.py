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
	cv2.imshow('imgOriginalGrey', imgOriginalGrey)

	# find the relevant corners
	# findCorners(imgOriginal)

	# find the relevant edges
	findEdges(imgOriginal)
	
	# wait for preview to finish
	cv2.waitKey(10000)


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


# run main program
if __name__ == "__main__":
	main()