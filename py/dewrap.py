import sys
import numpy as np
import cv2

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

	# find the relevant corners
	findCorners(imgOriginalGrey)

	# show
	cv2.imshow('imgOriginalGrey', imgOriginalGrey)
	cv2.waitKey(10000)

# find corners in image and finds the four page corners
# parameters: 	imgGrey:			grey scale image
# returns:		( (x, y), ... ):	four corners that are the page
def findCorners(imgGrey):
	imgGrey = imgGrey.copy()
	imgGrey = np.float32(imgGrey)

	dst = cv2.cornerHarris(imgGrey, 2, 5, 0.01)

	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst, None)

	# Threshold for an optimal value, it may vary depending on the image.
	imgColor = cv2.cvtColor(np.uint8(imgGrey), cv2.COLOR_GRAY2BGR)
	imgColor[dst > 0.01 * dst.max()] = [0,0,255]

	cv2.imshow('corners', imgColor)


# run main program
if __name__ == "__main__":
	main()