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
	imgOriginalGrey = img1 = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)


	# show
	cv2.imshow('imgOriginalGrey', imgOriginalGrey)

# run main program
if __name__ == "__main__":
	main()