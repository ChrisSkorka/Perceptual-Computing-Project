import sys, math, random, itertools, cv2
import numpy as np
from common import *
import inputs

# global variables
manualMode = True
showProgress = True
saveProgress = False
applyBimeans = False
testOutput = []

# represents a possible page in the image
# holds the corners, size, area and confidence realted measurements
class Page:
	def __init__(self, corners = None):
		self.corners = corners
		self.size = (0, 0)
		self.area = 0
		self.confidence = 0
		self.cornerCornerMatchCount = 0

		self.setCorners(corners)

	def setCorners(self, corners):
		points = sorted(corners, key=lambda p: p[1])
		topleft, topright = 		sorted(points[:2], key = lambda p: p[0])
		bottomleft, bottomright = 	sorted(points[2:], key = lambda p: p[0])

		# calculate bounding box size
		self.size = max(topright[0] - topleft[0], bottomright[0] - bottomleft[0]), max(bottomleft[1] - topleft[1], bottomright[1] - topright[1])
		self.area = self.size[0] * self.size[1]
		
		self.corners = [topleft, topright, bottomleft, bottomright]

	def computeConfidence(self, imgSize):
		if type(imgSize) == np.ndarray:
			imgSize = imgSize.shape[:2]
		
		normArea = self.area / (imgSize[0] * imgSize[1])
		normCornerMatchCount = (self.cornerCornerMatchCount + 1) / 5
		self.confidence = normArea * normCornerMatchCount

	def __repr__(self):
		d = {}
		d['corners'] = self.corners
		d['size'] = self.size
		d['area'] = self.area
		d['confidence'] = self.confidence
		d['cornerCornerMatchCount'] = self.cornerCornerMatchCount
		return str(d)

# represents the preoperties if a parameters for the image processing methods
# it holds, the default value, minimum, maximum and steps in which the parameter
# can be altered
class Property:
	def __init__(self, value, vmin, vmax, steps = 1, fixed = False):
		self.value = value
		self.min = vmin
		self.max = vmax
		self.steps = steps
		self.fixed = fixed

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

	# read file and convert to gray scale
	img = cv2.imread(filename)
	testOutput = img.copy()

	# show
	cv2.imshow('imgOriginal', img)

	if manualMode:
		# test all parameters
		test(img)

	else:

		# find optimal parameters
		parameters = findOptimalParameters(img)

		# transform image accordingly
		transformed = applyTransformation(img, parameters)

		# if selected apply bi means to produce a black and white image
		if applyBimeans:
			gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
			transformed = binmeans(gray)

			# transformed = cv2.adaptiveThreshold(transformed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

		# show final image
		index = filename.rfind('\\')
		newfilename = "output"+filename[index:]
		logImg(transformed, 'final', True)
		joinImages(newfilename)
		# cv2.waitKey(10000)

	

# local search algorithm to optimise parameters, implements steepest ascent algorithm
def findOptimalParameters(img):

	properties = {
		'blur_kernel_size': 			Property(value = 3, 	vmin = 1, vmax = 15, 	steps = 2, 	fixed = True		),
		'blur_diviation': 				Property(value = 1, 	vmin = 1, vmax = 9,						fixed = True	),
		'close_size': 					Property(value = 5, 	vmin = 1, vmax = 50,					),
		'harris_block_size': 			Property(value = 25, 	vmin = 1, vmax = 31, 	steps = 2		),
		'harris_ksize': 				Property(value = 3, 	vmin = 1, vmax = 31,  	steps = 2, 		fixed = True	),
		'harris_k': 					Property(value = 0.04, 	vmin = 0, vmax = 1, 	steps = 0.001, 	fixed = True	),
		'harris_threashold':			Property(value = 0.03, 	vmin = 0, vmax = 1, 	steps = 0.001	),
		'canny_lower': 					Property(value = 50, 	vmin = 1, vmax = 255,					),
		'canny_upper': 					Property(value = 100, 	vmin = 1, vmax = 255,					),
		'hough_rho': 					Property(value = 1, 	vmin = 1, vmax = 32,					),
		'hough_theta': 					Property(value = 1,	 	vmin = 1, vmax = 360,					),
		'hough_threshold': 				Property(value = 60, 	vmin = 40, vmax = 256,					),
		'hough_srn': 					Property(value = 0, 	vmin = 0, vmax = 255,	fixed = True	),
		'hough_stn': 					Property(value = 0, 	vmin = 0, vmax = 255,	fixed = True	),
		'hough_group_threshold_rho': 	Property(value = 10,	vmin = 0, vmax = 100,					),
		'hough_group_threshold_theta': 	Property(value = 0.1, 	vmin = 0, vmax = 0.5, 	steps = 0.01	),
		'perpective_threshold_theta': 	Property(value = 0.4,	vmin = 0, vmax = 0.7, 	steps = 0.01	),
	}

	parameters = {name:properties[name].value for name in properties}

	confidence = measureConfidenceOfParameters(img, parameters)
	cv2.waitKey(100)

	while confidence == 0:
		print("try random")
		for parname in parameters:
			if not properties[parname].fixed:
				randomInRange = random.randint(0, (properties[parname].max - properties[parname].min ) / properties[parname].steps)
				parameters[parname] = properties[parname].min + randomInRange * properties[parname].steps
		confidence = measureConfidenceOfParameters(img, parameters)
	
		cv2.waitKey(100)

	progress = True
	while progress:
		progress = False
		print("next")
		nextParameters = parameters.copy()

		for name in parameters:
			if not properties[name].fixed:

				# cahnge = random.randint(0, 10)
				cahnge = 1
				upperValue = parameters[name] + cahnge * properties[name].steps
				lowerValue = parameters[name] - cahnge * properties[name].steps

				if upperValue <= properties[name].max:
					testParameters = parameters.copy()
					testParameters[name] = upperValue
					if measureConfidenceOfParameters(img, testParameters) > confidence:
						nextParameters[name] = upperValue
						progress = True

				if lowerValue >= properties[name].min:
					testParameters = parameters.copy()
					testParameters[name] = lowerValue
					if measureConfidenceOfParameters(img, testParameters) > confidence:
						nextParameters[name] = lowerValue
						progress = True

		parameters = nextParameters
		confidence = measureConfidenceOfParameters(img, parameters)
		print(confidence, parameters, sep = "\n")
		cv2.waitKey(100)

	# 	cv2.waitKey(1000)

	return parameters

# measure confidence in the results with the specified parameters
# parameters: 	imgOriginal: nparray	image to test
#				parameters: {}			parameters to apply 
# returns:		float: confidence
def measureConfidenceOfParameters(img, parameters):
	global testOutput

	pages = findPages(img, parameters)

	if len(pages) > 0:
		best = max(pages, key = lambda p:p.confidence)
		imgRectangles = darwRectanglesImg(img, [best.corners], (255, 128, 0))
		logImg(imgRectangles, 'box', True)
		testOutput.append(imgRectangles)

	# overall confidence = average confidence / num pages
	overallConfidence = 0
	if len(pages) > 0:
		for page in pages:
			overallConfidence += page.confidence
		overallConfidence /= len(pages)

	return overallConfidence

# find all possible pages in the image gien the parameters
def findPages(img, parameters):
	global testOutput
	testOutput = [img]

	# blur
	imgSmooth = smoothImg(img, **parameters)
	logImg(imgSmooth, 'smoothed')

	# dilate image
	imgDilated = dilateImg(imgSmooth, **parameters)

	# erode image
	imgEroded = erodeImg(imgDilated, **parameters)
	logImg(imgEroded, 'closed')
	testOutput.append(imgEroded)
	
	erodedGray = grayFlaotImg(imgEroded)

	# harris corner detection
	harrisIntensity = harrisCornerImg(erodedGray, **parameters)
	harrisThreshold = harrisThresholdImg(harrisIntensity, **parameters);
	harrisCorners = paintPixelsImg(img, harrisThreshold, [0, 0, 255])
	logImg(harrisCorners, 'harris corners')
	testOutput.append(harrisCorners)

	# canny edges
	imgEdges = cannyEdgeImg(erodedGray, **parameters)
	logImg(imgEdges, 'canny edges')
	testOutput.append(imgEdges)

	# # probabilistic hough lines
	# imgProbLines = probabilisticHoughTransformImg(imgEdges, **state)

	# hough lines
	houghLines = houghTransformLines(imgEdges, **parameters)
	testOutput.append(drawLinesImg(black(img), houghLines, 255))
	groupedLines = groupLines(houghLines, **parameters)
	horizontalLines, verticalLines = filterHorizontalAnfVerticalLines(groupedLines, **parameters)
	imgLines = drawLinesImg(black(img), horizontalLines, 255)
	imgLines = drawLinesImg(imgLines, verticalLines, 255)
	logImg(imgLines, 'hough lines')
	testOutput.append(imgLines)

	# find 4 sided contours
	# contours = contours4Img(imgLines)
	# imgContours = darwContousImg(img, contours)
	# logImg(imgContours, 'contours')

	# find line intersections and corresponding rectangles
	rectangles = findRectangles(horizontalLines, verticalLines, img.shape[:2], **parameters)
	imgRectangles = darwRectanglesImg(img, rectangles, (0, 255, 0))
	logImg(imgRectangles, 'rectangles')
	testOutput.append(imgRectangles)

	# points = filterPointsOnCorners(points, harrisThreshold)
	# imgPoints = drawPointsImg(imgLines, points, radius = 5)
	# logImg(imgPoints, 'filtered points')

	# create page objects
	pages = [Page(r) for r in rectangles]

	# count number of corners of page that match with corners form Harris results
	countCornerMatchesPages(harrisThreshold, pages)

	for page in pages:
		page.computeConfidence(img)

	return pages

# find page and apply transformations according to parameters
def applyTransformation(img, parameters):
	
	pages = findPages(img, parameters)
	best = max(pages, key = lambda p:p.confidence)
	imgRectangles = darwRectanglesImg(img, [best.corners], (0, 128, 255))
	logImg(imgRectangles, 'box', True)
	testOutput.append(imgRectangles)

	imgTransformed = perspectiveTransformImg(img, best.corners)
	logImg(imgTransformed, 'transformed')

	return imgTransformed

# compose a wide image of all stages side by side
def joinImages(filename):
	global testOutput

	formated = []

	print(len(testOutput))
	for img in testOutput:
		img = np.uint8(img)
		if len(img.shape) == 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		print(img.shape)
		formated.append(img)

	for i in range(len(formated), 8):
		formated.append(np.zeros(testOutput[0].shape, dtype = np.uint8))

	row1 = np.concatenate(formated[:4], axis = 1)
	row2 = np.concatenate(formated[4:8], axis = 1)
	img = np.concatenate((row1, row2), axis = 0)
	print('test', img.shape)
	print('file', filename)
	cv2.imshow('test', img)
	cv2.imwrite(filename, img)

# show and or save an image	
def logImg(img, name, overrideShow = False, overrideSave = False):
	if showProgress or overrideShow:
		cv2.imshow(name, img)

	if saveProgress or overrideSave:
		cv2.imwrite('output/'+name+".png", img)

# black image with given size or size of given image
def black(size):
	if type(size) == np.ndarray:
		return np.zeros(size.shape[:2], dtype=np.uint8)
	else:
		return np.zeros(size[:2], dtype=np.uint8)

# convert image to gray scale
def grayIntImg(img):
	return np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
	
# convert image to float values
def grayFlaotImg(img):
	return np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

# apply smoothing according to parameters
def smoothImg(img, blur_kernel_size, blur_diviation, **args):

	imgSmooth = cv2.GaussianBlur(
		img, 
		(blur_kernel_size, blur_kernel_size), 
		blur_diviation
	)

	return imgSmooth

# apply dilation according to parameters
def dilateImg(img, close_size, **args):

	dilationKernel = cv2.getStructuringElement(
		cv2.MORPH_ELLIPSE, 
		(close_size, close_size)
	)

	imgDilated = cv2.dilate(img, dilationKernel)

	return imgDilated

# apply ertion according to parameters
def erodeImg(img, close_size, **args):

	erotionKernel = cv2.getStructuringElement(
		cv2.MORPH_ELLIPSE, 
		(close_size, close_size)
		# (erode_size, erode_size) # local seach increase dilation without increaing erotion
	)

	imgEroded = cv2.erode(img, erotionKernel)

	return imgEroded

# apply harris corner detection according to parameters
def harrisCornerImg(img, harris_block_size, harris_ksize, harris_k, **args):

	intensity = cv2.cornerHarris(
		img, 
		harris_block_size, 
		harris_ksize, 
		harris_k
	)

	if intensity.max() > 0:
		intensity = intensity / intensity.max()

	return intensity

# apply a threashold to harris results according to parameters
def harrisThresholdImg(img, harris_threashold, **args):

	threshold = img > harris_threashold * img.max()

	return threshold

# paint pixels according to a threshold map
def paintPixelsImg(img, pixel, color = (0, 0, 255), **args):

	copy = img.copy()
	copy[pixel] = color
	return copy

# apply canny edge detection according to parameters
def cannyEdgeImg(img, canny_lower, canny_upper, **args):
	
	imgEdges = cv2.adaptiveThreshold(np.uint8(img), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

	imgEdges = cv2.Canny(
		np.uint8(imgEdges), 
		canny_lower, 
		canny_upper
	)

	return imgEdges

# apply probabilistic hough transform according to parameters
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

# apply hough transform according to parameters
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

# filter horizontal and vertical lines from all lines according to parameters
def filterHorizontalAnfVerticalLines(lines, perpective_threshold_theta, **args):

	t = math.sin(perpective_threshold_theta)

	horizontalLines = sorted([l for l in lines if -t < math.cos(l[1]) < t], key = lambda l:l[0])
	verticalLines =   sorted([l for l in lines if -t < math.sin(l[1]) < t], key = lambda l:l[0])

	return horizontalLines, verticalLines

# group togeather similar lines according to parameters
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

# draw lines into image
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

# find 4 sided contours
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

# draw contours
def darwContousImg(img, contours, color = (0, 255, 0), **args):
	
	imgContours = img.copy()

	cv2.drawContours(imgContours, contours, -1, color, 1)

	return imgContours

# find intersection between two lines
def findLineIntersection(lineA, lineB, **args):
	
	def sin(x):
		y = math.sin(x)
		if y == 0:
			return 0.001
		else:
			return y

	def cos(x):
		y = math.cos(x)
		if y == 0:
			return 0.001
		else:
			return y

	p1, t1, p2, t2 = *lineA, *lineB

	x = int( ( p2 / sin(t2) - p1 / sin(t1) ) / ( cos(t2) / sin(t2) - cos(t1) / sin(t1) ) )
	y = int( ( p2 / cos(t2) - p1 / cos(t1) ) / ( sin(t2) / cos(t2) - sin(t1) / cos(t1) ) )

	return (x, y)

# find all possible rectangles from all vertical and horizontal lines
def findRectangles(horizontalLines, verticalLines, boundry, **args):

	horizontalLinePairs = [l for l in itertools.combinations(horizontalLines, 2)]
	verticalLinePairs =   [l for l in itertools.combinations(verticalLines,   2)]

	lineQuadruplets = [(x, y) for x in verticalLinePairs for y in horizontalLinePairs]

	rectangles = []

	for quad in lineQuadruplets:
		corner = (
			findLineIntersection(quad[0][0], quad[1][0]),
			findLineIntersection(quad[0][0], quad[1][1]),
			findLineIntersection(quad[0][1], quad[1][0]),
			findLineIntersection(quad[0][1], quad[1][1]),
		)
		rectangles.append(corner)

	return rectangles

# draw all rectangles
def darwRectanglesImg(img, rectangles, color = 255, **args):

	imgRectangles = img.copy()

	for rect in rectangles:
		pts = np.array([rect[0], rect[1], rect[3], rect[2]], np.int32)
		pts = pts.reshape((-1,1,2))
		cv2.polylines(imgRectangles, [pts], True, color, 3)

	return imgRectangles

# counte the number of corners of each rectangle that match with harris results
def countCornerMatchesPages(harrisThreshold, pages, **args):
	
	h, w = harrisThreshold.shape
	for page in pages:
		for (x, y) in page.corners:
			if 0 < y < h and 0 < x < w and harrisThreshold[y, x]:
				page.cornerCornerMatchCount += 1

# draw all points
def drawPointsImg(img, points, color = 255, radius = 1):

	imgPoints = img.copy()

	for point in points:
		cv2.circle(imgPoints, point, radius, color)

	return imgPoints

# filters out points that match with harris results
def filterPointsOnCorners(points, cornerMap):

	filtered = []

	for (x, y) in points:
		if cornerMap[y, x]:
			filtered.append((x, y))

	return filtered

# apply perspective transform on image according the corners of the page
def perspectiveTransformImg(img, points):

	# top to bottom
	points.sort(key=lambda p: p[1])
	topleft, topright = sorted(points[:2], key = lambda p: p[0])
	bottomleft, bottomright = sorted(points[2:], key = lambda p: p[0])

	# calculate bounding box size
	minWidth = min(topright[0] - topleft[0], bottomright[0] - bottomleft[0])
	maxWidth = max(topright[0] - topleft[0], bottomright[0] - bottomleft[0])
	minHeight = min(bottomleft[1] - topleft[1], bottomright[1] - topright[1])
	maxHeight = max(bottomleft[1] - topleft[1], bottomright[1] - topright[1])

	minToMaxWidthRatio = maxWidth / minWidth
	minToMaxHeightRatio = maxHeight / minHeight

	print(minToMaxWidthRatio, minToMaxHeightRatio)

	w = int(maxWidth * minToMaxHeightRatio)
	h = int(maxHeight * minToMaxWidthRatio)

	# transformation matrix parameters
	pointsSource = np.float32([topleft, topright, bottomleft, bottomright])
	pointsDestination = np.float32([(0, 0), (w, 0), (0, h), (w, h)])

	inset = 0.01
	winset = w*inset
	hinset = h*inset
	insetMatrix = np.float32([[winset, hinset], [-winset, hinset], [winset, -hinset], [-winset, -hinset]])
	pointsSource = pointsSource + insetMatrix

	matrix = cv2.getPerspectiveTransform(pointsSource, pointsDestination)
	imgTransform = cv2.warpPerspective(img, matrix, (w, h))

	return imgTransform

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
    basethresholds[0][0], baseCenters[0][0] = getThresholdAndCenters(img[:yRange//2, :xRange//2])
    basethresholds[0][1], baseCenters[0][1] = getThresholdAndCenters(img[:yRange//2, xRange//2:])
    basethresholds[1][0], baseCenters[1][0] = getThresholdAndCenters(img[yRange//2:, :xRange//2])
    basethresholds[1][1], baseCenters[1][1] = getThresholdAndCenters(img[yRange//2:, xRange//2:])

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
    img2 = img2array.reshape(img.shape)

    return img2

# start testing mode with sliders for parameters
def test(img):

	def show(state):
		print(measureConfidenceOfParameters(img, state))

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
		'close_size': 				var(v = 6, 		vmin = 1, vmax = 50						),
		'erode_size': 					var(v = 6, 		vmin = 1, vmax = 50						),
		'harris_block_size': 			var(v = 25, 	vmin = 1, vmax = 31, 	steps = 2		),
		'harris_ksize': 				var(v = 3, 		vmin = 1, vmax = 31,  	steps = 2		),
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
		'perpective_threshold_theta': 	var(v = 0.5,	vmin = 0, vmax = 3.0, 	steps = 0.01	),
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