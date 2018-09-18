# Perceptual-Computing-Project
Programming Assignment for Perceptual Computing

A python program that accepts images of documents (i.e. images taken with a camera), detects the main document in the image, its corners and edges to then de-wrap and transform the image to flatten the document and crop it using corner and edge data. Lastly It should convert the resulting image into a black and white image using Otsu’s method.

## Requirements
- read image specified in argument
- display image
- overlay image with data such as points and lines
- detect corners
  - display corners
  - find important corners
- detect edges
  - display edges
  - find edges of the page
- calculate transformation matrix
- apply transformation
- apply black and white filter
- save image to file

## Milestones
1. Basic IO – loading images into workable formats, displaying them with additional information such as corners and edges detected and storing them.
2. Corner detection – Detect corners, their orientation and size.
3. Filter out the corner quadruplet that roughly represents the outline of the page (excluding skewing and bends).
4. Detect edges – Detect edges, their direction and intensity.
5. Filter out edges that pass through the four corners and represent the outline of the page.
6. Calculating transformation matrices and operations necessary.
7. Perform Transformation to de-skew, crop image.
8. Apply Otsu’s method to turn the document black and white using an appropriately selected threshold.