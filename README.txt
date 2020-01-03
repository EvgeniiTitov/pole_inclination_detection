Task: estimate concrete pole's inclination

MEAN ERROR on 99 test images is 15%

Algorithm:
1. Read/Open an image (-> np array)

2. Find all lines on the image
    2.1 Apply mask to remove background (drastically decrease the total number of lines that get detected)
    2.2 Generate edges (Canny edge detector)
    2.3 Generate lines using the edges (HoughLinesP)

3. Discard all lines that are not vertical (not within 40 degrees cone)

4. Merge lines into longer ones where possible

5. Create a vertical line in the middle of the image (to extract lines representing pole's edges out of all
   the lines detected

6. Using this vertical line pick 2 lines (one to the left, one to the right).

TO DO:
- When picking 2 lines out of all lines generated, we could also check if they are parallel, play with
their lenght etc.
- Implement metadata reading and taking it into account in case camera was tilted when an image was taken