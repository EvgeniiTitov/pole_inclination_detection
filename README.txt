Task: estimate concrete pole inclination

MEAN ERROR on 99 test images is 11.5%

Algorithm:
1. Read/Open an image (-> np array)

2. Find all lines on the image
    2.1 Apply mask to remove background (drastically decrease the total number of lines that get detected)
    2.2 Generate edges (Canny edge detector)
    2.3 Generate lines using the edges (HoughLinesP)

3. Discard all lines that are not vertical (not within 40 degrees cone)

4. Merge lines into longer ones where possible

5. Use imaginary vertical line in the middle of the image to separate the generated lines lying to the right
   and to the left of that imaginary line. This ensures we do not pick 2 lines located too close to one another

6. Among the separated lines pick the ones that are most parallel

7. Use these two lines to calculate the angle

TO DO:
- Implement metadata reading and taking it into account in case camera was tilted when an image was taken
- Potential memory leak identified. Same image gives different results when getting processed alone or in a bulk

Commands:
python main.py --image= OR --folder= --save_path= "to save images" --retrieve= "to extract the polygon"

# FOR INTEGRATION
Detector returns: angle calculated, coordinates of the line(s) detected and used for angle calculation
concrete_polygon takes an image and the line(s), outputs only the area confined by the lines provided
