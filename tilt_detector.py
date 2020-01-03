import math
import cv2
import os
import numpy as np

path_to_images = r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection_inclination\testing_images"
save_path = r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection_inclination\output"


class TiltDetector:

    def __init__(self,
                 save_path,
                 line_thickness,
                 draw_and_save=True):

        self.destination_path = save_path

        self.line_thickness = line_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_colour = (255, 255, 255)
        self.line_type = 3

        # Discarding non vertical lines
        self.angle_thresh = 70

        # In case user wants to draw and save images with lines on them or just
        # calculate and return angle
        self.draw_and_save = draw_and_save

    def process_image(self, path_to_image):

        image = cv2.imread(path_to_image)
        image_name = os.path.split(path_to_image)[-1]

        # Find all lines on the image
        raw_lines = self.generate_lines(image)

        if raw_lines is not None:
            # Rewrite lines in a proper form (x1,y1), (x2,y2)
            lines_to_merge = list()
            for line in self.get_lines(raw_lines):
                lines_to_merge.append(
                        [(line[0], line[1]), (line[2], line[3])]
                                      )
        else:
            print("No lines detected")
            return

        # Process results: merge lines where possible
        merged_lines = self.merge_lines(lines_to_merge)

        # Lines merging visualization
        #self.draw_lines(lines_to_merge, image, "NOT_merged_lines")
        #self.draw_lines(merged_lines, image, "merged_lines")
        #print("Before merging:", len(lines_to_merge), "After merging:", len(merged_lines))

        # Pick lines based on which the angle will be calculated. Ideally we are looking for 2 lines
        # which represent both pole's edges. If there is 1, warn user and calculate angle based on it.
        # Pick two opposite and parrallel lines within the merged ones. This will be a pole
        if len(merged_lines) > 0 and len(merged_lines) != 1:
            the_lines = self.find_pole(merged_lines, image)

        elif len(merged_lines) == 1:
            print("WARNING: Only one line found, angle will be calculated based on it")
            the_lines = merged_lines

        else:
            print("No lines found, angle cannot be calculated")
            return

        assert len(the_lines) == 1 or len(the_lines) == 2, "ERROR: Wrong number of lines found"

        # Calculate inclination angle
        angle = self.calculate_angle(the_lines)

        # If user wants save an images with lines and angle calculated shown on it
        if self.draw_and_save:
            self.draw_lines(lines=the_lines,
                            image=image,
                            window_name="pole lines",
                            image_name=image_name,
                            angle=angle)

        return angle

    def calculate_angle(self, the_lines):

        if len(the_lines) == 2:
            x1_1 = the_lines[0][0][0]
            y1_1 = the_lines[0][0][1]
            x2_1 = the_lines[0][1][0]
            y2_1 = the_lines[0][1][1]

            angle_1 = round(90 - np.rad2deg(np.arctan2(abs(y2_1 - y1_1), abs(x2_1 - x1_1))), 2)

            x1_2 = the_lines[1][0][0]
            y1_2 = the_lines[1][0][1]
            x2_2 = the_lines[1][1][0]
            y2_2 = the_lines[1][1][1]

            angle_2 = round(90 - np.rad2deg(np.arctan2(abs(y2_2 - y1_2), abs(x2_2 - x1_2))), 2)

            return round((angle_1 + angle_2) / 2, 2)

        else:
            x1 = the_lines[0][0][0]
            y1 = the_lines[0][0][1]
            x2 = the_lines[0][1][0]
            y2 = the_lines[0][1][1]

            return round(90 - np.rad2deg(np.arctan2(abs(y2 - y1), abs(x2 - x1))), 2)

    def find_pole(self, merged_lines, image):

        pole_lines = list()

        # A vertical line in the middle of the image.
        dividing_line = [
            (image.shape[1] // 2, 0), (image.shape[1] // 2, image.shape[0])
                         ]
        # ------------------------------------------------------------------------
        # YOU CAN CATCH NOT PARALLEL LINES HERE! THEY NEED TO BE WITHIN SOME ANGLE
        # Within 2 degrees. Check for parallelizm, consider lenght as well?
        # ------------------------------------------------------------------------

        # Find a line to the left the dividing line
        for line in merged_lines:
            if line[0][0] <= dividing_line[0][0] and line[1][0] <= dividing_line[0][0]:
                pole_lines.append(line)
                break

        # Find a line to the right of the dividing line
        for line in merged_lines:
            if line[0][0] > dividing_line[0][0] and line[1][0] > dividing_line[0][0]:
                pole_lines.append(line)
                break

        return pole_lines

    def draw_lines(self,
                   lines,
                   image,
                   window_name,
                   image_name,
                   angle):

        for line in lines:
            cv2.line(image,
                     (line[0][0], line[0][1]),
                     (line[1][0], line[1][1]),
                     (0, 0, 255),
                     self.line_thickness)

        cv2.putText(image,
                    str(angle),
                    (int(image.shape[1]*0.35), int(image.shape[0]*0.95)),
                    self.font,
                    self.font_scale,
                    self.font_colour,
                    self.line_type)

        cv2.imwrite(os.path.join(self.destination_path, image_name), image)

        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # cv2.imshow(window_name, image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def generate_lines(self, image):

        # Apply mask to remove background
        image_masked = self.apply_mask(image)

        # Generate edges
        edges = cv2.Canny(image_masked,
                          threshold1=50,
                          threshold2=200,
                          apertureSize=3)
        # Based on the edges found, find lines
        lines = cv2.HoughLinesP(edges,
                                rho=1,
                                theta=np.pi / 180,
                                threshold=100,
                                minLineLength=100,
                                maxLineGap=100)

        return lines

    def apply_mask(self, image):

        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # start_x, start_y, width, height
        rect = (int(image.shape[1] * 0.1),
                0,
                image.shape[1] - int(image.shape[1] * 0.2),
                image.shape[0])

        cv2.grabCut(image,
                    mask,
                    rect,
                    bgd_model,
                    fgd_model,
                    10,
                    cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        img = image * mask2[:, :, np.newaxis]

        ret, thresh = cv2.threshold(img,
                                    0,
                                    255,
                                    cv2.THRESH_BINARY)

        return thresh

    def get_lines(self, lines_in):

        if cv2.__version__ < "3.0":
            return lines_in[0]

        return [line[0] for line in lines_in]

    def merge_lines(self, lines_to_merge):

        # Discard horizontal lines (no point merging lines that are not what we need)
        vertical_lines = list()

        for line in lines_to_merge:
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[1][0]
            y2 = line[1][1]
            angle = abs(round(np.rad2deg(np.arctan2((y2 - y1), (x2 - x1))), 2))

            if angle < self.angle_thresh:
                continue

            vertical_lines.append(
                [(line[0][0], line[0][1]), (line[1][0], line[1][1])]
                                  )

        # Sort and get line orientation
        lines_x, lines_y = list(), list()
        for line in vertical_lines:
            orientation = math.atan2(
                (line[0][1] - line[1][1]), (line[0][0] - line[1][0])
                                     )

            if (abs(math.degrees(orientation)) > 45) and\
                                abs(math.degrees(orientation)) < (90 + 45):
                lines_y.append(line)
            else:
                lines_x.append(line)

        lines_x.sort(key=lambda line: line[0][0])
        lines_y.sort(key=lambda line: line[0][1])

        merged_lines_x = self.merge_lines_pipeline_2(lines_x)
        merged_lines_y = self.merge_lines_pipeline_2(lines_y)

        merged_lines_all = list()
        merged_lines_all.extend(merged_lines_x)
        merged_lines_all.extend(merged_lines_y)

        return merged_lines_all

    def merge_lines_pipeline_2(self, lines):

        super_lines_final = []
        super_lines = []
        min_distance_to_merge = 30
        min_angle_to_merge = 30

        # check if a line has angle and enough distance to be considered similar
        for line in lines:
            create_new_group = True
            group_updated = False

            for group in super_lines:
                for line2 in group:

                    if self.get_distance(line2, line) < min_distance_to_merge:
                        # check the angle between lines
                        orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))
                        orientation_j = math.atan2((line2[0][1] - line2[1][1]), (line2[0][0] - line2[1][0]))

                        if int(abs(abs(math.degrees(orientation_i)) - abs(
                                math.degrees(orientation_j)))) < min_angle_to_merge:
                            # print("angles", orientation_i, orientation_j)
                            # print(int(abs(orientation_i - orientation_j)))
                            group.append(line)

                            create_new_group = False
                            group_updated = True
                            break

                if group_updated:
                    break

            if create_new_group:
                new_group = list()
                new_group.append(line)

                for idx, line2 in enumerate(lines):
                    # check the distance between lines
                    if self.get_distance(line2, line) < min_distance_to_merge:
                        # check the angle between lines
                        orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))
                        orientation_j = math.atan2((line2[0][1] - line2[1][1]), (line2[0][0] - line2[1][0]))

                        if int(abs(abs(math.degrees(orientation_i)) - abs(
                                math.degrees(orientation_j)))) < min_angle_to_merge:
                            # print("angles", orientation_i, orientation_j)
                            # print(int(abs(orientation_i - orientation_j)))

                            new_group.append(line2)

                            # remove line from lines list
                            # lines[idx] = False
                # append new group
                super_lines.append(new_group)

        for group in super_lines:
            super_lines_final.append(self.merge_lines_segments1(group))

        return super_lines_final

    def merge_lines_segments1(self, lines, use_log=False):

        if len(lines) == 1:
            return lines[0]

        line_i = lines[0]

        # orientation
        orientation_i = math.atan2((line_i[0][1] - line_i[1][1]), (line_i[0][0] - line_i[1][0]))

        points = []
        for line in lines:
            points.append(line[0])
            points.append(line[1])

        if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90 + 45):

            # sort by y
            points = sorted(points, key=lambda point: point[1])

            if use_log:
                print("use y")
        else:

            # sort by x
            points = sorted(points, key=lambda point: point[0])

            if use_log:
                print("use x")

        return [points[0], points[len(points) - 1]]

    def lines_close(self, line1, line2):

        dist1 = math.hypot(line1[0][0] - line2[0][0], line1[0][0] - line2[0][1])
        dist2 = math.hypot(line1[0][2] - line2[0][0], line1[0][3] - line2[0][1])
        dist3 = math.hypot(line1[0][0] - line2[0][2], line1[0][0] - line2[0][3])
        dist4 = math.hypot(line1[0][2] - line2[0][2], line1[0][3] - line2[0][3])

        if min(dist1, dist2, dist3, dist4) < 100:
            return True
        else:
            return False

    def lineMagnitude(self, x1, y1, x2, y2):
        lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
        return lineMagnitude

    def DistancePointLine(self, px, py, x1, y1, x2, y2):
        # http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        LineMag = self.lineMagnitude(x1, y1, x2, y2)

        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            # // closest point does not fall within the line segment, take the shorter distance
            # // to an endpoint
            ix = self.lineMagnitude(px, py, x1, y1)
            iy = self.lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = self.lineMagnitude(px, py, ix, iy)

        return DistancePointLine

    def get_distance(self, line1, line2):

        dist1 = self.DistancePointLine(line1[0][0], line1[0][1],
                                       line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist2 = self.DistancePointLine(line1[1][0], line1[1][1],
                                       line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist3 = self.DistancePointLine(line2[0][0], line2[0][1],
                                       line1[0][0], line1[0][1], line1[1][0], line1[1][1])
        dist4 = self.DistancePointLine(line2[1][0], line2[1][1],
                                       line1[0][0], line1[0][1], line1[1][0], line1[1][1])

        return min(dist1, dist2, dist3, dist4)


def main():

    detector = TiltDetector(save_path=save_path,
                            line_thickness=2)

    total_error = 0

    for image_name in os.listdir(path_to_images):

        if not any(image_name.endswith(ext) for ext in [".jpg", ".png", ".jpeg", ".JPG", ".JPEG", ".PNG"]):
            continue

        print(image_name)

        path_to_image = os.path.join(path_to_images, image_name)

        truth_angle = float(image_name[3:7])
        predicted_tilt_angle = detector.process_image(path_to_image)

        difference = abs(truth_angle - predicted_tilt_angle)
        error = round(difference / truth_angle, 3)
        print("Error:", error)

        total_error += error

    mean_error = round(total_error / len(os.listdir(path_to_images)), 3)
    print("\nMEAN ERROR:", mean_error * 100, "%")

if __name__ == "__main__":
    main()
