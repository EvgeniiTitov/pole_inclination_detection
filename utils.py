import cv2
import os
import sys
import numpy as np


class ResultsHandler:

    def __init__(
            self,
            save_path,
            line_thickness=2,
            window_name='window'
    ):
        self._window_name = window_name
        self._is_window_created = False
        self._sleeping_time = 1

        self._line_thickness = line_thickness
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 1
        self._font_colour = (255, 0, 255)
        self._line_type = 3

        self.save_path = save_path

    @property
    def is_window_created(self):
        return self._is_window_created

    def create_window(self):
        cv2.namedWindow(self._window_name)
        self._is_window_created = True

    def show(self, image):
        cv2.imshow(self._window_name, image)

    def destroy_windows(self):
        cv2.destroyWindow(self._window_name)
        self._is_window_created = False

    def draw_lines_write_text(self,
                              lines,
                              image,
                              angle):

        for line in lines:
            cv2.line(image,
                     (line[0][0], line[0][1]),
                     (line[1][0], line[1][1]),
                     (0, 0, 255),
                     self._line_thickness)

        cv2.putText(image,
                    str(angle),
                    (int(image.shape[1]*0.35), int(image.shape[0]*0.95)),
                    self._font,
                    self._font_scale,
                    self._font_colour,
                    self._line_type)

        return image

    def save_image(self,
                   lines,
                   image,
                   image_name,
                   angle):

        cv2.imwrite(os.path.join(self.save_path, image_name),
                    self.draw_lines_write_text(lines, image, angle))

    def show_image(self,
                   lines,
                   image,
                   angle):

        pass


class LineExtender:

    def extend_lines(self,
                     image,
                     the_lines):

        """
        Extends the lines provided within an image (until the lines reach
        image's boundaries)
        :param image: an image on which the lines were detetced
        :param the_lines: list of lists, lines
        :return: coordinates of extended lines, list of lists
        """
        import math

        lines_extended = list()

        for line in the_lines:

            x1, y1 = line[0][0], line[0][1]
            x2, y2 = line[1][0], line[1][1]

            curr_lenght = math.sqrt((x1 - x2)**2 + (y2 - y1)**2)

            # y = 0
            x_top = int(round(x1 + (x1 - x2) / curr_lenght * y1))

            # y = image.shape[0]
            x_bottom = int(
                round(x2 + (x2 - x1) / curr_lenght * (image.shape[0] - y2))
                           )
            # Dots are intentionally appended *flat* to the list, not typical
            # syntax (x1,y1), (x2,y2) etc
            #lines_extended.append([(x_top, 0), (x_bottom, image.shape[0])])
            lines_extended.append([x_top, 0])
            lines_extended.append([x_bottom, image.shape[0]])

        return lines_extended


class PolygonRetriever:
    """
    Class tasked with retrieving image section between the lines (pole edges)
    detected in order to calculate pole's inclination angle.
    Since the lines are never full image height, it employs the LineExtender
    class to extend those lines.
    """
    def __init__(
            self,
            line_extender
    ):

        self.line_extender = line_extender
        self.results_processor = ResultsHandler(save_path=r"D:\Desktop\system_output\TILT_TESTING\extended_lines")

    def retrieve_polygon(self,
                         path_to_image,
                         the_lines):
        """

        :param path_to_image:
        :param the_lines:
        :return:
        """
        image = cv2.imread(path_to_image)
        image_name = os.path.split(path_to_image)[-1]

        # Since lines are usually of varying length and almost always are
        # shorter than image's height, extend them first
        extended_lines = list()
        if len(the_lines) == 2:
            extended_lines += self.line_extender.extend_lines(image=image,
                                                              the_lines=the_lines)
        else:
            # Address an issue when only one line is found
            pass

        # Once line's been extended, use them to extract the image section
        # restricted, defined by them

        support_point = extended_lines[2]
        extended_lines.append(support_point)

        points = np.array(extended_lines)

        mask = np.zeros((image.shape[0], image.shape[1]))

        cv2.fillConvexPoly(img=mask,
                           points=points,
                           color=1)

        mask = mask.astype(np.bool)

        output = np.zeros_like(image)

        output[mask] = image[mask]

        cv2.imwrite(
            os.path.join(r'D:\Desktop\system_output\TILT_TESTING\extracted', image_name),
                         output
                    )

        return
