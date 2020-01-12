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

            current_lenght = math.sqrt((x1 - x2)**2 + (y2 - y1)**2)

            print()
            new_lenght_multiplier = int((100 * current_lenght) / (image.shape[0] - x1))
            print(line)
            print(new_lenght_multiplier)
            continue


            new_lenght = image.shape[0]

            x3_bottom = int(x2 + (x2 - x1) / current_lenght * new_lenght)
            y3_bottom = int(y2 + (y2 - y1) / current_lenght * new_lenght)

            x3_top = int()
            y3_top = 0

            lines_extended.append((x3_bottom, y3_bottom))

        sys.exit()

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

        image = cv2.imread(path_to_image)
        image_name = os.path.split(path_to_image)[-1]

        extended_lines = self.line_extender.extend_lines(image=image,
                                                         the_lines=the_lines)

        # print()
        # print(image.shape)
        # print("LINES BEFORE BEING EXTENDED:", the_lines)
        # print("AFTER EXTENSION:", extended_lines)

        new_lines = [
            [the_lines[0][0], extended_lines[0]],
            [the_lines[1][0], extended_lines[1]]
        ]

        self.results_processor.save_image(lines=new_lines,
                                          image=image,
                                          image_name=image_name,
                                          angle=0)

        return
