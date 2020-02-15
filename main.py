from tilt_detector import TiltDetector, LineMerger
from utils import ResultsHandler
from concrete_polygon_extractor import LineExtender, PolygonRetriever
import os
import argparse
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', type=str, help='Path to an image.')
    parser.add_argument('--folder', type=str, help="Path to a folder with images to process")
    parser.add_argument('--save_path', type=str, default=None, help="If None, don't save results, show them")

    parser.add_argument('--retrieve', type=int, default=0, help="Retrieve image section defined by the two pole"
                                                                "edge lines detected")
    arguments = parser.parse_args()

    return arguments


def main():

    arguments = parse_args()

    assert arguments.image or arguments.folder, "No input data provided"

    images_to_process = list()

    if arguments.image:

        if os.path.isfile(arguments.image):
            images_to_process.append(arguments.image)
        else:
            print("ERROR: Provided image is not an image")
            sys.exit()

    else:
        if not os.path.isdir(arguments.folder):
            print("ERROR: Provided folder is not a folder")
            sys.exit()

        for image_name in os.listdir(arguments.folder):
            if not any(image_name.endswith(ext) for ext in [".jpg", ".png", ".jpeg",
                                                            ".JPG", ".JPEG", ".PNG"]):
                continue

            images_to_process.append(os.path.join(arguments.folder, image_name))

    # If save path has been provided, all images will be processed and saved there
    # Otherwise, after each image gets processed it will be shown to a user until
    # he clicks a button to proceed to the next image if any.
    # In order to just calculate and receive angle = flag is 0
    if arguments.save_path is not None:
        if not os.path.exists(arguments.save_path):
            os.mkdir(arguments.save_path)

        results_handling = 1, arguments.save_path
        handler = ResultsHandler(save_path=arguments.save_path)
    else:
        results_handling = 0, ''
        handler = None

    merger = LineMerger()

    detector = TiltDetector(results_handling_way=results_handling,
                            line_merger=merger,
                            results_processor=handler)

    total_error = 0.0
    images_with_calculated_angles = 0
    images_without_angle_calculated = list()
    performance_tracker = list()

    for path_to_image in images_to_process:

        image_name = os.path.split(path_to_image)[-1]
        print('\n', image_name)

        # Find lines, calculate the angle
        start_time = time.time()
        predicted_tilt_angle, the_lines, img_res = detector.process_image(path_to_image)
        inference_time = time.time() - start_time

        # Keep track of the error
        if predicted_tilt_angle:

            performance_tracker.append(
                (image_name, img_res, inference_time)
                                       )

            try:
                # Extract angle from the image name (naming convention)
                truth_angle = float(image_name.split("_")[-1][:-4])
            except:
                print(f"\nERROR: Naming convention error: {image_name}")
                images_with_calculated_angles += 1
                continue

            difference = abs(truth_angle - predicted_tilt_angle)
            error = round(difference / truth_angle, 3)
            print(f"Predicted: {predicted_tilt_angle}, Truth: {truth_angle}, Error: {error}")

            total_error += error
            images_with_calculated_angles += 1

        else:
            images_without_angle_calculated.append(image_name)

        if not the_lines:
            print("Failed to detect any lines for:", image_name)
            continue

        assert 1 <= len(the_lines) <= 2, "Wrong number of lines!"

        # Retrieve area defined by the lines for future cracks detection
        if arguments.retrieve:
            line_extender = LineExtender()
            polygon_retriever = PolygonRetriever(line_extender=line_extender)

            concrete_polygon = polygon_retriever.retrieve_polygon(path_to_image,
                                                                  the_lines)


            # DELETE ME I AM FOR TESTING
            handler.save_image_2(image_name,
                                 concrete_polygon,
                                 arguments.save_path)
            #handler.show_image(concrete_polygon)

    if images_with_calculated_angles > 0:
        mean_error = round(total_error / images_with_calculated_angles, 3)
        print("\nMEAN ERROR:", mean_error * 100, "%")

        total_time = sum(t[-1] for t in performance_tracker)
        print("TOTAL TIME TAKEN:", total_time)
        print("AVERAGE TIME PER IMAGE:", round((total_time / images_with_calculated_angles), 3))

        print()
        for name, res, t in performance_tracker:
            print(name, "Res:", res, "Time:", round(t, 2), " seconds")

    else:
        print("\nCannot calculate MEAN ERROR. Failed to calculate the angle for"
              " any images")

    if images_without_angle_calculated:
        print("\nFAILED TO CALCULATE ANGLE FOR:",
              ' '.join(images_without_angle_calculated))


if __name__ == "__main__":
    main()
