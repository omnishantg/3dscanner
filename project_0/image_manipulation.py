"""Project 0: Image Manipulation with OpenCV.

In this assignment, you will implement a few basic image
manipulation tasks using the OpenCV library.

Use the unit tests is image_manipulation_test.py to guide
your implementation, adding functions as needed until all
unit tests pass.
"""

# TODO: Implement!

import cv2
import numpy


def flip_image(image, horizontal, vertical):
    if horizontal == 1 and vertical == 1:
        flip = cv2.flip(image, -1)
    elif horizontal == 0 and vertical == 0:
        flip = image
    else:
        flip = cv2.flip(image, horizontal, vertical)
    return flip

def negate_image(image):
    negate = 255 - image
    return negate

def swap_blue_and_green(image):
    b, g, r = cv2.split(image)
    x = b
    b = g
    g = x
    img = cv2.merge((b, g, r))
    return img

if __name__ == '__main__':
    image = cv2.imread("test_data/nyc.jpg")
    image2 = flip_image(image, 0, 0)
    cv2.imshow("cat", image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image3 = negate_image(image)
    cv2.imshow("cat", image3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image4 = swap_blue_and_green(image)
    cv2.imshow("cat", image4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
