"""Project 0: Image Manipulation with OpenCV.

In this assignment, you will implement a few basic image
manipulation tasks using the OpenCV library.

Use the unit tests is image_manipulation_test.py to guide
your implementation, adding functions as needed until all
unit tests pass.
"""
import cv2


def flip_image(img, horizontal, vertical):
    flip_code = None

    if horizontal and vertical:
        flip_code = -1
    elif horizontal:
        flip_code = 1
    elif vertical:
        flip_code = 0

    if flip_code is not None:
        return cv2.flip(img, flip_code)
    else:
        return img


def negate_image(img):
    return cv2.bitwise_not(img)


def swap_blue_and_green(img):
    ch = cv2.split(img)
    swapped = cv2.merge([ch[1], ch[0], ch[2]])

    return swapped
