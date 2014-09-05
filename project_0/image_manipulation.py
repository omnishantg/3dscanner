"""Project 0: Image Manipulation with OpenCV.

In this assignment, you will implement a few basic image
manipulation tasks using the OpenCV library.

Use the unit tests is image_manipulation_test.py to guide
your implementation, adding functions as needed until all
unit tests pass.
"""
import cv2
import numpy
import unittest
# TODO: Implement!


def flip_image(img, x, y):
    rows = img.shape[0]
    cols = img.shape[1]
    copy = numpy.copy(img)
    if (x):
        if (y):
            for w in range(0, rows):
                for z in range(0, cols):
                    copy[w, z] = img[rows-w-1, cols-z-1]
            return copy
        else:
            for w in range(0, rows):
                for z in range(0, cols):
                    copy[w, z] = img[w, cols-z-1]
            return copy
    else:
        if (y):
            for w in range(0, rows):
                for z in range(0, cols):
                    copy[w, z] = img[rows-w-1, z]
            return copy
    return img


def negate_image(img):
    rows = img.shape[0]
    cols = img.shape[1]
    copy = numpy.copy(img)
    for w in range(0, rows):
        for z in range(0, cols):
            copy[w, z] = [255, 255, 255] - img[w, z]
    return copy


def swap_blue_and_green(img):
    rows = img.shape[0]
    cols = img.shape[1]
    copy = numpy.copy(img)
    for w in range(0, rows):
        for z in range(0, cols):
            copy[w, z] = [img[w, z][1], img[w, z][0], img[w, z][2]]
    return copy
