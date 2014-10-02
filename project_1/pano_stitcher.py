"""Project 1: Panorama stitching.

In this project, you'll stitch together images to form a panorama.

A shell of starter functions that already have tests is listed below.
"""

import cv2
import numpy as np


def homography(image_a, image_b):
    """Returns the homography mapping image_b into alignment with image_a.

    Arguments:
      image_a: A grayscale input image.
      image_b: A second input image that overlaps with image_a.

    Returns: the 3x3 perspective transformation matrix (aka homography)
             mapping points in image_b to corresponding points in image_a.
    """

    # Detect keypoints in both images using SIFT
    sift = cv2.SIFT()
    kp_a, des_a = sift.detectAndCompute(image_a, None)
    kp_b, des_b = sift.detectAndCompute(image_b, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    NUM_TREES = 5
    NUM_CHECKS = 100

    # Find matches between keypoints using FLANN
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=NUM_TREES)
    search_params = dict(checks=NUM_CHECKS)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_a, des_b, k=2)

    RATIO = 0.75
    good = [m for m, n in matches if m.distance < RATIO * n.distance]

    # Return None if there are not enough matches betwen the two images
    if len(good) < 3:
        return None

    dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good])
    src_pts = np.float32([kp_b[m.trainIdx].pt for m in good])

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M


def warp_image(image, homography):
    """Warps 'image' by 'homography'

    Arguments:
      image: a 3-channel image to be warped.
      homography: a 3x3 perspective projection matrix mapping points
                  in the frame of 'image' to a target frame.

    Returns:
      - a new 4-channel image containing the warped input, resized to contain
        the new image's bounds. Translation is offset so the image fits exactly
        within the bounds of the image. The fourth channel is an alpha channel
        which is zero anywhere that the warped input image does not map in the
        output, i.e. empty pixels.
      - an (x, y) tuple containing location of the warped image's upper-left
        corner in the target space of 'homography', which accounts for any
        offset translation component of the homography.
    """

    top_left = np.array([0, 0, 1])
    bottom_left = np.array([image.shape[0], 0, 1])
    top_right = np.array([0, image.shape[1], 1])
    bottom_right = np.array([image.shape[0], image.shape[1], 1])

    # Find the new origin from the scale parameters in the matrix
    origin = (int(homography[0][2]), int(homography[1][2]))

    # Remove the scale parameters from the matrix transform
    homography[0][2] = 0
    homography[1][2] = 0

    # Perform the homography on each of the corners to get the new image size
    top_left_warped = np.dot(homography, top_left)
    bottom_left_warped = np.dot(homography, bottom_left)
    top_right_warped = np.dot(homography, top_right)
    bottom_right_warped = np.dot(homography, bottom_right)

    top_left = top_left_warped / top_left_warped[2]
    bottom_left = bottom_left_warped / bottom_left_warped[2]
    top_right = top_right_warped / top_right_warped[2]
    bottom_right = bottom_right_warped / bottom_right_warped[2]

    # The new image size is the maximum of the x and y of our new bounds
    new_width = max(top_left[1], bottom_left[1], top_right[1], bottom_right[1])
    new_height = max(top_left[0], bottom_left[0], top_right[0],
                     bottom_right[0])

    new_size = (int(new_width), int(new_height))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    warped = cv2.warpPerspective(image, homography, new_size)

    return (warped, origin)


def create_mosaic(images, origins):
    """Combine multiple images into a mosaic.

    Arguments:
      images: a list of 4-channel images to combine in the mosaic.
      origins: a list of the locations upper-left corner of each image in
               a common frame, e.g. the frame of a central image.

    Returns: a new 4-channel mosaic combining all of the input images. pixels
             in the mosaic not covered by any input image should have their
             alpha channel set to zero.
    """

    # Find the top left most point on the panorama and set it equal to 0
    origin_x, origin_y = zip(*origins)
    min_x = min(origin_x)
    min_y = min(origin_y)

    # Set other points relative to the new top left point
    new_origins = [(x - min_x, y - min_y) for x, y in origins]

    pano_width = 0
    pano_height = 0

    # Calculate the output panorama's dimensions
    for image, (x, y) in zip(images, new_origins):
        pano_width = max(pano_width, image.shape[1] + x)
        pano_height = max(pano_height, image.shape[0] + y)

    # Copy the image onto the panorama
    panorama = np.zeros((pano_height, pano_width, 4), np.uint8)

    for image, (origin_x, origin_y) in zip(images, new_origins):
        end_x = origin_x + image.shape[0]
        end_y = origin_y + image.shape[1]
        panorama[origin_x:end_x, origin_y:end_y] = image

    return panorama
