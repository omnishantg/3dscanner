"""Project 2: Stereo vision.

In this project, you'll extract dense 3D information from stereo image pairs.
"""

import cv2
import math
import numpy as np


def rectify_pair(image_left, image_right, viz=False):
    """Computes the pair's fundamental matrix and rectifying homographies.

    Arguments:
      image_left, image_right: 3-channel images making up a stereo pair.

    Returns:
      F: the fundamental matrix relating epipolar geometry between the pair.
      H_left, H_right: homographies that warp the left and right image so
        their epipolar lines are corresponding rows.
    """

    # Detect keypoints in both images using SIFT
    sift = cv2.SIFT()
    kp_left, des_left = sift.detectAndCompute(image_left, None)
    kp_right, des_right = sift.detectAndCompute(image_right, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    NUM_TREES = 5
    NUM_CHECKS = 100

    # Find matches between keypoints using FLANN
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=NUM_TREES)
    search_params = dict(checks=NUM_CHECKS)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_left, des_right, k=2)

    THRESHOLD = 0.75
    good = [m for m, n in matches if m.distance < THRESHOLD * n.distance]

    # Return None if there are not enough matches between the two images
    if len(good) < 3:
        return None

    left_pts = np.float32([kp_left[m.queryIdx].pt for m in good])
    right_pts = np.float32([kp_right[m.trainIdx].pt for m in good])

    F, status = cv2.findFundamentalMat(left_pts, right_pts)

    status = status.ravel()
    mp1 = left_pts[status].reshape(1, -1, 2)
    mp2 = right_pts[status].reshape(1, -1, 2)

    _, H_left, H_right = cv2.stereoRectifyUncalibrated(mp1,
                                                       mp2,
                                                       F,
                                                       image_left.shape[:2])

    return (F, H_left, H_right)


def disparity_map(image_left, image_right):
    """Compute the disparity images for image_left and image_right.

    Arguments:
      image_left, image_right: rectified stereo image pair.

    Returns:
      an single-channel image containing disparities in pixels,
        with respect to image_left's input pixels.
    """
    pass


def point_cloud(disparity_image, image_left, focal_length):
    """Create a point cloud from a disparity image and a focal length.

    Arguments:
      disparity_image: disparities in pixels.
      image_left: BGR-format left stereo image, to color the points.
      focal_length: the focal length of the stereo camera, in pixels.

    Returns:
      A string containing a PLY point cloud of the 3D locations of the
        pixels, with colors sampled from left_image. You may filter low-
        disparity pixels or noise pixels if you choose.
    """
    pass
