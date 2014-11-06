"""Project 2: Stereo vision.

In this project, you'll extract dense 3D information from stereo image pairs.
"""

import cv2
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
    NUM_CHECKS = 50

    # Find matches between keypoints using FLANN
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=NUM_TREES)
    search_params = dict(checks=NUM_CHECKS)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_left, des_right, k=2)

    # Remove matches that are outside this threshold
    THRESHOLD = 0.75
    good = [m for m, n in matches if m.distance < THRESHOLD * n.distance]

    # Make sure that there are at least 3 matching keypoints
    # Return None if there are not enough matches between the two images
    if len(good) < 3:
        return None

    left_pts = np.float32([kp_left[m.queryIdx].pt for m in good])
    right_pts = np.float32([kp_right[m.trainIdx].pt for m in good])

    F, status = cv2.findFundamentalMat(left_pts, right_pts)

    # Resize the points into a matrix that stereo rectify will accept
    status = status.ravel()
    mp1 = left_pts[status].reshape(1, -1, 2)
    mp2 = right_pts[status].reshape(1, -1, 2)

    # Calculate rectifying homographies for left and right images
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

    # Convert both images to 8-bit images
    image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

    window_size = 1
    stereo = cv2.StereoSGBM(
            minDisparity=15,
            numDisparities=112,
            SADWindowSize=window_size,
            uniquenessRatio=7,
            speckleWindowSize=50,
            speckleRange=1,
            disp12MaxDiff=100,
            fullDP=True,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2
    )

    # Blur for better disparity map
    image_left = cv2.GaussianBlur(image_left, (3, 3), 5)
    image_right = cv2.GaussianBlur(image_right, (3, 3), 5)

    # Calculate disparities and change resulting type to 8-bit
    disparity = stereo.compute(image_left, image_right)
    disparity /= 16
    disparity = disparity.astype(np.uint8)

    return disparity


def make_ply(verts, colors):
    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

    # Reshape matrix
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])

    # Add size of verts to the ply header
    ply = ply_header % dict(vert_num=len(verts))

    # Loop over image and print out elements in .ply format
    for vector in verts:
        v_list = vector.tolist()

        ply += " ".join(map(str, v_list[:3]))
        ply += " "
        ply += " ".join(map(str, map(int, v_list[3:])))
        ply += "\n"

    return ply


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

    # Setup projection matrix
    projection_matrix = np.matrix([[1, 0, 0, image_left.shape[1] * 0.5],
                                   [0, 1, 0, image_left.shape[0] * 0.5],
                                   [0, 0, focal_length, 0],
                                   [0, 0, 0, 1]])

    # Reproject image to 3D
    points = cv2.reprojectImageTo3D(disparity_image, projection_matrix)
    colors = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)

    # Make a .ply string
    return make_ply(points, colors)