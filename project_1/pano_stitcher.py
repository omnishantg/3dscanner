"""Project 1: Panorama stitching.

In this project, you'll stitch together images to form a panorama.

A shell of starter functions that already have tests is listed below.

TODO: Implement!
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
    #initiate SIFT detector
    sift = cv2.SIFT()

    #find the keypoints and descriptors with SIFT
    kp_a, des_a = sift.detectAndCompute(image_a, None)
    kp_b, des_b = sift.detectAndCompute(image_b, None)

    #create matches
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des_a, trainDescriptors=des_b, k=2)

    #filter matches with ratio test
    ratio = .75
    mkp_a, mkp_b = [], []
    for m in raw_matches:
        if len(m) == 2 and (m[0].distance < (m[1].distance * ratio)):
            m = m[0]
            mkp_a.append(kp_a[m.queryIdx])
            mkp_b.append(kp_b[m.trainIdx])

    # #number of matches
    # kp_pairs = zip(mkp_a, mkp_b)
    # print len(kp_pairs)

    # convert good matches into points
    img_a = np.float32([kp.pt for kp in mkp_a])
    img_b = np.float32([kp.pt for kp in mkp_b])

    #find homography matrix
    M, status = cv2.findHomography(img_b, img_a, cv2.RANSAC, 5.0)
    # print '%d / %d  inliers/matched' % (np.sum(status), len(status))
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
    topleft = (0, 0)
    #create alpha channel in image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    height, width, depth = img.shape
    a1 = [0, 0, width, width]
    a2 = [0, height, 0, height]
    a3 = [1, 1, 1, 1]
    box = np.matrix([a1, a2, a3])
    res = np.dot(homography, box)

    div1 = np.divide(res[0], res[2])
    div2 = np.divide(res[1], res[2])
    topleft = (int(np.amin(div1)), int(np.amin(div2)))

    minlength = np.amin(div1)
    minheight = np.amin(div2)

    l = np.amax(div1) - np.amin(div1)
    h = np.amax(div2) - np.amin(div2)

    homography[0][2] = homography[0][2] - minlength
    homography[1][2] = homography[1][2] - minheight
    BT = cv2.BORDER_TRANSPARENT
    warp = cv2.warpPerspective(img, homography,
                               (int(l), int(h)), borderMode=BT)
    (cb, cg, cr, ca) = cv2.split(warp)
    warp = cv2.merge([cb, cg, cr, ca])
    return warp, topleft


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
    #shows images in order of dominance
    x = []
    y = []
    #get minimum x and y
    for i in origins:
        x.append(i[0])
        y.append(i[1])
    x_min = np.amin(x)
    y_min = np.amin(y)
    x = []
    y = []
    #get largest x and y
    for i in range(len(origins)):
        x.append(origins[i][0] + images[i].shape[1])
        y.append(origins[i][1] + images[i].shape[0])
    #calculate width by subtracting top right x from top left x
    width = np.amax(x) - x_min
    #calculate height by subtracting bottom left y from top left y
    height = np.amax(y) - y_min
    result = np.zeros((height, width, 4))
    for i in range(len(origins)):
        rows = images[i].shape[0]
        cols = images[i].shape[1]
        img = images[i]
        x_o = origins[i][0]
        y_o = origins[i][1]
        for w in range(0, rows):
            for z in range(0, cols):
                if (img[w, z][3] != 0):
                    result[(w + y_o - y_min), z + x_o - x_min] = img[w, z]
    return result

if __name__ == '__main__':
  # Read initial images
    img1 = cv2.imread("my_panos/photo1.JPG")
    img2 = cv2.imread("my_panos/photo2.JPG")
    img3 = cv2.imread("my_panos/photo3.JPG")

  # Run homography to change img1 to match img2. Add alpha channel
    M = homography(img2, img1)
    img1, topleft = warp_image(img1, M)
    cv2.imwrite("my_panos/test.png", img1)
  # Run homography to change img3 to match img2. Add alpha channel
    M2 = homography(img2, img3)
    img3, topleft2 = warp_image(img3, M2)
    cv2.imwrite("my_panos/test2.png", img3)
  # Add alpha channel to img2
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)

  # Create the panorama mosaic.
    images = (img1, img2, img3)
    origins = (topleft, (0, 0), topleft2)
    pano = create_mosaic(images, origins)

    cv2.imwrite("my_panos/pano.png", pano)
