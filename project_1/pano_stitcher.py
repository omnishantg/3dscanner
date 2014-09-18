"""Project 1: Panorama stitching.

In this project, you'll stitch together images to form a panorama.

A shell of starter functions that already have tests is listed below.

TODO: Implement!
"""

import cv2
import numpy as np


def homography(image_a, image_b, bff_match=False):
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
    raw_matches = matcher.knnMatch(des_a, trainDescriptors = des_b, k=2)

    #filter matches with ratio test
    ratio = .75
    mkp_a, mkp_b = [], []
    for m in raw_matches:
        if len(m) == 2 and (m[0].distance < (m[1].distance * ratio)):
            m = m[0]
            mkp_a.append( kp_a[m.queryIdx] )
            mkp_b.append( kp_b[m.trainIdx] )

    # #number of matches
    # kp_pairs = zip(mkp_a, mkp_b)
    # print len(kp_pairs)
    
    # convert good matches into points
    img_a = np.float32([kp.pt for kp in mkp_a])
    img_b = np.float32([kp.pt for kp in mkp_b])

    #find homography matrix
    M, status = cv2.findHomography(img_b, img_a, cv2.RANSAC,5.0)
    # print '%d / %d  inliers/matched' % (np.sum(status), len(status))
    return M

    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks = 50)

    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1,des2,k=2)

    # good = []
    # for m,n in matches:
    #     if m.distance < 0.7*n.distance:
    #         good.append(m)

    # print len(good)        

    # if len(good)>10:
    #     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    #     matchesMask = mask.ravel().tolist()
    #     print '%d / %d  inliers/matched' % (np.sum(mask), len(mask))

    # return M

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
    pass


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
    pass

if __name__ == '__main__':
  img1 = cv2.imread("test_data/books_1.png")
  img2 = cv2.imread("test_data/books_2.png")
  M = homography(img1, img2)
  
