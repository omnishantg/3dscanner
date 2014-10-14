#!/usr/bin/env python
import cv2
import sys
import os

sys.path.append(os.path.abspath(".."))

import stereo


if len(sys.argv) < 3:
    print "Usage: {} image_a image_b".format(sys.argv[0])
    exit(0)

image_a = cv2.imread(sys.argv[1])
image_b = cv2.imread(sys.argv[2])

disparity = stereo.disparity_map(image_a, image_b)

focal_length = 10
ply = stereo.point_cloud(disparity, image_a, focal_length)

out_file = "stereo.ply"
with open(out_file, 'w') as f:
    f.write(ply)
