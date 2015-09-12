import numpy as np
import cv2
import sys

imfile = sys.argv[1]

img = cv2.imread(imfile)
img = cv2.resize(img, None, fx=0.5, fy=0.5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB()

keys, des = orb.detectAndCompute(gray, None)

for key in keys:
    cv2.circle(img, (int(key.pt[0]), int(key.pt[1])), 5, (0,0,1))

cv2.imshow("image", img)
cv2.waitKey(0)
