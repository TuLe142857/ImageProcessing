import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("petter.png", cv2.IMREAD_GRAYSCALE)

t = 0
rerender = True
while True:
    if rerender:
        print(t)
        mask = (img < t).astype(np.uint8)
        thresholded = mask * img
        frame = np.hstack([img, thresholded])
        cv2.imshow("hi", frame)
        rerender = False

    k = cv2.waitKey(1)
    if k == 27:
        break
    if k  == ord('w'):
        t += 1
        rerender = True
    if k  == ord('s'):
        t -= 1
        rerender = True
