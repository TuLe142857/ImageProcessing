import cv2
import matplotlib.pyplot as plt
import numpy as np

def put_text(frame, text):
    new_frame = frame.copy()
    cv2.putText(new_frame,
                text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                255, 2,
                cv2.LINE_AA)
    return new_frame

def subtract_frame(f1, f2):
    return np.abs(f1.astype(np.int32) - f2.astype(np.int32)).astype(np.uint8)

def handle_thresholded(frame, t):
    new_frame = frame.copy()
    h, w = frame.shape
    for i in range(h):
        for j in range(w):
            new_frame[i][j] = 255 if (frame[i][j] >= t) else (0)
    return new_frame

# open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Can not open camera!")
    exit(1)

alpha = 0.5
thresholded = 10
background = None
cv2.namedWindow("Background Subtraction", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # convert to gray
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # update background
    background = (alpha * background + ((1-alpha) * frame)) if background is not None else (frame)
    background = background.astype(np.uint8)

    # subtract
    diff_frame = subtract_frame(background, frame)
    diff_frame = cv2.medianBlur(diff_frame, 5) # remove noise = median blur
    thresholded_diff = handle_thresholded(diff_frame, thresholded)

    # combine frame
    row_1 = np.hstack([put_text(frame, "current frame"), put_text(background, "background")])
    row_2 = np.hstack([put_text(diff_frame, "Difference frame"), put_text(thresholded_diff, "Thresholded Difference Frame")])
    combined = np.vstack([row_1, row_2])

    # render
    cv2.imshow("Background Subtraction", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()