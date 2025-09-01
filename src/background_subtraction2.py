import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Can not open camera!")
    exit(1)

background = None
fg_box = None
alpha = 0.5
thresholded = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background = (alpha*background + (1 -  alpha)*gray_frame).astype(np.uint8) if background is not None else gray_frame


    diff = np.abs(gray_frame.astype(np.int32) - background.astype(np.int32)).astype(np.uint8)
    diff = cv2.medianBlur(diff, 5)
    h, w = diff.shape
    for i in range(h):
        for j in range(w):
            diff[i][j] = 255 if (diff[i][j] > thresholded) else 0

    # calc bounding box
    ys, xs = np.where(diff == 255)
    if len(xs) != 0 and len(ys) != 0:
        fg_box = ((xs.min(), ys.min()), (xs.max(), ys.max()))

    if fg_box is not None:
        cv2.rectangle(frame, fg_box[0], fg_box[1], (255, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow("Background Subtraction", np.hstack([frame, cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)]))


    # press esc/q to break
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    