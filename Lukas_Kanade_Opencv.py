import numpy as np
import cv2

from LucasKanade import LucasKanadeScale

cap = cv2.VideoCapture("slow_traffic_small.mp4")

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

rect = np.array([220, 120, 280, 170], dtype=float)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #### This part is modified
    #### Original code
    # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #### New code
    p = LucasKanadeScale(old_gray, frame_gray, rect)
    rect[0] = (1 + p[2]) * rect[0] + p[0] 
    rect[2] = (1 + p[2]) * rect[2] + p[0] 
    rect[1] = (1 + p[3]) * rect[1] + p[1] 
    rect[3] = (1 + p[3]) * rect[3] + p[1] 
    ####

    rect2 = rect.astype(int)
    frame = cv2.rectangle(frame, (rect2[0],rect2[1]), (rect2[2],rect2[3]), (0,255,0), 2)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    old_gray = frame_gray.copy()

cv2.destroyAllWindows()
cap.release()