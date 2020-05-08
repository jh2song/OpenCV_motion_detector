import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,1080)
cap.set(4,720)

thresh = 25
max_diff=100
if cap.isOpened():
    while True:
        ret1, frame1 = cap.read()
        draw = frame1.copy()
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if ret1:
            ret2, frame2 = cap.read()
            draw = frame2.copy()
            frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            diff1 = cv2.absdiff(frame1_gray, frame2_gray)   # 같은 점이면 0, 다른 점이면 1~255
            if ret2:
                ret3, frame3 = cap.read()
                draw = frame3.copy()
                frame3_gray = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
                diff2 = cv2.absdiff(frame2_gray, frame3_gray)   # 같은 점이면 0, 다른 점이면 1~255


                ret4, diff1_th = cv2.threshold(diff1, thresh, 255, cv2.THRESH_BINARY)
                ret5, diff2_th = cv2.threshold(diff2, thresh, 255, cv2.THRESH_BINARY)
                diff_bin = cv2.bitwise_and(diff1_th, diff2_th)
                
                # 여기서부턴 motion detect탐지 
                whiteDotCnt = np.count_nonzero(diff_bin)
                if(whiteDotCnt>max_diff):
                    nzero = np.nonzero(diff_bin)
                    cv2.rectangle(draw, (min(nzero[1]), min(nzero[0])),
                    (max(nzero[1]), max(nzero[0])), (0,255,0), 3)
                    cv2.putText(draw, "Motion Detected!!", (min(nzero[1]), min(nzero[0]))
                    ,cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)    
                
        
        cv2.imshow('test',draw)        
        if(cv2.waitKey(1)==27):
            break
else:
    print("can't open camera")
