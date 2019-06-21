import numpy as np
import cv2

cap = cv2.VideoCapture()

cap.open("rtsp://admin:Secur1ty@10.100.1.37/doc/page/preview.asp")
count = 0
while(True):
     # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imwrite("frame%d.jpg" % count, frame)
    count += 1
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('Frame',gray)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

