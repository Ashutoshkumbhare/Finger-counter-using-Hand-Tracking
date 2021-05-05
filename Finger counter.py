import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
fingTip = [4, 8, 12, 16, 20]
detection = htm.handdetection(detectionconf=0.75)

while True:
    success, img = cap.read()
    img = detection.findHands(img)
    lmlist = detection.findPosition(img)
    if len(lmlist) != 0:
        finger_status = []
        if lmlist[fingTip[0]][1] > lmlist[fingTip[0] - 1][1]:
            finger_status.append(1)
        else:
            finger_status.append(0)
        for id in range(1, 5):
            if lmlist[fingTip[id]][2] < lmlist[fingTip[id] - 2][2]:
                finger_status.append(1)
            else:
                finger_status.append(0)

        # print(finger_status)
        total_fingers = finger_status.count(1)  # count how many 1 are in the list
        # print(total_fingers)

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(total_fingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime


    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)
    cv2.imshow("Images", img)
    cv2.waitKey(1)
