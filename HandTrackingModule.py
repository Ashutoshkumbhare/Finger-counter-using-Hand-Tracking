import cv2
import mediapipe as mp
import time


class handdetection():
    def __init__(self, mode=False, maxHand=2, detectionconf=0.5, trackconf=0.5):
        self.mode = mode
        self.maxHand = maxHand
        self.detectionconf = detectionconf
        self.trackconf = trackconf

        self.mphands = mp.solutions.hands  # creating object
        self.hands = self.mphands.Hands(self.mode, self.maxHand, self.detectionconf, self.trackconf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)  # will process the RGB image through media-pipeline and return.
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_lms, self.mphands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, num_of_hand=0, draw=True):

        li = []
        if self.results.multi_hand_landmarks:
            myhands = self.results.multi_hand_landmarks[num_of_hand]
            for i, xyz in enumerate(myhands.landmark):
                # print(i, xyz)

                # getting actual size of img
                h, w, c = img.shape  # hight, width, center of the image
                cx, cy = int(xyz.x * w), int(xyz.y * h)
                # print(i, cx, cy)
                li.append([i, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        return li


def main():
    pTime = 0
    detector = handdetection()
    cap = cv2.VideoCapture(1)
    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        landmark_list = detector.findPosition(img)

        if len(landmark_list) != 0:
            print(landmark_list[12])

        # FPS calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)

        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
