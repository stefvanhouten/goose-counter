import cv2
import copy
from tracker import Tracker
import imutils
import numpy as np


def main():
  cap = cv2.VideoCapture('geese.mp4')
  tracker = Tracker(160, 30, 5, 1)

  pause = False
  firstFrame = None
  counted = []
  while cap.isOpened():
    ret, frame = cap.read()
    if ret:
      frame = imutils.resize(frame, width=800)
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      gray = cv2.GaussianBlur(gray, (3, 3), 0)

      if firstFrame is None:
        firstFrame = gray
        continue

      frameDelta = cv2.absdiff(firstFrame, gray)
      thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

      thresh = cv2.dilate(thresh, None, iterations=2)
      cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)

      centers = []
      for i, c in enumerate(cnts):
        if cv2.contourArea(c) > 500:
          continue
        (x, y, w, h) = cv2.boundingRect(c)

        centers.append(np.array([[x + w / 2], [y + h / 2]]))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

      if (len(centers) > 0):
        tracker.Update(centers)
        for track in tracker.tracks:
          x = track.prediction[0]
          y = track.prediction[1]
          id = track.track_id
          if x > 400 and id not in counted:
            counted.append(id)
          cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                      (0, 255, 0), 1)

          cv2.putText(frame, str(len(counted)), (20, 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        cv2.imshow('Tracking', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
