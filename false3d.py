import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def main():
    tracking = EyeFaceTracking()
    cap = cv2.VideoCapture(1)
    frameCount = 0

    while True:
        frame, gray = nextFrame(cap)
        frameCount += 1

        if tracking.tracking:
            tracking.track(gray)
        if not tracking.tracking:
            tracking.detect(gray)

        displayFrame(frame, tracking)
        if frameCount > 200:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracking.reportStats()
    cap.release()
    cv2.destroyAllWindows()

"""
Capture and parse a frame from the webcam.
"""
def nextFrame(cap):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray

"""
Display frame and everything it is supposed to display.
"""
def displayFrame(frame, tracking):
    displayFeatures(frame, tracking)
    cv2.imshow("Video", frame)

def displayFeatures(frame, tracking):
    for (ex,ey,ew,eh) in tracking.eyes:
        cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
    (fx,fy,fw,fh) = tracking.face
    cv2.rectangle(frame, (fx,fy), (fx+fw,fy+fh), (255,0,0), 2)

"""
Class performing all feature detection and tracking.
"""
class EyeFaceTracking:
    def __init__(self):
        classifDir = "classifiers/"
        facecas = classifDir + "haarcascade_frontalface_default.xml"
        eyecas = classifDir + "haarcascade_eye.xml"
        self.faceCascade = cv2.CascadeClassifier(facecas)
        self.eyeCascade = cv2.CascadeClassifier(eyecas)
        self.orb = cv2.ORB()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.tracking = False
        self.eyes = [(0,0,0,0), (0,0,0,0)]
        self.face = (0,0,0,0)
        self.smoothN = 2
        self.faceHistory = deque()
        self.eyeHistory = deque()
        self.setupStats()

    def setupStats(self):
        self.attemptedDetect = 0
        self.successfulDetect = 0
        self.attemptedRedetect = 0
        self.successfulRedetect = 0
        self.attemptedFaceDetect = 0
        self.successfulFaceDetect = 0
        self.attemptedFaceRedetect = 0
        self.successfulFaceRedetect = 0

    """
    Detect eyes using haar cascade classifier.
    First detect face and within it search for eyes.
    """
    def detect(self, gray):
        self.attemptedDetect += 1
        self.attemptedFaceDetect += 1
        ret, faces = self.searchForFaces(gray)
        if not ret:
            self.tracking = False
            return
        self.successfulFaceDetect += 1
        (fx,fy,fw,fh) = faces[0]
        self.face = faces[0]
        self.updateFace(faces[0])
        ret, eyes = self.searchForEyes(gray[fy:fy+fh, fx:fx+fw])
        if not ret:
            self.tracking = False
            return
        eyes = map(lambda (ex,ey,ew,eh):(ex+fx,ey+fy,ew,eh), eyes)
        self.eyes = eyes
        self.tracking = True
        self.successfulDetect += 1

    """
    Detect eyes in a small region around the previous position using
    haar cascades.
    """
    def track(self, gray):
        newEyes = []
        (fx, fy, fw, fh) = self.face
        #search for the face in an area around the previous face
        xMargin = fw // 3
        yMargin = fh // 4
        roiFace = gray[fy-yMargin:fy+fh+yMargin, fx-xMargin:fx+fw+xMargin]
        ret, faces = self.searchForFaces(roiFace)
        self.attemptedFaceRedetect += 1
        if not ret:
            self.tracking = False
            return
        self.successfulFaceRedetect += 1
        (newfx,newfy,fw,fh) = faces[0]
        fx, fy = newfx+fx-xMargin, newfy+fy-yMargin
        self.face = (fx, fy, fw, fh)
        ret, eyes = self.searchForEyes(gray[fy:fy+fh, fx:fx+fw])
        self.attemptedRedetect += 1
        if not ret:
            self.tracking = False
            return
        self.successfulRedetect += 1
        eyes = map(lambda (ex,ey,ew,eh):(ex+fx,ey+fy,ew,eh), eyes)
        self.eyes = eyes

    def searchForFaces(self, roi, scaleF=1.3, minNeighbors=5):
        faces = self.faceCascade.detectMultiScale(roi, scaleF, minNeighbors)
        return len(faces) == 1, faces

    def searchForEyes(self, roi, scaleF=1.3, minNeighbors=5):
        eyes = self.eyeCascade.detectMultiScale(roi, scaleF, minNeighbors)
        return len(eyes) == 2, eyes

    def updateFace(self, face):
        self.faceHistory.append(face)
        #keep most recent face for fast access
        self.face = face
        if len(self.faceHistory) > self.smoothN:
            self.faceHistory.popleft()

    def reportStats(self):
        eyeDetect = float(self.successfulDetect)/self.attemptedDetect
        eyeRedetect = float(self.successfulRedetect)/(self.attemptedRedetect + 0.01)
        faceDetect = float(self.successfulFaceDetect)/self.attemptedFaceDetect
        faceRedetect = float(self.successfulFaceRedetect)/(self.attemptedFaceRedetect + 0.01)
        print "face detections: " + str(faceDetect)
        print "face redetections: " + str(faceRedetect)
        print "eye detections: " + str(eyeDetect)
        print "eye redetections: " + str(eyeRedetect)

    def getSmoothedFace(self):
        if len(self.faceHistory) == 0:
            return self.face
        return self.average(list(self.faceHistory))

    def getSmoothedEyes(self):
        return

    """Takes a list of tuples and returns their average"""
    def average(self, tuples):
        array = np.array(tuples, dtype=np.float64)
        mean = np.mean(array, 0).astype(np.int32)
        return tuple(mean)
            


if __name__ == "__main__":
    main()

