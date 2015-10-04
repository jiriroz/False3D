import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def main():
    tracking = EyeFaceTracking()
    cap = cv2.VideoCapture(0)
    frameCount = 0

    while True:
        frameBGR, gray = nextFrame(cap)
        frameCount += 1

        if tracking.tracking:
            #tracking.track(gray)
            tracking.trackCamshift(frameBGR)
        if not tracking.tracking:
            tracking.detect(gray)
        tracking.storeData(frameBGR)

        displayFrame(frameBGR, tracking)
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
    #for (ex,ey,ew,eh) in tracking.eyePositions:
    #    cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
    (fx,fy,fw,fh) = tracking.facePosition
    #(fx, fy, fw, fh) = tracking.getSmoothedFace()
    cv2.rectangle(frame, (fx,fy), (fx+fw,fy+fh), (255,0,0), 2)

"""
Class performing all feature detection and tracking.
"""
class EyeFaceTracking:
    def __init__(self):
        classifDir = "classifiers/"
        facecas = classifDir + "haarcascade_frontalface_default.xml"
        eyecas = classifDir + "haarcascade_eye.xml"
        self.facePositionCascade = cv2.CascadeClassifier(facecas)
        self.eyeCascade = cv2.CascadeClassifier(eyecas)
        self.orb = cv2.ORB()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.tracking = False
        self.eyePositions = [(0,0,0,0), (0,0,0,0)]
        self.facePosition = (0,0,0,0)
        self.smoothN = 2
        #termination criteria of meanshift
        self.termCrit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.facePositionHistory = deque()
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
    Store the detected features so that we can access them
    in the next iteration.
    """
    def storeData(self, frame):
        if self.tracking:
            (fx,fy,fw,fh) = self.facePosition
            self.face = frame[fy:fy+fh, fx:fx+fw]

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
        self.facePosition = faces[0]
        self.updateFace(faces[0])
        ret, eyes = self.searchForEyes(gray[fy:fy+fh, fx:fx+fw])
        if not ret:
            self.tracking = False
            return
        eyes = map(lambda (ex,ey,ew,eh):(ex+fx,ey+fy,ew,eh), eyes)
        self.eyePositions = eyes
        self.tracking = True
        self.successfulDetect += 1

    """
    Detect eyes and face in a small region around the previous
    position using haar cascades.
    """
    def track(self, gray):
        (fx, fy, fw, fh) = self.facePosition
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
        self.facePosition = (fx, fy, fw, fh)
        ret, eyes = self.searchForEyes(gray[fy:fy+fh, fx:fx+fw])
        self.attemptedRedetect += 1
        if not ret:
            self.tracking = False
            return
        self.successfulRedetect += 1
        eyes = map(lambda (ex,ey,ew,eh):(ex+fx,ey+fy,ew,eh), eyes)
        self.eyePositions = eyes

    """
    Search for the face using histogram backprojection and camshift.
    """
    def trackCamshift(self, frame):
        faceHsv = cv2.cvtColor(self.face, cv2.COLOR_BGR2HSV)
        #could discard low values using cv2.inRange() function
        dims = [0] #what components of hsv do we want to use?
        histSizes = [180] #corresponding to hsv
        ranges = [0, 180]
        faceHist = cv2.calcHist([faceHsv], dims, None, histSizes, ranges)
        cv2.normalize(faceHist, faceHist, 0, 255, cv2.NORM_MINMAX)

        frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backProj = cv2.calcBackProject([frameHsv], dims, faceHist, ranges, 1)
        facePos = tuple(self.facePosition)
        ret, newFacePos = cv2.meanShift(backProj, facePos, self.termCrit)

        self.attemptedFaceRedetect += 1
        if not ret:
            self.tracking = False
            return
        self.successfulFaceRedetect += 1

        self.facePosition = newFacePos
        self.updateFace(newFacePos)



    def searchForFaces(self, roi, scaleF=1.3, minNeighbors=5):
        faces = self.facePositionCascade.detectMultiScale(roi, scaleF, minNeighbors)
        return len(faces) == 1, faces

    def searchForEyes(self, roi, scaleF=1.3, minNeighbors=5):
        eyes = self.eyeCascade.detectMultiScale(roi, scaleF, minNeighbors)
        return len(eyes) == 2, eyes

    def updateFace(self, face):
        self.facePositionHistory.append(face)
        #keep most recent face for fast access
        self.facePosition = face
        if len(self.facePositionHistory) > self.smoothN:
            self.facePositionHistory.popleft()

    def reportStats(self):
        eyeDetect = float(self.successfulDetect)/self.attemptedDetect
        eyeRedetect = float(self.successfulRedetect)/(self.attemptedRedetect + 0.01)
        faceDetect = float(self.successfulFaceDetect)/self.attemptedFaceDetect
        faceRedetect = float(self.successfulFaceRedetect)/(self.attemptedFaceRedetect + 0.01)
        print "face detections: " + str(faceDetect)
        print "face redetections: " + str(faceRedetect)
        print "eye detections: " + str(eyeDetect)
        print "eye redetections: " + str(eyeRedetect)

    """
    Get the average of several previous posisions of the face
    to make the shift look smooth.
    """
    def getSmoothedFace(self):
        if len(self.facePositionHistory) == 0:
            return self.facePosition
        return self.average(list(self.facePositionHistory))

    def getSmoothedEyes(self):
        return

    """Takes a list of tuples and returns their average"""
    def average(self, tuples):
        array = np.array(tuples, dtype=np.float64)
        mean = np.mean(array, 0).astype(np.int32)
        return tuple(mean)
            


if __name__ == "__main__":
    main()

