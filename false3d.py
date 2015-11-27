import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class False3D:
    def run(self, camera):
        self.tracker = EyeFaceTracker(eyeMode = True, smoothMode = False)
        self.displayer = ObjectDisplayer()
        cap = cv2.VideoCapture(camera)
        frameCount = 0

        while True:
            frameBGR, gray = self.nextFrame(cap)
            frameCount += 1
            if self.tracker.isTracking:
                self.tracker.trackMeanShift(frameBGR, gray)
            if not self.tracker.isTracking:
                self.tracker.detect(gray)
            self.tracker.storeData(frameBGR)
            self.tracker.computePerspective()
            self.displayer.computeAndDisplayAngle(self.tracker.perspective, frameBGR)
    
            self.displayFrame(frameBGR)
            if frameCount > 400:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        #self.tracker.reportStats()
        cap.release()
        cv2.destroyAllWindows()

    """
    Capture and parse a frame from the webcam.
    """
    def nextFrame(self, cap):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame, gray
    
    """
    Display frame and everything it is supposed to display.
    """
    def displayFrame(self, frame):
        self.displayFeatures(frame)
        cv2.imshow("Video", frame)
    
    def displayFeatures(self, frame):
        (fx,fy,fw,fh) = self.tracker.facePosition
        if self.tracker.smoothMode:
            (fx,fy,fw,fh) = self.tracker.getSmoothedFace()
        cv2.rectangle(frame, (fx,fy), (fx+fw,fy+fh), (255,0,0), 2)
        if self.tracker.eyeMode:
            for (ex,ey,ew,eh) in self.tracker.eyePositions:
                cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
        cv2.circle(frame, self.tracker.perspective, 5, (0,0,255), thickness=-1)

"""
Class performing feature detection and tracking.
"""
class EyeFaceTracker:
    """
    Class contructor. Loads classifiers and matchers, initializes
    property variables.
    @param eyeMode boolean determining whether we should track eyes
    """
    def __init__(self, eyeMode=False, smoothMode=False):
        #denotes whether we should track eyes or not
        self.eyeMode = eyeMode
        self.smoothMode = smoothMode
        classifDir = "classifiers/"
        facecas = classifDir + "haarcascade_frontalface_default.xml"
        eyecas = classifDir + "haarcascade_eye.xml"
        self.facePositionCascade = cv2.CascadeClassifier(facecas)
        self.eyeCascade = cv2.CascadeClassifier(eyecas)
        self.orb = cv2.ORB()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.isTracking = False
        self.eyePositions = [(0,0,0,0), (0,0,0,0)]
        self.facePosition = (0,0,0,0)
        self.perspective = (0,0)
        self.smoothN = 2
        #termination criteria for meanshift
        self.maxIterations = 10
        self.termCrit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.maxIterations, 1)
        self.facePositionHistory = deque()
        self.eyeHistory = deque()
        self.setupStats()

    def setupStats(self):
        self.attemptedEyeDetect = 0
        self.successfulEyeDetect = 0
        self.attemptedEyeRedetect = 0
        self.successfulEyeRedetect = 0
        self.attemptedFaceDetect = 0
        self.successfulFaceDetect = 0
        self.attemptedFaceRedetect = 0
        self.successfulFaceRedetect = 0

    """
    Store the detected features so that we can access them
    in the next iteration.
    """
    def storeData(self, frame):
        if self.isTracking:
            (fx,fy,fw,fh) = self.facePosition
            self.face = frame[fy:fy+fh, fx:fx+fw]

    """
    Detect features using haar cascade classifier.
    First detect face and within it search for eyes.
    """
    def detect(self, gray):
        ret, face = self.detectFace(gray)
        if not ret:
            self.isTracking = False
            return
        self.isTracking = True
        if self.eyeMode:
            self.detectEyes(gray, face)

    def detectFace(self, gray):
        self.attemptedFaceDetect += 1
        ret, faces = self.searchForFaces(gray)
        if not ret:
            return False, None
        self.successfulFaceDetect += 1
        face = faces[0]
        self.facePosition = face
        self.updateFace(face)
        return True, face

    def detectEyes(self, gray, face):
        (fx,fy,fw,fh) = face
        self.attemptedEyeDetect += 1
        ret, eyes = self.searchForEyes(gray[fy:fy+fh, fx:fx+fw])
        if not ret:
            self.eyePositions = [(0,0,0,0), (0,0,0,0)]
            return
        eyes = map(lambda (ex,ey,ew,eh):(ex+fx,ey+fy,ew,eh), eyes)
        self.eyePositions = eyes
        self.successfulEyeDetect += 1
        
    """Search for the face using histogram backprojection and
    meanshift/camshift."""
    def trackMeanShift(self, frameBGR, gray):
        faceHsv = cv2.cvtColor(self.face, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(faceHsv, np.array((0., 60., 32.)), np.array((180.,255.,255.)))
        dims = [0] #what components of hsv do we want to use?
        histSizes = [180] #corresponding to hsv
        ranges = [0, 180]
        faceHist = cv2.calcHist([faceHsv], dims, mask, histSizes, ranges)
        faceHist = cv2.GaussianBlur(faceHist, (13,13), 5)
        cv2.normalize(faceHist, faceHist, 0, 255, cv2.NORM_MINMAX)

        frameHsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)
        backProj = cv2.calcBackProject([frameHsv], dims, faceHist, ranges, 1)
        facePos = tuple(self.facePosition)
        ret, newFacePos = cv2.meanShift(backProj, facePos, self.termCrit)

        #cv2.imshow("Back Projected", backProj)

        self.attemptedFaceRedetect += 1
        if ret == self.maxIterations:
            self.isTracking = False
            self.facePosition = (0,0,0,0)
            return
        self.successfulFaceRedetect += 1

        self.updateFace(newFacePos)

        if self.eyeMode:
            self.trackEyes(gray)

    """Track eyes using viola jones"""
    def trackEyes(self, gray):
        (fx,fy,fw,fh) = self.facePosition
        ret, eyes = self.searchForEyes(gray[fy:fy+fh, fx:fx+fw])
        self.attemptedEyeRedetect += 1
        if not ret:
            return
        self.successfulEyeRedetect += 1
        eyes = map(lambda (ex,ey,ew,eh):(ex+fx,ey+fy,ew,eh), eyes)
        self.eyePositions = eyes

    """Track one eye using meanshift"""
    def trackEyeMeanShift(self, frameBGR, gray):
        eyeHsv = cv2.cvtColor(eye, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(eyeHsv, np.array((0., 60., 32.)), np.array((180.,255.,255.)))
        dims = [0] #what components of hsv do we want to use?
        histSizes = [180] #corresponding to hsv
        ranges = [0, 180]
        eyeHist = cv2.calcHist([eyeHsv], dims, mask, histSizes, ranges)
        eyeHist = cv2.GaussianBlur(eyeHist, (13,13), 5)
        cv2.normalize(eyeHist, eyeHist, 0, 255, cv2.NORM_MINMAX)
        frameHsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)
        backProj = cv2.calcBackProject([frameHsv], dims, eyeHist, ranges, 1)
        eyePos = tuple(eyePosition)
        ret, newPos = cv2.meanShift(backProj, eyePos, self.termCrit)
        #cv2.imshow("Back Projected", backProj)

        if ret == self.maxIterations:
            #self.isTracking = False
            self.eyePosition = (0,0,0,0)
            return
        self.eyePosition = newEyePos
        #self.updateEye(newEyePos)


    def searchForFaces(self, roi, scaleF=1.3, minNeighbors=5):
        faces = self.facePositionCascade.detectMultiScale(roi, scaleF, minNeighbors)
        return len(faces) == 1, faces

    def searchForEyes(self, roi, scaleF=1.3, minNeighbors=5):
        eyes = self.eyeCascade.detectMultiScale(roi, scaleF, minNeighbors)
        return len(eyes) == 2, eyes

    """
    Store the current face. Also, as an experimental feature, keep a queue
    of faces to smooth the tracking rectangle movement.
    """
    def updateFace(self, face):
        self.facePositionHistory.append(face)
        #keep most recent face for fast access
        self.facePosition = face
        #make space for next face
        if len(self.facePositionHistory) > self.smoothN:
            self.facePositionHistory.popleft()

    """
    Compute the approximate point of perspective.
    It will be exactly between the two eyes.
    """
    def computePerspective(self):
        (fx,fy,fw,fh) = self.facePosition
        self.perspective = (fx + fw/2, fy + fh/2)

    def reportStats(self):
        faceDetect = float(self.successfulFaceDetect)/self.attemptedFaceDetect
        faceRedetect = float(self.successfulFaceRedetect)/(self.attemptedFaceRedetect + 0.01)
        print ("face detections: " + str(faceDetect))
        print ("face redetections: " + str(faceRedetect))
        if self.eyeMode:
            eyeDetect = float(self.successfulEyeDetect)/self.attemptedEyeDetect
            eyeRedetect = float(self.successfulEyeRedetect)/self.attemptedEyeRedetect
            print ("eye detections: " + str(eyeDetect))
            print ("eye redetections: " + str(eyeRedetect))

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

"""
This class will display and rotate 3D projection of an image
based on the perspective.
Change the name to something less stupid.
"""
class ObjectDisplayer:
    def __init__(self):
        self.angleMarkerPos = np.array([400, 30])
        self.pointOfView = (0, 0)
        self.xMarkerOrigin = np.array([50, 0]) + self.angleMarkerPos
        self.yMarkerOrigin = np.array([130, 40]) + self.angleMarkerPos
        self.markerLength = 60.0
        self.innerMarkerRatio = 0.7
        self.fieldX = 120.0
        self.fieldY = 90.0 #both twice the actual (presumed) value
        self.disp = False
        self.angleX = 0.0
        self.angleY = 0.0
        self.frameDims = None

    def computeAndDisplayAngle(self, pointOfView, frame):
        if self.frameDims == None:
            self.frameDims = frame.shape[:2][::-1]
        self.pointOfView = pointOfView
        self.computeAngle()
        self.displayMarker(frame, horiz = True)
        self.displayMarker(frame, horiz = False)

    def computeAngle(self):
        x, y = self.pointOfView[0], self.pointOfView[1]
        width, height = self.frameDims[0], self.frameDims[1]
        self.angleX = (float(x)/width - 0.5) * self.fieldX
        self.angleY = (float(y)/height - 0.5) * self.fieldY

    def displayMarker(self, frame, horiz=True):
        if horiz:
            direction = np.array([0, self.markerLength])
            angle = -self.angleX
            origin = self.xMarkerOrigin
            field = self.fieldX
        else:
            direction = np.array([self.markerLength, 0])
            angle = self.angleY
            origin = self.yMarkerOrigin
            field = self.fieldY
        endPointRight = np.dot(direction, cv2.getRotationMatrix2D((2,2), -field/2, 1))[:2] + origin
        endPointLeft = np.dot(direction, cv2.getRotationMatrix2D((2,2), field/2, 1))[:2] + origin
        endPointTransl = direction + origin
        origin = tuple(origin.astype(np.int32))
        left = tuple(endPointLeft.astype(np.int32))
        right = tuple(endPointRight.astype(np.int32))
        cv2.line(frame, origin, left, (50,50,50), 2)
        cv2.line(frame, origin, right, (50,50,50), 2)
        markerEnd = direction * self.innerMarkerRatio
        markerEnd = np.dot(markerEnd, cv2.getRotationMatrix2D((2,2), angle/2, 1))[:2]
        arrowEnd = tuple((markerEnd + origin).astype(np.int32))
        cv2.line(frame, origin, arrowEnd, (50,50,50), 2)
        cv2.circle(frame, arrowEnd, 4, (50,50,50), thickness=-1)

        
if __name__ == "__main__":
    try:
        camera = int(sys.argv[1])
    except Exception:
        camera = 0
    false3D = False3D()
    false3D.run(camera)
