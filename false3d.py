import cv2
import numpy as np
import matplotlib.pyplot as plt
import visual as vis
import sys
from board import *
from collections import deque

class False3D:

    def __init__(self, test, mode=0):
        self.isTestRun = test
        self.displayEyeSearchRegion = False
        self.tracker = EyeFaceTracker(eyeMode = True)
        self.displayer = ObjectDisplayer(self.isTestRun, mode)

    def run(self, camera):
        cap = cv2.VideoCapture(camera)
        frameCount = 0
        firstTracking = False

        while True:
            frameBGR, gray = self.nextFrame(cap)
            frameCount += 1
            self.tracker.track(frameBGR, gray)
            self.tracker.storeData(frameBGR)
            self.tracker.computePerspective(firstTracking)
            self.displayer.computeAndDisplayAngle(self.tracker.perspective, self.tracker.distanceEyes, self.tracker.dDistanceEyes, frameBGR, firstTracking)
            if self.tracker.isTrackingEyes:
                firstTracking = True
            if self.isTestRun:
                self.displayFrame(frameBGR)
            if frameCount > 10000:
                break
            if self.isTestRun:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

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
        cv2.rectangle(frame, (fx,fy), (fx+fw,fy+fh), (255,0,0), 2)
        if self.displayEyeSearchRegion:
            topMargin = int(self.tracker.topMargin * fh)
            bottomMargin = int((1-self.tracker.bottomMargin) * fh)
            cv2.rectangle(frame, (fx,fy), (fx+fw, fy+topMargin), (255,0,0), -1)
            cv2.rectangle(frame, (fx,fy+bottomMargin), (fx+fw, fy+fh), (255,0,0), -1)
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
    def __init__(self, eyeMode=False):
        #denotes whether we should track eyes or not
        self.eyeMode = eyeMode
        classifDir = "classifiers/"
        facecasDefault = "haarcascade_frontalface_default.xml"
        facecasAlt = "haarcascade_frontalface_alt.xml"
        eyecasDefault = "haarcascade_eye.xml"
        eyecasGlasses = "haarcascade_eye_tree_eyeglasses.xml"
        facecas = classifDir + facecasDefault
        eyecas = classifDir + eyecasGlasses
        self.facePositionCascade = cv2.CascadeClassifier(facecas)
        self.eyeCascade = cv2.CascadeClassifier(eyecas)
        self.orb = cv2.ORB()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.isTrackingFace = False
        self.isTrackingEyes = False
        self.eyePositions = [(0,0,0,0), (0,0,0,0)]
        self.facePosition = (0,0,0,0)
        self.perspective = (0,0)
        self.distanceEyes = 0.0
        self.dDistanceEyes = 0.0
        self.distanceEyesSmooth = deque()
        self.distanceEyesSmooth.append(0)
        self.distanceEyesSmooth.append(0)
        self.distanceEyesSmooth.append(0) 
        #Portion of face from the top to exclude
        self.topMargin = 0.2
        #Portion of face from the bottom to exclude
        self.bottomMargin = 0.3
        self.face = None
        self.eyes = [None, None]
        self.xFaceShift = 0
        self.yFaceShift = 0
        #above this threshold any shift in eyes or face is rejected
        #it is two standard deviations above the observed mean
        self.shiftThreshold = 10
        #termination criteria for meanshift
        self.maxIterations = 10
        self.termCrit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.maxIterations, 1)

    """Search for and track face and eyes."""
    def track(self, frameBGR, gray):
        if self.isTrackingFace:
            self.isTrackingFace = self.trackFace(frameBGR)
        if not self.isTrackingFace:
            self.isTrackingFace = self.detectFace(gray)

        if self.eyeMode and not self.isTrackingEyes and self.isTrackingFace:
            self.isTrackingEyes = self.detectEyes(gray)
        elif self.eyeMode and self.isTrackingEyes:
            self.isTrackingEyes = self.trackEyes(frameBGR, gray)

    def detectFace(self, gray):
        ret, faces = self.searchForFaces(gray)
        if not ret:
            return False
        face = faces[0]
        self.updateFace(face)
        return True

    def detectEyes(self, gray):
        (fx,fy,fw,fh) = self.facePosition
        ret, eyes = self.searchForEyes(gray[fy:fy+fh, fx:fx+fw])
        if not ret:
            self.updateEyes(eyes, shift = True)
            return False

        eyes = map(lambda (ex,ey,ew,eh):(ex+fx,ey+fy,ew,eh), eyes)
        self.updateEyes(eyes)
        return True
        
    """Track face using meanshift."""
    def trackFace(self, frameBGR):
        faceHsv = cv2.cvtColor(self.face, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(faceHsv, np.array((0., 60., 32.)), np.array((180.,255.,255.)))
        #dims: Components of HSV, histSizes: corresponding to HSV
        dims, histSizes, ranges = [0], [180], [0, 180]
        faceHist = cv2.calcHist([faceHsv], dims, mask, histSizes, ranges)
        faceHist = cv2.GaussianBlur(faceHist, (13,13), 5)
        cv2.normalize(faceHist, faceHist, 0, 255, cv2.NORM_MINMAX)

        frameHsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)
        backProj = cv2.calcBackProject([frameHsv], dims, faceHist, ranges, 1)
        facePos = tuple(self.facePosition)
        ret, newFacePos = cv2.meanShift(backProj, facePos, self.termCrit)

        if ret == self.maxIterations:
            return False
        self.updateFace(newFacePos)
        return True

    """Track eyes using viola jones"""
    def trackEyes(self, frame, gray):
        (fx,fy,fw,fh) = self.facePosition
        ret, eyes = self.searchForEyes(gray[fy:fy+fh, fx:fx+fw])
        if not ret:
            self.updateEyes(eyes, shift=True)
            return False
        eyes = map(lambda (ex,ey,ew,eh):(ex+fx,ey+fy,ew,eh), eyes)
        self.updateEyes(eyes)
        return True

    """
    Search for face using viola jones. Return boolean indicating
    whether valid face found and the face.
    """
    def searchForFaces(self, roi, scaleF=1.3, minNeighbors=5):
        faces = self.facePositionCascade.detectMultiScale(roi, scaleF, minNeighbors)
        if len(faces) != 1:
            return False, faces
        return self.validateFace(faces[0]), faces

    def validateFace(self, face):
        dx = face[0] - self.facePosition[0]
        dy = face[1] - self.facePosition[1]
        validX = abs(dx) < self.shiftThreshold
        validY = abs(dy) < self.shiftThreshold
        #apply threshold iff tracking face
        return not self.isTrackingFace or (validX and validY)

    """
    Search for eyes using viola jones within the face.
    Exclude an upper and lower portion of the face.
    """
    def searchForEyes(self, face, scaleF=1.3, minNeighbors=5):
        height = face.shape[0]
        top = height * self.topMargin
        bottom = height * (1 - self.bottomMargin)
        faceRestricted = face[:bottom, :]
        faceRestricted = faceRestricted[top:, :]
        eyes = self.eyeCascade.detectMultiScale(faceRestricted, scaleF, minNeighbors)
        eyes = map(lambda (ex,ey,ew,eh):(int(ex),int(ey+top),int(ew),int(eh)), eyes)
        if len(eyes) < 2:
            return False, None
        return self.validateEyes([eyes[0], eyes[1]]), [eyes[0], eyes[1]]

    def validateEyes(self, eyes):
        area1 = eyes[0][2] * eyes[0][3]
        area2 = eyes[1][2] * eyes[1][3]
        dx1 = (eyes[0][0] - self.eyePositions[0][0])
        dx2 = (eyes[1][0] - self.eyePositions[1][0])
        dy1 = (eyes[0][1] - self.eyePositions[0][1])
        dy2 = (eyes[1][1] - self.eyePositions[1][1])
        thres = self.shiftThreshold
        shiftCheck = all([abs(dx1), abs(dx2), abs(dy1), abs(dy2)]) < thres
        areaCheck =  area1 > 800 and area2 > 800 and area1 < 6000 and area2 < 6000
        return shiftCheck and areaCheck

    """
    Store the detected features so that we can access them
    in the next iteration.
    """
    def storeData(self, frame):
        if self.isTrackingFace:
            (fx,fy,fw,fh) = self.facePosition
            self.face = frame[fy:fy+fh, fx:fx+fw]
        if self.isTrackingEyes:
            for i in range(len(self.eyePositions)):
                (ex,ey,ew,eh) = self.eyePositions[i]
                self.eyes[i] = frame[ey:ey+eh, ex:ex+ew]

    """
    Store the current position of the face and the shift.
    """
    def updateFace(self, face):
        self.xFaceShift = face[0] - self.facePosition[0]
        self.yFaceShift = face[1] - self.facePosition[1]
        self.facePosition = face

    """
    Check if new eye position and dimension is valid and replace
    them.
    """
    def updateEyes(self, eyes, shift=False):
        if shift:
            dx, dy = self.xFaceShift, self.yFaceShift
            eyes = self.eyePositions
            eyes = map(lambda (ex,ey,ew,eh):(ex+dx,ey+dy,ew,eh), eyes)
        self.eyePositions = eyes

    """
    Compute the approximate point of perspective.
    It will be exactly between the two eyes.
    """
    def computePerspective(self, firstTracking):
        eye1, eye2 = self.eyePositions[0], self.eyePositions[1]
        x1, y1 = eye1[0] + eye1[2]/2, eye1[1] + eye1[3]/2
        x2, y2 = eye2[0] + eye2[2]/2, eye2[1] + eye2[3]/2
        self.perspective = (x1 + (x2-x1)/2, y1 + (y2-y1)/2)
        distanceEyes = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
        dD = distanceEyes - self.distanceEyes
        self.distanceEyes = distanceEyes
        if firstTracking:
            self.distanceEyesSmooth.popleft()
            self.distanceEyesSmooth.append(dD)
            self.dDistanceEyes = sum([x for x in self.distanceEyesSmooth]) / 3.0
        print self.dDistanceEyes

"""
This class will display and rotate 3D projection of an image
based on the perspective.
Change the name to something less stupid.
"""
class ObjectDisplayer:
    def __init__(self, isTest, mode):
        self.isTestRun = isTest
        self.angleMarkerPos = np.array([400, 30])
        self.viewPoint = (0, 0)
        self.xMarkerOrigin = np.array([50, 0]) + self.angleMarkerPos
        self.yMarkerOrigin = np.array([130, 40]) + self.angleMarkerPos
        self.markerLength = 60.0
        self.innerMarkerRatio = 0.7
        self.fieldX = 120.0
        self.fieldY = 90.0 #both twice the actual (presumed) value
        self.disp = False
        self.angleX = 0.0
        self.angleY = 0.0
        self.dAngleX = 0.0
        self.dAngleY = 0.0
        self.dDistanceEyes = 0
        self.distanceEyes = 0
        self.frameDims = None
        if not self.isTestRun:
            self.createObjects(mode)

    def createObjects(self, mode):
        vis.scene.autoscale = False

        if mode == 0:
            self.board = Board()
            self.objects = self.board.frame
            self.objects.pos = (0,0,-3)
        elif mode == 1:
            self.objects = vis.frame()
            arrow = vis.arrow(frame=self.objects, pos=vis.vector(0,0,0), axis = vis.vector(0,0,5), color = vis.color.red)
            self.objects.pos = (0,0,0)
        self.objects.velocity = vis.vector(0,0,0)

    def computeAndDisplayAngle(self, viewPoint, distanceEyes, dDistanceEyes, frame, rotate):
        if self.frameDims == None:
            self.frameDims = frame.shape[:2][::-1]
        self.viewPoint = viewPoint
        self.distanceEyes = distanceEyes
        self.dDistanceEyes = dDistanceEyes
        self.computeAngle()
        self.displayMarker(frame, horiz = True)
        self.displayMarker(frame, horiz = False)
        if rotate and not self.isTestRun:
            self.displayObject()

    def computeAngle(self):
        x, y = self.viewPoint[0], self.viewPoint[1]
        width, height = self.frameDims[0], self.frameDims[1]
        angleX = (float(x)/width - 0.5) * self.fieldX
        angleY = (float(y)/height - 0.5) * self.fieldY
        self.dAngleX = angleX - self.angleX
        self.dAngleY = angleY - self.angleY
        self.angleX = angleX
        self.angleY = angleY

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

    def displayObject(self):
        dax = -self.dAngleX/40
        day = -self.dAngleY/40
        self.objects.velocity = (0,0, self.dDistanceEyes/3)
        self.objects.rotate(angle=dax, axis = vis.vector(0,1,0), origin=vis.vector(0,0,0))
        self.objects.rotate(angle=day, axis = vis.vector(1,0,0), origin=vis.vector(0,0,0))
        self.objects.pos += self.objects.velocity

        
if __name__ == "__main__":
    try:
        mode = sys.argv[1]
    except Exception:
        mode = "--run"
    try:
        camera = int(sys.argv[2])
    except Exception:
        camera = 0
    if mode == "--test":
        false3D = False3D(test=True)
    elif mode == "--run":
        false3D = False3D(test=False)
    false3D.run(camera)
