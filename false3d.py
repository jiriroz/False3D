import cv2
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import visual as vis
import sys
from board import *
from collections import deque

"""
To test eye tracking:
python false3D.py --test [webCam number]

To run:
python false3D.py --run [webCam number]

"""

#Ideas TODO:
#Argparse
#One eye tracking
#Decrease variability of eyes' size
#Change the size of the search region based on distance
#Infer next position of the eyes
#Verify eyes based on their relative position
#Camshift instead of meanshift? Compare thoroughly

class False3D:

    def __init__(self, test, mode=0):
        self.isTestRun = test
        self.tracker = EyeFaceTracker()
        self.displayer = ObjectDisplayer(self.isTestRun, mode)

    def run(self, camera):
        cap = cv2.VideoCapture(camera)
        frameCount = 0

        while True:
            frameBGR, gray = self.nextFrame(cap)
            frameCount += 1
            self.tracker.track(frameBGR, gray)
            self.tracker.storeData(frameBGR)
            self.tracker.computePerspective()
            self.displayer.computeAndDisplayAngle(self.tracker.perspective, self.tracker.distanceEyes, self.tracker.dDistanceEyes, frameBGR, self.tracker.startedTracking)
            if self.tracker.isTrackingEyes:
                self.tracker.startedTracking = True
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
        (sx,sy,sw,sh) = self.tracker.eyeroi
        cv2.rectangle(frame, (sx,sy), (sx+sw,sy+sh), (255,0,0), 2)

        (ex,ey,ew,eh) = self.tracker.eyePositions[0]
        cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
        (ex,ey,ew,eh) = self.tracker.eyePositions[1]
        cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (0,0,255), 2)
        cv2.circle(frame, self.tracker.perspective, 5, (0,0,255), thickness=-1)

"""
Class performing feature detection and tracking.
"""
class EyeFaceTracker:
    """
    Class contructor. Loads classifiers and matchers, initializes
    property variables.
    """
    def __init__(self):
        self.initClassifiers()
        self.isTrackingFace = False
        self.isTrackingEyes = False
        self.startedTracking = False
        self.eyePositions = [(0,0,0,0), (0,0,0,0)]
        self.facePosition = (0,0,640,480)
        self.perspective = (0,0)
        self.distanceEyes = 0.0
        self.dDistanceEyes = 0.0
        self.distanceEyesSmooth = deque()
        self.smoothN = 4
        for x in range(self.smoothN):
            self.distanceEyesSmooth.append(0)
        #Portion of face from the top to exclude
        self.topMargin = 0
        #Portion of face from the bottom to exclude
        self.bottomMargin = 0
        #Portion of face from the left to exclude
        self.leftMargin = 0.10
        #Portion of face from the right to exclude
        self.rightMargin = 0.10
        self.face = None
        self.eyes = [None, None]
        self.xFaceShift = 0
        self.yFaceShift = 0
        self.eyeShift = [(0,0), (0,0)]
        self.maxEyeShift = 15.0
        #above this threshold any shift in eyes or face is rejected
        #it is two standard deviations above the observed mean
        self.shiftThreshold = 10.0
        #minimum face shift to search for eyes again
        self.minFaceShift = 1.0
        self.frameShape = (480, 640)
        self.foreground = np.zeros(self.frameShape)
        self.countBg = 0
        self.successSequence = 0
        self.failSequence = 0
        self.minSuccessSequence = 1
        self.minFailSequence = 4
        self.eyeroi = (0,0,0,0)

    """Search for and track face and eyes."""
    def track(self, frameBGR, gray):
        self.processFrame(frameBGR, gray)

        self.isTrackingEyes = self.trackEyes()
        if self.isTrackingEyes:
            self.successSequence += 1
            self.failSequence = 0
        else:
            self.failSequence += 1
        if self.failSequence > self.minFailSequence:
            self.successSequence = 0

    """
    Process frame.
    """
    def processFrame(self, frameBGR, gray):
        self.roiImgBGR = frameBGR
        self.roiImgGray = gray

    def detectFace(self):
        ret, faces = self.searchForFaces(self.roiImgGray)
        if not ret:
            return False
        face = faces[0]
        self.updateFace(face)
        return True
        
    """Track face using meanshift."""
    def trackFace(self):
        faceHsv = cv2.cvtColor(self.face, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(faceHsv, np.array((0., 60., 32.)), np.array((180.,255.,255.)))
        #dims: Components of HSV, histSizes: corresponding to HSV
        dims, histSizes, ranges = [0], [180], [0, 180]
        faceHist = cv2.calcHist([faceHsv], dims, mask, histSizes, ranges)
        faceHist = cv2.GaussianBlur(faceHist, (13,13), 5)
        cv2.normalize(faceHist, faceHist, 0, 255, cv2.NORM_MINMAX)

        frameHsv = cv2.cvtColor(self.roiImgBGR, cv2.COLOR_BGR2HSV)
        backProj = cv2.calcBackProject([frameHsv], dims, faceHist, ranges, 1)
        facePos = tuple(self.facePosition)
        ret, newFacePos = cv2.meanShift(backProj, facePos, self.termCrit)

        if ret == self.maxIterations:
            return False
        self.updateFace(newFacePos)
        return True

    """
    Track eyes and update them.
    @return boolean whether the eyes were successfully tracked
    """
    def trackEyes(self):
        if self.successSequence >= self.minSuccessSequence:
            #If we've been tracking long enough, search around eyes.
            (x,y,w,h) = self.getEyeRoi()
            scaleF = 1.20
        else:
            #If haven't been tracking long enough, search whole image.
            (x,y,w,h) = (0,0,self.frameShape[1],self.frameShape[0])
            scaleF = 1.30
        eyes = self.searchForEyes(self.roiImgGray[y:y+h, x:x+w], scaleF=scaleF)
        if len(eyes) < 2:
            self.updateEyes(self.eyePositions, [], [0, 1])
            return False
        eyes = map(lambda (ex,ey,ew,eh):(ex+x,ey+y,ew,eh), eyes)
        self.updateEyes(self.fixEyeOrder(eyes), [0, 1], [])
        return True

    """Get roi where we should search for eyes in the next frame.
    Assume the eye at 0 position is the left one."""
    def getEyeRoi(self):
        lookAheadX = 0
        lookAheadY = lookAheadX
        dx = (self.eyeShift[0][0] + self.eyeShift[1][0])/2 * lookAheadX
        dy = (self.eyeShift[0][1] + self.eyeShift[1][1])/2 * lookAheadY
        xExtent = 100
        yExtent = 70
        lEye, rEye = self.eyePositions[0], self.eyePositions[1]
        left = max(lEye[0] - xExtent + dx, 0)
        right = min(rEye[0] + rEye[2] + xExtent + dx, self.frameShape[1])
        top = max(min(lEye[1], rEye[1]) - yExtent + dy, 0)
        fooDown = max(lEye[1] + lEye[3], rEye[1] + rEye[3])
        down = min(fooDown + yExtent + dy, self.frameShape[0])
        self.eyeroi = (left, top, right-left, down-top)
        return (left, top, right-left, down-top)

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
    def searchForEyes(self, searchRoi, scaleF=1.25, minNeighbors=5):
        height = searchRoi.shape[0]
        width = searchRoi.shape[1]
        eyes = self.eyeCascade.detectMultiScale(searchRoi, scaleF, minNeighbors)
        eyes = map(lambda (ex,ey,ew,eh):(int(ex),int(ey),int(ew),int(eh)), eyes)
        if not self.validateEyes(eyes[:2]):
            return []
        return eyes[:2]

    def validateEyes(self, eyes):
        thres = self.shiftThreshold
        for i in range(len(eyes)):
            area = eyes[i][2] * eyes[i][3]
            dx = (eyes[i][0] - self.eyePositions[i][0])
            dy = (eyes[i][1] - self.eyePositions[i][1])
            shiftCheck = abs(dx) < thres and abs(dy) < thres
            areaCheck = area > 800 and area < 6000
            if not (shiftCheck or areaCheck):
                return False
        return True

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
    def updateEyes(self, eyes, detected, notDetected):
        for i in notDetected:
            eyes[i] = self.shiftEye(eyes[i], i)
        for i in detected:
            if not self.startedTracking:
                continue
            self.eyeShift[i] = self.computeEyeShift(eyes[i], i)
            (exOld,eyOld,ewOld,ehOld) = self.eyePositions[i]
            (exNew,eyNew,ewNew,ehNew) = eyes[i]
            dist = self.hypotenuse(exOld - exNew, eyOld - eyNew)
            dist = dist/10
            #make the eyes approach slower as they get closer
            dex, dey = self.setLength(exNew - exOld, eyNew - eyOld, self.maxEyeShift*dist)
            eyes[i] = [int(exOld + dex), int(eyOld + dey), ewNew, ehNew]
        self.eyePositions = eyes

    def computeEyeShift(self, eyeNew, index):
        dx = eyeNew[0] - self.eyePositions[index][0]
        dy = eyeNew[1] - self.eyePositions[index][1]
        return (dx, dy)

    def shiftEye(self, eye, index):
        #dx, dy = self.eyeShift[index][0], self.eyeShift[index][1]
        #return [eye[0] + dx, eye[1] + dy, eye[2], eye[3]]
        return eye

    def fixEyeOrder(self, eyes):
        (ex1,ey1,ew1,eh1) = eyes[0]
        (ex2,ey2,ew2,eh2) = eyes[1]
        if ex1 > ex2:
             temp = eyes[1]
             eyes[1] = eyes[0]
             eyes[0] = temp
        return eyes

    def setLength(self, x, y, mag):
        length = self.hypotenuse(x, y)
        if length <= mag:
            return x, y
        alpha = length / mag
        return x/alpha, y/alpha

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
    Compute the approximate point of perspective.
    It will be exactly between the two eyes.
    """
    def computePerspective(self):
        eye1, eye2 = self.eyePositions[0], self.eyePositions[1]
        x1, y1 = eye1[0] + eye1[2]/2, eye1[1] + eye1[3]/2
        x2, y2 = eye2[0] + eye2[2]/2, eye2[1] + eye2[3]/2
        self.perspective = (x1 + (x2-x1)/2, y1 + (y2-y1)/2)
        distanceEyes = self.hypotenuse(x1 - x2, y1 - y2)
        diffDist = distanceEyes - self.distanceEyes
        self.distanceEyes = distanceEyes
        if self.startedTracking:
            self.distanceEyesSmooth.popleft()
            self.distanceEyesSmooth.append(diffDist)
            self.dDistanceEyes = sum([x for x in self.distanceEyesSmooth]) / self.smoothN

    """
    Initialize classifiers used for detection/tracking.
    """
    def initClassifiers(self):
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
        #termination criteria for meanshift
        self.maxIterations = 10
        self.termCrit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.maxIterations, 1)
        self.bgsub = cv2.BackgroundSubtractorMOG2(history=50, varThreshold=16)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    """Compute hypotenuse in a right triangle given by legs x and y."""
    def hypotenuse(self, x, y):
        return math.sqrt(x*x + y*y)


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
        self.fieldX = 60.0 #angle of the horizontal field of the camera
        self.fieldY = 45.0 #angle of the vertical field of the camera
        self.angleScale = 2.0 #scale the angle by this value to make it look nicer
        self.disp = False
        self.angleX = 0.0
        self.angleY = 0.0
        self.dAngleX = 0.0
        self.dAngleY = 0.0
        self.distanceEyes = 0
        self.dDistanceEyes = 0
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
        angleX = (float(x)/width - 0.5) * self.fieldX * self.angleScale
        angleY = (float(y)/height - 0.5) * self.fieldY * self.angleScale
        self.dAngleX = angleX - self.angleX
        self.dAngleY = angleY - self.angleY
        self.angleX = angleX
        self.angleY = angleY

    def displayMarker(self, frame, horiz=True):
        if horiz:
            direction = np.array([0, self.markerLength])
            angle = -self.angleX
            origin = self.xMarkerOrigin
            field = self.fieldX * self.angleScale
        else:
            direction = np.array([self.markerLength, 0])
            angle = self.angleY
            origin = self.yMarkerOrigin
            field = self.fieldY * self.angleScale
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
        self.objects.velocity = (0,0,self.dDistanceEyes/5)
        self.objects.rotate(angle=dax, axis = vis.vector(0,1,0), origin=self.objects.pos)
        self.objects.rotate(angle=day, axis = vis.vector(1,0,0), origin=self.objects.pos)
        self.objects.pos += self.objects.velocity
        axis = self.objects.axis
        self.objects.axis = (axis[0], 0, axis[2]) #fix x axis

        
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
        false3D.run(camera)
    elif mode == "--run":
        false3D = False3D(test=False)
        false3D.run(camera)
    else:
        print "Option not recognized. To test, run with"
        print "python false3D.py --test [webcam number]"
        print "To run, run"
        print "python false3D.py --run [webcam number]"
