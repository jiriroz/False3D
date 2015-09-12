import cv2
import numpy
import matplotlib.pyplot as plt


def main():
    
    eyeData = EyeData()

    cap = cv2.VideoCapture(1)
    frameCount = 0

    while True:
        frame, gray = nextFrame(cap)
        frameCount += 1

        if eyeData.tracking:
            track("haar", gray, eyeData)
        if not eyeData.tracking:
            detectEyes(gray, eyeData)

        displayState(eyeData.tracking)
        displayFrame(frame, eyeData.eyes)

        if frameCount > 100:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print "detection rate", eyeData.detectionRate()
    print "track rate", eyeData.redetectionRate()

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
def displayFrame(frame, eyes):
    displayEyes(frame, eyes)
    cv2.imshow("Video", frame)

def displayEyes(frame, eyes):
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

"""
Displays state of the program to the command line.
"""
def displayState(tracking):
    return
    if tracking:
        print "Eye detection succesful"
    else:
        print "Eye detection not succesful"

"""
Detect eyes using haar cascade classifier.
First detect face and within it search for eyes.
Return positive iff len(faces) == 1 and len(eyes) == 2.
"""
def detectEyes(gray, eyeData):
    eyeData.attemptedDetections += 1
    #print "attempting to detect eyes"
    faces = eyeData.faceCascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) != 1:
        eyeData.tracking = False
        return
    (x,y,w,h) = faces[0]
    roiGray = gray[y:y+h, x:x+w]
    eyes = eyeData.eyeCascade.detectMultiScale(roiGray)
    if len(eyes) != 2:
        eyeData.tracking = False
        return
    eyes = map(lambda (ex,ey,ew,eh):(ex+x,ey+y,ew,eh), eyes)
    keypts = []
    descr = []
    for (ex,ey,ew,eh) in eyes:
        roiEye = gray[ey:ey+eh,ex:ex+ew]
        localPts, localDes = eyeData.orb.detectAndCompute(roiEye, None)
        keypts.append(localPts)
        descr.append(localDes)
    #print keypts
    eyeData.eyes = eyes
    eyeData.keypts = keypts
    eyeData.descr = descr
    eyeData.tracking = True
    eyeData.successfulDetections += 1

"""
Attempt eye tracking according to the selected method.
"""
def track(mode, gray, eyeData):
    assert mode in ["haar", "keypoints"], "Tracking mode doesn't exist."
    eyeData.attemptedRedetections += 1
    if mode == "haar":
        trackEyesHaar(gray, eyeData)
    elif mode == "keypoints":
        trackEyesKeypt(gray, eyeData)
    if eyeData.tracking:
        eyeData.successfulRedetections += 1

"""
Detect eyes in a small region around the previous position using
haar cascades.
"""
def trackEyesHaar(gray, eyeData):
    newEyes = []
    horiz = 10
    vert = horiz / 2
    for (ex,ey,ew,eh) in eyeData.eyes:
        x, y = ex-horiz, ey-vert
        roiGray = gray[x:ex+ew+horiz, y:ey+eh+vert]
        eye = eyeData.eyeCascade.detectMultiScale(roiGray)
        if len(eye) != 1:
            eyeData.tracking = False
            return
        eye = eye[0]
        eye = (eye[0] + x, eye[1] + y, eye[2], eye[3])
        newEyes.append(eye)
    eyeData.tracking = True
    eyeData.eyes = newEyes

"""
Detect eyes in a small region around the previous position
using keypoint descriptor.
"""
def trackEyesKeypt(gray, eyeData):
    #print "attempting to track eyes"
    newEyes = []
    horiz = 10
    vert = horiz / 2
    avgdx = 0
    avgdy = 0
    count = 0
    for i in range(len(eyeData.eyes)):
        (ex,ey,ew,eh) = eyeData.eyes[i]
        x, y = ex-horiz, ey-vert
        roiGray = gray[x:ex+ew+horiz, y:ey+eh+vert]
        newKp, newDes = eyeData.orb.detectAndCompute(roiGray, None)
        if len(newKp) == 0:
            eyeData.tracking = False
            return
        matches = eyeData.matcher.match(eyeData.descr[i], newDes)
        matches = sorted(matches, key = lambda x:x.distance)
        matches = matches[:5]
        for match in matches:
            count += 1
            oldPt = eyeData.keypts[i][match.trainIdx]
            newPt = newKp[match.queryIdx]
            dx = oldPt[0] - newPt[0]
            dy = oldPt[1] - newPt[1]
            avgdx += dx
            avgdy += dy
    #avgdx /= count
    #avgdy /= count
    eyeData.tracking = True
    eyeData.eyes = newEyes

"""
Class holding all data needed for eye detection and tracking,
independent of the method.
"""
class EyeData:
    def __init__(self):
        facecas = "haarcascade_frontalface_default.xml"
        eyecas = "haarcascade_eye.xml"
        self.faceCascade = cv2.CascadeClassifier(facecas)
        self.eyeCascade = cv2.CascadeClassifier(eyecas)
        self.orb = cv2.ORB()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.tracking = False
        self.eyes = []
        self.keypts = None
        self.descr = None
        self.attemptedDetections = 0
        self.successfulDetections = 0
        self.attemptedRedetections = 0
        self.successfulRedetections = 0

    def redetectionRate(self):
        return float(self.successfulRedetections)/self.attemptedRedetections

    def detectionRate(self):
        return float(self.successfulDetections)/self.attemptedDetections




if __name__ == "__main__":
    main()


