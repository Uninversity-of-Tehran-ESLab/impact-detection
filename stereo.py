import cv2
import ImpactUtils
import glob
import os
import numpy as np

from typing import Sequence, Tuple, Optional

CAMWIDTH = 640
CAMHEIGHT = 480
NUM_OF_CALPOINTS = 6



CAMDIR = "/dev/v4l/by-path/pci-0000:00:14.0-usbv2-*"


CalibrationPoints = np.array([[0,0,0],[180,0,0],[180,0,-90],[0,0,-90]],dtype="double")
mtx = np.array([[ 554.2563,   0,          320],
 [  0,        554.2563,      240],
 [  0,          0,           1,        ]])
dist = np.array([[-0.10771770030260086, 0.1213262677192688,  0.00091733073350042105, 0.00010589254816295579]],dtype="double")


def detect_markers(
    frame: cv2.typing.MatLike,
    draw_markers: bool = False,
    aruco_dictionary: int = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
) -> Tuple[Sequence[cv2.typing.MatLike], cv2.typing.MatLike, cv2.typing.MatLike | None]:
    """
    Detects the markers on the given frame and can optionally draw them
    on the frame as well!
    """

    gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        gray_scale_frame,
        aruco_dictionary,
    )
    frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    return corners, ids, (frame_markers if draw_markers else None)

class Camera:
    def __init__(self,name,path):
        global numOfTrackers, availableColors
        self.name = name
        self.cap = cv2.VideoCapture(path)
        self.calPointsCords = []
        self.extMat  = np.zeros((3,4))
        self.T = None
        self.R = None
        self.path = path
        self.tvec = None
        self.rvec = None
        self.calibrated = False
        self.projectionMatrix = None
        self.inMatrix = None
        self.distortCoeff = None
        self.lastFrame = None
        for j in range(NUM_OF_CALPOINTS):
            self.calPointsCords.append(np.array([[0,0]]))
    def normalSettings(self):
        os.system(f"v4l2-ctl -d {self.path} -c auto_exposure=0")
        os.system(f"v4l2-ctl -d {self.path} -c gain_automatic=0")
        os.system(f"v4l2-ctl -d {self.path} -c white_balance_automatic=0")

    def trackSettings(self):
        os.system(f"v4l2-ctl -d {self.path} -c auto_exposure=1")
        os.system(f"v4l2-ctl -d {self.path} -c gain_automatic=0")
        os.system(f"v4l2-ctl -d {self.path} -c white_balance_automatic=0")
        os.system(f"v4l2-ctl -d {self.path} -c gain=15")
        os.system(f"v4l2-ctl -d {self.path} -c exposure=30")
    
    def getFrame(self):
        ret, self.lastFrame  = self.cap.read()
        return ret
        
        

        
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= Get all cameras =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
cameras = []
i = 0
for camPath in glob.glob(CAMDIR):
    print(camPath)
    cameras.append(Camera(str(i),camPath))
    cameras[-1].normalSettings()
    cameras[-1].inMatrix = mtx
    cameras[-1].distortCoeff = dist
    i += 1
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= Gather Calibration Data  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
for cam in cameras:
        ids = None
        while ids == None:
            cam.getFrame()
            corners, ids, frame = detect_markers(cam.lastFrame,True)
        print(ids)
      
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= Calibrate  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if not os.path.exists(f"./calibrationData"):
    for cam in cameras:
        print(f"-=-=-=-=-=-=-=-=-=-=-=-= Cam {cam.name} =-=-=-=-=-=-=-=-=-=-=-=-=-")
        points = {}
        camSpacePoints = []
        for imgName in os.listdir(f"./STC/{cam.name}/"):
            cords = imgName.split('.')[0].split('_')
            points[int(cords[0])] = np.array([int(cords[1]),int(cords[2])],dtype="double")

        for i in range(NUM_OF_CALPOINTS):
            camSpacePoints.append(points[i])
        camSpacePoints = np.array(camSpacePoints,dtype="double")
        
        success, rvec, tvec  = cv2.solvePnP(CalibrationPoints,camSpacePoints,cam.inMatrix,cam.distortCoeff)
        if success:
            R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to matrix
            T = -np.matrix(R).T @ np.matrix(tvec)
            cam.T = T
            cam.R = R
            cam.tvec = tvec
            cam.rvec = rvec
            cam.calibrated = True
            projected_points, _ = cv2.projectPoints(CalibrationPoints, rvec, tvec, mtx, dist)
            print("Cam Space Points : ",camSpacePoints)
            print("Projected Points : ",projected_points)
            print("Position: ",T)
            print("Rotation: ",rvec)
            cam.projectionMatrix = np.dot(mtx, np.hstack((R,tvec)))

    

while True:
    for cam in cameras:
        if cam.calibrated:
            cam.getFrame()
            
    results = []
    for cam1 in cameras:
        for cam2 in cameras:
            if cam1.name == cam2.name:
                continue
            f, loc, cont = GetTrackerCordFromImge(cam1.lastFrame)
            
            if not f:
                continue
    
            pt1_hom = np.array([[loc[0]], [loc[1]]], dtype=np.float32)
            
            f, loc, cont = GetTrackerCordFromImge(cam2.lastFrame)
            
            if not f:
                continue
            
            pt2_hom = np.array([[loc[0]], [loc[1]]], dtype=np.float32)

            # Perform triangulation
            point_4d_hom = cv2.triangulatePoints(cam1.projectionMatrix, cam2.projectionMatrix, pt1_hom, pt2_hom)
            
            # Convert from homogeneous coordinates to (X, Y, Z)
            point_3d = point_4d_hom[:3] / point_4d_hom[3]

            results.append(point_3d.flatten())
    #print(results)
    results = np.array(results)
    results = np.average(results,axis=0)
    print(results)