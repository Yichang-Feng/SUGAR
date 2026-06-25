#!/usr/bin/env python3
import zmq
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation 

class VisionNode:
    def __init__(self):
        self.context = zmq.Context()
        
        self.img_sub = self.context.socket(zmq.SUB)
        self.img_sub.connect("tcp://127.0.0.1:5555")
        self.img_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        
        self.pose_pub = self.context.socket(zmq.PUB)
        self.pose_pub.connect("tcp://127.0.0.1:5556")

        # Camera intrinsics based on MuJoCo fovy=58.76, 640x480
        # fy = (480/2) / tan(58.76 deg / 2) = 240 / 0.5626 = 426.5
        self.camera_matrix = np.array([
            [426.5, 0.0, 320.0],
            [0.0, 426.5, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4,1))

        # Tag size is 0.06m (since MuJoCo half-size is 0.03)
        self.tag_size = 0.045
        
        try:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        print("Vision node initialized with ZMQ. Waiting for images...")

    def detect_tags(self, gray):
        if hasattr(self, 'detector'):
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        return corners, ids

    def estimate_pose(self, corners):
        obj_points = np.array([
            [-self.tag_size/2, -self.tag_size/2, 0],
            [ self.tag_size/2, -self.tag_size/2, 0],
            [ self.tag_size/2,  self.tag_size/2, 0],
            [-self.tag_size/2,  self.tag_size/2, 0]
        ], dtype=np.float32)
        
        success, rvec, tvec = cv2.solvePnP(obj_points, corners[0], self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        return rvec, tvec

    def rvec_to_quat(self, rvec):
        R, _ = cv2.Rodrigues(rvec)
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return w, x, y, z

    def run(self):
        while True:
            try:
                # Receive metadata then image
                md = self.img_sub.recv_json()
                data = self.img_sub.recv()
                
                img = np.frombuffer(data, dtype=md['dtype']).reshape(md['shape'])
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                corners, ids = self.detect_tags(gray)
                
                if ids is not None:
                    poses = {}
                    for i, tag_id in enumerate(ids.flatten()):
                        rvec, tvec = self.estimate_pose(corners[i])
                        w, x, y, z = self.rvec_to_quat(rvec)
                        
                        pose_data = {
                            "pos": [float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])],
                            "quat": [float(w), float(x), float(y), float(z)]
                        }
                        
                        if tag_id == 0:
                            poses["tag_0"] = pose_data
                        elif tag_id == 1:
                            poses["tag_1"] = pose_data
                            
                    if poses:
                        self.pose_pub.send_json(poses)
            except Exception as e:
                print(f"Error processing image: {e}")

if __name__ == '__main__':
    node = VisionNode()
    node.run()
