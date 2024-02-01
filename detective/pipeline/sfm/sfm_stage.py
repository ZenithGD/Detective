from typing import Tuple, List, TypeAlias
import numpy as np
import cv2

import os

from detective.pipeline import Pipeline, Stage

from detective.utils.spatial import *

from .triangulation import *
from detective.utils.plot import *

_ImageList : TypeAlias = List[np.array]
_SFMInput : TypeAlias = Tuple[np.array, _ImageList, np.array]
_SFMOutput : TypeAlias = Tuple[np.array, np.array]

class SFMStage(Stage):
    
    def __init__(self, callback = None, full_ba : bool = True):
        # Initialize parent class
        super().__init__(callback)
        self.full_ba = full_ba

    def __repr__(self):
        return f"SFMStage : (images, target, image points, target points) -> (3d, pose)"
    
    def __pose_with_calib(self, keypoints : list, npoints : int, context : Pipeline):
        """Computes for each camera from 2 to n the relative pose with respect to camera 1.
        index 0 of keypoints is camera 1.

        Args:
            keypoints (np.array): _description_
        """
        K_c = context.input_calib
        poses = [ np.eye(4) ]

        ## 1. first compute points by triangulation
        
        # obtain correspondences
        kp0 : np.array = keypoints[0]
        kp1 : np.array = keypoints[-1]

        # obtain F from correspondences
        F = create_F_from_matches(kp0, kp1)

        # obtain pose from F knowing the calibration
        E = K_c.T @ F @ K_c

        # Find pose and triangulated 3d points from camera 1 and i
        points3d, pose = getPose(kp0, kp1, K_c, K_c, E)

        print(points3d.shape)

        ## 2. compute pose with PnP solve for every other camera
        for i in range(1, len(keypoints) - 1):
            print("PnP estimation of camera 3 pose")
            imagePoints = np.ascontiguousarray(keypoints[i][:, :2]).reshape((npoints, 1, 2))
            objectPoints = np.ascontiguousarray(points3d[:, :3]).reshape((npoints, 1, 3))
            distCoeffs = None
            retval, pnp_rvec, pnp_tvec = cv2.solvePnP(
                objectPoints=objectPoints, 
                imagePoints=imagePoints, 
                cameraMatrix=K_c, 
                distCoeffs=distCoeffs,
                flags=cv2.SOLVEPNP_EPNP)
            
            if not retval:
                print("PnP problem can't be solved!!")
                os.exit(1)

            # get pose from PnP solution
            pnp_rotation = sc.linalg.expm(crossMatrix(pnp_rvec.flatten()))
            T_ci_c1_pnp = create_T(pnp_rotation, pnp_tvec.flatten())
            poses.append(T_ci_c1_pnp)
        
        poses.append(pose)

        if self.full_ba:
            # refinement via BA
            return BA_optimize(points3d.T, poses, keypoints, context.input_calib)
        
        else:
            return poses, points3d


    def run(self, input: _SFMInput, context : Pipeline) -> _SFMOutput:
        kp_imgs, kp_target, matches_imgs, matches_target = input

        # NOTE: 
        # All poses will be computed with respect to the first new camera,
        # that is, index 1 in keypoints/imgs array

        # obtain set of common points of all *new* cameras
        common_ids = set(matches_imgs[0][:, 0])
        print(common_ids)
        for i in range(1, len(matches_imgs)):
            common_ids = set(filter(
                lambda x : x in common_ids, matches_imgs[i][:, 0]
            ))
            print(common_ids)

        # find common matches between new cameras and compute 3d and pose
        matches_common_new = [ np.array(list(filter(lambda x: x[0] in common_ids, m))) for m in matches_imgs ]

        kp_0 = kp_imgs[0][list(common_ids)]
        kp_common_new = [ kp_0 ]
        for i, mp in enumerate(matches_common_new):
            points1 = kp_imgs[i+1][mp[..., 1]]
            kp_common_new.append(points1)

        # triangulate points from new photos
        poses, points3d = self.__pose_with_calib(kp_common_new, len(common_ids), context)

        # pose for old camera
        K_old, T_old, P = DLT_pose(points3d, kp_target)

        # poses serve as parameters for BA refinement
        if super().has_callback():
            super().get_callback()(context.images, context.target, kp_common_new, kp_target, points3d, poses, T_old, P)

        # points and old pose
        return points3d, T_old, kp_imgs, kp_target