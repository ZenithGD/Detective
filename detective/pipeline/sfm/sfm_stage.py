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
    
    def __init__(self, callback = None, full_ba : bool = True, refine_old : bool = True):
        # Initialize parent class
        super().__init__(callback)
        self.full_ba = full_ba
        self.refine_old = refine_old
        self.reuse_result = False

    def __repr__(self):
        return f"SFMStage : (images, target, image points, target points) -> (3d, pose)"
    
    def __pose_with_calib(self, keypoints : list, npoints : int, context : Pipeline):
        """Computes for each camera from 2 to n the relative pose with respect to camera 1.
        index 0 of keypoints is camera 1.

        Args:
            keypoints (np.array): _description_
        """
        if self.reuse_result:
            poses = [ np.loadtxt(f"results/ba/pose-{i}.txt") for i in range(len(keypoints)) ]
            points3d = np.loadtxt(f"results/ba/points3d.txt")

            return poses, points3d, None

        K_c = context.input_calib
        poses = [ np.eye(4) ]

        ## 1. first compute points by triangulation
        
        # obtain correspondences
        kp0 : np.array = keypoints[0]
        kp1 : np.array = keypoints[-1]

        # obtain F from correspondences
        pts1 = np.int32(kp0)
        pts2 = np.int32(kp1)
        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC)
        print(mask.shape)
        #F = create_F_from_matches(kp0, kp1)

        # obtain pose from F knowing the calibration
        E = K_c.T @ F @ K_c

        # Find pose and triangulated 3d points from camera 1 and i
        points3d, pose = getPose(kp0, kp1, K_c, K_c, E)

        mask = mask.flatten() > 0

        ## 2. compute pose with PnP solve for every other camera
        for i in range(1, len(keypoints) - 1):
            print("PnP estimation of camera 3 pose")
            kpi = keypoints[i]
            print(kpi.shape)
            imagePoints = np.ascontiguousarray(kpi[..., :2]).reshape((-1, 1, 2))
            objectPoints = np.ascontiguousarray(points3d[..., :3]).reshape((-1, 1, 3))
            distCoeffs = None
            retval, pnp_rvec, pnp_tvec, mask_i = cv2.solvePnPRansac(
                objectPoints=objectPoints, 
                imagePoints=imagePoints, 
                cameraMatrix=K_c, 
                distCoeffs=distCoeffs)
            
            # transform ransac mask into boolean mask
            mask_i = np.isin(np.arange(points3d.shape[0]),mask_i)
            print(mask_i.shape)
            
            if not retval:
                print("PnP problem can't be solved!!")
                os.exit(1)

            # get pose from PnP solution
            pnp_rotation = sc.linalg.expm(crossMatrix(pnp_rvec.flatten()))
            T_ci_c1_pnp = create_T(pnp_rotation, pnp_tvec.flatten())
            poses.append(T_ci_c1_pnp)

            print(np.count_nonzero(np.logical_and(mask, mask_i.flatten() > 0)))

            mask = np.logical_and(mask, mask_i.flatten() > 0)
        
        print(mask.shape)
        poses.append(pose)

        if self.full_ba:
            # refinement via BA
            kps = [ kp[mask] for kp in keypoints ]
            points3d_all = np.zeros_like(points3d)
            points3d_all[:, 3] = 1
            poses, points3d = BA_optimize(points3d[mask].T, poses, kps, context.input_calib)
            points3d_all[mask] = points3d
            return poses, points3d, points3d_all, mask
        else:
            
            points3d_all = np.zeros_like(points3d)
            points3d_all[:, 3] = 1
            points3d_all[mask] = points3d[mask].copy()
            return poses, points3d[mask], points3d_all, mask

    def __old_camera_pose(self, points3d, kp):
        
        K, R, t, P = DLT_pose(points3d, kp)

        if self.refine_old:
            # refine pose and return
            return refine_pose(points3d, kp, K, R, t, P)
        else:
            return K, create_T(R, t), P


    def run(self, input: _SFMInput, context : Pipeline) -> _SFMOutput:
        kp_imgs, kp_target, matches_imgs, matches_target = input

        # NOTE: 
        # All poses will be computed with respect to the first new camera,
        # that is, index 1 in keypoints/imgs array

        # obtain set of common points of all *new* cameras
        common_ids = set(matches_imgs[0][:, 0])
        for i in range(1, len(matches_imgs)):
            common_ids = set(filter(
                lambda x : x in common_ids, matches_imgs[i][:, 0]
            ))

        # find common matches between new cameras and compute 3d and pose
        matches_common_new = [ np.array(list(filter(lambda x: x[0] in common_ids, m))) for m in matches_imgs ]

        kp_0 = kp_imgs[0][matches_common_new[0][..., 0]]
        kp_common_new = [ kp_0 ]
        for i, mp in enumerate(matches_common_new):
            points1 = kp_imgs[i+1][mp[..., 1]]
            kp_common_new.append(points1)

        # triangulate points from new photos
        poses, points3d, points3d_all, mask = self.__pose_with_calib(kp_common_new, len(kp_common_new[0]), context)

        common_target_ids = common_ids.copy()
        common_target_ids = set(filter(
            lambda x : x in common_target_ids, matches_target[:, 0]
        ))
        
        # # find ids of masked keypoints
        # mask for values common with old photo
        mask_target = np.isin(matches_common_new[0][..., 0], np.array(list(common_target_ids)))
        
        print(len(common_target_ids))

        matches_common_target = np.array(list(filter(lambda x: x[0] in common_target_ids, matches_target)))
        kp_common_target = kp_target[matches_common_target[..., 1]]

        print(kp_common_target.shape)
        
        # pose for old camera
        points3d_old = points3d_all[mask_target]
        print(points3d_old)
        pmask = points3d_old[:, 0] != 0
        print(pmask)

        points3d_old_m = points3d_old[pmask]
        print(points3d_old_m)
        kp_common_target_m = kp_common_target[pmask]

        K_old, T_old, P = self.__old_camera_pose(points3d_old, kp_common_target)

        kpt_proj = P @ points3d_old_m.T
        kpt_proj /= kpt_proj[2]

        # poses serve as parameters for BA refinement
        if super().has_callback():
            kpcn = [ kp[mask] for kp in kp_common_new ]
            super().get_callback()(context.images, context.target, kpcn, kp_common_target_m, points3d, poses, points3d_old_m, T_old, P)

        # next stage should be sparse on a pair of images
        return kp_common_new[0], kpt_proj.T