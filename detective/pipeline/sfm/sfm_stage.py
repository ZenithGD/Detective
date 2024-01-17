from typing import Tuple, List, TypeAlias
import numpy as np
import cv2

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
    
    def __pose_with_calib(self, keypoints : np.array, context : Pipeline):
        """Computes for each camera from 2 to n the relative pose with respect to camera 1.
        index 0 of keypoints is camera 1.

        Args:
            keypoints (np.array): _description_
        """
        K_c = context.input_calib
        poses = [ np.eye(4) ]
        points3d = None
        
        for i in range(1, len(keypoints)):
            # obtain F from correspondences
            F = create_F_from_matches(keypoints[0], keypoints[i])

            # obtain pose from F knowing the calibration
            E = K_c.T @ F @ K_c

            # Find pose and triangulated 3d points from camera 1 and i
            points3d_i, pose = getPose(keypoints[0], keypoints[i], K_c, K_c, E)
            if points3d is None:
                points3d = points3d_i

            poses.append(pose)

        if self.full_ba:
            # refinement via BA
            return BA_optimize(points3d.T, poses, keypoints, context.input_calib)
        
        else:
            return poses, points3d


    def run(self, input: _SFMInput, context : Pipeline) -> _SFMOutput:
        imgs, target, img_keypoints, target_keypoints = input

        # NOTE: 
        # All poses will be computed with respect to the first new camera,
        # that is, index 1 in keypoints/imgs array

        # triangulate points from new photos
        poses, points3d = self.__pose_with_calib(img_keypoints, context)

        # pose for old camera
        K_old, T_old, P = DLT_pose(points3d, target_keypoints)

        # find projection matrix of a camera
        xi_proj = P @ points3d.T
        xi_proj /= xi_proj[2]
        
        fig, ax = plt.subplots()
        plot_image_residual(ax, context.target, target_keypoints.T, xi_proj[:2])
        ax.set_title(f"residuals old camera")

        # poses serve as parameters for BA refinement
        if super().has_callback():
            super().get_callback()(imgs, target, img_keypoints, points3d, poses, T_old)

        # points and old pose
        return points3d, T_old, img_keypoints, target_keypoints