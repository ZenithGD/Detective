from typing import Tuple, List, TypeAlias
import numpy as np
import cv2

import matplotlib.pyplot as plt

from detective.pipeline import Pipeline, Stage

_ImageList : TypeAlias = List[np.array]
_SDInput : TypeAlias = Tuple[np.array, np.array]
_SDOutput : TypeAlias = Tuple[np.array, np.array, np.array]

class SparseDiffStage(Stage):

    def __repr__(self):
        return f"DiffStage : (images, target, image points, target points) -> (diffs)"

    def run(self, input: _SDInput, context : Pipeline) -> _SDOutput:
        kp_old,kp_new = input


        # extract SIFT descriptors for comparison
        sift = cv2.SIFT_create()

        kp_new_cv = [cv2.KeyPoint(x[0], x[1], 12) for x in kp_new]
        kp_old_cv = [cv2.KeyPoint(x[0], x[1], 12) for x in kp_old]

        _, ds_new = sift.compute(context.target, kp_new_cv)
        _, ds_old = sift.compute(context.target, kp_old_cv)

        bf = cv2.BFMatcher()

        dist = 200
        matches = bf.radiusMatch(ds_new, ds_old, dist)

        common_pts, diff_pts = [], []

        common_idx = []
        # ratio test as per Lowe's paper
        for i, m in enumerate(matches):
            if len(m) > 0:
                match_obj = m[0]
                # compute distance in image space
                kp0, kp1 = kp_new[match_obj.queryIdx], kp_old[match_obj.trainIdx]
                mdist = match_obj.distance
                #if m.distance < 0.7*n.distance:
                if mdist < 150:
                    common_pts.append(kp0)
                else:
                    diff_pts.append(kp0)

                common_idx.append(match_obj.queryIdx)

        diff_idx = np.logical_not(np.isin(np.arange(len(kp_new)), common_idx))
        for p in kp_new[diff_idx]:
            diff_pts.append(p)

        common_pts = np.array(common_pts)
        diff_pts = np.array(diff_pts)

        plt.figure()
        plt.imshow(context.target, cmap='gray')
        plt.title("Sparse difference")
        plt.plot(kp_new[:,0], kp_new[:,1],'b.', markersize=3, label="All new projected markers")
        plt.plot(common_pts[:,0], common_pts[:,1],color="chartreuse", linestyle='None', marker="x", markersize=10, label="Common points")
        plt.plot(diff_pts[:,0], diff_pts[:,1],'rx', markersize=10, label="Different points")
        
        plt.legend()
        plt.show()
        return None