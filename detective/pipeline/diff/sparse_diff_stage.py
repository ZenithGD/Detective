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
        kp_new, kp_old = input

        # extract SIFT descriptors for comparison
        sift = cv2.SIFT_create()

        kp_new_cv = [cv2.KeyPoint(x[0], x[1], 64) for x in kp_new]
        kp_old_cv = [cv2.KeyPoint(x[0], x[1], 64) for x in kp_old]

        _, ds_new = sift.compute(context.images[0], kp_new_cv)
        _, ds_old = sift.compute(context.target, kp_old_cv)

        # # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(ds_new,ds_old,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = cv2.DrawMatchesFlags_DEFAULT)
        img3 = cv2.drawMatchesKnn(context.images[0],kp_new_cv,context.target,kp_old_cv,matches,None,**draw_params)
        plt.imshow(img3)
        plt.show()
        return None