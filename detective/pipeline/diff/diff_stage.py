from typing import Tuple, List, TypeAlias
import numpy as np
import cv2

import matplotlib.pyplot as plt

from detective.pipeline import Pipeline, Stage

_ImageList : TypeAlias = List[np.array]
_SFMInput : TypeAlias = Tuple[np.array, np.array]
_SFMOutput : TypeAlias = Tuple[np.array, np.array, np.array]

class DiffStage(Stage):

    def __repr__(self):
        return f"DiffStage : (images, target, image points, target points) -> (diffs)"

    def run(self, input: _SFMInput, context : Pipeline) -> _SFMOutput:
        images, target_image, keypoints, target_keypoints = input

        target_image = cv2.bilateralFilter(target_image,9,75,75)

        for i in range(len(images)):
            new_img = cv2.bilateralFilter(images[i],9,75,75)

            src_pts = np.float32(keypoints[i]).reshape(-1,1,2)
            dst_pts = np.float32(target_keypoints).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

            h,w = target_image.shape[:2]

            im_dst = cv2.warpPerspective(new_img, M, (w, h))

            ret, thresh_old = cv2.threshold(target_image,0,255,cv2.THRESH_OTSU)
            ret, thresh_new = cv2.threshold(im_dst,0,255,cv2.THRESH_OTSU)

            thresh_new = thresh_new.astype(np.int8)
            thresh_old = thresh_old.astype(np.int8)

            diff = (thresh_new - thresh_old)
            seg_labels = np.where(diff < 0, 2, diff)

            colours = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0]])
            
            fig, ax = plt.subplots(1, 3)
            ax[0].set_title("target otsu binarization")
            ax[0].imshow(thresh_old,cmap='hot')
            ax[1].set_title("warped new otsu binarization")
            ax[1].imshow(thresh_new,cmap='hot')
            ax[2].set_title("difference")
            ax[2].imshow(colours[seg_labels],cmap='hot')

        plt.show()
        return np.zeros((10, 10))