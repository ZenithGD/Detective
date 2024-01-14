from typing import Tuple, List, TypeAlias
import numpy as np
import cv2

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import numpy_image_to_torch, rbd
from lightglue import viz2d
from lightglue import match_pair

from detective.pipeline import Pipeline, Stage
from detective.logger import Logger

from enum import Enum

# type signatures
_ImageList : TypeAlias = List[np.array]
_KeypointMatch : TypeAlias = List[Tuple[np.array, np.array]]
_MSInput : TypeAlias = Tuple[_ImageList, np.array]
_MSOutput : TypeAlias = Tuple[_ImageList, np.array, _KeypointMatch]

class ExtractorType(Enum):
    SUPERPOINT = 0,
    DISK = 1,
    ALIKED = 2,
    SIFT = 3

class MatchingStage(Stage):

    def __init__(self, callback = None, extractor_type = ExtractorType.SUPERPOINT):
        # Initialize parent class
        super().__init__(callback)

        # extractor
        match extractor_type:
            case ExtractorType.SUPERPOINT:
                self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
            case ExtractorType.DISK:
                self.extractor = DISK(max_num_keypoints=2048).eval().cuda()
            case ExtractorType.ALIKED:
                self.extractor = ALIKED(max_num_keypoints=2048).eval().cuda()
            case ExtractorType.SIFT:
                self.extractor = SIFT(max_num_keypoints=2048).eval().cuda()

        # matcher is always lightglue
        self.matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher


    def __repr__(self):
        return f"MatchingStage : (Photos, Target) -> (Photos, Target, Matches)"
    
    def __compute_matches(self, image0, image1):

        # match pair of images
        feats0, feats1, matches01 = match_pair(self.extractor, self.matcher, image0, image1)

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        
        Logger.info(f"Found {matches.shape[0]} matches.")
        return kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    def __match(self, input_images : _ImageList, target_image : np.array, context : Pipeline) -> _KeypointMatch:
        """Match images and return list of matches.
        Each list of matches is given by a tensor of shape (M, 2)
        Each match is a pair of keypoints [idx0, idx1].

        Args:
            input_images (_ImageList): New image list 
            target_image (np.array): The target image
            context (Pipeline): The pipeline context

        Returns:
            List[(np.array, np.array)]: The list of matches in 2d image space.
        """

        tensor_imgs = [ numpy_image_to_torch(target_image).cuda() ] + [ numpy_image_to_torch(img).cuda() for img in input_images ]

        match_list = []
        for i in range(len(tensor_imgs) - 1):
            if i == 0:
                Logger.info(f"Matching target image to image {i + 1}")
            else:
                Logger.info(f"Matching image {i} to image {i + 1}")
                
            match_list.append(self.__compute_matches(tensor_imgs[i], tensor_imgs[i + 1]))

        return tensor_imgs, match_list

    def run(self, input: _MSInput, context : Pipeline) -> _MSOutput:
        cal_input_images, target_image = input

        tensor_images, matches = self.__match(cal_input_images, target_image, context)

        if super().has_callback():
            super().get_callback()(tensor_images, matches)

        return matches