from typing import Tuple, List, TypeAlias
import numpy as np
import cv2

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import numpy_image_to_torch, rbd
from lightglue import viz2d, match_pair

from detective.pipeline import Pipeline, Stage
from detective.logger import Logger

import matplotlib.pyplot as plt

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

class FullMatchingStage(Stage):

    def __init__(self, callback = None, match_thresh = 40, common_thresh = 50, extractor_type = ExtractorType.SUPERPOINT):
        # Initialize parent class
        super().__init__(callback)

        self.match_thresh = match_thresh
        self.common_thresh = common_thresh

        # extractor
        match extractor_type:
            case ExtractorType.DISK:
                Logger.info("Setting type of extractor to DISK")
                self.extractor = DISK(max_num_keypoints=2048).eval().cuda()
                fst = 'disk'
            case ExtractorType.ALIKED:
                Logger.info("Setting type of extractor to ALIKED")
                self.extractor = ALIKED(max_num_keypoints=2048).eval().cuda()
                fst = 'aliked'
            case ExtractorType.SIFT:
                Logger.info("Setting type of extractor to SIFT")
                self.extractor = SIFT(max_num_keypoints=2048).eval().cuda()
                fst = 'sift'
            case _:
                Logger.info("Setting type of extractor to SUPERPOINT")
                self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
                fst = 'superpoint'

        # matcher is always lightglue
        self.matcher = LightGlue(features=fst).eval().cuda()  # load the matcher

    def __repr__(self):
        return f"MatchingStage : (Photos, Target) -> (images, target, image points, target points) "

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

        tensor_imgs = [ numpy_image_to_torch(img).cuda() for img in input_images ]
        tensor_target = numpy_image_to_torch(target_image).cuda()

        # extract local features
        feats_imgs = [ self.extractor.extract(ten) for ten in tensor_imgs ]
        feats_target = self.extractor.extract(tensor_target)

        # match features from image 0 to target
        matches01 = self.matcher({'image0': feats_imgs[0], 'image1': feats_target})
        f0, feats_target, matches01 = [rbd(x) for x in [feats_imgs[0], feats_target, matches01]]  # remove batch dimension

        matches_target = matches01['matches']
        
        matches_imgs = []
        kp_imgs = [ f0['keypoints'] ]
        # match the features from image 0 to i-th image
        for fi in feats_imgs[1:]:
            matches01 = self.matcher({'image0': feats_imgs[0], 'image1': fi})
            f0, fi, matches01 = [rbd(x) for x in [feats_imgs[0], fi, matches01]]  # remove batch dimension
            matches_imgs.append(matches01['matches'])  # indices with shape (K,2)
            kp_imgs.append(fi['keypoints'])
            #points0 = f0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
            #points1 = f1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
        
        kp_target = feats_target['keypoints']

        print(kp_target.shape)
        print(matches_target.shape)
        print(matches_imgs[0].shape)
        print(kp_imgs[0].shape)
        return tensor_imgs, tensor_target, kp_imgs, kp_target, matches_imgs, matches_target
          
    def run(self, input: _MSInput, context : Pipeline) -> _MSOutput:
        cal_input_images, target_image = input

        # set reference to matcher
        context.extractor = self.extractor
        context.matcher = self.matcher

        tensor_images, tensor_target, kp_imgs, kp_target, matches_imgs, matches_target = self.__match(cal_input_images, target_image, context)

        if super().has_callback():
            super().get_callback()(tensor_images, tensor_target, kp_imgs, kp_target, matches_imgs, matches_target)

        return (
            [ k.cpu().numpy().squeeze() for k in kp_imgs ], 
            kp_target.cpu().numpy().squeeze(), 
            [ m.cpu().numpy().squeeze() for m in matches_imgs ], 
            matches_target.cpu().numpy().squeeze()
        )