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

class MatchingStage(Stage):

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
        return f"MatchingStage : (Photos, Target) -> (Photos, Target, points)"

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
        target_tensor = numpy_image_to_torch(target_image).cuda()
        sel = []

        #Get keypoints of target image
        feats_target = self.extractor.extract(target_tensor)

        common_ids : list = None
        match_list = []
        feats_list = []
        
        feats_image_0 = self.extractor.extract(tensor_imgs[0])

        # match image 0 with target
        Logger.info(f"Matching image 0 to target")
        # matches from keypoints for target and image i
            # match the features
        matches01 = self.matcher({'image0': feats_image_0, 'image1': feats_target})
        _, _, matches01 = [rbd(x) for x in [feats_image_0, feats_target, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)

        n_matches = matches.shape[0]
        Logger.info(f"Found {n_matches} matches.")

        # start with matches of the first new image with the target
        common_ids = matches[:, 0].cpu().numpy()
    
        for i in range(1, len(tensor_imgs)):

            Logger.info(f"Matching image 0 to image {i}")
            # matches from keypoints for target and image i

            feats_image_i = self.extractor.extract(tensor_imgs[i])
             # match the features
            matches01 = self.matcher({'image0': feats_image_0, 'image1': feats_image_i})
            _, _, matches01 = [rbd(x) for x in [feats_image_0, feats_image_i, matches01]]  # remove batch dimension
            matches = matches01['matches']  # indices with shape (K,2)

            n_matches = matches.shape[0]
            Logger.info(f"Found {n_matches} matches.")

            if n_matches >= self.match_thresh:

                tmp_common_ids = list(filter(lambda x : x in common_ids, matches[:, 0].cpu().numpy()))
                
                if len(tmp_common_ids) >= self.common_thresh:
                    sel.append(i)
                    common_ids = tmp_common_ids
                    feats_list.append(feats_image_i)
                    match_list.append(matches.cpu().numpy())
                    
                    Logger.info(f"{len(common_ids)} common matches between all images so far.")
                else:
                    Logger.warning(f"Discarding image {i} due to few common matches ({len(tmp_common_ids)} < {self.common_thresh})")
            else:
                Logger.warning(f"Discarding image {i} due to few matches ({n_matches} < {self.match_thresh})")
        
        # find common points
        common_ids_set = set(common_ids)
        for i in range(len(match_list)):
            match_list[i] = np.array(list(filter(lambda x: x[0] in common_ids_set, match_list[i])))

        tensor_imgs_sel = [ tensor_imgs[i] for i in sel ]
        # translate matches into keypoint pairs
        kp_target = feats_target['keypoints'].cpu().numpy().squeeze()
        all_keypoints = [  ]

        for i, mp in enumerate(match_list):
            kp_i = feats_list[i]['keypoints'].cpu().numpy().squeeze()
            points1 = kp_i[mp[..., 1]]
            all_keypoints.append(points1)

        # set selection for the future
        self.selection = sel
        return tensor_imgs_sel, target_tensor, all_keypoints, kp_target
          
    def run(self, input: _MSInput, context : Pipeline) -> _MSOutput:
        cal_input_images, target_image = input

        tensor_images, target_tensor, img_keypoints, target_keypoints = self.__match(cal_input_images, target_image, context)

        if super().has_callback():
            super().get_callback()(tensor_images, target_tensor, img_keypoints, target_keypoints)

        return [ cal_input_images[i] for i in self.selection ], target_image, img_keypoints, target_keypoints