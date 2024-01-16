from typing import Tuple, List, TypeAlias
import numpy as np
import cv2

from detective.pipeline import Pipeline, Stage

_ImageList : TypeAlias = List[np.array]
_SFMInput : TypeAlias = Tuple[np.array, np.array]
_SFMOutput : TypeAlias = Tuple[np.array, np.array, np.array]

class DiffStage(Stage):
    
    def __init__(self, full_ba : bool = True):
        self.full_ba = full_ba

    def __repr__(self):
        return f"DiffStage : (3d, pose) -> (3d, pose, diffs)"

    def run(self, input: _SFMInput, context : Pipeline) -> _SFMOutput:
        points3d, pose = input

        return points3d, pose, np.zeros((10, 10))