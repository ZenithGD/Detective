from __future__ import annotations
from abc import ABC, abstractmethod 
from typing import List

from .calib import *
from .sfm import *

from detective.utils.images import *

class Stage(ABC):
    """Base stage class. A pipeline is composed of many stages,
    and each stages takes some input, which is feeded into the
    next stage.
    """
    def __init__(self, callback = None):
        self.__callback = callback

    def has_callback(self):
        return not self.__callback is None
    
    def get_callback(self):
        return self.__callback

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def run(self, input : object, context : Pipeline) -> object:
        pass

class Pipeline:
    def __init__(self, stages : list):
        self.__stages : List[Stage] = stages

        # context
        self.input_calib = None
        self.input_dist = None

    def __repr__(self) -> str:

        return (
            f'Pipeline layout\n'
            '===================================\n'
            f"{os.linesep.join([ repr(stage) for stage in self.__stages ])}\n"
            '===================================\n'
        )

    def run(self, cal_path : str, photo_path : str, target_path : str):
        
        # load images
        cal_images = read_images(cal_path)
        photo_images = read_images(photo_path)
        target_image = read_image(target_path)

        current = (cal_images, photo_images, target_image)

        for stage in self.__stages:
            output = stage.run(current, self)
            current = output

        return current