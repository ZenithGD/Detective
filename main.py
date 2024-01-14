import argparse
import json
import logging

from lightglue import viz2d
import matplotlib.pyplot as plt

from detective.logger import Logger, ANSIFormatter

from detective.pipeline import Pipeline, CalibrationStage, SFMStage, MatchingStage, DiffStage
from detective.pipeline.matching import ExtractorType

def parse_arguments():
    parser = argparse.ArgumentParser(description="Detective: A tool for inferring camera pose and changes in the scene from a set of pictures")
    parser.add_argument("-c", "--config", help="Config file, in JSON format.")
    parser.add_argument("-k", "--calibration", help="Calibration path. Will override config value, if present.")
    parser.add_argument("-p", "--photos", help="Photos. Will override config value, if present.")
    parser.add_argument("-t", "--target", help="Target photo. Will override config value, if present.")

    return parser.parse_args()

def setup_logging():
    # Setup logger
    logger = logging.getLogger("detective")
    logger.setLevel(logging.DEBUG)
    
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(ANSIFormatter())

    logger.addHandler(ch)

    Logger.initialize(logger)
    
def get_properties(args):

    cal_path = None
    photo_path = None
    target_path = None

    if args.config:
        with open(args.config) as conf:
            json_conf = json.load(conf)

            if "calibration" in json_conf:
                cal_path = json_conf["calibration"]

            if "photos" in json_conf:
                photo_path = json_conf["photos"]

            if "target" in json_conf:
                target_path = json_conf["target"]

    if args.calibration:    
        cal_path = args.calibration
        Logger.warning("Calibration path specified on CLI")
    
    if args.photos:
        photo_path = args.photos
        Logger.warning("Photo path specified on CLI")
        
    if args.target:
        target_path = args.target
        Logger.warning("Target photo path specified on CLI")

    # check that all params were specified
    if cal_path is None:
        Logger.error("Calibration path was not specified, exiting...")
        exit(1)

    if photo_path is None:
        Logger.error("Photo path was not specified, exiting...")
        exit(1)

    if target_path is None:
        Logger.error("Target photo path was not specified, exiting...")
        exit(1)

    Logger.info(f"Calibration path: '{cal_path}'")
    Logger.info(f"Photo path: '{photo_path}'")
    Logger.info(f"Target photo path: '{target_path}'")

    return cal_path, photo_path, target_path
    

def matching_callback(image_tensors, matches):

    for i in range(len(image_tensors) - 1):
        axes = viz2d.plot_images([image_tensors[i], image_tensors[i+1]])

        mkp0, mkp1 = matches[i]
        viz2d.plot_matches(mkp0, mkp1, color="lime", lw=0.2)
        plt.show()


def main(args):
    # setup logging
    setup_logging()

    # get all properties
    cal_path, photo_path, target_path = get_properties(args) 

    # run pipeline
    dp = Pipeline([
        CalibrationStage(),
        MatchingStage(callback=matching_callback, extractor_type=ExtractorType.SIFT),
        SFMStage(),
        DiffStage()
    ])

    print(dp)
    
    points3d_w, pose3d_w, diffs = dp.run(
        cal_path=cal_path,
        photo_path=photo_path,
        target_path=target_path,
    )

if __name__ == "__main__":
    args = parse_arguments()

    main(args)