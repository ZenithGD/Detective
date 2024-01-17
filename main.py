import argparse
import json
import logging

import cv2

from lightglue import viz2d
import matplotlib.pyplot as plt

from detective.logger import Logger, ANSIFormatter
from detective.pipeline import Pipeline, CalibrationStage, SFMStage, MatchingStage, DiffStage
from detective.pipeline.matching import ExtractorType
from detective.utils.plot import *
from detective.utils.spatial import *

def parse_arguments():
    parser = argparse.ArgumentParser(description="Detective: A tool for inferring camera pose and changes in the scene from a set of pictures")
    parser.add_argument("-c", "--config", help="Config file, in JSON format.")
    parser.add_argument("-k", "--calibration", help="Calibration path. Will override config value, if present.")
    parser.add_argument("-p", "--photos", help="Photos. Will override config value, if present.")
    parser.add_argument("-t", "--target", help="Target photo. Will override config value, if present.")
    parser.add_argument(
        "--match-threshold", 
        help="threshold of matches to discard an image. Defaults to 50", 
        type=int, 
        default=50)
    
    parser.add_argument(
        "--common-threshold", 
        help="threshold of common matches with other images to discard an image. Defaults to 40", 
        type=int, 
        default=40)

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
    match_thresh = args.match_threshold
    common_thresh = args.common_threshold

    if args.config:
        with open(args.config) as conf:
            json_conf = json.load(conf)

            if "calibration" in json_conf:
                cal_path = json_conf["calibration"]

            if "photos" in json_conf:
                photo_path = json_conf["photos"]

            if "target" in json_conf:
                target_path = json_conf["target"]

            if "match_threshold" in json_conf:
                match_thresh = int(json_conf["match_threshold"])

            if "common_threshold" in json_conf:
                common_thresh = int(json_conf["common_threshold"])

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
    Logger.info(f"Match threshold: {match_thresh}")
    Logger.info(f"Common threshold: {common_thresh}")

    return cal_path, photo_path, target_path, match_thresh, common_thresh
    

def matching_callback(image_tensors, target_tensor, image_keypoints, target_keypoints):

    for i in range(1, len(image_keypoints)):
        axes = viz2d.plot_images([image_tensors[0], image_tensors[i-1]])

        mkp0, mkp1 = image_keypoints[0], image_keypoints[i-1]
        viz2d.plot_matches(mkp0, mkp1, color="lime", lw=0.2)
        viz2d.add_text(0, f"{mkp0.shape[0]} common matches")
        viz2d.save_plot(f"match0-{i}.png")
        plt.show()

def sfm_callback(imgs, target, keypoints, points3d, poses):
    ax_ini = plot_3dpoints(
        refs=[ p for p in poses],
        points=[points3d],
        ref_labels=[ f"C{i + 1}" for i in range(len(poses))],
        point_labels=["Initial triangulation"])
    
    ax_ini.set_title("Initial estimation")
    plt.show()


    for i, p in enumerate(poses):
        fig, ax = plt.subplots()
        K_c = np.loadtxt("K_c.txt")
        # find projection matrix of a camera
        P = create_P(K_c, p)
        xi_proj = P @ points3d.T
        xi_proj /= xi_proj[2]
        
        plot_image_residual(ax, imgs[i], keypoints[i].T, xi_proj[:2])
        ax.set_title(f"residuals 0")

    plt.show()

def sfm_callback(imgs, target, keypoints, points3d, poses):
    ax_ini = plot_3dpoints(
        refs=poses,
        points=[],
        ref_labels=[ f"C{i}" for i in range(len(poses))],
        point_labels=[])
    
    ax_ini.set_title("Initial estimation")
    plt.show()

    K_c = np.loadtxt("K_c.txt")
    for i, p in enumerate(poses):
        # pose i corresponds to camera i+1's pose with respect to camera 1

        # find projection matrix of a camera
        P = create_P(K_c, p)
        xi_proj = P @ points3d
        xi_proj /= xi_proj[2]
        
        fig, ax = plt.subplots()
        plot_image_residual(ax, imgs[i], keypoints[i].T, xi_proj[:2])
        ax.set_title(f"residuals {i}")

    fig, ax = plt.subplots()
    ax.imshow(imgs[0])
    plotNumberedImagePoints(keypoints[1], 'g', (5,5))
    plt.show()

def main(args):
    # setup logging
    setup_logging()

    # get all properties
    cal_path, photo_path, target_path, match_thresh, common_thresh = get_properties(args) 

    # run pipeline
    dp = Pipeline([
        CalibrationStage(),
        MatchingStage(
            #callback=matching_callback, 
            match_thresh=match_thresh,
            common_thresh=common_thresh, 
            extractor_type=ExtractorType.SUPERPOINT),
        SFMStage(
            callback=sfm_callback
        ),
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