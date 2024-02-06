import argparse
import json
import logging

import cv2

from lightglue import viz2d
import matplotlib.pyplot as plt

from detective.logger import Logger, ANSIFormatter
from detective.pipeline import Pipeline, CalibrationStage, SFMStage, FullMatchingStage, DiffStage
from detective.pipeline.matching import ExtractorType
from detective.utils.plot import *
from detective.utils.spatial import *
from detective.utils.images import *

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
    extractor = ExtractorType.SUPERPOINT

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

            if "extractor" in json_conf:
                match json_conf["extractor"]:
                    case "ALIKED":
                        extractor = ExtractorType.ALIKED
                    case "DISK":
                        extractor = ExtractorType.DISK
                    case "SIFT":
                        extractor = ExtractorType.SIFT
                    case _:
                        extractor = ExtractorType.SUPERPOINT

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

    return cal_path, photo_path, target_path, match_thresh, common_thresh, extractor

def load_data(subf : str):
    
    # get points
    points3d_img = np.loadtxt(f"results/{subf}/points3d.txt")
    points3d_target = np.loadtxt(f"results/{subf}/points3d_target.txt")

    # poses
    pose_img = [
        np.loadtxt(f"results/{subf}/pose-0.txt"),
        np.loadtxt(f"results/{subf}/pose-1.txt"),
        np.loadtxt(f"results/{subf}/pose-2.txt")
    ]

    pose_target = np.loadtxt(f"results/{subf}/pose-target.txt")

    P_old = np.loadtxt(f"results/{subf}/P-old.txt")

    return points3d_img, points3d_target, pose_img, pose_target, P_old

def main(args):
    # setup logging
    setup_logging()

    # get all properties
    cal_path, photo_path, target_path, match_thresh, common_thresh, extractor = get_properties(args) 

    # read images
    dp = Pipeline([
        CalibrationStage()
    ])
    photo_images, target_image = dp.run(
        cal_path=cal_path,
        photo_path=photo_path,
        target_path=target_path,
    )

    # get subfolder data
    p3d_img, p3d_target, poses, pose_target, P_old = load_data("tri")

    # get keypoints
    kp_img = [
        np.loadtxt(f"results/kp-0.txt"),
        np.loadtxt(f"results/kp-1.txt"),
        np.loadtxt(f"results/kp-2.txt")
    ]
    kp_target = np.loadtxt(f"results/kp-target.txt")
    K_c = np.loadtxt("K_c.txt")

    parr = [ np.linalg.inv(p) for p in poses ] + [ np.linalg.inv(pose_target) ]
    ax_ini = plot_3dpoints(
        refs=parr,
        points=[p3d_img],
        ref_labels=[ f"C{i}" for i in range(len(poses))] + [ "old" ],
        point_labels=["3d sparse reconstruction"])
    
    ax_ini.set_title("Initial estimation")

    for i, p in enumerate(poses):
        # pose i corresponds to camera i+1's pose with respect to camera 1
        print(photo_images[i].shape)
        # find projection matrix of a camera
        P = create_P(K_c, p)
        xi_proj = P @ p3d_img.T
        xi_proj /= xi_proj[2]
        
        fig, ax = plt.subplots()
        plot_image_residual(ax, photo_images[i], kp_img[i].T, xi_proj[:2])
        ax.set_title(f"residuals {i}")
        
        dists = np.linalg.norm(kp_img[i].T - xi_proj[:2], axis=0)
        mde = np.mean(dists)
        Logger.info(f"MDE of residuals for camera {i} = {mde} pixels")

    # # find projection matrix of a camera
    pm = p3d_target
    xi_proj = P_old @ pm.T
    xi_proj /= xi_proj[2]
    
    fig, ax = plt.subplots()
    plot_image_residual(ax, target_image, kp_target.T, xi_proj[:2])
    ax.set_title(f"residuals target")
    dists = np.linalg.norm(kp_target.T - xi_proj[:2], axis=0)
    mde = np.mean(dists)
    Logger.info(f"MDE of residuals for camera {i} = {mde} pixels")
    
    plt.show()

if __name__ == '__main__':
    args = parse_arguments()

    main(args)