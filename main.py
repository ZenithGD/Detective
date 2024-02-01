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
    

def matching_callback(tensor_images, tensor_target, kp_imgs, kp_target, matches_imgs, matches_target):

    for i in range(1, len(kp_imgs)):
        axes = viz2d.plot_images([tensor_images[0], tensor_images[i]])

        mkp0, mkp1 = kp_imgs[0], kp_imgs[i]
        mkp0 = mkp0[matches_imgs[i-1][..., 0]]
        mkp1 = mkp1[matches_imgs[i-1][..., 1]]

        viz2d.plot_matches(mkp0, mkp1, color="lime", lw=0.2)
        viz2d.add_text(0, f"{mkp0.shape[0]} common matches")
        viz2d.save_plot(f"match0-{i}.png")
        plt.show()

    axes = viz2d.plot_images([tensor_images[0], tensor_target])

    mkp0, mkp1 = kp_imgs[0], kp_target
    mkp0 = mkp0[matches_target[..., 0]]
    mkp1 = mkp1[matches_target[..., 1]]
    viz2d.plot_matches(mkp0, mkp1, color="lime", lw=0.2)
    viz2d.add_text(0, f"{mkp0.shape[0]} common matches")
    viz2d.save_plot(f"match0-target.png")
    plt.show()

def sfm_callback(imgs, target, keypoints, target_keypoints, points3d, poses, old_pose, old_proj):
    ax_ini = plot_3dpoints(
        refs=poses + [ old_pose ],
        points=[points3d],
        ref_labels=[ f"C{i}" for i in range(len(poses))] + [ "old" ],
        point_labels=["3d sparse reconstruction"])
    
    ax_ini.set_title("Initial estimation")
    plt.show()

    np.savetxt("points3d.txt", points3d)
    for i, p in enumerate(poses):
        np.savetxt(f"pose-{i}.txt", p)

    K_c = np.loadtxt("K_c.txt")
    for i, p in enumerate(poses):
        # pose i corresponds to camera i+1's pose with respect to camera 1

        # find projection matrix of a camera
        P = create_P(K_c, p)
        xi_proj = P @ points3d.T
        xi_proj /= xi_proj[2]
        
        fig, ax = plt.subplots()
        plot_image_residual(ax, imgs[i], keypoints[i].T, xi_proj[:2])
        ax.set_title(f"residuals {i}")
        
        dists = np.linalg.norm(keypoints[i].T - xi_proj[:2], axis=0)
        rmse = np.sqrt(np.mean(np.square(dists)))
        Logger.info(f"RMSE of residuals for camera {i} = {rmse}")
    
    # # find projection matrix of a camera
    # xi_proj = old_proj @ points3d.T
    # xi_proj /= xi_proj[2]
    
    # fig, ax = plt.subplots()
    # plot_image_residual(ax, target, target_keypoints.T, xi_proj[:2])
    # ax.set_title(f"residuals target")
    # dists = np.norm(target_keypoints.T - xi_proj[:2], axis=0)
    # rmse = np.sqrt(np.mean(np.square(dists)))
    # Logger.info(f"RMSE of residuals for target camera = {rmse}")
    
    plt.show()

def main(args):
    # setup logging
    setup_logging()

    # get all properties
    cal_path, photo_path, target_path, match_thresh, common_thresh, extractor = get_properties(args) 

    # run pipeline
    dp = Pipeline([
        CalibrationStage(),
        FullMatchingStage(
            callback=matching_callback, 
            match_thresh=match_thresh,
            common_thresh=common_thresh, 
            extractor_type=extractor),
        SFMStage(
            full_ba=False,
            callback=sfm_callback
        ),
    ])

    print(dp)
    
    diffs = dp.run(
        cal_path=cal_path,
        photo_path=photo_path,
        target_path=target_path,
    )

if __name__ == "__main__":
    args = parse_arguments()

    main(args)