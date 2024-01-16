from typing import Tuple, List, TypeAlias
import numpy as np
import cv2

from detective.pipeline import Pipeline, Stage
from detective.logger import Logger

# type signatures
_ImageList : TypeAlias = List[np.array]
_TaggedImageList : TypeAlias = List[Tuple[str, np.array]]
_CSInput : TypeAlias = Tuple[_TaggedImageList, _TaggedImageList, np.array]
_CSOutput : TypeAlias = Tuple[_ImageList, np.array]

class CalibrationStage(Stage):

    def __repr__(self):
        return f"CalibrationStage : (Calibration, Photos, Target) -> (Photos, Target)"
    
    def __calibrate(self, cal_images: _TaggedImageList, input_images : _TaggedImageList, context : Pipeline):
        # parameters of the camera calibration pattern
        pattern_num_rows = 9
        pattern_num_cols = 6
        pattern_size= (pattern_num_rows, pattern_num_cols)

        #mobile phone cameras can have a very high resolution.
        # It can be reduced to reduce the computing overhead
        image_downsize_factor = 4

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((pattern_num_rows*pattern_num_cols,3), np.float32)
        objp[:,:2] = np.mgrid[0:pattern_num_rows,0:pattern_num_cols].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        for img_tup in cal_images:
            name, img = img_tup
            img_rows = img.shape[1]
            img_cols = img.shape[0]
            new_img_size = (int(img_rows / image_downsize_factor), int(img_cols / image_downsize_factor))
            img = cv2.resize(img, new_img_size, interpolation = cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                #corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)

                # Draw and display the corners
                # cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
                # cv2.imshow('img', img)
                # cv2.waitKey(500)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=(cv2.CALIB_ZERO_TANGENT_DIST))

        # reprojection error for the calibration images
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        Logger.info("Total error: {}".format(mean_error / len(objpoints)))

        # undistorting the images
        cal_input_images = []
        for img_tup in input_images:
            name, img = img_tup
            img_rows = img.shape[1]
            img_cols = img.shape[0]
            new_img_size = (int(img_rows / image_downsize_factor), int(img_cols / image_downsize_factor))
            img = cv2.resize(img,new_img_size, interpolation = cv2.INTER_CUBIC)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            Logger.info(f"Undistorting image '{name}'")
            undist_image = cv2.undistort(img, mtx, dist)
            cal_input_images.append(undist_image)

        # update context
        context.input_calib = mtx
        context.input_dist = dist

        np.savetxt("K_c.txt", context.input_calib)
        np.savetxt("dist.txt", context.input_dist)

        return cal_input_images


    def run(self, input: _CSInput, context : Pipeline) -> _CSOutput:
        cal_images, input_images, target_image = input

        cal_input_images = self.__calibrate(cal_images, input_images, context)

        self.images = cal_input_images
        self.target = target_image

        if super().has_callback():
            super().get_callback()(cal_input_images)

        return cal_input_images, target_image