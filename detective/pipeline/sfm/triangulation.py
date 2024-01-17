from detective.logger import Logger
from detective.utils.plot import *
from detective.utils.spatial import *
import scipy as sc
import scipy.optimize as scOptim

import cv2

from itertools import chain

def triangulate(x1, x2, P_c1, P_c2):
    points3d = []
    for i in range(x1.shape[0]):
        match_1 = x1[i,:]
        match_2 = x2[i,:]
        eq1 = create_tri_eq(P_c1, match_1)
        eq2 = create_tri_eq(P_c2, match_2)

        eq = np.concatenate([eq1, eq2], axis=0)

        _, _, V = np.linalg.svd(eq)
    
        points3d.append(V[-1] / V[-1, 3])
    return np.array(points3d)


def pointsInFront(x1, x2, R, t, K1, K2): 
    Icanon = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    t2D = t.reshape(t.shape[0], 1)

    P1 = K1 @ Icanon
    P2 = K2 @ np.concatenate([R, t2D], axis=1)

    points3d = triangulate(x1, x2, P1, P2)

    infront = 0

    tr_w_c2 = create_T(R, t)
    for point3d in points3d:
        if point3d[2] < 0:
            continue
        
        # z > 0 in C1 frame (assuming it's in the origin)
        pointFrame2 = np.squeeze(tr_w_c2 @ point3d.reshape(-1, 1))
        pointFrame2 /= pointFrame2[3]

        if pointFrame2[2] > 0:
            infront += 1

    return points3d, infront

def getAllPossibleT(x1, x2, K1, K2, E : np.array):
    U, S, V = np.linalg.svd(E)
    t = U[:, 2] 

    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])

    uwvt = U @ W @ V
    uwvtn = -U @ W @ V

    R_p90_t = uwvt if np.linalg.det(uwvt) > 0 else uwvtn

    uwtvt = U @ W.T @ V
    uwtvtn = -U @ W.T @ V
    R_n90_t = uwtvt if np.linalg.det(uwtvt) > 0 else uwtvtn

    # 4 solutions
    sols = [ (R_p90_t, t), (R_p90_t, -t), (R_n90_t, t), (R_n90_t, -t) ]

    return [create_T(R, t) for R, t in sols]

def getPose(x1, x2, K1, K2, E : np.array):
    U, S, V = np.linalg.svd(E)
    t = U[:, 2] 

    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])

    uwvt = U @ W @ V
    uwvtn = -U @ W @ V

    R_p90_t = uwvt if np.linalg.det(uwvt) > 0 else uwvtn

    uwtvt = U @ W.T @ V
    uwtvtn = -U @ W.T @ V
    R_n90_t = uwtvt if np.linalg.det(uwtvt) > 0 else uwtvtn

    # 4 solutions
    sols = [ (R_p90_t, t), (R_p90_t, -t), (R_n90_t, t), (R_n90_t, -t) ]
    points3d_all, points_in_front = list(zip(*[ pointsInFront(x1, x2, R, t, K1, K2) for (R, t) in sols ]))

    bidx = np.argmax(points_in_front)
    R, t = sols[bidx]
    points3d = points3d_all[bidx]
    
    return points3d, create_T(R, t)

def resBundleProjection(Op, data, K_c, nPoints, nCams):
    """
    -input:
        Op: Optimization parameters: this must include a
        paramtrization for T_i1 (reference 1 seen from reference i) list [X3D1, X3D2, ..., X3DnPoints,tx1, ty1, tz1, theta_x1, theta_y1, theta_z1, ...]
        in a proper way and for X1 (3D points in ref 1)
        data: list((2xnPoints)) list of 2D points on image i (w/o homogeneous
        coordinates)
        K_c: (3x3) Intrinsic calibration matrix for all cameras
        nPoints: Number of points 
        nCams: Number of cameras (including camera 1)
    -output: 
        res: residuals from the error between the 2D matched points and the 
        projected points from the 3D points (2 equations/residuals per 2D point) 
    """

    points3d = np.array(Op[0:3 * nPoints])
    opt_params = np.array(Op[3 * nPoints:])

    projs = []
    res = []
    params = [np.eye(4)]

    X1_3d = np.concatenate([points3d.reshape(3, nPoints), np.ones((1, nPoints))])
    for i in range(0, nCams - 1):
        # 6 parameters per camera
        params_i = opt_params[6 * i: 6 * (i + 1)]
        t = params_i[:3]
        R = sc.linalg.expm(crossMatrix(params_i[3:]))

        Ti1 = create_T(R, t)
        params.append(Ti1)

    for i in range(nCams):
        Pi = create_P(K_c, params[i])
        xi_proj = Pi @ X1_3d
        xi_proj /= xi_proj[2]

        projs.append(xi_proj[:2])

        res_i = xi_proj[:2].flatten() - data[i].flatten()
        res.append(res_i)

    return np.concatenate(res)

def BA_optimize(points3d, poses, keypoints, K_c):
    """Optimize points in 3d and poses with multiview BA.

    Args:
        points3d (np.array): _description_
        poses (list): _description_
        keypoints (list): _description_
    """

    vec3d = points3d[:3].flatten()

    pose_tup = [ pose_from_T(np.linalg.inv(pose)) for pose in poses ]
    pose_list = []
    for R, t in pose_tup:
        pose_list.append(crossMatrixInv(sc.linalg.logm(R)))
        pose_list.append(t)

    # first solution [X3D1, X3D2, ..., X3DnPoints,tx, ty, tz, theta_x, theta_y, theta_z]
    Op = np.concatenate([ vec3d ] + pose_list).astype(float)

    npoints = points3d.shape[1]
    ncams = len(poses)

    Logger.info(f"Multiview BA ({ncams} views) for {npoints} points")

    # Optimization with L2 norm and Levenberg-Marquardt
    OpOptim = scOptim.least_squares(
        resBundleProjection, Op, 
        args=([ kp.T for kp in keypoints ], K_c, npoints, ncams), 
        method='trf', jac='3-point', loss='huber',
        verbose=2)

    print("Plotting 3d BA-refined reconstruction")

    p3d_ba = np.concatenate([OpOptim.x[: 3 * npoints].reshape(3, -1), np.ones((1, npoints))])
    transf = [ np.eye(4) ]
    ba_params = OpOptim.x[3 * npoints:]
    for i in range(ncams - 1):
        ba_params_i = ba_params[6 * i: 6 * (i + 1)]
        t_ba = ba_params_i[:3].reshape(1, 3)
        th_ba = ba_params_i[3:]
        R_ba = sc.linalg.expm(crossMatrix(th_ba))
        transf.append(create_T(R_ba, t_ba))

    ax_ba = plot_3dpoints(
        refs=transf, 
        points=[points3d.T, p3d_ba.T],
        ref_labels=[ f"C{i}" for i in range(len(poses))],
        point_labels=["Initial triangulation", "BA optimized"])
    
    plt.show()
    
    return transf, p3d_ba.T

def DLT_eq(point3d, point2d):

    X, Y, Z, W = point3d
    x, y = point2d

    return np.array([
        [-X, -Y, -Z, -W,  0,  0,  0,  0, x * X, x * Y, x * Z, x * W],
        [ 0,  0,  0,  0, -X, -Y, -Z, -W, y * X, y * Y, y * Z, y * W]
    ])

def DLT_pose(points3d, kp_old):

    # create equations
    DLT_eq_mat = np.concatenate(
        [ DLT_eq(points3d[i], kp_old[i]) for i in range(points3d.shape[0]) ], 
        axis = 0)
    
    U, S, V = np.linalg.svd(DLT_eq_mat)
    P = V[-1].reshape(3, 4)

    # decompose matrix
    M = P[:, :3]
    P_signed = P * np.sign(np.linalg.det(M))

    out = cv2.decomposeProjectionMatrix(P_signed)
    K, R, t = out[0], out[1], out[2]

    R = np.linalg.inv(R)
    t = np.linalg.inv(P[:3, :3]) @ P[:3, 3]
    t = t.ravel() / t[-1]
    return K, create_T(R, t[:3]), P