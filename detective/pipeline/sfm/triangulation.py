from detective.utils.spatial import *

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