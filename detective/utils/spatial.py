import numpy as np

# create T matrix
def create_T(R_w_c, t_w_c) -> np.array:
    """
    create the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4), dtype=np.float32)
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c

def create_P(K, T):
    can_proj = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])
    P = K @ can_proj @ T
    return P


def create_F_eq(p0, p1):
    m0 = p0[0] * p1[0]
    m1 = p0[1] * p1[0]
    m2 = p1[0]
    m3 = p0[0] * p1[1]
    m4 = p0[1] * p1[1]
    m5 = p1[1]
    m6 = p0[0]
    m7 = p0[1]
    m8 = 1

    return np.array([m0, m1, m2, m3, m4, m5, m6, m7, m8])
    
def create_F_from_matches(matches1 : np.array, matches2 : np.array):
    """Create fundamental matrix from a set of matches.

    Args:
        matches1 (np.array): Points in first image space
        matches2 (np.array): Points in second image space
    """
    F_eq = np.concatenate(
        [ create_F_eq(matches1[i], matches2[i]).reshape(1, 9) for i in range(matches1.shape[0]) ],
        axis = 0)
    
    U, S, V = np.linalg.svd(F_eq)
    F = V[-1].reshape(3,3)
    U, S, V = np.linalg.svd(F)
    return U @ np.diag(np.array([S[0], S[1], 0])) @ V

def DLT_projection(matches1, matches2):
    pass

def create_tri_eq(P, point2d):
    mat00 = P[2][0] * point2d[0] - P[0][0]
    mat01 = P[2][1] * point2d[0] - P[0][1]
    mat02 = P[2][2] * point2d[0] - P[0][2]
    mat03 = P[2][3] * point2d[0] - P[0][3]
    
    mat10 = P[2][0] * point2d[1] - P[1][0]
    mat11 = P[2][1] * point2d[1] - P[1][1]
    mat12 = P[2][2] * point2d[1] - P[1][2]
    mat13 = P[2][3] * point2d[1] - P[1][3]
    
    mat = np.array([
        [mat00, mat01, mat02, mat03],
        [mat10, mat11, mat12, mat13]
    ])

    return mat

def pose_from_T(T):
    """Return rotation and translation from T matrix

    Args:
        T (np.array): Transformation matrix

    Returns:
        np.array, np.array: The rotation matrix and translation vector
    """
    t = T[0:3,3]
    R = T[0:3, 0:3]

    return R, t

def create_E(T):
    t = T[0:3,3]
    R = T[0:3, 0:3]
    tx = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    return tx @ R

def create_F_GT(T, K0, K1):
    E = create_E(T)
    K1invT = np.linalg.inv(K1).T
    K0inv = np.linalg.inv(K0)
    return K1invT @ E @ K0inv

def essential_from_F(F, K0, K1):
    return K0.T @ F @ K1

def crossMatrixInv(M):
    x = np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=np.float64)
    return x
    
def crossMatrix(x):
    M = np.array([
        [  0.0, -x[2],  x[1]],
        [ x[2],   0.0, -x[0]],
        [-x[1],  x[0],   0.0]
    ], dtype=float)
    return M