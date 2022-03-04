import numpy as np


def quat2mat(q):
    qi, qj, qk, qr = q[0], q[1], q[2], q[3]
    assert abs(qi ** 2 + qj ** 2 + qk ** 2 + qr ** 2 - 1) < 1e-5
    mat = np.zeros((3, 3))
    mat[0][0] = 1 - 2 * (qj * qj + qk * qk)
    mat[0][1] = 2 * (qi * qj - qk * qr)
    mat[0][2] = 2 * (qi * qk + qj * qr)
    mat[1][0] = 2 * (qi * qj + qk * qr)
    mat[1][1] = 1 - 2 * (qi * qi + qk * qk)
    mat[1][2] = 2 * (qj * qk - qi * qr)
    mat[2][0] = 2 * (qi * qk - qj * qr)
    mat[2][1] = 2 * (qj * qk + qi * qr)
    mat[2][2] = 1 - 2 * (qi * qi + qj * qj)
    return mat


def mat2quat(mat):
    if not isinstance(mat, np.ndarray):
        mat = np.array(mat)
    assert mat.shape == (3, 3)
    assert 1 + mat[0, 0] + mat[1, 1] + mat[2, 2] >= 0, mat
    qr = 0.5 * np.sqrt(1 + mat[0, 0] + mat[1, 1] + mat[2, 2])
    if qr > 1e-6:
        qi = (mat[2, 1] - mat[1, 2]) / (4 * qr)
        qj = (mat[0, 2] - mat[2, 0]) / (4 * qr)
        qk = (mat[1, 0] - mat[0, 1]) / (4 * qr)
    else:
        qi_square = (mat[0, 0] + 1) / 2
        qj_square = (mat[1, 1] + 1) / 2
        qk_square = (mat[2, 2] + 1) / 2
        qi = np.sqrt(qi_square)
        if mat[0, 1] > 0:
            qj = np.sqrt(qj_square)
        else:
            qj = -np.sqrt(qj_square)
        if mat[0, 2] > 0:
            qk = np.sqrt(qk_square)
        else:
            qk = -np.sqrt(qk_square)
    return np.array([qi, qj, qk, qr])


def quat_conjugate(q):
    if not isinstance(q, np.ndarray):
        q = np.array(q)
    inv_q = -q
    inv_q[..., -1] *= -1
    return inv_q


def quat_mul(q0, q1):
    if not isinstance(q0, np.ndarray):
        q0 = np.array(q0)
    if not isinstance(q1, np.ndarray):
        q1 = np.array(q1)
    assert q0.shape == q1.shape
    assert q0.shape[-1] == 4
    assert q1.shape[-1] == 4
    assert np.all(abs(np.linalg.norm(q0, axis=-1) - 1) < 1e-5)
    assert np.all(abs(np.linalg.norm(q1, axis=-1) - 1) < 1e-5)
    
    w0 = q0[..., 3]
    x0 = q0[..., 0]
    y0 = q0[..., 1]
    z0 = q0[..., 2]
    
    w1 = q1[..., 3]
    x1 = q1[..., 0]
    y1 = q1[..., 1]
    z1 = q1[..., 2]
    
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
    z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
    q = np.array([x, y, z, w])
    if q.ndim == 2:
        q = q.swapaxes(0, 1)
    assert q.shape == q0.shape
    return q
    
    
def quat_diff(q1, q2):
    '''
    q1 - q2
    :param q1:
    :param q2:
    :return:
    '''
    assert np.all(abs(np.linalg.norm(q1) - 1) < 1e-5)
    assert np.all(abs(np.linalg.norm(q2) - 1) < 1e-5)
    q1 = np.array(q1)
    q2 = np.array(q2)
    if q2[3] < 0:
        q2 *= -1
    if q1[3] < 0:
        q1 *= -1
    inv_q2 = quat_conjugate(q2)
    q_diff = quat_mul(q1, inv_q2)
    if q_diff[3] < 0:
        q_diff *= -1
    return q_diff