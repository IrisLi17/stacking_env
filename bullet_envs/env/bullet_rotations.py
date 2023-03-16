import numpy as np
import bullet_envs.env.rotations_c.rotations as rotations
import pybullet as p


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
    assert 1 + mat[0, 0] + mat[1, 1] + mat[2, 2] >= -1e-6, mat
    qr = 0.5 * np.sqrt(max(1 + mat[0, 0] + mat[1, 1] + mat[2, 2], 0.0))
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
    assert (np.all(abs(np.linalg.norm(q1) - 1) < 1e-5)), (np.linalg.norm(q1), q1)
    assert (np.all(abs(np.linalg.norm(q2) - 1) < 1e-5)), (np.linalg.norm(q2), q2)
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

def quat2euler(q):
    '''
    :param q: [qi, qj, qk, qr]
    :return: [alpha, beta, gamma] = [roll, pitch, yaw]
    '''
    return np.asarray(p.getEulerFromQuaternion(q))
    # # TODO: deal with singularities
    # qi, qj, qk, qr = q[0], q[1], q[2], q[3]
    # assert abs(qi ** 2 + qj ** 2 + qk ** 2 + qr ** 2 - 1) < 1e-5
    #
    # sqx = qi * qi
    # sqy = qj * qj
    # sqz = qk * qk
    # squ = qr * qr
    # sarg = -2 * (qi * qk - qr * qj)
    #
    # # If the pitch angle is PI / 2 or -PI / 2, there are infinite many solutions. We set roll = 0
    # if sarg <= -0.99999:
    #     roll = 0
    #     pitch = -0.5 * np.pi
    #     yaw = 2 * np.arctan2(qi, -qj)
    # elif sarg >= 0.99999:
    #     roll = 0
    #     pitch = 0.5 * np.pi
    #     yaw = 2 * np.arctan2(-qi, qj)
    # else:
    #     roll = np.arctan2(2 * (qj * qk + qr * qi), squ - sqx - sqy + sqz)
    #     pitch = np.arcsin(sarg)
    #     yaw = np.arctan2(2 * (qi * qj + qr * qk), squ + sqx - sqy - sqz)
    # return np.array([roll, pitch, yaw])


def euler2quat(euler):
    return np.asarray(p.getQuaternionFromEuler(euler))
    # alpha, beta, gamma = euler[0], euler[1], euler[2]
    # qi = np.sin(alpha / 2) * np.cos(beta / 2) * np.cos(gamma / 2) - np.cos(alpha / 2) * np.sin(beta / 2) * np.sin(gamma / 2)
    # qj = np.cos(alpha / 2) * np.sin(beta / 2) * np.cos(gamma / 2) + np.sin(alpha / 2) * np.cos(beta / 2) * np.sin(gamma / 2)
    # qk = np.cos(alpha / 2) * np.cos(beta / 2) * np.sin(gamma / 2) - np.sin(alpha / 2) * np.sin(beta / 2) * np.cos(gamma / 2)
    # qr = np.cos(alpha / 2) * np.cos(beta / 2) * np.cos(gamma / 2) + np.sin(alpha / 2) * np.sin(beta / 2) * np.sin(gamma / 2)
    # assert abs(qi ** 2 + qj ** 2 + qk ** 2 + qr ** 2 - 1) < 1e-5
    # return np.array([qi, qj, qk, qr])

def is_rotation_mat(mat):
    if np.all(np.abs(mat.transpose() @ mat - np.eye(3)) < 1e-4) and np.all(np.abs(mat @ mat.transpose() - np.eye(3)) < 1e-4):
        return True
    print(mat.transpose() @ mat, mat @ mat.transpose())
    return False


def quat_rot_vec_py(q, v0):
    if not isinstance(q, np.ndarray):
        q = np.array(q)
    v0_norm = np.linalg.norm(v0)
    q_v0 = np.array([v0[0], v0[1], v0[2], 0]) / v0_norm
    q_v = quat_mul_py(q, quat_mul_py(q_v0, quat_conjugate(q)))
    v = q_v[:-1] * v0_norm
    return v


def quat_rot_vec(q, v0):
    if not isinstance(q, np.ndarray):
        q = np.array(q)
    v0_norm = np.linalg.norm(v0)
    q_v0 = np.array([v0[0], v0[1], v0[2], 0]) / v0_norm
    q_v = quat_mul(q, quat_mul(q_v0, quat_conjugate(q)))
    v = q_v[:-1] * v0_norm
    return v
