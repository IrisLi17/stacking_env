import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import os
import pybullet as p

from bullet_envs.env.primitive_stacking import ArmStackwLowPlanner


def log_state(state, space=24):
    state_str=""
    for i, s in enumerate(state):
        arr_str = "[ "
        for v in s:
            arr_str += "{:8>.3f}".format(v) + "  "
        arr_str += ']'
        state_str += "".join([" " for _ in range(space)]) + arr_str + "\n"
    return state_str

def compute_action(state, next_state):
    cur_pos = np.asarray(state['qpos'])
    nxt_pos = np.asarray(next_state['qpos'])
    pos_err = abs(cur_pos - nxt_pos).sum(axis=-1)
    block_id = np.argmax(pos_err)
    position = nxt_pos[block_id][:3]
    orientation = nxt_pos[block_id][3:]
    euler = np.asarray(p.getEulerFromQuaternion(orientation)) / (np.pi / 2)
    action = np.concatenate([[block_id], position, euler])
    action[3] += 0.01
    print("[DEBUG] pos_err={}".format(pos_err))
    return action

def test_armstack_low(_states_history, length=None, n_object=None):
    # load trajectory data
    fig, ax = plt.subplots(1, 1)
    os.makedirs("debug_tmp", exist_ok=True)

    # enviroment
    env = ArmStackwLowPlanner(
        n_object=n_object, reward_type="sparse", action_dim=7, generate_data=True, primitive=True,
        n_to_stack=np.array([[1, 2, 3]]), name="allow_rotation", 
        robot="xarm", invisible_robot=False, compute_path=True,
    )
    env.reset()
    if length is None:
        length = len(_states_history) - 1
    robot_state = env.robot.get_state()
    for i in range(length):
        action = None
        state = _states_history[i]
        next_state = _states_history[i + 1] if i + 1 < len(_states_history) else None

        if next_state is None:
            break

        action = compute_action(state, next_state)

        env.set_state({"objects": state}, set_robot_state=False)
        env.robot.set_state(robot_state)
        env._sim_until_stable()

        blocks_orn = []
        for j in range(n_object):
            pos, orn = env.p.getBasePositionAndOrientation(env.blocks_id[j])
            blocks_orn.append(orn)

        img = env.render(mode="rgb_array")
        plt.imsave("debug_tmp/tmp%d_a.png" % i, img)

        env.step(action)
        robot_state = env.robot.get_state()

        img = env.render(mode="rgb_array")
        plt.imsave("debug_tmp/tmp%d_b.png" % i, img)

        print(f'''At step {i}:
        state:\n{log_state(np.asarray(state['qpos']), 24)}
        object_orn:\n{log_state(np.asarray(blocks_orn), 24)}
        action={action}
        expected next_state:\n{log_state(np.asarray(next_state['qpos']), 24)}
        ground-truth next_state:\n{log_state(np.asarray(env.get_state()['objects']['qpos']), 24)}''')

        if next_state is not None:
            env.set_state({"objects": next_state}, set_robot_state=False)
            img = env.render(mode="rgb_array")
            plt.imsave("debug_tmp/tmp%d_c.png" % i, img)


if __name__ == "__main__":
    _states_history = [
        {'qpos': np.array([
            [ 4.34501553e-01,  9.43058518e-02,  7.59898826e-02, -6.36704107e-03,  7.07075171e-01,  6.37165515e-03, 7.07081018e-01],
            [ 4.84671661e-01,  7.85358590e-02,  1.75979315e-01, -3.22369385e-06,  8.23487197e-06,  1.85790643e-03, 9.99998274e-01],
            [ 5.35906241e-01,  7.84825423e-02,  7.59888060e-02, -7.05939383e-03,  7.07093422e-01,  7.05622384e-03, 7.07049692e-01]]), 
         'qvel': np.array([
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.]])
        }, 
        {'qpos': np.array([
            [ 4.34653711e-01,  9.48761487e-02,  7.62308748e-02, -1.22306019e-02,  7.07000852e-01,  5.32138559e-03, 7.07086905e-01],
            [ 4.36713370e-01, -5.70031576e-02,  2.59862702e-02, 2.88288599e-06, -3.46144222e-05,  1.18585019e-02, 9.99929685e-01],
            [ 5.35917695e-01,  7.84797272e-02,  7.59850418e-02, -7.05273564e-03,  7.07140760e-01,  7.06586889e-03, 7.07002318e-01]]), 
         'qvel': np.array([
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.]])
        }, 
        {'qpos': np.array([
            [ 4.34467020e-01,  1.04449431e-01,  7.86844842e-02, -5.50572712e-02,  7.04935726e-01, -4.11924792e-02, 7.05930237e-01],
            [ 4.37769387e-01, -5.48871927e-02,  2.59697532e-02, 1.28121566e-04,  1.82288527e-06,  9.70659067e-03, 9.99952882e-01],
            [ 5.57954235e-01, -2.39693695e-01,  2.59791428e-02, 1.43912480e-04, -1.49825946e-05,  3.08403912e-05, 9.99999989e-01]]), 
         'qvel': np.array([
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.]])
        }, 
        {'qpos': np.array([
            [ 2.98413446e-01,  2.32736620e-01,  2.58403374e-02, 8.63630244e-05, -8.99600554e-06,  6.69754026e-06, 9.99999996e-01],
            [ 4.37912461e-01, -3.52341358e-02,  2.42539084e-02, 2.12873102e-05, -2.22642986e-06,  1.17989084e-06, 1.00000000e+00],
            [ 5.57956544e-01, -2.39684773e-01,  2.53539220e-02, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]), 
         'qvel': np.array([
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.]])
        }
        ]
    test_armstack_low(_states_history, n_object=3)