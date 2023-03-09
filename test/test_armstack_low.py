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
    cur_pos = np.asarray(state['objects']['qpos'])
    nxt_pos = np.asarray(next_state['objects']['qpos'])
    pos_err = abs(cur_pos - nxt_pos).sum(axis=-1)
    block_id = np.argmax(pos_err)
    position = nxt_pos[block_id][:3]
    orientation = nxt_pos[block_id][3:]
    euler = np.asarray(p.getEulerFromQuaternion(orientation)) / (np.pi / 2)
    action = np.concatenate([[block_id], position, euler])
    action[3] += 0.001
    print("[DEBUG] pos_err={}".format(pos_err))
    return action

def test_armstack_low(traj_path, length=None, n_object=None):
    # load trajectory data
    traj_data = pickle.load(open(traj_path, 'rb'))
    _actions_history = traj_data['actions']
    _states_history = traj_data['states']

    _actions_history = _actions_history[:3]
    _states_history = _states_history[0:4]

    if n_object is None:
        n_object = len(_states_history[0]['objects']['qpos'])
    for i, a in enumerate(_actions_history):
        if a[0] >= n_object:
            a = -torch.ones_like(a)
            _actions_history[i] = a
    for i, s in enumerate(_states_history):
        _states_history[i]['objects']['qpos'] = np.array(_states_history[i]['objects']['qpos'][:n_object])
        _states_history[i]['objects']['qvel'] = np.zeros_like(_states_history[i]['objects']['qvel'][:n_object])
        _states_history[i]['objects']['qpos'][:, 2] += 0.001
        '''for qpos in _states_history[i]['objects']['qpos']:
            theta = np.pi / 4 * 3
            qpos[3:7][3] = np.cos(theta)
            qpos[3:7][:3] = np.array([0., 1., 0.]) * np.sin(theta)
            qpos[0:3][2] += 0.05
    _states_history[1]['objects']['qpos'][0][:2] = _states_history[0]['objects']['qpos'][0][:2]
    _states_history[1]['objects']['qpos'][0][2] -= 0.075
    _states_history[1]['objects']['qpos'][0][3:7] = np.array([0., 0., 0., 1.])'''
    '''_states_history[0]['objects']['qpos'][0][2] = 0.078
    _states_history[0]['objects']['qpos'][0][3:7] = np.asarray([0., np.sin(np.pi/4), 0., np.cos(np.pi / 4)])
    _states_history[0]['objects']['qpos'][1][0] += 0.05
    _states_history[1]['objects']['qpos'][1][0] += 0.05
    _states_history[0]['objects']['qpos'][1][1] += 0.3
    _states_history[1]['objects']['qpos'][1][1] += 0.3'''
    _states_history[0]['objects']['qpos'][1][1] -= 0.1
    _states_history[1]['objects']['qpos'][1][1] -= 0.1
    _states_history[2]['objects']['qpos'][1][1] -= 0.1
    # _states_history[1]['objects']['qpos'][:, 2] += 0.05

    _states_history = list(reversed(_states_history))

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
        length = len(_actions_history)
    robot_state = env.robot.get_state()
    for i in range(length):
        action = None # _actions_history[i]
        state = _states_history[i]
        next_state = _states_history[i + 1] if i + 1 < len(_states_history) else None

        if next_state is None:
            break

        action = compute_action(state, next_state)

        env.set_state(state, set_robot_state=False)
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
        state:\n{log_state(np.asarray(state['objects']['qpos']), 24)}
        object_orn:\n{log_state(np.asarray(blocks_orn), 24)}
        action={action}
        expected next_state:\n{log_state(np.asarray(next_state['objects']['qpos']), 24)}
        ground-truth next_state:\n{log_state(np.asarray(env.get_state()['objects']['qpos']), 24)}''')

        if next_state is not None:
            env.set_state(next_state, set_robot_state=False)
            img = env.render(mode="rgb_array")
            plt.imsave("debug_tmp/tmp%d_c.png" % i, img)


if __name__ == "__main__":
    test_armstack_low('test_demo_random.pkl', n_object=3)