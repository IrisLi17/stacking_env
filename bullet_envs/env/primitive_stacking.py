import pybullet as p
from pybullet_utils import bullet_client as bc
import pybullet_data
import gym, os
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch
from bullet_envs.env.bullet_rotations import mat2quat, quat_mul  # env.
import random
import pickle
import matplotlib.pyplot as plt
import shutil
import math
import copy
from collections import OrderedDict
import pkgutil
egl = pkgutil.get_loader('eglRenderer')
from itertools import combinations
from functools import partial
from bullet_envs.env.primitive_env import render

DATAROOT = pybullet_data.getDataPath()
# raw image training
COLOR = [[1.0, 0, 0], [1, 1, 0], [0.2, 0.8, 0.8], [0.8, 0.2, 0.8], [0.2, 0.8, 0.2], [0.0, 0.0, 1.0], [0.5, 0.2, 0.0],
         [0.2, 0, 0.5], [0, 0.2, 0.5]]
# mvp training
# COLOR = [[1.0, 0, 0], [1, 1, 0], [0.2, 0.8, 0.8], [0.8, 0.2, 0.8], [0, 0, 0], [0.0, 0.0, 1.0], [0.5, 0.2, 0.0],
#          [0.2, 0, 0.5], [0, 0.2, 0.5]]
# FAKE
# COLOR = [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0.0], [0.5, 0.2, 0.0],
#          [0.2, 0, 0.5], [0, 0.2, 0.5]]
FRAMECOUNT = 0

class ArmGoalEnv(gym.Env):
    def __init__(self, robot="panda", seed=None, action_dim=4, generate_data=None, use_gpu_render=True):
        self.seed(seed)
        self.use_gpu_render = use_gpu_render

        self._setup_env(robot)
        self.goal = self._sample_goal()
        obs = self._get_obs()
        if isinstance(obs, dict):
            self.observation_space = spaces.Dict(
                OrderedDict([(key, spaces.Box(low=-np.inf, high=np.inf, shape=obs[key].shape)) for key in obs])
            )
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape)
        self.action_space = spaces.Box(-1., 1., shape=(action_dim,))
        self.frame_count = 0
        self.used_object = None
        self.action_dim = action_dim
        self.generate_data = generate_data
        self.gen_action = None
        self.n_step = 0

    @property
    def dt(self):
        return 1. / 240

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _create_simulation(self):
        self.p = bc.BulletClient(connection_mode=p.DIRECT)
        if self.use_gpu_render:
            plugin = self.p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            print("plugin=", plugin)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 0)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
    
    def _setup_env(self, robot, init_qpos=None, init_end_effector_pos=(1.0, 0.6, 0.4), init_end_effector_orn=(0, -np.pi, np.pi / 2),
                 useNullSpace=True, base_position=(0, 0, 0)):
        self._create_simulation()
        self.p.resetSimulation()
        self.p.setTimeStep(self.dt)
        self.p.setGravity(0., 0., -9.8)
        # self.p.resetDebugVisualizerCamera(1.0, 40, -20, [0, 0, 0, ])
        self.plane_id = self.p.loadURDF(os.path.join(os.path.dirname(__file__), "assets/plane.urdf"), [0, 0, -0.795])
        self.table_id = self.p.loadURDF(
            os.path.join(DATAROOT, "table/table.urdf"),
            [0.40000, 0.00000, -.625000], [0.000000, 0.000000, 0.707, 0.707]
        )
        for shape in self.p.getVisualShapeData(self.plane_id):
            self.p.changeVisualShape(self.plane_id, shape[1], rgbaColor=(1, 1, 1, 1))
        # for shape in self.p.getVisualShapeData(self.table_id):
        #     self.p.changeVisualShape(self.table_id, shape[1], rgbaColor=(0, 0, 0, 1))
        if robot == "xarm":
            from bullet_envs.env.robots import XArm7Robot
            self.robot = XArm7Robot(self.p)
        elif robot == "panda":
            from bullet_envs.env.robot import PandaRobot  # env.
            self.robot = PandaRobot(self.p, init_qpos, base_position)
        elif robot == "ur":
            from bullet_envs.env.robots import UR2f85Robot
            self.robot = UR2f85Robot(self.p, init_qpos=init_qpos, init_end_effector_pos=init_end_effector_pos,
                                     init_end_effector_orn=init_end_effector_orn, useOrientation=True,
                                     useNullSpace=useNullSpace)
        else:
            raise NotImplementedError
        self._setup_callback()

    def reset(self):
        self._reset_sim()
        self.goal = self._sample_goal().copy()
        # """
        # for creating last step env
        step = False
        while step:
            cur_action = self.act()[-1]
            if (cur_action == -1).all():
                step = False
            else:
                _, _, _, _ = self.step(cur_action)
        # """
        if random.uniform(0, 1) < self.multi_goal_prob:
            self.multi_goal = True
            available_idx = []
            for i in range(self.n_active_object):
                if i != np.argmax(self.goal[0][6:12]):
                    available_idx.append(i)
            if random.uniform(0, 1) < 0.5:
                sec_goal_idx = random.choice(available_idx)
                pos, orn = self.p.getBasePositionAndOrientation(self.blocks_id[sec_goal_idx])
                self.goal[0][12:15] = np.array([0, 1, 0])
                self.goal[0][15:18] = pos
                self.goal[0][18:24] = 0
                self.goal[0][18 + sec_goal_idx] = 1
            else:
                sec_thi_idx = random.sample(available_idx, 2)
                self.goal[0][12:15] = np.array([0, 1, 0])
                self.goal[0][15:18] = self.p.getBasePositionAndOrientation(self.blocks_id[sec_thi_idx[0]])[0]
                self.goal[0][18:24] = 0
                self.goal[0][18 + sec_thi_idx[0]] = 1
                self.goal[0][24:27] = np.array([0, 1, 0])
                self.goal[0][27:30] = self.p.getBasePositionAndOrientation(self.blocks_id[sec_thi_idx[1]])[0]
                self.goal[0][30:36] = 0
                self.goal[0][30 + sec_thi_idx[1]] = 1
            self.visualize_goal(self.goal, True)
        else:
            self.multi_goal = False
        obs = self._get_obs()
        self.n_step = 0
        return obs

    def check_collision(self, pos, orn, obj_id):
        """
        assert self.n_active_object is not None
        block_positions = []
        block_halfextents = []
        for i in range(self.n_active_object):
            cur_pos, _ = self.p.getBasePositionAndOrientation(self.blocks_id[i])
            block_halfextent = np.array(self.p.getCollisionShapeData(self.blocks_id[i], -1)[0][3]) / 2
            block_positions.append(cur_pos)
            block_halfextents.append(block_halfextent)
        eef_pos = pos
        """
        cur_state_id = self.p.saveState()
        self.robot.control(
            pos, orn,
            (self.robot.finger_range[0] + self.robot.finger_range[1]) / 2,
            relative=False, teleport=True,
        )
        self.p.performCollisionDetection()
        contact1 = self.p.getContactPoints(bodyA=self.robot.id, linkIndexA=8)
        contact2 = self.p.getContactPoints(bodyA=self.robot.id, linkIndexA=9)
        contact3 = self.p.getContactPoints(bodyA=self.robot.id, linkIndexA=10)
        if len(contact1) > 0:
            for contact in contact1:
                if contact[2] != self.blocks_id[obj_id]:
                    self.p.restoreState(stateId=cur_state_id)
                    self.p.removeState(cur_state_id)
                    return True
        self.p.restoreState(stateId=cur_state_id)
        self.p.removeState(cur_state_id)
        # eef_halfextent = np.array(self.p.getCollisionShapeData(self.robot.id, self.robot.eef_index)[0][3]) / 2
        return False

    def act(self):
        if self.generate_data:
            """
            if not self.generate_data:
                self.gen_action = self.offline_datasets[self.traj_idx]["actions"]
                cur_action = self.gen_action[0, :]
                print(self.gen_action)
                return torch.tensor(cur_action).reshape((1, -1))
                if random.uniform(0, 1) < 1:
                    self.goal_orn = [self.p.getQuaternionFromEuler([0, math.pi / 2, 0])]
                else:
                    self.goal_orn = [self.gen_obj_quat()]
            """
            assert self.n_goal == 1
            goal_pos = np.reshape(self.goal, (-1, 6 + self.n_object))[0, 3:6]
            # print("n_step", self.n_step, "goal", self.goal)
            if self.n_step == 0:  # design all the actions later
                # print(goal_pos)
                self.n_step += 1
                obj_id = self._get_achieved_goal()[1][0]
                available_id = []
                for i in range(self.n_active_object):
                    if i != obj_id:
                        available_id.append(i)
                all_pos = np.array([self.p.getBasePositionAndOrientation(self.blocks_id[i])[0] for i in range(self.n_object)])
                all_orn = np.array([self.p.getBasePositionAndOrientation(self.blocks_id[i])[1] for i in range(self.n_object)])
                block_half_extent = np.array(self.p.getCollisionShapeData(self.blocks_id[0], -1)[0][3]) / 2

                if self.goal_orn[0][1] == 0:  # (0, 0, 0, 1) / (0, 0, theta/2, -theta/2), horizontal goal
                    for i in range(self.n_active_object):
                        rot_matrix = _quaternion2RotationMatrix(all_orn[i])
                        goal_pos_b_abs = np.abs(np.matmul(rot_matrix, all_pos[i] - goal_pos))
                        if i != obj_id and all(goal_pos_b_abs[:2] <= block_half_extent[:2]):
                            self.n_step = 0
                            return -torch.ones((1, 7))
                else:  # vertical goal
                    for i in range(self.n_active_object):
                        if i != obj_id and all(np.abs(all_pos[i][:2] - goal_pos[:2]) <= 0.025):
                            self.n_step = 0
                            return -torch.ones((1, 7))
                if self.goal_orn[0][1] == 0:
                    if goal_pos[2] <= 0.08:
                        orn = self.p.getEulerFromQuaternion(self.gen_obj_quat())
                        action1 = np.concatenate(([available_id.pop(0), goal_pos[0], goal_pos[1], 0.025],
                                                 orn))
                        action2 = np.concatenate(([obj_id], goal_pos, self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        self.gen_action = [action1, action2]
                    elif goal_pos[2] <= 0.13:
                        action1 = np.concatenate(([available_id.pop(0), goal_pos[0], goal_pos[1], 0.025],
                                                 self.p.getEulerFromQuaternion(self.gen_obj_quat())))
                        action2 = np.concatenate(([available_id.pop(0), goal_pos[0], goal_pos[1], 0.075],
                                                 self.p.getEulerFromQuaternion(self.gen_obj_quat())))
                        action3 = np.concatenate(([obj_id], goal_pos, self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        self.gen_action = [action1, action2, action3]
                    elif goal_pos[2] <= 0.18:
                        theta = self.p.getEulerFromQuaternion(self.goal_orn[0])[2]
                        action1 = np.array([available_id.pop(0),
                                            min(max(-0.05*math.cos(theta) + goal_pos[0], self.robot.x_workspace[0]), self.robot.x_workspace[1]),
                                            min(max(-0.05*math.sin(theta) + goal_pos[1], self.robot.y_workspace[0]), self.robot.y_workspace[1]),
                                            0.075, 0, math.pi / 2, 0])
                        action2 = np.array([available_id.pop(0),
                                            max(min(0.05*math.cos(theta) + goal_pos[0], self.robot.x_workspace[1]), self.robot.x_workspace[0]),
                                            max(min(0.05*math.sin(theta) + goal_pos[1], self.robot.y_workspace[1]), self.robot.y_workspace[0]),
                                            0.075, 0, math.pi / 2, 0])
                        action3 = np.concatenate(([obj_id], goal_pos, self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        self.gen_action = [action1, action2, action3]
                    elif goal_pos[2] <= 0.23:
                        action1 = np.concatenate(([available_id.pop(0)], goal_pos[:2], [0.025]))
                        action1 = np.concatenate((action1, self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        theta = self.p.getEulerFromQuaternion(self.goal_orn[0])[2]
                        action2 = np.array([available_id.pop(0),
                                            min(max(-0.05 * math.cos(theta) + goal_pos[0], self.robot.x_workspace[0]), self.robot.x_workspace[1]),
                                            min(max(-0.05 * math.sin(theta) + goal_pos[1], self.robot.y_workspace[0]), self.robot.y_workspace[1]),
                                            0.125, 0, math.pi / 2, 0])
                        action3 = np.array([available_id.pop(0),
                                            max(min(0.05 * math.cos(theta) + goal_pos[0], self.robot.x_workspace[1]), self.robot.x_workspace[0]),
                                            max(min(0.05 * math.sin(theta) + goal_pos[1], self.robot.y_workspace[1]), self.robot.y_workspace[0]),
                                            0.125, 0, math.pi / 2, 0])
                        action4 = np.concatenate(([obj_id], goal_pos, self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        self.gen_action = [action1, action2, action3, action4]
                    elif goal_pos[2] <= 0.28:
                        action1 = np.concatenate(([available_id.pop(0)], goal_pos[:2], [0.025]))
                        action1 = np.concatenate((action1, self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        action2 = np.concatenate(([available_id.pop(0)], goal_pos[:2], [0.075]))
                        action2 = np.concatenate((action2, self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        theta = self.p.getEulerFromQuaternion(self.goal_orn[0])[2]
                        action3 = np.array([available_id.pop(0),
                                            min(max(-0.05 * math.cos(theta) + goal_pos[0], self.robot.x_workspace[0]), self.robot.x_workspace[1]),
                                            min(max(-0.05 * math.sin(theta) + goal_pos[1], self.robot.y_workspace[0]), self.robot.y_workspace[1]),
                                            0.175, 0, math.pi / 2, 0])
                        action4 = np.array([available_id.pop(0),
                                            max(min(0.05 * math.cos(theta) + goal_pos[0], self.robot.x_workspace[1]), self.robot.x_workspace[0]),
                                            max(min(0.05 * math.sin(theta) + goal_pos[1], self.robot.y_workspace[1]), self.robot.y_workspace[0]),
                                            0.175, 0, math.pi / 2, 0])
                        action5 = np.concatenate(([obj_id], goal_pos, self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        self.gen_action = [action1, action2, action3, action4, action5]
                    else:
                        # print("error if not equal to 0.325", goal_pos[2])
                        theta = self.p.getEulerFromQuaternion(self.goal_orn[0])[2]
                        action1 = np.array([available_id.pop(0),
                                            min(max(-0.05 * math.cos(theta) + goal_pos[0], self.robot.x_workspace[0]), self.robot.x_workspace[1]),
                                            min(max(-0.05 * math.sin(theta) + goal_pos[1], self.robot.y_workspace[0]), self.robot.y_workspace[1]),
                                            0.075, 0, math.pi / 2, 0])
                        action2 = np.array([available_id.pop(0),
                                            max(min(0.05 * math.cos(theta) + goal_pos[0], self.robot.x_workspace[1]), self.robot.x_workspace[0]),
                                            max(min(0.05 * math.sin(theta) + goal_pos[1], self.robot.y_workspace[1]), self.robot.y_workspace[0]),
                                            0.075, 0, math.pi / 2, 0])
                        action3 = np.array([available_id.pop(0),
                                            min(max(-0.05 * math.cos(theta) + goal_pos[0], self.robot.x_workspace[0]), self.robot.x_workspace[1]),
                                            min(max(-0.05 * math.sin(theta) + goal_pos[1], self.robot.y_workspace[0]), self.robot.y_workspace[1]),
                                            0.225, 0, math.pi / 2, 0])
                        action4 = np.array([available_id.pop(0),
                                            max(min(0.05 * math.cos(theta) + goal_pos[0], self.robot.x_workspace[1]), self.robot.x_workspace[0]),
                                            max(min(0.05 * math.sin(theta) + goal_pos[1], self.robot.y_workspace[1]), self.robot.y_workspace[0]),
                                            0.225, 0, math.pi / 2, 0])
                        action5 = np.concatenate(([obj_id], goal_pos, self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        self.gen_action = [action1, action2, action3, action4, action5]
                    action = self.gen_action.pop(0)
                    if action[1] > self.robot.x_workspace[1] or action[1] < self.robot.x_workspace[0]:
                        print(action[1], self.robot.x_workspace, goal_pos)
                        raise ValueError
                    if action[2] > self.robot.y_workspace[1] or action[2] < self.robot.y_workspace[0]:
                        print(action[2], self.robot.y_workspace, goal_pos)
                        raise ValueError
                    action[1] = (action[1] - self.robot.x_workspace[0]) * 2. / \
                                (self.robot.x_workspace[1] - self.robot.x_workspace[0]) - 1
                    action[2] = (action[2] - self.robot.y_workspace[0]) * 2. / \
                                (self.robot.y_workspace[1] - self.robot.y_workspace[0]) - 1
                    if self.name == "allow_rotation":
                        action[3] = (action[3] - 0.025 - self.robot.base_pos[2]) * 5. - 1
                    elif self.name == "default":
                        action[3] = (action[3] - 0.025 - self.robot.base_pos[2]) * 5. - 1
                    else:
                        raise NotImplementedError
                    action[4:] = action[4:] * 2. / np.pi
                    if action[-1] > 1:
                        action[-1] = action[-1] - 2
                    elif action[-1] < -1:
                        action[-1] = action[-1] + 2
                    return torch.tensor(action).reshape(1, -1)
                else:
                    if goal_pos[2] <= 0.08:
                        action1 = np.concatenate(([obj_id], goal_pos, self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        self.gen_action = [action1]
                    elif goal_pos[2] <= 0.13:
                        action1 = np.concatenate(([available_id.pop(0), goal_pos[0], goal_pos[1], 0.025],
                                                   self.p.getEulerFromQuaternion(self.gen_obj_quat())))
                        action2 = np.concatenate(([obj_id], goal_pos, self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        self.gen_action = [action1, action2]
                    elif goal_pos[2] <= 0.18:
                        action1 = np.concatenate(([available_id.pop(0), goal_pos[0], goal_pos[1], 0.025],
                                                  self.p.getEulerFromQuaternion(self.gen_obj_quat())))
                        action2 = np.concatenate(([available_id.pop(0), goal_pos[0], goal_pos[1], 0.075],
                                                  self.p.getEulerFromQuaternion(self.gen_obj_quat())))
                        action3 = np.concatenate(([obj_id], goal_pos, self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        self.gen_action = [action1, action2, action3]
                    elif goal_pos[2] <= 0.23:
                        action1 = np.concatenate(([available_id.pop(0), goal_pos[0], goal_pos[1], 0.075],
                                                  self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        action2 = np.concatenate(([obj_id], goal_pos, self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        self.gen_action = [action1, action2]
                    elif goal_pos[2] <= 0.28:
                        action1 = np.concatenate(([available_id.pop(0), goal_pos[0], goal_pos[1], 0.025],
                                                  self.p.getEulerFromQuaternion(self.gen_obj_quat())))
                        action2 = np.concatenate(([available_id.pop(0), goal_pos[0], goal_pos[1], 0.125],
                                                  self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        action3 = np.concatenate(([obj_id], goal_pos, self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        self.gen_action = [action1, action2, action3]
                    else:
                        # print("error if not equal to 0.325", goal_pos[2])
                        action1 = np.concatenate(([available_id.pop(0), goal_pos[0], goal_pos[1], 0.025],
                                                  self.p.getEulerFromQuaternion(self.gen_obj_quat())))
                        action2 = np.concatenate(([available_id.pop(0), goal_pos[0], goal_pos[1], 0.075],
                                                  self.p.getEulerFromQuaternion(self.gen_obj_quat())))
                        action3 = np.concatenate(([available_id.pop(0), goal_pos[0], goal_pos[1], 0.125],
                                                  self.p.getEulerFromQuaternion(self.gen_obj_quat())))
                        action4 = np.concatenate(([available_id.pop(0), goal_pos[0], goal_pos[1], 0.175],
                                                  self.p.getEulerFromQuaternion(self.gen_obj_quat())))
                        action5 = np.concatenate(([available_id.pop(0), goal_pos[0], goal_pos[1], 0.225],
                                                  self.p.getEulerFromQuaternion(self.gen_obj_quat())))
                        action6 = np.concatenate(([obj_id], goal_pos, self.p.getEulerFromQuaternion(self.goal_orn[0])))
                        self.gen_action = [action1, action2, action3, action4, action5, action6]
                    action = self.gen_action.pop(0)
                    if action[1] > self.robot.x_workspace[1] or action[1] < self.robot.x_workspace[0]:
                        print(action[1], self.robot.x_workspace)
                        raise ValueError
                    if action[2] > self.robot.y_workspace[1] or action[2] < self.robot.y_workspace[0]:
                        print(action[2], self.robot.y_workspace)
                        raise ValueError
                    action[1] = (action[1] - self.robot.x_workspace[0]) * 2. / \
                                (self.robot.x_workspace[1] - self.robot.x_workspace[0]) - 1
                    action[2] = (action[2] - self.robot.y_workspace[0]) * 2. / \
                                (self.robot.y_workspace[1] - self.robot.y_workspace[0]) - 1
                    if self.name == "allow_rotation":
                        action[3] = (action[3] - 0.025 - self.robot.base_pos[2]) * 5. - 1
                    elif self.name == "default":
                        action[3] = (action[3] - 0.025 - self.robot.base_pos[2]) * 5. - 1
                    else:
                        raise NotImplementedError
                    action[4:] = action[4:] * 2. / np.pi
                    if action[-1] > 1:
                        action[-1] = action[-1] - 2
                    elif action[-1] < -1:
                        action[-1] = action[-1] + 2
                    return torch.tensor(action).reshape(1, -1)
            else:  # choose one action in the action sequence
                # print("gen actions", self.gen_action)
                self.n_step += 1
                assert self.gen_action is not None
                if random.uniform(0, 1) < self.multi_step_goal_prob:
                    self.multi_step_goal = True
                    threshold = random.choice([2, 3, 4, 5, 6])
                else:
                    threshold = 0 # 1
                    self.multi_step_goal = False
                if len(self.gen_action) <= threshold:  # == 0, # <= 1, for creating last step env
                    self.n_step = 0
                    return -torch.ones((1, 7))
                action = self.gen_action.pop(0)
                if action[1] > self.robot.x_workspace[1] or action[1] < self.robot.x_workspace[0]:
                    print(action[1], self.robot.x_workspace, goal_pos)
                    raise ValueError
                if action[2] > self.robot.y_workspace[1] or action[2] < self.robot.y_workspace[0]:
                    print(action[2], self.robot.y_workspace, goal_pos)
                    raise ValueError
                action[1] = (action[1] - self.robot.x_workspace[0]) * 2. / \
                            (self.robot.x_workspace[1] - self.robot.x_workspace[0]) - 1
                action[2] = (action[2] - self.robot.y_workspace[0]) * 2. / \
                            (self.robot.y_workspace[1] - self.robot.y_workspace[0]) - 1
                if self.name == "allow_rotation":
                    action[3] = (action[3] - 0.025 - self.robot.base_pos[2]) * 5. - 1
                elif self.name == "default":
                    action[3] = (action[3] - 0.025 - self.robot.base_pos[2]) * 5. - 1
                else:
                    raise NotImplementedError
                action[4:] = action[4:] * 2. / np.pi
                if action[-1] > 1:
                    action[-1] = action[-1] - 2
                elif action[-1] < -1:
                    action[-1] = action[-1] + 2
                return torch.tensor(action).reshape(1, -1)
        else:
            raise NotImplementedError

    def step(self, action):
        assert self.name is not None
        if (action == -1).all():
            obs = self._get_obs()
            reward, info = self.compute_reward_and_info()
            done = True
            return obs, reward, done, info
        if self.primitive:
            assert len(action) == 7
            assert self.n_active_object is not None
            if any(np.abs(action[1:]) > 1):
                print(action)
                raise ValueError
            # get target position & orientation
            # todo: let generate action & not generate action be the same
            obj_id = int(action[0])
            tgt_pos = torch.zeros(3)
            tgt_orn = action[4:]
            tgt_pos[0] = (action[1] + 1) * (self.robot.x_workspace[1] - self.robot.x_workspace[0]) / 2 \
                            + self.robot.x_workspace[0]
            tgt_pos[1] = (action[2] + 1) * (self.robot.y_workspace[1] - self.robot.y_workspace[0]) / 2 \
                            + self.robot.y_workspace[0]
            if self.name == "allow_rotation":
                tgt_pos[2] = (action[3] + 1) * 0.4 / 2 + self.robot.base_pos[2] + 0.025
            elif self.name == "default":
                tgt_pos[2] = (action[3] + 1) * 0.4 / 2 + self.robot.base_pos[2] + 0.025
            else:
                raise NotImplementedError
            tgt_orn = tgt_orn * np.pi / 2.
            tgt_orn = p.getQuaternionFromEuler(tgt_orn)

            stable = True
            # if self._robot_feasible(self.blocks_id[obj_id], tgt_pos, tgt_orn):
            if True:
                # get current position & orientation of the target object
                # cur_pos, cur_orn = self.p.getBasePositionAndOrientation(self.blocks_id[obj_id])
                # self.robot.control(
                #     tgt_pos, tgt_orn,
                #     (self.robot.finger_range[0] + self.robot.finger_range[1]) / 2,
                #     relative=False, teleport=True,
                # )
                # state_id = self.p.saveState()
                self.p.resetBasePositionAndOrientation(self.blocks_id[obj_id], tgt_pos, tgt_orn)
                #fig, ax = plt.subplots(1, 1)
                # if os.path.exists("tmp_roll"):
                #     shutil.rmtree("tmp_roll")
                #os.makedirs("tmp_roll", exist_ok=True)
                # img = self.render(mode="rgb_array")
                # ax.cla()
                # ax.imshow(img)
                # plt.imsave("tmp_roll/tmp%d.png" % self.frame_count, img)
                # self.frame_count += 1
                # print(self.frame_count)
                for frame_count in range(40):
                    self.p.stepSimulation()
                    # self.frame_count += 1
                # judge whether stable
                cur_pos = self._get_achieved_goal()[0]
                for _ in range(10):
                    # img = self.render(mode="rgb_array")
                    # ax.cla()
                    # ax.imshow(img)
                    # plt.imsave("tmp_roll/tmp%d.png" % self.frame_count, img)
                    self.p.stepSimulation()
                    # self.frame_count += 1
                #img = self.render(mode="rgb_array")
                #ax.cla()
                #ax.imshow(img)
                #plt.imsave("tmp_roll/tmp%d.png" % self.frame_count, img)
                #self.frame_count += 1
                # print(self.frame_count)
                future_pos = self._get_achieved_goal()[0]
                stable = not any(np.linalg.norm(future_pos - cur_pos, axis=-1) >= 1e-3)
            obs = self._get_obs()
            reward, info = self.compute_reward_and_info()
            done = False
            if not stable:
                reward -= 0.001
            #     # self.p.removeState(state_id)
            return obs, reward, done, info

    def _robot_feasible(self, body_id, target_pos, target_quat):
        state_id = self.p.saveState()
        import pybullet_planning as pp
        # generate grasp pose from object pose
        all_eef_T_obj = []
        for i in range(4):
            for j in range(2):
                eef_pos_obj, eef_quat_obj = self.p.multiplyTransforms(
                    [0, 0, 0], [np.cos(np.pi / 4 * i), 0, 0, np.sin(np.pi / 4 * i)], 
                    [0, 0, 0], [0, 0, np.cos(np.pi / 2 * j), np.sin(np.pi / 2 * j)]
                )
                all_eef_T_obj.append((eef_pos_obj, eef_quat_obj))
        O_T_eef = (self.robot.get_eef_position(), self.robot.get_eef_orn())
        O_T_obj = self.p.getBasePositionAndOrientation(body_id)
        all_grasp_eef_pose = []
        all_grasp_conf = []
        for i in range(len(all_eef_T_obj)):
            obj_T_eef = self.p.invertTransform(all_eef_T_obj[i][0], all_eef_T_obj[i][1])
            grasp_eef_pose = self.p.multiplyTransforms(O_T_obj[0], O_T_obj[1], obj_T_eef[0], obj_T_eef[1])
            all_grasp_eef_pose.append(grasp_eef_pose) 
            all_grasp_conf.append(
                self.p.calculateInverseKinematics(
                    self.robot.id, self.robot.eef_index, grasp_eef_pose[0], grasp_eef_pose[1],
                    self.robot.joint_ll[:7], self.robot.joint_ul[:7], self.robot.joint_ranges[:7], self.robot.rest_poses[:7],
                    maxNumIterations=20,
                )
            )
        # plan path
        ik_joints= pp.get_movable_joints(self.robot.id)
        robot_self_collision_disabled_link_names = [('panda_link0', 'panda_link1'),
            ('panda_link1', 'panda_link2'), ('panda_link2', 'panda_link3'),
            ('panda_link3', 'panda_link4'), ('panda_link4', 'panda_link5'),
            ('panda_link5', 'panda_link6'), ('panda_link6', 'panda_link7'),
            ('panda_link7', 'panda_link8'), ('panda_link8', 'panda_hand'),
            ('panda_hand', 'panda_leftfinger'), ('panda_hand', 'panda_rightfinger'),
            ] 
        self_collision_links = pp.get_disabled_collisions(self.robot.id, robot_self_collision_disabled_link_names)
        path_found = False
        for i in range(len(all_grasp_conf)):
            # TODO: check whether the environment has changed
            path = pp.plan_joint_motion(
                self.robot.id, ik_joints, all_grasp_conf[i], obstacles=self.blocks_id + [self.table_id], attachments=[],
                self_collisions=True, disabled_collisions=self_collision_links, extra_disabled_collisions=set(),
                weights=None, resolutions=None, custom_limits={}, diagnosis=False
            )
            if path is not None:
                path_found = True
                pp.set_joint_position(self.robot.id, ik_joints, path[-1])
                break
        if not path_found:
            self.p.restoreState(state_id)
            self.p.removeState(state_id)
            return False
        # go to target
        all_release_conf = []
        for i in range(len(all_eef_T_obj)):
            obj_T_eef = self.p.invertTransform(all_eef_T_obj[i][0], all_eef_T_obj[i][1])
            release_eef_pose = self.p.multiplyTransforms(target_pos[0], target_quat[1], obj_T_eef[0], obj_T_eef[1])
            all_release_conf.append(
                self.p.calculateInverseKinematics(
                    self.robot.id, self.robot.eef_index, release_eef_pose[0], release_eef_pose[1],
                    self.robot.joint_ll[:7], self.robot.joint_ul[:7], self.robot.joint_ranges[:7], self.robot.rest_poses[:7],
                    maxNumIterations=20,
                )
            )
        # create attachment
        block_attach = pp.create_attachment(self.robot.id, self.robot.eef_index, body_id)
        block_attach.assign()
        path_found = False
        for i in range(len(all_release_conf)):
            path = pp.plan_joint_motion(
                self.robot.id, ik_joints, all_release_conf[i], 
                obstacles=[block_id for block_id in self.blocks_id if block_id != body_id] + [self.table_id],
                attachments=[block_attach], self_collisions=True, disabled_collisions=self_collision_links,
            )
            if path is not None:
                path_found = True
                pp.set_joint_position(self.robot.id, ik_joints, path[-1])
                break
        if not path_found:
            self.p.restoreState(state_id)
            self.p.removeState(state_id)
            return False
        # move to some reset pose
        
        self.p.restoreState(state_id)
        self.p.removeState(state_id)
        return True
        
            
    def render(self, mode="rgb_array", width=500, height=500):
        '''from bullet_envs.env.primitive_env import render
        scene = render(self.p, width=128, height=128, robot=self.robot, view_mode="third",
                       pitch=-45, distance=0.6,
                       camera_target_position=(0.5, 0.0, 0.1)).transpose((2, 0, 1))[:3]
        return scene'''
        if mode == 'rgb_array':
            '''view_matrix = self.p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=(0.3, 0, 0.2),
                distance=1.0,
                yaw=60,
                pitch=-10,
                roll=0,
                upAxisIndex=2,
            )'''
            
            view_matrix = self.p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0.0, 0.1],
                                                                   distance=0.6,
                                                                   yaw=90,
                                                                   pitch=-45,
                                                                   roll=0,
                                                                   upAxisIndex=2)
            proj_matrix = self.p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
            )
            (_, _, px, _, _) = self.p.getCameraImage(
                width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            )
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (height, width, 4))

            return rgb_array
        else:
            raise NotImplementedError

    def _setup_callback(self):
        # Other things except plane, table and robot
        pass

    def _sample_goal(self):
        # Should be inherited
        return None

    def visualize_goal(self, goal, token_type=True):
        # changed
        if self.body_goal is not None:
            for goal_id in self.body_goal:
                self.p.removeBody(goal_id)
        self.body_goal = []
        if token_type:
            goal = np.reshape(goal, (-1, 6 + self.n_object))
            for i in range(goal.shape[0]):
                if all(goal[i, :] == -1):
                    goal = goal[:i, :]
                    break
            n_goal = np.shape(goal)[0]
            goal_idx = np.zeros(n_goal)
            for i in range(n_goal):
                if goal[i][2] == 1:
                    goal_idx[i] = -1
                else:
                    goal_idx[i] = int(np.argmax(goal[i, 6:]))
            for id, goal_id in enumerate(goal_idx):
                vis_id = self.p.createVisualShape(self.p.GEOM_SPHERE, radius=0.03,
                                                  rgbaColor=COLOR[int(goal_id % len(COLOR))] + [0.2])
                tmp = self.p.createMultiBody(0, baseVisualShapeIndex=vis_id,
                                             basePosition=goal[id, 3:6])  # self.body_goal =
                self.body_goal.append(tmp)
            return
        goal = np.reshape(goal, (-1, 3 + self.n_object))
        n_max_goal = np.shape(goal)[0]
        goal_idx = np.argmax(goal[:n_max_goal, 3:], axis=1)
        for id, goal_id in enumerate(goal_idx):
            vis_id = self.p.createVisualShape(self.p.GEOM_SPHERE, radius=0.04,
                                              rgbaColor=COLOR[goal_id % len(COLOR)] + [0.2])
            tmp = self.p.createMultiBody(0, baseVisualShapeIndex=vis_id, basePosition=goal[id, :3])  # self.body_goal =
            self.body_goal.append(tmp)

    def _reset_sim(self):
        pass

    def _get_obs(self):
        obs = self.robot.get_obs_old()  # robot_obs? TODO: check shape
        return dict(observation=obs, achieved_goal=self.goal.copy(), desired_goal=self.goal.copy())

    def get_obs(self):
        return self._get_obs()

    def compute_reward_and_info(self):
        return None, {}


class ArmPickAndPlace(ArmGoalEnv):
    def __init__(self, robot="panda", seed=None, n_object=6, reward_type="sparse", primitive=False, action_dim=4, generate_data=False, use_gpu_render=True, invisible_robot=True):
        self.env_id = "BulletPickAndPlace-v1"
        self.name = "allow_rotation"
        self.generate_data = generate_data
        self.action_dim = action_dim
        if self.name == "allow_rotation":
            self.n_max_goal = 6
            self.n_goal = 1
            # self.n_goal = random.choice(range(1, self.n_max_goal + 1))
        elif self.name == "default":
            self.n_max_goal = 6
            self.n_goal = self.n_max_goal
        else:
            raise NotImplementedError
        self.n_object = n_object
        # print("n_object:", n_object)
        self.n_active_object = n_object
        self.primitive = primitive
        self.blocks_id = []
        self.robot_dim = None
        self.object_dim = None
        self.body_goal = None
        self.env_number = None
        self.reward_type = reward_type
        self._previous_distance = None
        self.inactive_xy = (10, 10)
        self.goal_type = "concat"
        self.goal_pos_type = "ground"
        self.n_goal_prob = 0.5  # 0.3, 0.5, 0.7 for curriculum learning
        self.invisible_robot = invisible_robot
        # n_goal becomes 2 with some prob

        self.stack = False
        super(ArmPickAndPlace, self).__init__(robot, seed, action_dim, generate_data, use_gpu_render)

    def _setup_callback(self):
        for i in range(self.n_object):
            if self.name == "allow_rotation":
                self.blocks_id.append(
                    _create_block(
                        self.p, [0.075, 0.025, 0.025], [10, 0, 1 + 0.5 * i], [0, 0, 0, 1], 0.1, COLOR[i % len(COLOR)]
                    )
                )
            else:
                self.blocks_id.append(
                    _create_block(
                        self.p, [0.025, 0.025, 0.025], [10, 0, 1 + 0.5 * i], [0, 0, 0, 1], 0.1, COLOR[i % len(COLOR)]
                    )
                )
            for block_id in self.blocks_id:
                print(f"[DEBUG] block{block_id}:", self.p.getCollisionShapeData(block_id, -1), flush=True)
        # Make the robot invisible
        if self.invisible_robot:
            for shape in self.p.getVisualShapeData(self.robot.id):
                self.p.changeVisualShape(self.robot.id, shape[1], rgbaColor=(0, 0, 0, 0))

    def reset(self):
        obs_dict = super().reset()
        if self.reward_type == "dense":
            achieved_goal = np.reshape(obs_dict["achieved_goal"], (-1, 3 + self.n_object))
            desired_goal = np.reshape(obs_dict["desired_goal"], (-1, 3 + self.n_object))
            self._previous_distance = np.linalg.norm(achieved_goal[:3] - desired_goal[:3])
        return obs_dict

    def _sample_goal(self):
        # Changed
        assert self.goal_type == "concat"
        robot_pos = self.robot.base_pos
        x_range = self.robot.x_workspace
        y_range = self.robot.y_workspace
        z_range = (robot_pos[2] + 0.025, robot_pos[2] + 0.425)
        goal = []

        if self.n_goal > self.n_active_object:
            raise ValueError("Not enough objects to satisfy goals!")
        if self.n_object > 1:
            if self.stack and self.n_active_object - self.n_max_goal >= 2:  # highest object
                choices = []
                for i in range(0, self.n_active_object):
                    if i not in self.stack_obj_idx:
                        choices.append(i)
                goal_idx = random.sample(choices, self.n_goal)
            else:
                if self.name == "allow_rotation":  # -1 means the position should not be occupied by any obj
                    goal_idx = random.sample(range(0, self.n_active_object), self.n_goal)
                    for i in range(self.n_goal):
                        if random.uniform(0, 1) < 0.5:
                            goal_idx[i] = -1
                else:
                    goal_idx = random.sample(range(0, self.n_active_object), self.n_goal)
            goal_onehot = np.zeros((self.n_goal, self.n_object))
            goal_type_onehot = np.zeros((self.n_goal, 3))
            for id, idx in enumerate(goal_idx):
                if self.name == "allow_rotation":
                    if idx == -1:
                        goal_type_onehot[id, 2] = 1
                    else:
                        goal_onehot[id, idx] = 1
                        goal_type_onehot[id, 1] = 1
                elif self.primitive:
                    goal_onehot[id, idx] = 1
                    goal_type_onehot[id, 1] = 1
                else:
                    goal_onehot[id, idx] = 1
        else:
            goal_onehot = np.ones((self.n_goal, 1))
            if self.name == "allow_rotation":
                if random.uniform(0, 1) < 0.5:
                    goal_onehot = np.zeros((self.n_goal, 1))
                    goal_type_onehot = torch.tensor([[0, 0, 1]])
                    goal_idx = [-1]
                else:
                    goal_type_onehot = torch.tensor([[0, 1, 0]])
                    goal_idx = [0]
            elif self.primitive:
                goal_type_onehot = torch.tensor([[0, 1, 0]])
                goal_idx = [0]
        for goal_id in range(self.n_goal):
            if self.goal_pos_type == "space":
                if self.np_random.uniform() < 0.5 and goal_id == 0:
                    if self.stack and self.n_active_object - self.n_max_goal >= 2:
                        goal_tmp = np.array([self.stack_obj_pos[0][0], self.stack_obj_pos[0][1], z_range[0]+2*0.05])
                    else:
                        goal_tmp = np.array([self.np_random.uniform(*x_range), self.np_random.uniform(*y_range),
                                     self.np_random.uniform(*z_range)])
                else:
                    goal_tmp = np.array(
                        [self.np_random.uniform(*x_range), self.np_random.uniform(*y_range), z_range[0]])
                    for goal_j in range(goal_id):
                        while np.linalg.norm(goal_tmp - goal[goal_j], ord=2) <= 0.05:
                            goal_tmp = np.array(
                                [self.np_random.uniform(*x_range), self.np_random.uniform(*y_range), z_range[0]])

            elif self.goal_pos_type == "ground":
                goal_tmp = np.array([self.np_random.uniform(*x_range), self.np_random.uniform(*y_range), z_range[0]])
                for goal_j in range(goal_id):
                    while np.linalg.norm(goal_tmp - goal[goal_j], ord=2) <= 0.05:
                        goal_tmp = np.array(
                            [self.np_random.uniform(*x_range), self.np_random.uniform(*y_range), z_range[0]])
                if self.name == "allow_rotation":
                    if goal_idx[goal_id] == -1:
                        goal_tmp, _ = self.p.getBasePositionAndOrientation(self.blocks_id[goal_id])
            else:
                raise NotImplementedError
            goal.append(goal_tmp)

        goal = np.array(goal)
        if self.name == "allow_rotation" or self.primitive:
            goal = np.concatenate([goal_type_onehot, goal, goal_onehot], axis=1)
        else:
            goal = np.concatenate([goal, goal_onehot], axis=1)  # goal size: n_goal * (3 + n_object)
        # print("n_active:", self.n_active_object)
        # print("goal:", goal)
        # print("x range:", x_range)
        # print("y range:", y_range)
        # print("z range:", z_range)

        if self.reward_type == "sparse" and self.np_random.uniform() < 0.5 and \
                ((all(goal[:, 2] > np.ones(self.n_goal) * z_range[0]) and not self.primitive) or
                 (all(goal[:, 5] > np.ones(self.n_goal) * z_range[0]) and self.primitive)):
            # Let the robot hold a random goal block
            goal_id = random.choice(goal_idx)
            self.robot.control(
                [0, 0, 0], [1, 0, 0, 0],
                (self.robot.finger_range[0] + self.robot.finger_range[1]) / 2,
                relative=True, teleport=True,
            )
            if goal_id >= 0:
                self.p.resetBasePositionAndOrientation(self.blocks_id[goal_id], self.robot.get_eef_position(), (0, 0, 0, 1))

        if self.primitive:
            goal = np.reshape(goal, (-1, self.n_goal * (6 + self.n_object)))
            goal = np.concatenate([goal, -np.ones((goal.shape[0], (self.n_max_goal - self.n_goal) * (6 + self.n_object)))], axis=-1)
            self.visualize_goal(goal, True)
            return goal

        goal = np.reshape(goal, (-1, self.n_goal * (3 + self.n_object)))  # [1, n_goal*(3+n_object)]
        goal_idx_available = []
        for i in range(self.n_object):
            if i not in goal_idx:
                goal_idx_available.append(i)
        # pad goals till max_goal via setting goal pos at object pos
        if self.n_max_goal > self.n_goal:
            goal_idx_buffer = random.sample(goal_idx_available, self.n_max_goal - self.n_goal)
            for goal_id in goal_idx_buffer:
                cur_pos, _ = self.p.getBasePositionAndOrientation(self.blocks_id[goal_id])
                goal_onehot = np.zeros((1, self.n_object))
                goal_onehot[0][goal_id] = 1
                cur_pos = np.reshape(cur_pos, (1, -1))
                pad = np.concatenate([cur_pos, goal_onehot], axis=1)
                goal = np.concatenate([goal, pad], axis=1)
            # pad = np.ones((np.shape(goal)[0], (self.n_max_goal - self.n_goal) * (3+self.n_object))) * (-1)
        #print(self.n_max_goal, self.n_goal, goal_idx, self.n_active_object)
        #print("ArmPickAndPlace/log sampled goal:", goal)
        # print(self.n_goal, self.n_active_object)
        self.visualize_goal(goal)
        return goal

    def _reset_sim(self):
        robot_pos = self.robot.base_pos
        x_range = self.robot.x_workspace
        y_range = self.robot.y_workspace
        z_range = (robot_pos[2] + 0.025, robot_pos[2] + 0.425)
        eef_pos = np.array([self.np_random.uniform(*x_range), self.np_random.uniform(*y_range),
                            self.robot.init_eef_height])
        self.robot.control(eef_pos, (1, 0, 0, 0), 0., relative=False, teleport=True)
        self.n_active_object = self.np_random.randint(self.n_goal, self.n_object + 1)
        # self.n_active_object = 1
        self.n_goal = random.sample(range(1, min(self.n_max_goal + 1, self.n_active_object + 1)), 1)[0]
        # self.n_goal = 1
        # Randomize initial position of blocks
        block_positions = np.array([[self.np_random.uniform(*x_range), self.np_random.uniform(*y_range), z_range[0]]
                                    for _ in range(self.n_active_object)]
                                   + [[eef_pos[0], eef_pos[1], z_range[0]]])
        '''block_positions = np.array([[x_range[0] + (x_range[1] - x_range[0])* ( _ / self.n_active_object + 0.5), 
                                     y_range[0] + (y_range[1] - y_range[0])* ( _ / self.n_active_object + 0.5), 
                                     z_range[0]]
                                    for _ in range(self.n_active_object)]
                                   + [[eef_pos[0], eef_pos[1], z_range[0]]])'''
        block_halfextent = np.array(self.p.getCollisionShapeData(self.blocks_id[0], -1)[0][3]) / 2
        while _in_collision(block_positions, block_halfextent):
            block_positions = np.array([[self.np_random.uniform(*x_range), self.np_random.uniform(*y_range), z_range[0]]
                                        for _ in range(self.n_active_object)]
                                       + [[eef_pos[0], eef_pos[1], z_range[0]]])
        block_positions = block_positions[:-1]
        if self.stack and self.n_active_object - self.n_max_goal >= 2:
            stack_id = random.sample(list(np.arange(self.n_active_object)), 2)
            block_positions[stack_id[1]][:2] = block_positions[stack_id[0]][:2]
            block_positions[stack_id[1]][2] = z_range[0] + 0.05
            self.stack_obj_idx = stack_id
            self.stack_obj_pos = [block_positions[stack_id[0]], block_positions[stack_id[1]]]
        for n in range(self.n_active_object):
            # randomize orientation if allow rotation
            self.p.resetBasePositionAndOrientation(self.blocks_id[n], block_positions[n], self.gen_obj_quat())
        for n in range(self.n_active_object, self.n_object):
            self.p.resetBasePositionAndOrientation(
                self.blocks_id[n], list(self.inactive_xy) + [z_range[0] + (n - self.n_active_object) * 0.05],
                (0, 0, 0, 1)
            )
            tmp, _ = self.p.getBasePositionAndOrientation(self.blocks_id[n])
            # print(z_range[0] + (n - self.n_active_object) * 0.05)
            # print(tmp)

    # randomize orientation if allow rotation
    def gen_obj_quat(self):
        return (0, 0, 0, 1)
        if self.action_space.shape[0] == 7:
            angle = self.np_random.uniform(-np.pi, np.pi)
            obj_quat = (0, 0, np.cos(angle / 2), np.sin(angle / 2))
        else:
            obj_quat = (0, 0, 0, 1)
        return obj_quat

    def _get_obs(self):
        # changed
        obs = self.robot.get_obs_old()  # robot_obs
        # print("robot_obs:", obs)
        eef_pos = obs[:3]
        if self.robot_dim is None:
            self.robot_dim = len(obs)
        for i in range(self.n_active_object):
            object_pos, object_quat = self.p.getBasePositionAndOrientation(self.blocks_id[i])
            import copy
            original_obj_quat = copy.copy(object_quat)
            # object_quat = convert_symmetric_rotation(np.array(object_quat))
            object_quat = np.array(object_quat)
            object_euler = self.p.getEulerFromQuaternion(object_quat)
            object_velp, object_velr = self.p.getBaseVelocity(self.blocks_id[i])
            object_pos, object_euler, object_velp, object_velr = map(np.array, [object_pos, object_euler, object_velp,
                                                                                object_velr])
            object_velp *= self.dt * self.robot.num_substeps
            object_velr *= self.dt * self.robot.num_substeps
            object_rel_pos = object_pos - eef_pos
            obs = np.concatenate([obs, object_pos, object_rel_pos, object_euler, object_velp, object_velr])
            # print("obs1:", obs)
            if self.n_object > 1:
                if self.primitive:
                    goal_indicator = (i in np.argmax(np.reshape(self.goal, (-1, 6 + self.n_object))[:self.n_goal, 6:],
                                                     axis=1))
                else:
                    goal_indicator = (i in np.argmax(np.reshape(self.goal, (-1, 3 + self.n_object))[:, 3:],
                                                     axis=1))  # changed, bool
                obs = np.concatenate([obs, [goal_indicator]])
                # print("obs2:", obs)
            if self.object_dim is None:
                self.object_dim = len(obs) - self.robot_dim
        for i in range(self.n_active_object, self.n_object):
            obs = np.concatenate([obs, -np.ones(self.object_dim)])
        if self.primitive:
            achieved_goal = np.concatenate([np.reshape(self.goal, (-1, 6 + self.n_object))[:, :3],
                                            self._get_achieved_goal()[0],
                                            np.reshape(self.goal, (-1, 6 + self.n_object))[:, 6:]], axis=-1)
            achieved_goal = np.reshape(achieved_goal, (-1, (6 + self.n_object) * self.n_max_goal))
            # achieved_goal = np.concatenate([achieved_goal, -np.ones((1, (self.n_max_goal - self.n_goal) * (6 + self.n_object)))], axis=1)

            obs_dict = dict(observation=obs, achieved_goal=achieved_goal,
                            desired_goal=self.goal.copy())
        else:
            achieved_goal = np.concatenate(
                [self._get_achieved_goal()[0], np.reshape(self.goal, (-1, 3 + self.n_object))[:, 3:]], axis=1)
            achieved_goal = np.reshape(achieved_goal, (-1, (3 + self.n_object) * self.n_max_goal))
            obs_dict = dict(observation=obs, achieved_goal=achieved_goal,
                            desired_goal=self.goal.copy())
        # print("return obs")
        return obs_dict

    def _get_achieved_goal(self):
        # changed
        if self.primitive:
            goal = np.reshape(self.goal, (-1, 6 + self.n_object))
            n_goal = None
            for i in range(goal.shape[0]):
                if all(goal[i, :] == -1):
                    goal = goal[:i, :]
                    n_goal = i
                    break
            if n_goal is None:
                n_goal = goal.shape[0]
            goal_idx = np.zeros(n_goal)
            if self.n_object == 1:
                goal_idx = np.array([0])
            else:
                for i in range(n_goal):
                    if goal[i][2] == 1:
                        goal_idx[i] = -1
                    else:
                        goal_idx[i] = np.argmax(goal[i, 6:])
        else:
            goal = np.reshape(self.goal, (-1, 3 + self.n_object))
            goal_idx = np.array([0]) if self.n_object == 1 else np.argmax(goal[:self.n_max_goal, 3:], axis=1)
        cur_pos_all = []
        obj_id = []
        for n, goal_id in enumerate(goal_idx):
            if goal_id == -1:
                all_pos = np.array([self.p.getBasePositionAndOrientation(self.blocks_id[i])[0] for i in range(self.n_object)])
                nearest_id = np.argmin(np.linalg.norm(all_pos - goal[n, 3:6]))
                cur_pos = all_pos[nearest_id]
                cur_pos_all.append(cur_pos)
                obj_id.append(nearest_id)
                continue
            cur_pos, _ = self.p.getBasePositionAndOrientation(self.blocks_id[int(goal_id)])
            cur_pos = list(cur_pos)
            if goal_id >= self.n_active_object:
                cur_pos[2] = 0.025 + (
                            goal_id - self.n_active_object) * 0.05  # fix subtle bug with inactive objects out of table
            cur_pos_all.append(cur_pos)
            obj_id.append(int(goal_id))
        if self.primitive:
            for i in range(n_goal, self.n_max_goal):
                cur_pos_all.append([-1, -1, -1])
                obj_id.append(-1)
        return np.array(cur_pos_all), np.array(obj_id)  # with length n_goal

    def compute_reward_and_info(self):
        # changed
        if self.primitive:
            goal = np.reshape(self.goal, (-1, 6 + self.n_object))
            goal_pos = goal[:, 3:6]
        else:
            goal = np.reshape(self.goal, (-1, 3 + self.n_object))
            goal_pos = goal[:, :3]
        cur_pos, obj_id = self._get_achieved_goal()
        n_goal = None
        for i in range(cur_pos.shape[0]):
            if all(cur_pos[i, :] == -1):
                n_goal = i
                break
        if n_goal is None:
            n_goal = cur_pos.shape[0]
        distance = np.linalg.norm(goal_pos - cur_pos, axis=1)
        if self.reward_type == "sparse":
            if self.primitive:
                is_success = True
                for i in range(n_goal):
                    pos, orn = self.p.getBasePositionAndOrientation(self.blocks_id[obj_id[i]])
                    rot_matrix = _quaternion2RotationMatrix(orn)
                    goal_pos_b_abs = np.abs(np.matmul(rot_matrix, goal_pos[i] - pos))
                    block_half_extent = np.array(self.p.getCollisionShapeData(self.blocks_id[obj_id[i]], -1)[0][3]) / 2
                    if goal[i, 2] == 1 and all(goal_pos_b_abs <= block_half_extent):
                        is_success = False
                    if goal[i, 1] == 1 and any(goal_pos_b_abs > block_half_extent):
                        is_success = False
                reward = float(is_success)
            else:
                reward = float(all(distance < 0.05 * np.ones(self.n_max_goal)))
                is_success = all((distance < 0.05 * np.ones(self.n_max_goal)))
        elif self.reward_type == "dense":
            reward = np.sum(self._previous_distance - distance)
            self._previous_distance = distance
            is_success = all((distance < 0.05 * np.ones(self.n_max_goal)))
        else:
            raise NotImplementedError
        # print(reward, is_success)
        info = {'is_success': is_success}
        return reward, info

    def compute_reward(self, obs, goal):
        # not changed
        # For HER
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(goal, torch.Tensor):
            goal = goal.cpu().numpy()
        assert isinstance(obs, np.ndarray) and isinstance(goal, np.ndarray)
        goal = np.reshape(goal, (-1, 3 + self.n_object))
        goal_pos = goal[:, :3]
        goal_idx = np.argmax(goal[:, 3:], axis=1)
        # TODO
        cur_pos_all = []
        for goal_id in goal_idx:
            cur_pos = obs[self.robot_dim + self.object_dim * goal_id: self.robot_dim + self.object_dim * goal_id + 3]
            cur_pos_all.append(cur_pos)
        cur_pos_all = np.array(cur_pos_all)
        distance = np.linalg.norm(cur_pos_all - goal_pos, axis=1)
        # Since it is inconvenient to get previous reward when performing HER, we only accept sparse reward here
        assert self.reward_type == "sparse"
        reward = (distance < 0.05 * np.ones(self.n_goal))
        return float(reward), reward

    def relabel_obs(self, obs, goal):
        assert isinstance(obs, np.ndarray) and isinstance(goal, np.ndarray)
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        if len(goal.shape) == 1:
            goal = np.expand_dims(goal, axis=0)
        assert len(obs.shape) == 2 and len(goal.shape) == 2
        obs = obs.copy()
        obs_len = obs.shape[1]
        # print("ArmPickAndPlace/relabel_obs: obs.shape =", np.shape(obs))
        if self.primitive:
            goal = np.reshape(goal, (goal.shape[0], -1, 6 + self.n_object))
            n_goal = [0] * goal.shape[0]
            goal_idx = []
            for idx in range(goal.shape[0]):
                for goal_id in range(goal.shape[1]):
                    if all(goal[idx, goal_id, 6:] == -1):
                        n_goal[idx] = goal_id
                        break
                if n_goal[idx] == 0:
                    n_goal[idx] = goal.shape[1]
                goal_idx.append(np.argmax(goal[idx, :n_goal[idx], 6:], axis=-1))
            # goal_idx = np.argmax(goal[:, :n_goal, 6:], axis=-1)
        else:
            goal = np.reshape(goal, (goal.shape[0], -1, 3 + self.n_object))
            goal_idx = np.argmax(goal[:, :, 3:], axis=-1)
            n_goal = [goal.shape[1]] * goal.shape[0]
        goal_dim = goal.shape[-1]
        n_max_goal = self.n_max_goal
        # print(goal.shape, n_goal, n_max_goal)
        goal_indicator_idx = np.arange(
            self.robot_dim + self.object_dim - 1, obs.shape[-1] - 2 * goal_dim * n_max_goal, self.object_dim
        )
        obs[:, goal_indicator_idx] = 0.
        # for each random sampled point
        for idx in range(obs.shape[0]):
            # target obj idx
            for goal_id in range(n_goal[idx]):
                obs[idx, self.robot_dim + goal_idx[idx][goal_id] * self.object_dim + self.object_dim - 1] = 1.
                if self.primitive:
                    obs[idx, -2 * goal_dim * n_max_goal + goal_dim * goal_id + 3: -2 * goal_dim * n_max_goal + goal_dim * goal_id + 6] = \
                        obs[idx, self.robot_dim + goal_idx[idx][goal_id] * self.object_dim:
                                self.robot_dim + goal_idx[idx][goal_id] * self.object_dim + 3]
                    obs[idx, -2 * goal_dim * n_max_goal + goal_dim * goal_id + 6: -2 * goal_dim * n_max_goal + goal_dim * (
                            goal_id + 1)] = goal[idx, goal_id, 6:]
                    obs[idx, -2 * goal_dim * n_max_goal + goal_dim * goal_id: -2 * goal_dim * n_max_goal + goal_dim * goal_id + 3] = np.array([0, 1, 0])
                else:
                    obs[idx,
                            -2 * goal_dim * n_goal + goal_dim * goal_id: -2 * goal_dim * n_goal + goal_dim * goal_id + 3] = \
                            obs[idx, self.robot_dim + goal_idx[idx][goal_id] * self.object_dim:
                                    self.robot_dim + goal_idx[idx][goal_id] * self.object_dim + 3]
                    obs[idx, -2 * goal_dim * n_goal + goal_dim * goal_id + 3: -2 * goal_dim * n_goal + goal_dim * (
                            goal_id + 1)] = goal[idx, goal_id, 3:]
                obs[idx, obs.shape[1] - goal_dim * n_max_goal + goal_dim * goal_id: obs.shape[1] - goal_dim *
                        n_max_goal + goal_dim * (goal_id + 1)] = goal[idx, goal_id, :]
            for goal_id in range(n_goal[idx], n_max_goal):
                tmp = -np.ones_like(goal[idx, 0, :])
                obs[idx, obs.shape[1] - goal_dim * n_max_goal + goal_dim * goal_id: obs.shape[1] - goal_dim *
                        n_max_goal + goal_dim * (goal_id + 1)] = tmp
                obs[idx, -2 * goal_dim * n_max_goal + goal_dim * goal_id: -2 * goal_dim *
                        n_max_goal + goal_dim * (goal_id + 1)] = tmp
        return obs

    def imagine_obs(self, obs, goal, info):
        # imagine a goal has been achieved
        # TODO: further check shape
        # todo: try to imagine multiple objects movement, ideally with a generative model?
        assert isinstance(obs, np.ndarray) and isinstance(goal, np.ndarray)
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        if len(goal.shape) == 1:
            goal = np.expand_dims(goal, axis=0)
        assert len(obs.shape) == 2 and len(goal.shape) == 2
        goal = np.reshape(goal, (goal.shape[0], -1, 3 + self.n_object))
        # print("ArmPickAndPlace/imagine_obs: obs.shape =", np.shape(obs))
        obs = obs.copy()
        goal_idx = np.argmax(goal[:, :, 3:], axis=-1)
        goal_dim = goal.shape[-1]
        n_max_goal = goal.shape[1]
        ultimate_goal_idx = []
        for goal_id in range(n_max_goal):
            tmp_idx = np.argmax(obs[:, -goal_dim * (n_max_goal - goal_id) + 3: -goal_dim * (n_max_goal - goal_id - 1)],
                                axis=-1)
            tmp_idx = np.reshape(tmp_idx, (1, -1))
            ultimate_goal_idx.append(tmp_idx)
        ultimate_goal_idx = np.transpose(np.array(ultimate_goal_idx))  # shape: [1024, 2]

        # ultimate_goal_idx = np.argmax(obs[:, -goal_dim + 3:], axis=-1)

        # not used yet?
        def is_in_tower(pos: torch.Tensor, goal: torch.Tensor):
            if torch.norm(pos[:2] - goal[:2]) > 0.01:
                return False
            maybe_n_height = (pos[2] - self.robot.z_workspace[0] - 0.025) / 0.05
            if (maybe_n_height - torch.round(maybe_n_height)).abs() > 1e-5:
                return False
            return True

        for idx in range(obs.shape[0]):
            # original objects positions
            max_height = [self.robot.base_pos[2] - 0.025 + 0.05 * info["n_base"][i] for i in range(n_max_goal)]
            # n_imagine_to_stack_old = int(torch.round((goal[idx, 2] - max_height_old) / 0.05).item())
            n_imagine_to_stack = \
                [int(np.round((goal[idx, i, 2] - self.robot.base_pos[2] - 0.025) / 0.05)) + 1 - info["n_base"][i] for i
                 in range(n_max_goal)]
            if any(n_imagine_to_stack > np.zeros(n_max_goal)):
                # n_base_old = int(torch.round((max_height_old - self.robot.z_workspace[0] - 0.025) / 0.05).item()) + 1
                n_base = info["n_base"]
                move_objects_candidates = list(range(np.sum(n_base), self.n_object))
                for i in range(n_max_goal):
                    if ultimate_goal_idx[idx][i] in move_objects_candidates:
                        move_objects_candidates.remove(ultimate_goal_idx[idx][i])
                for i in range(n_max_goal):
                    if goal_idx[idx, i] in move_objects_candidates:
                        move_objects_candidates.remove(goal_idx[idx, i])  # imagined top id
                # sample move_objects_id
                all_move_objects_id = list(
                    np.random.choice(move_objects_candidates, size=np.sum(n_imagine_to_stack) - n_max_goal,
                                     replace=False))
                move_objects_id = []
                for i in range(n_max_goal):
                    tmp_sum = np.sum(n_imagine_to_stack[:i])
                    move_objects_id.append(
                        all_move_objects_id[tmp_sum: tmp_sum + n_imagine_to_stack[i]] + [goal_idx[idx, i]])
                # move_objects_id = list(np.random.choice(move_objects_candidates, size=n_imagine_to_stack - 1, replace=False)) + [goal_idx[idx]]
                for i in range(n_max_goal):
                    assert abs(max_height[i] + len(move_objects_id[i]) * 0.05 - goal[idx, i, 2]) < 1e-3
                for i in range(n_max_goal):
                    for h_idx in range(len(move_objects_id[i])):
                        obs[idx, self.robot_dim + move_objects_id[i][h_idx] * self.object_dim:
                                 self.robot_dim + move_objects_id[i][h_idx] * self.object_dim + 3] = \
                            np.concatenate([goal[idx, i, :2], [max_height[i] + (h_idx + 1) * 0.05]])
                        obs[idx, self.robot_dim + move_objects_id[i][h_idx] * self.object_dim + 3:
                                 self.robot_dim + move_objects_id[i][h_idx] * self.object_dim + 6] = \
                            np.concatenate([goal[idx, i, :2], [max_height[i] + (h_idx + 1) * 0.05]]) - obs[idx, :3]
            else:
                for i in range(n_max_goal):
                    assert abs(goal[idx, i, 2] - (self.robot.base_pos[2] + 0.025)) < 1e-3
                for i in range(n_max_goal):
                    obs[idx, self.robot_dim + goal_idx[idx][i] * self.object_dim:
                             self.robot_dim + goal_idx[idx][i] * self.object_dim + 3] = goal[idx, i, :3]
                    obs[idx, self.robot_dim + goal_idx[idx][i] * self.object_dim + 3:
                             self.robot_dim + goal_idx[idx][i] * self.object_dim + 6] = goal[idx, i, :3] - obs[idx, :3]
            '''
            # object position
            obs[idx, self.robot_dim + goal_idx[idx] * self.object_dim:
                     self.robot_dim + goal_idx[idx] * self.object_dim + 3] = goal[idx, :3]
            # relative position
            obs[idx, self.robot_dim + goal_idx[idx] * self.object_dim + 3:
                     self.robot_dim + goal_idx[idx] * self.object_dim + 6] = goal[idx, :3] - obs[idx, :3]
            '''
            # achieved goal
            for i in range(n_max_goal):
                obs[idx, -2 * goal_dim * n_max_goal + i * goal_dim: -2 * goal_dim * n_max_goal + i * goal_dim + 3] = \
                    obs[idx, self.robot_dim + ultimate_goal_idx[idx][i] * self.object_dim:
                             self.robot_dim + ultimate_goal_idx[idx][i] * self.object_dim + 3]
        return obs

    def get_state(self):
        state_dict = dict(robot=None, objects=dict(qpos=[], qvel=[]))
        state_dict["robot"] = self.robot.get_state()
        for body_id in self.blocks_id:
            body_pos, body_quat = self.p.getBasePositionAndOrientation(body_id)
            vel, vela = self.p.getBaseVelocity(body_id)
            state_dict["objects"]["qpos"].append(
                np.concatenate([np.array(body_pos), np.array(body_quat)])
            )
            state_dict["objects"]["qvel"].append(
                np.concatenate([np.array(vel), np.array(vela)])
            )
        return state_dict

    def set_state(self, state_dict, env_number=None, set_robot_state=True, reset_robot=False):
        if env_number is not None:
            self.env_number = env_number
        if set_robot_state:
            self.robot.set_state(state_dict["robot"])
        if reset_robot:
            self.robot.reset()
        for i in range(self.n_object):
            qpos = state_dict["objects"]["qpos"][i]
            qvel = state_dict["objects"]["qvel"][i]
            self.p.resetBasePositionAndOrientation(self.blocks_id[i], qpos[:3], qpos[3:])
            print(f"Set block {i} to {qpos[:3]}")
            self.p.resetBaseVelocity(self.blocks_id[i], qvel[:3], qvel[3:])

    def get_env_number(self):
        return self.env_number

    def get_achieved_goal(self):
        return self._get_achieved_goal()

    def get_n_object(self):
        return self.n_object

    def get_goal(self):
        return self.goal

# changed, generalization: each alternation has several goals on the same level
# n_max_goal
class ArmStack(ArmPickAndPlace):
    def __init__(self, *args, n_to_stack=[[1, 2, 3], [1, 2, 3]], name=None, generate_data=False, action_dim=4, 
                 use_expand_goal_prob=0, expand_task_path=None, permute_object=False, **kwargs):
        self.env_id = "BulletStack-v1"
        self.name = name
        self.generate_data = generate_data
        self.action_dim = action_dim
        if self.action_space is None:
            self.action_space = spaces.Box(-1., 1., shape=(self.action_dim,))
        if self.name == "allow_rotation":
            self.n_object = 6
            self.n_max_goal = 6
            self.change_height_prob = 0  # whether use rectangle
            # self.use_expand_goal_prob = 1  # for rl tuning, use saved expand trajectory as goal or not
            self.use_expand_goal_prob = use_expand_goal_prob
            self.put_goal_aside_prob = 0  # use pyramid expansion or not
            self.multi_step_goal_prob = 0
            self.multi_goal_prob = 0
            self.pyramid_evaluation = False
        elif self.name == "default":
            self.n_object = 6
            self.n_max_goal = 6
            self.change_height_prob = 0  # whether use rectangle
            self.use_expand_goal_prob = use_expand_goal_prob  # for rl tuning, use saved expand trajectory as goal or not
            self.put_goal_aside_prob = 0  # use pyramid expansion or not
            self.pyramid_evaluation = False
        self.n_to_stack_choices = n_to_stack  # shape: [n_goal, choices]
        self.n_to_stack_probs = [[1. / len(n_to_stack[i])] * len(n_to_stack[i]) for i in range(np.shape(n_to_stack)[0])]
        self.n_to_stack = [len(n_to_stack[i]) for i in range(np.shape(n_to_stack)[0])]
        self.n_to_stack_all = np.sum(self.n_to_stack)
        # number of objects on the same base at present
        self.n_base = np.zeros(self.n_max_goal)
        self.n_base_all = np.sum(self.n_base)
        # position of each base
        self.base_xy = [[0.08 * i + 0.3, 0] for i in range(self.n_max_goal)]
        self.cl_ratio = 0

        self.use_expand_goal = False  # randomly choose in reset_sim
        self.put_goal_aside = False
        self.inside_pyramid = False  # # precisely classify goal genre, new goals
        self.stack_straight = False  # precisely classify goal genre, new goals
        self.multi_goal = False
        self.multi_step_goal = False

        self.expand_traj = None  # if use saved expand traj, the current goal and simulated objects
        if self.use_expand_goal_prob > 0 and expand_task_path is not None:
            with open(expand_task_path, "rb") as f:  # collect_data_last_step
                data = pickle.load(f)
                self.offline_datasets = data["expansion"]  # 4300 data in form of obs, actions, rewards
                # self.classified_data = self.gen_data_classify(data)
                # self.n_inside_pyramid = self.data_statistics(data["expansion"])
        self.permute_object = permute_object
        super(ArmStack, self).__init__(*args, generate_data=generate_data, action_dim=action_dim, **kwargs)

    def gen_data_classify(self, data):
        dict = {"up_075": [], "up_125": [], "up_175": [], "up_225": [], "up_275": [], "up_325": [],
                "hor_075": [], "hor_125": [], "hor_175": [], "hor_225": [], "hor_275": [], "hor_325": []}
        for i in range(len(data)):
            height = data[i]["obs"][0, -(6 + self.n_object) * self.n_max_goal:][5]
            y_orn = data[i]["actions"][-1][5]
            if y_orn > 0.9:
                if height <= 0.08:
                    dict["up_075"].append(i)
                elif height <= 0.13:
                    dict["up_125"].append(i)
                elif height <= 0.18:
                    dict["up_175"].append(i)
                elif height <= 0.23:
                    dict["up_225"].append(i)
                elif height <= 0.28:
                    dict["up_275"].append(i)
                else:
                    dict["up_325"].append(i)
            else:
                if height <= 0.08:
                    dict["hor_075"].append(i)
                elif height <= 0.13:
                    dict["hor_125"].append(i)
                elif height <= 0.18:
                    dict["hor_175"].append(i)
                elif height <= 0.23:
                    dict["hor_225"].append(i)
                elif height <= 0.28:
                    dict["hor_275"].append(i)
                else:
                    dict["hor_325"].append(i)
        return dict

    def env_statistics(self, goal):
        # goal shape: n_goal * (3 + n_object)
        goal = np.reshape(goal, (-1, 3 + self.n_object))
        goal_height = goal[:, 2]
        maybe_inpy = False
        inpy_idx = []
        for m in range(self.n_max_goal):
            for n in range(m):
                if np.linalg.norm(goal[m, :2] - goal[n, :2]) < 0.05:
                    maybe_inpy = True
                    if m not in inpy_idx:
                        inpy_idx.append(m)
                    if n not in inpy_idx:
                        inpy_idx.append(n)
        if maybe_inpy:
            for m in range(self.n_max_goal):
                for n in range(m):
                    if m in inpy_idx and n in inpy_idx and np.abs(goal_height[m] - goal_height[n]) > 0.03:
                        self.inside_pyramid = True
                        return
        return

    def data_statistics(self, data):
        possible_in_pyr = []
        for j in range(len(data)):
            obs = data[j]["obs"]
            obs_end = obs[-1]
            goal = [obs_end[13+16*self.n_object+(self.n_max_goal+m)*(3+self.n_object)] for m in range(self.n_max_goal)]
            goal_pos = [obs_end[11+16*self.n_object+(self.n_max_goal+m)*(3+self.n_object):
                                13+16*self.n_object+(self.n_max_goal+m)*(3+self.n_object)] for m in range(self.n_max_goal)]
            maybe_inpy = False
            inpy_idx = []
            for m in range(self.n_max_goal):
                for n in range(m):
                    if np.linalg.norm(goal_pos[m] - goal_pos[n]) < 0.05:
                        maybe_inpy = True
                        if m not in inpy_idx:
                            inpy_idx.append(m)
                        if n not in inpy_idx:
                            inpy_idx.append(n)
            if maybe_inpy:
                for m in range(self.n_max_goal):
                    for n in range(m):
                        if m in inpy_idx and n in inpy_idx and np.abs(goal[m] - goal[n]) > 0.03 \
                                and j not in possible_in_pyr:
                            possible_in_pyr.append(j)
                            break
        return possible_in_pyr

    def _reset_sim(self):
        # todo: more distracting tasks, e.g. distracting towers
        # self.robot.control([self.robot.base_pos[0] + 0.2, self.robot.base_pos[1] - 0.2, self.robot.base_pos[2] + 0.3],
        #                    (1, 0, 0, 0), 0., relative=False, teleport=True)
        try:
            self.robot.set_state(dict(qpos=np.array([-0.35720248, -0.75528038, -0.36600858, -2.77078997, -0.27654494,
                                                  2.02585467, 0.28351196, 0., 0., 0., 0., 0.]),
                                  qvel=np.zeros(12)))
        except:
            pass
        try:
            self.robot.reset()
        except:
            pass
        if random.uniform(0, 1) < self.use_expand_goal_prob:
            self.use_expand_goal = True
            # if random.uniform(0, 1) < 0.4:  # 0.4, 0.5
            #     traj_idx = int(random.sample(list(self.n_inside_pyramid), 1)[0])
            # else:
            #     traj_idx = int(random.sample(range(len(self.offline_datasets)), 1)[0])  # random.randint(0, len(self.offline_datasets) - 1)
            # if traj_idx in self.n_inside_pyramid:
            #     self.inside_pyramid, self.stack_straight = True, False
            # else:
            #     self.inside_pyramid, self.stack_straight = False, True
            #possibility = random.uniform(0, 1)
            #if possibility < 1:
            #    # traj_idx = int(random.choice(range(128)))
            #    traj_idx = int(random.choice(self.classified_data["up_075"]))
            #elif possibility < 0.5:
            #    traj_idx = int(random.choice(self.classified_data["up_125"]))
            #elif possibility < 0.75:
            #    traj_idx = int(random.choice(self.classified_data["up_225"]))
            #else:
            #    traj_idx = int(random.choice(self.classified_data["hor_075"]))
            if hasattr(self, "presampled_obs"):
                self.expand_traj = self.presampled_obs
            else:
                traj_idx = int(random.sample(range(len(self.offline_datasets)), 1)[0])
                self.traj_idx = traj_idx
                self.expand_traj = self.offline_datasets[traj_idx]["obs"][0]
                print("traj idx", self.traj_idx)
            self.object_permutation = np.arange(self.n_object)
            if self.permute_object:
                np.random.shuffle(self.object_permutation)
            for n in range(self.n_object):
                self.p.resetBasePositionAndOrientation(
                    self.blocks_id[self.object_permutation[n]], self.expand_traj[11 + 16 * n: 14 + 16 * n],
                    self.p.getQuaternionFromEuler(self.expand_traj[17 + 16 * n: 20 + 16 * n])  # todo
                )
            self.robot.control(self.expand_traj[:3], (1, 0, 0, 0), 0., relative=False, teleport=True)
            return
        else:
            self.use_expand_goal = False
            self.inside_pyramid, self.stack_straight = False, False

        if self.primitive:  # if primitive, initial objects are all on the ground/on other objects
            self.n_goal = 1
            # self.n_goal = random.choice(range(1, self.n_max_goal + 1))
            self.n_active_object = self.n_object
            if self.name == "allow_rotation":
                all_position = np.array([[self.np_random.uniform(*self.robot.x_workspace),
                                             self.np_random.uniform(*self.robot.y_workspace),
                                             self.robot.base_pos[2] + self.np_random.uniform(0.025, 0.075)]
                                            for _ in range(self.n_active_object)])
            elif self.name == "default":
                all_position = np.array([[self.np_random.uniform(*self.robot.x_workspace),
                                          self.np_random.uniform(*self.robot.y_workspace),
                                          self.robot.base_pos[2] + 0.025]
                                         for _ in range(self.n_active_object)])
            else:
                raise NotImplementedError
            for n in range(self.n_active_object):
                self.p.resetBasePositionAndOrientation(
                    self.blocks_id[n], all_position[n], self.gen_obj_quat()
                )
            eef_pos = np.array([self.np_random.uniform(*self.robot.x_workspace),
                                self.np_random.uniform(*self.robot.y_workspace),
                                self.robot.init_eef_height + 0.4])  # +0.1
            self.robot.control(eef_pos, (1, 0, 0, 0), 0., relative=False, teleport=True)
            for frame_count in range(15):
                self.p.stepSimulation()
            return

        # self.n_goal = random.choice(range(1, self.n_max_goal + 1))
        self.n_goal = 1

        self.n_to_stack = [self.np_random.choice(self.n_to_stack_choices[i], p=self.n_to_stack_probs[i]) for i in
                           range(self.n_goal)]
        if any(self.n_to_stack == -np.ones(self.n_goal)):
            print("n_to_stack_choices:", self.n_to_stack_choices)
            print("n_to_stack_probs:", self.n_to_stack_probs)
            print("n_to_stack", self.n_to_stack)
        # self.n_active_object = self.np_random.randint(self.n_to_stack, self.n_object + 1)
        #print(self.n_to_stack)
        if all(self.n_to_stack == np.ones(self.n_goal)) and \
                self.cl_ratio > 0 and self.np_random.uniform() < self.cl_ratio:
            self.n_active_object = self.np_random.randint(min(4, self.n_object), self.n_object + 1)
        else:
            self.n_active_object = self.n_object
        #self.n_active_object = 1
        if self.n_max_goal > self.n_goal:
            self.n_to_stack = np.concatenate([self.n_to_stack, [0] * (self.n_max_goal - self.n_goal)])
        # randomly select present number of base blocks
        # this selection may cause trivial policy for base 2&3?
        self.n_to_stack_all = int(np.sum(self.n_to_stack))
        self.n_base = np.zeros(self.n_max_goal)
        for id in range(self.n_goal, self.n_max_goal):
            self.n_base[id] = 1
        for id in range(self.n_goal):
            tmp_sum = np.sum(self.n_base)
            self.n_base[id] = int(
                self.np_random.randint(0, min(self.n_active_object - self.n_to_stack_all - tmp_sum + 1, 3)))  # self.n_active_object - self.n_to_stack + 1
        if self.name == "allow_rotation" or self.primitive:
            self.n_base = np.zeros(self.n_max_goal)  # TODO: delete for diversity

        if self.pyramid_evaluation:
            for id in range(self.n_goal):
                self.n_base[id] = 0
                self.n_to_stack[id] = 1

        if random.uniform(0, 1) < self.put_goal_aside_prob:
            self.put_goal_aside = True
            for id in range(self.n_goal):
                self.n_base[id] = 0
                self.n_to_stack[id] = 1
        else:
            self.put_goal_aside = False
        self.n_base_all = int(np.sum(self.n_base))
        self.n_to_stack_all = int(np.sum(self.n_to_stack))

        if self.n_base_all + self.n_goal > self.n_active_object:
            print(self.n_base)
            print(self.n_to_stack)
            print(self.n_goal)
            print(self.n_active_object)
            raise ArithmeticError("Not enough objects to be sampled other than bases")
        # other objects distribute on the ground randomly
        # first 2 rows are positions of 2 bases
        base_and_other_position = np.array([[self.np_random.uniform(*self.robot.x_workspace),
                                             self.np_random.uniform(*self.robot.y_workspace),
                                             self.robot.base_pos[2] + 0.025]
                                            for _ in range(self.n_max_goal + self.n_active_object - self.n_base_all)])
        block_halfextent = np.array(self.p.getCollisionShapeData(self.blocks_id[0], -1)[0][3]) / 2
        _count = 0
        while _in_collision(base_and_other_position, block_halfextent * 2) and _count < 10:
            base_and_other_position = np.array([[self.np_random.uniform(*self.robot.x_workspace),
                                                 self.np_random.uniform(*self.robot.y_workspace),
                                                 self.robot.base_pos[2] + 0.025]
                                                for _ in
                                                range(self.n_max_goal + self.n_active_object - self.n_base_all)])
            _count += 1
        self.base_xy = [base_and_other_position[i][:2] for i in range(self.n_max_goal)]
        for i in range(self.n_max_goal):
            for n in range(int(np.sum(self.n_base[:i])), int(np.sum(self.n_base[:i + 1]))):
                self.p.resetBasePositionAndOrientation(
                    self.blocks_id[n],
                    (self.base_xy[i][0] + random.uniform(-0.01, 0.01), self.base_xy[i][1] + random.uniform(-0.01, 0.01),
                     self.robot.base_pos[2] + 0.025 + 0.05 * (n - np.sum(self.n_base[:i]))),
                    self.gen_obj_quat()
                )
                #print(n, self.base_xy[i][0], self.base_xy[i][1],
                #     self.robot.base_pos[2] + 0.025 + 0.05 * (n - np.sum(self.n_base[:i])))
        for n in range(self.n_base_all, self.n_active_object):
            self.p.resetBasePositionAndOrientation(
                self.blocks_id[n], base_and_other_position[self.n_max_goal + n - self.n_base_all], self.gen_obj_quat()
            )
            #print(n, base_and_other_position[self.n_max_goal + n - self.n_base_all])
        for n in range(self.n_active_object, self.n_object):
            self.p.resetBasePositionAndOrientation(
                self.blocks_id[n],
                list(self.inactive_xy) + [self.robot.base_pos[2] + 0.025 + (n - self.n_active_object) * 0.05],
                (0, 0, 0, 1)
            )
            #print(n, list(self.inactive_xy) + [self.robot.base_pos[2] + 0.025 + (n - self.n_active_object) * 0.05])
        eef_pos = np.array([self.np_random.uniform(*self.robot.x_workspace),
                            self.np_random.uniform(*self.robot.y_workspace),
                            self.robot.init_eef_height + 0.1])
        while any(np.linalg.norm(np.tile(eef_pos[:2], (self.n_max_goal, 1)) - self.base_xy, axis=1) < 0.05):
            eef_pos = np.array([self.np_random.uniform(*self.robot.x_workspace),
                                self.np_random.uniform(*self.robot.y_workspace),
                                self.robot.init_eef_height + 0.1])
        self.robot.control(eef_pos, (1, 0, 0, 0), 0., relative=False, teleport=True)
        # if self.p.getDynamicsInfo(self.robot.id, 9)[1] != 1:
        #     print("unusual friction case")
        #     print(self.robot.get_eef_position())
        #     for n in range(self.n_object):
        #         print(self.p.getBasePositionAndOrientation(self.blocks_id[n]))

    def _sample_goal(self):
        if self.use_expand_goal:
            if self.primitive:
                assert (len(self.expand_traj) - self.robot_dim - self.n_object * self.object_dim) % (6 + self.n_object) == 0
                n_goal = (len(self.expand_traj) - self.robot_dim - self.n_object * self.object_dim) / (6 + self.n_object)
                if int(n_goal / 2) != self.n_max_goal:
                    print("Warning: data n_goal different from env n_goal")
                    goal = np.concatenate([self.expand_traj[-(6 + self.n_object) * int(n_goal / 2):], -1 * np.ones((6 + self.n_object) * (self.n_max_goal - int(n_goal / 2)))])
                else:
                    goal = self.expand_traj[-(6 + self.n_object) * self.n_max_goal:]
                if self.permute_object:
                    # change onehot according to object permutation
                    goal = np.reshape(goal, (self.n_max_goal, 6 + self.n_object))
                    for j in range(self.n_max_goal):
                        if np.max(goal[j][6:]) == 1:
                            obj_idx = self.object_permutation[np.argmax(goal[j][6:])]
                            goal[j][6:] = 0.
                            goal[j][6 + obj_idx] = 1.
                goal = np.reshape(goal, (-1, (6 + self.n_object) * self.n_max_goal))
                self.visualize_goal(goal, True)
            else:
                goal = self.expand_traj[-(3 + self.n_object) * self.n_max_goal:]
                goal = np.reshape(goal, (-1, (3 + self.n_object) * self.n_goal))
                self.visualize_goal(goal)
            return goal

        goal = np.zeros((self.n_goal, 3))

        goal_onehot = np.zeros((self.n_goal, self.n_object))
        goal_type_onehot = torch.zeros((self.n_goal, 3))
        goal_type_onehot[:, 1] = 1
        # print(self.n_goal, type(self.n_goal))
        # print(self.n_base_all, type(self.n_base_all))
        # print(self.n_active_object, type(self.n_active_object))
        if self.n_base_all + self.n_goal > self.n_active_object:
            raise ArithmeticError("Not enough objects to be sampled other than bases")
        goal_idx = random.sample(range(int(self.n_base_all), self.n_active_object), self.n_goal)
        for idx, goal_id in enumerate(goal_idx):
            goal_onehot[idx, goal_id] = 1

        for idx in range(self.n_goal):
            if random.uniform(0, 1) < self.change_height_prob:
                goal[idx, :] = np.array([self.base_xy[idx][0], self.base_xy[idx][1],
                                         self.robot.base_pos[2] + 0.065 + (
                                                 self.n_base[idx] + self.n_to_stack[idx] - 1) * 0.05])
            elif self.put_goal_aside:
                # put new goal aside some tower on the ground
                prob = random.uniform(0, 1)
                if prob <= 0.25:
                    goal[idx, :] = np.array([self.base_xy[idx][0] + 0.05, self.base_xy[idx][1],
                                             self.robot.base_pos[2] + 0.025])
                elif prob <= 0.5:
                    goal[idx, :] = np.array([self.base_xy[idx][0] - 0.05, self.base_xy[idx][1],
                                             self.robot.base_pos[2] + 0.025])
                elif prob <= 0.75:
                    goal[idx, :] = np.array([self.base_xy[idx][0], self.base_xy[idx][1] + 0.05,
                                             self.robot.base_pos[2] + 0.025])
                else:
                    goal[idx, :] = np.array([self.base_xy[idx][0], self.base_xy[idx][1] - 0.05,
                                             self.robot.base_pos[2] + 0.025])
                # self.n_base[idx] = 0
                # self.n_to_stack[idx] = 1
            elif self.pyramid_evaluation:
                available = []
                for i in range(self.n_active_object):
                    if i not in goal_idx:
                        available.append(i)
                target = random.sample(available, 1)[0]
                pos = self.p.getBasePositionAndOrientation(target)[0]
                goal[idx, :] = np.array([pos[0] + random.uniform(-0.02, 0.02), pos[1] + random.uniform(-0.02, 0.02),
                                         pos[2] + 0.025])
            elif self.primitive and not self.generate_data:
                available_obj = []
                for n in range(self.n_active_object):
                    if n not in goal_idx:
                        available_obj.append(n)
                all_pos = [self.p.getBasePositionAndOrientation(self.blocks_id[n])[0] for n in available_obj]
                if self.name == "allow_rotation":
                    if random.uniform(0, 1) < 0.5:
                        base_id = random.choice(range(len(available_obj)))
                        goal[idx, :] = np.array([all_pos[base_id][0],
                                                 all_pos[base_id][1],
                                                 all_pos[base_id][2] + self.np_random.uniform(0.025, 0.075)])
                    else:
                        goal[idx, :] = np.array([self.np_random.uniform(*self.robot.x_workspace),
                                            self.np_random.uniform(*self.robot.y_workspace),
                                            self.robot.base_pos[2] + self.np_random.uniform(0.025, 0.075)])
                elif self.name == "default":
                    if random.uniform(0, 1) < 0.5:
                        base_id = random.choice(available_obj)
                        goal[idx, :] = np.array([all_pos[base_id][0],
                                                 all_pos[base_id][1],
                                                 all_pos[base_id][2] + 0.05])
                    else:
                        goal[idx, :] = np.array([self.np_random.uniform(*self.robot.x_workspace),
                                                 self.np_random.uniform(*self.robot.y_workspace),
                                                 self.robot.base_pos[2] + 0.025])
                else:
                    raise NotImplementedError
            elif self.generate_data and self.primitive:
                z_range = [0.075, 0.125, 0.175, 0.225, 0.275, 0.325]  # [0.075, 0.125, 0.225]
                goal[idx, :] = np.array([self.np_random.uniform(*self.robot.x_workspace),
                                         self.np_random.uniform(*self.robot.y_workspace),
                                         random.choice(z_range)])
            else:
                self.stack_straight = True
                goal[idx, :] = np.array([self.base_xy[idx][0], self.base_xy[idx][1],
                                         self.robot.base_pos[2] + 0.025 + (
                                                 self.n_base[idx] + self.n_to_stack[idx] - 1) * 0.05])
        if self.primitive:
            goal = np.concatenate([goal_type_onehot, goal, goal_onehot], axis=1)
        else:
            goal = np.concatenate([goal, goal_onehot], axis=1)

        if self.generate_data:
            assert self.primitive
            if self.name == "allow_rotation":
                self.goal_orn = []
                for i in range(self.n_goal):
                    if random.uniform(0, 1) < 0.5:
                        self.goal_orn.append(self.p.getQuaternionFromEuler([0, math.pi / 2, 0]))
                    else:
                        self.goal_orn.append(self.gen_obj_quat())
            else:
                self.goal_orn = [(0, 0, 0, 1) for _ in range(self.n_goal)]

        """
        if self.reward_type == "sparse" and all(self.n_to_stack >= np.ones(self.n_goal)):
            for i in range(self.n_goal):
                if self.n_base[i] > 0 and self.np_random.uniform() < 0.5:
                    self.robot.control(
                        [0, 0, 0], [1, 0, 0, 0],
                        (self.robot.finger_range[0] + self.robot.finger_range[1]) / 2,
                        relative=True, teleport=True
                    )
                    self.p.resetBasePositionAndOrientation(
                        self.blocks_id[goal_idx[i]], self.robot.get_eef_position(), (0, 0, 0, 1))
        """
        if any(np.array(self.n_to_stack) > 2) and self.np_random.uniform() < 0.5:
            # generate block position very close to base position
            all_block_positions = [np.array(self.p.getBasePositionAndOrientation(i)[0]) for i in self.blocks_id]
            block_halfextent = []
            for i in range(self.n_goal):
                all_block_positions[goal_idx[i]][0] = self.base_xy[i][0] + \
                                                      self.np_random.uniform(0.05, 0.1) * self.np_random.choice([1, -1])
                all_block_positions[goal_idx[i]][1] = self.base_xy[i][1] + \
                                                      self.np_random.uniform(0.05, 0.1) * self.np_random.choice([1, -1])
                block_halfextent.append(
                    np.array(self.p.getCollisionShapeData(self.blocks_id[i], -1)[0][3]) / 2)  # TODO: check inference
            _count = 0
            while any([_in_collision(all_block_positions[int(self.n_base_all): self.n_active_object],
                                     block_halfextent[i] * 2) for i in range(self.n_goal)]) and _count < 10:
                for i in range(self.n_goal):
                    all_block_positions[goal_idx[i]][0] = self.base_xy[i][0] + \
                                                          self.np_random.uniform(0.05, 0.1) * self.np_random.choice(
                        [1, -1])
                    all_block_positions[goal_idx[i]][1] = self.base_xy[i][1] + \
                                                          self.np_random.uniform(0.05, 0.1) * self.np_random.choice(
                        [1, -1])
                _count += 1
            for i in range(self.n_goal):
                self.p.resetBasePositionAndOrientation(
                    self.blocks_id[goal_idx[i]], all_block_positions[goal_idx[i]], (0, 0, 0, 1)
                )
        # for debug
        if ((all(goal[:, 2] <= 0.03) and not self.primitive) or (all(goal[:, 5] <= 0.03) and self.primitive)) and \
                any(self.n_base[:self.n_goal] >= 1):
            logger.log("--------")
            for n in range(self.n_active_object):
                cur_pos, _ = self.p.getBasePositionAndOrientation(self.blocks_id[n])
                logger.log(cur_pos)
            logger.log(goal)
            logger.log(self.n_base, self.n_to_stack)
            logger.log("--------")
            raise ArithmeticError

        if self.primitive:
            goal = np.reshape(goal, (-1, (6 + self.n_object) * self.n_goal))
            goal = np.concatenate(
                [goal, -np.ones((goal.shape[0], (self.n_max_goal - self.n_goal) * (6 + self.n_object)))], axis=-1)
            self.visualize_goal(goal, True)
            return goal

        goal = np.reshape(goal, (-1, (3 + self.n_object) * self.n_goal))

        if self.n_max_goal > self.n_goal:
            for i in range(self.n_goal, self.n_max_goal):
                for n in range(int(np.sum(self.n_base[:i])), int(np.sum(self.n_base[:i + 1]))):
                    cur_pos, _ = self.p.getBasePositionAndOrientation(self.blocks_id[n])
                    goal_onehot = np.zeros((1, self.n_object))
                    goal_onehot[0][n] = 1
                    cur_pos = np.reshape(cur_pos, (1, -1))
                    pad = np.concatenate([cur_pos, goal_onehot], axis=1)
                    goal = np.concatenate([goal, pad], axis=1)
        
        self.visualize_goal(goal)
        # self.env_statistics(goal)

        return goal

    def visualize_goal(self, goal, token_type=True):
        if self.body_goal is not None:
            for goal_id in self.body_goal:
                self.p.removeBody(goal_id)
        
    def set_choice_prob(self, n_to_stack, prob):
        # called in pair
        # n_to_stack shape: [n_max_goal, 1], prob shape: [n_max_goal, 1]
        # n_to_stack_probs shape: [n_max_goal, n_choices]
        # n_to_stack_choices shape: [n_max_goal, n_choices]
        # TODO: checking
        n_to_stack = np.reshape(n_to_stack, (-1, 1))
        prob = np.reshape(prob, (-1, 1))
        assert np.shape(self.n_to_stack_probs)[0] == self.n_max_goal
        for idx in range(self.n_max_goal):
            probs = self.n_to_stack_probs[idx]  # shape: [n_choices]
            assert len(n_to_stack) == len(prob)
            visited = [False] * len(probs)  # shape: [n_choices]
            for i in range(len(n_to_stack)):
                assert n_to_stack[i] in self.n_to_stack_choices[idx]
                idy = np.where(np.array(self.n_to_stack_choices[idx]) == n_to_stack[i])[0][0]
                # print(idy)
                visited[idy] = True
                probs[idy] = prob[i][0]
            for i in range(len(visited)):
                if not visited[i]:
                    probs[i] = (1 - np.sum(prob[idx])) / (len(visited) - len(n_to_stack[idx]))
            self.n_to_stack_probs[idx] = probs

    def sync_attr(self):
        if self.primitive:
            goal = np.reshape(self.goal, (-1, 6 + self.n_object))
            goal_pos = goal[:, 3:6]
        else:
            goal = np.reshape(self.goal, (-1, 3 + self.n_object))
            goal_pos = goal[:, :3]
        if hasattr(self, "goals") and isinstance(self.goals, list) and len(self.goals):
            if self.primitive:
                goal_pos = self.goals[0][3:6]
            else:
                goal_pos = self.goals[0][:3]
        # n_base, n_active_object = np.zeros(self.n_max_goal), 0
        n_active_object = 0
        for n in self.blocks_id:
            pos, orn = self.p.getBasePositionAndOrientation(n)
            if np.linalg.norm(np.array(pos[:2]) - np.array(self.inactive_xy)) > 0.01:
                n_active_object += 1
                # print(self.n_max_goal)
                #for i in range(self.n_max_goal):
                #    if np.linalg.norm(np.array(pos[:2]) - goal_pos[i, :2]) < 1e-3:
                #        n_base[i] += 1
                #        break
        #for i in range(self.n_max_goal):
        #    self.n_to_stack[i] = int(round((goal_pos[i, 2] - self.robot.base_pos[2] - 0.025) / 0.05)) + 1 - n_base[i]
        #    self.n_base[i] = n_base[i]
        #    self.base_xy[i, :] = goal_pos[i, :2]
        self.n_active_object = n_active_object

    def get_info_from_objects(self, objects_pos, goal):
        # changed
        #print(objects_pos)
        #print(goal)
        if isinstance(objects_pos, torch.Tensor):
            objects_pos = objects_pos.cpu().numpy()
        assert len(objects_pos.shape) == 2
        if self.primitive:
            goal = np.reshape(goal, (-1, 6 + self.n_object))
        else:
            goal = np.reshape(goal, (-1, 3 + self.n_object))

        # return which tower an object is in
        def in_tower(pos):
            threshold = 3e-2
            if self.primitive:
                goal_pos = goal[:, 3:5]
            else:
                goal_pos = goal[:, :2]
            if all(np.linalg.norm(np.tile(pos[:2], (goal_pos.shape[0], 1)) - goal_pos, axis=1) > threshold):
                return -1
            maybe_n = (pos[2] - self.robot.base_pos[2] - 0.025) / 0.05
            if abs(maybe_n - np.round(maybe_n)) > 1e-2:
                return -1
            for i in range(self.n_max_goal):
                if np.linalg.norm(pos[:2] - goal_pos) <= threshold:
                    return i
            raise NotImplementedError

        in_tower_heights = []
        max_height = []
        n_base = []
        n_to_stack = []
        n_goal = 0
        if self.primitive:
            goal_pos = goal[:, 3:6]
            n_goal = None
            for i in range(goal.shape[0]):
                if all(goal[i, :] == -1):
                    n_goal = i
                    break
            if n_goal is None:
                n_goal = goal.shape[0]
        else:
            goal_pos = goal[:, :3]
        for idx in range(np.shape(goal_pos)[0]):
            if not self.primitive and any([np.linalg.norm(goal_pos[idx] - objects_pos[i, :3]) < 0.03 for i in range(self.n_object)]):
                n_goal += 1
            in_tower_heights.append(
                [objects_pos[i][2] for i in range(objects_pos.shape[0]) if in_tower(objects_pos[i]) == idx])
            max_height.append(
                [np.max(in_tower_heights[idx]) if len(in_tower_heights[idx]) else self.robot.base_pos[2] - 0.025])
            n_base.append(int(np.round((max_height[idx][0] - (self.robot.base_pos[2] - 0.025)) / 0.05)))
            n_to_stack.append(int(np.round((goal_pos[idx][2] - (self.robot.base_pos[2] - 0.025)) / 0.05)) - n_base[idx])
        n_active = int(np.sum(np.linalg.norm(objects_pos[:, :2] + 1, axis=-1) > 1e-2))  # TODO: not legal?
        return dict(n_base=n_base, n_to_stack=n_to_stack, n_active=n_active, n_goal=n_goal)

    def compute_reward_and_info(self):
        if self.primitive:
            goal = np.reshape(self.goal, (-1, 6 + self.n_object))
            goal_pos = goal[:, 3:6]
        else:
            goal = np.reshape(self.goal, (-1, 3 + self.n_object))
            goal_pos = goal[:, :3]

        cur_pos, obj_id = self._get_achieved_goal()
        n_goal = None
        for i in range(cur_pos.shape[0]):
            if all(cur_pos[i, :] == -1):
                n_goal = i
                break
        if n_goal is None:
            n_goal = cur_pos.shape[0]
        goal_distance = np.linalg.norm(goal_pos - cur_pos, axis=1)

        # eef_distance = []
        is_stable = False
        # eef_threshold = []
        # for i in range(n_goal):
        #     eef_distance.append(np.linalg.norm(self.robot.get_eef_position() - goal_pos[i, :]))
        #     eef_threshold.append(0.1 * (1 - self.cl_ratio) if self.n_to_stack[i] == 1 else 0.1)

        if self.primitive:
            is_success = True
            for i in range(n_goal):
                pos, orn = self.p.getBasePositionAndOrientation(self.blocks_id[obj_id[i]])
                rot_matrix = _quaternion2RotationMatrix(orn)
                goal_pos_b_abs = np.abs(np.matmul(rot_matrix, goal_pos[i] - pos))
                block_half_extent = np.array(self.p.getCollisionShapeData(self.blocks_id[obj_id[i]], -1)[0][3]) / 2
                if goal[i, 2] == 1 and all(goal_pos_b_abs <= block_half_extent):
                    is_success = False
                if goal[i, 1] == 1 and any(goal_pos_b_abs > block_half_extent):
                    is_success = False
        else:
            if all(goal_distance < 0.03):
                is_success = True
            else:
                is_success = False

        # if isinstance((eef_distance > eef_threshold), np.ndarray):
        #     eef_indicator = all(eef_distance > eef_threshold)
        # else:
        #     eef_indicator = (eef_distance > eef_threshold)
        if is_success:  # tighten the threshold from 0.3 to 0.015 for pyramid stacking
            state_id = self.p.saveState()
            for _ in range(10):
                self.p.stepSimulation()
            future_pos = self._get_achieved_goal()[0]
            if all(np.linalg.norm(future_pos - cur_pos, axis=-1) < 1e-3):
                is_stable = True
            self.p.restoreState(stateId=state_id)
            self.p.removeState(state_id)

        if self.reward_type == "sparse":
            reward = float(is_stable)  # receive reward only when all goals are stable
        elif self.reward_type == "dense":
            raise NotImplementedError
            # TODO
            reward = self._previous_distance - (np.sum(goal_distance) - (np.sum(goal_distance) < 0.15) * np.sum(
                eef_distance)) + float(is_stable)
            self._previous_distance = (np.sum(goal_distance) - (np.sum(goal_distance) < 0.15) * np.sum(eef_distance))
        else:
            raise NotImplementedError
        is_success = is_stable
        img = self.render()
        info = {'is_success': is_success, "n_to_stack": self.n_to_stack, "n_base": self.n_base,
                'put_goal_aside': self.put_goal_aside, 'use_expand_goal': self.use_expand_goal,
                'stack_straight': self.stack_straight, 'inside_pyramid': self.inside_pyramid,
                'multi_goal': self.multi_goal, 'multi_step_goal': self.multi_step_goal, "img": img.astype(np.uint8)}
        return reward, info

    def compute_reward(self, obs, goal):
        # For stacking, it seems meaningless to perform HER
        raise NotImplementedError

    def set_cl_ratio(self, cl_ratio):
        self.cl_ratio = cl_ratio
    
    def set_obs_debug(self, traj_obs):
        self.presampled_obs = traj_obs
    
    def create_generalize_task(self, shape="3T"):
        def objects_in_collision():
            self.p.performCollisionDetection()
            all_contact_points = self.p.getContactPoints()
            if len(all_contact_points):
                _, bodyA, bodyB, linkA, linkB, *_ = zip(*all_contact_points)
                set1 = set(zip(zip(bodyA, linkA), zip(bodyB, linkB)))
                set2 = set(zip(zip(bodyB, linkB), zip(bodyA, linkA)))
                all_contact_set = set.union(set1, set2)
                moving_body_and_links = [(body, -1) for body in self.blocks_id]
                check_link_pairs = set(combinations(moving_body_and_links, 2))
                is_collision = not all_contact_set.isdisjoint(check_link_pairs)
            else:
                is_collision = False
            return is_collision

        assert shape in ["3T", "I", "Y", "Y_v2", "2I"]
        robot_obs = self.robot.get_obs()
        all_position = np.array([
            [self.np_random.uniform(*self.robot.x_workspace),
            self.np_random.uniform(*self.robot.y_workspace),
            self.robot.base_pos[2] + self.np_random.uniform(0.025, 0.025)]
            for _ in range(self.n_object)])
        for i in range(self.n_object):
            self.p.resetBasePositionAndOrientation(
                self.blocks_id[i], all_position[i], self.gen_obj_quat()
            )
        while objects_in_collision():
            all_position = np.array([
                [self.np_random.uniform(*self.robot.x_workspace),
                self.np_random.uniform(*self.robot.y_workspace),
                self.robot.base_pos[2] + 0.025]
                for _ in range(self.n_object)
            ])
            for i in range(self.n_object):
                self.p.resetBasePositionAndOrientation(
                    self.blocks_id[i], all_position[i], self.gen_obj_quat()
                )
        for _ in range(50):
            self.p.stepSimulation()
        init_state = []
        for i in range(self.n_object):
            obj_pos, obj_quat = self.p.getBasePositionAndOrientation(self.blocks_id[i])
            obj_quat = np.array(obj_quat)
            if obj_quat[-1] < 0:
                obj_quat = -obj_quat
            init_state.append(np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        init_state = np.concatenate(init_state)
        obj_idxs = np.arange(self.n_object)
        np.random.shuffle(obj_idxs)
        # goal
        # Change to be consistent with state version
        x_ = self.np_random.uniform(*[0.3, 0.5])
        y_ = self.np_random.uniform(*[-0.1, 0.1])
        if shape == "3T":
            goal_poses = [
                (np.array([x_+0.142, y_, 0.175]), np.array([0., 0., np.sin(np.pi / 4), np.cos(np.pi / 4)])),
                (np.array([x_+0.142, y_, 0.075]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
                (np.array([x_-0.046, y_+0.12, 0.175]), np.array([0., 0., np.sin(np.pi / 12), np.cos(np.pi / 12)])),
                (np.array([x_-0.046, y_+0.12, 0.075]), np.array(self.p.getQuaternionFromEuler([0., np.pi / 2, -np.pi / 3]))), 
                (np.array([x_-0.046, y_-0.12, 0.175]), np.array([0., 0., np.sin(-np.pi / 12), np.cos(-np.pi / 12)])),
                (np.array([x_-0.046, y_-0.12, 0.075]), np.array(self.p.getQuaternionFromEuler([0., np.pi / 2, np.pi / 3]))),
            ]
        elif shape == "I":
            # check x_, y_ are not located on other objects
            def check_xy():
                for i in range(3, self.n_object):
                    if np.linalg.norm(init_state[7 * i: 7 * i + 2] - np.array([x_, y_])) < 0.075:
                        return False
                return True
            reset_count = 0
            while not check_xy() and reset_count < 50:
                x_ = self.np_random.uniform(*[0.3, 0.5])
                y_ = self.np_random.uniform(*[-0.1, 0.1])
                reset_count += 1
            goal_poses = [
                (np.array([x_, y_, 0.075]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
                (np.array([x_, y_, 0.175]), np.array([0., 0., np.sin(np.pi / 4), np.cos(np.pi / 4)])),
                (np.array([x_, y_, 0.275]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
            ]
        elif shape == "2I":
            goal_poses = [
                (np.array([x_, y_, 0.075]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
                (np.array([x_, y_, 0.175]), np.array([0., 0., np.sin(np.pi / 4), np.cos(np.pi / 4)])),
                (np.array([x_, y_, 0.275]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
                (np.array([0.8 - x_, -y_, 0.075]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
                (np.array([0.8 - x_, -y_, 0.175]), np.array([0., 0., np.sin(np.pi / 4), np.cos(np.pi / 4)])),
                (np.array([0.8 - x_, -y_, 0.275]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
            ]
        elif shape == "Y":
            goal_poses = [
                (np.array([x_, y_, 0.075]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
                (np.array([x_, y_+0.05, 0.075]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
                (np.array([x_, y_-0.05, 0.075]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
                (np.array([x_, y_, 0.175]), np.array([0., 0., np.sin(np.pi / 4), np.cos(np.pi / 4)])),
                (np.array([x_, y_+0.05, 0.275]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
                (np.array([x_, y_-0.05, 0.275]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
            ]
        elif shape == "Y_v2":
            # check x_, y_ are not located on other objects
            def check_xy():
                for i in range(5, self.n_object):
                    if init_state[7 * i] < x_ + 0.05:
                        return False
                return True
            reset_count = 0
            while not check_xy() and reset_count < 50:
                x_ = self.np_random.uniform(*[0.3, 0.5])
                y_ = self.np_random.uniform(*[-0.1, 0.1])
                reset_count += 1
            goal_poses = [
                (np.array([x_, y_, 0.075]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
                (np.array([x_, y_+0.05, 0.075]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
                (np.array([x_, y_-0.05, 0.075]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
                (np.array([x_, y_+0.05, 0.225]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
                (np.array([x_, y_-0.05, 0.225]), np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
            ]
        else:
            raise NotImplementedError
        # goal_poses = [
        #     (np.array([0.0, -0.2, 0.075]) + offset, np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
        #     (np.array([0.0, -0.12, 0.075]) + offset, np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
        #     (np.array([0.0, -0.04, 0.075]) + offset, np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
        #     (np.array([0.0, 0.04, 0.075]) + offset, np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
        #     (np.array([0.0, 0.12, 0.075]) + offset, np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
        #     (np.array([0.0, 0.2, 0.075]) + offset, np.array([0., np.sin(np.pi / 4), 0., np.cos(np.pi / 4)])),
        # ]
        for i in range(len(goal_poses)):
            self.p.resetBasePositionAndOrientation(self.blocks_id[obj_idxs[i]], goal_poses[i][0], goal_poses[i][1])
        goal_state = []
        for i in range(self.n_object):
            obj_pos, obj_quat = self.p.getBasePositionAndOrientation(self.blocks_id[i])
            obj_quat = np.array(obj_quat)
            if obj_quat[-1] < 0:
                obj_quat = -obj_quat
            goal_state.append(np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        goal_state = np.concatenate(goal_state)
        goal_image = self.render()
        # plt.imsave("tmp/tmp0.png", goal_image.transpose((1, 2, 0)))
        # exit()
        return robot_obs, init_state, goal_state, goal_image

    def create_canonical_view(self):
        # for shape in self.p.getVisualShapeData(self.table_id):
        #     self.p.changeVisualShape(self.table_id, shape[1], rgbaColor=(0, 0, 0, 1))
        images = []
        for obj_idx in range(self.n_object):
            for i in range(self.n_object):
                if i == obj_idx:
                    target_position = np.array([np.mean(self.robot.x_workspace), np.mean(self.robot.y_workspace), 0.025])
                    target_orientation = np.array([0., 0., 0., 1.])
                else:
                    target_position = np.array([self.robot.x_workspace[0], 10, 0.025 + 0.05 * i])
                    target_orientation = np.array([0., 0., 0., 1.])
                self.p.resetBasePositionAndOrientation(self.blocks_id[i], target_position, target_orientation)
            from bullet_envs.env.primitive_env import render
            image = render(self.p, width=128, height=128, robot=self.robot, view_mode="third",
                        pitch=-45, distance=0.6,
                        camera_target_position=(0.5, 0.0, 0.1)).transpose((2, 0, 1))[:3, 48: 80, 48: 80]
            # plt.imsave("tmp/tmp%d.png" % obj_idx, image.transpose((1, 2, 0)).astype(np.uint8))
            images.append(image)
        return images
    
class ArmStackwLowPlanner(ArmStack):
    def __init__(self, *args, actionRepeat=10, use_low_level_planner=True, force_scale=0., compute_path=False, **kwargs):
        self.action_repeat = actionRepeat
        self.use_low_level_planner = use_low_level_planner
        self.compute_path = compute_path

        self.__init_args = copy.deepcopy(args)
        self.__init_kwargs = copy.deepcopy(kwargs)

        super().__init__(*args, **kwargs)

        if self.use_low_level_planner:
            pass
            # self._create_low_level_planner(force_scale, kwargs.get('robot'))
    
    def _create_low_level_planner(self, force_scale, robot):
        from bullet_envs.env.primitive_stacking_lowlevel import LowLevelEnv
        # Create low level env for planning
        self.plan_low = LowLevelEnv(*self.__init_args, actionRepeat=self.action_repeat, **self.__init_kwargs)
        from bullet_envs.env.primitive_stacking_planner import Planner, Executor, Primitive
        planner = Planner(self.plan_low.robot, self.plan_low.p, self.p, smooth_path=self.compute_path)
        executor = Executor(self.robot, self.p, ctrl_mode="teleport", record=True)
        self._planner = Primitive(planner, executor, self.blocks_id, None, None,
                                   self.plan_low.blocks_id, [self.plan_low.table_id],
                                   teleport_arm=not self.compute_path, force_scale=force_scale)
    
    def reset(self):
        obs = super().reset()
        # if self.use_low_level_planner:
        #     self._planner.align_at_reset()
        return obs
    
    def step(self, action):
        """
        @gjx: copyed from ArmGoalEnv and added low-level planner
        """
        assert self.name is not None
        if (action == -1).all():
            obs = self._get_obs()
            reward, info = self.compute_reward_and_info()
            done = True
            return obs, reward, done, info
        if self.primitive:
            assert len(action) == 7
            assert self.n_active_object is not None
            # if any(np.abs(action[1:]) > 1):
            #     print(action)
            #     raise ValueError
            # get target position & orientation
            # todo: let generate action & not generate action be the same
            obj_id = int(action[0])
            tgt_pos = np.zeros(3)
            tgt_orn = action[4:]
            tgt_pos[0] = (action[1] + 1) * (self.robot.x_workspace[1] - self.robot.x_workspace[0]) / 2 \
                            + self.robot.x_workspace[0]
            tgt_pos[1] = (action[2] + 1) * (self.robot.y_workspace[1] - self.robot.y_workspace[0]) / 2 \
                            + self.robot.y_workspace[0]
            if self.name == "allow_rotation":
                tgt_pos[2] = (action[3] + 1) * 0.4 / 2 + self.robot.base_pos[2] + 0.025
            elif self.name == "default":
                tgt_pos[2] = (action[3] + 1) * 0.4 / 2 + self.robot.base_pos[2] + 0.025
            else:
                raise NotImplementedError
            tgt_orn = tgt_orn * np.pi / 2.
            tgt_orn = p.getQuaternionFromEuler(tgt_orn)

            stable = True
            # if self._robot_feasible(self.blocks_id[obj_id], tgt_pos, tgt_orn):
            if obj_id >= 0:
                if not self.use_low_level_planner:
                    # get current position & orientation of the target object
                    # cur_pos, cur_orn = self.p.getBasePositionAndOrientation(self.blocks_id[obj_id])
                    # self.robot.control(
                    #     tgt_pos, tgt_orn,
                    #     (self.robot.finger_range[0] + self.robot.finger_range[1]) / 2,
                    #     relative=False, teleport=True,
                    # )
                    # state_id = self.p.saveState()
                    self.p.resetBasePositionAndOrientation(self.blocks_id[obj_id], tgt_pos, tgt_orn)
                    #fig, ax = plt.subplots(1, 1)
                    # if os.path.exists("tmp_roll"):
                    #     shutil.rmtree("tmp_roll")
                    #os.makedirs("tmp_roll", exist_ok=True)
                    # img = self.render(mode="rgb_array")
                    # ax.cla()
                    # ax.imshow(img)
                    # plt.imsave("tmp_roll/tmp%d.png" % self.frame_count, img)
                    # self.frame_count += 1
                    # print(self.frame_count)
                    for frame_count in range(40):
                        self.p.stepSimulation()
                        # self.frame_count += 1
                    # judge whether stable
                    cur_pos = self._get_achieved_goal()[0]
                    for _ in range(10):
                        # img = self.render(mode="rgb_array")
                        # ax.cla()
                        # ax.imshow(img)
                        # plt.imsave("tmp_roll/tmp%d.png" % self.frame_count, img)
                        self.p.stepSimulation()
                        # self.frame_count += 1
                    #img = self.render(mode="rgb_array")
                    #ax.cla()
                    #ax.imshow(img)
                    #plt.imsave("tmp_roll/tmp%d.png" % self.frame_count, img)
                    #self.frame_count += 1
                    # print(self.frame_count)
                    future_pos = self._get_achieved_goal()[0]
                    stable = not any(np.linalg.norm(future_pos - cur_pos, axis=-1) >= 1e-3)
                else:
                    # print("[DEBUG] low level moving")
                    # _state = self.p.saveState()
                    # res, path = self._planner.move_one_object(obj_id, tgt_pos, tgt_orn)
                    print("[DEBUG] primitive moving")
                    _state = self.p.saveState()
                    if not hasattr(self.robot, "video_writer"):
                        import imageio
                        self.robot.video_writer = imageio.get_writer("robot_demo.mp4",
                                                fps=20,
                                                format='FFMPEG',
                                                codec='h264',)
                        self.robot.save_video = True
                    obj_init_pos, obj_init_quat = self.p.getBasePositionAndOrientation(self.blocks_id[obj_id])
                    # print("obj init quat", obj_init_quat)
                    obj_T_grasp = []
                    for x_offset in [0.]:
                        for x_rot in [0., np.pi / 2, np.pi, -np.pi / 2]:
                            for z_rot in [0., np.pi]:
                                local_grasp = np.eye(4)
                                local_grasp[:3, :3] = (np.array(self.p.getMatrixFromQuaternion([np.sin(x_rot / 2), 0., 0., np.cos(x_rot / 2)])).reshape(3, 3)) \
                                    @ (np.array(self.p.getMatrixFromQuaternion([0., 0., np.sin(z_rot / 2), np.cos(z_rot / 2)])).reshape(3, 3)) \
                                    @ (np.array(self.p.getMatrixFromQuaternion([1., 0., 0., 0.])).reshape(3, 3))
                                local_grasp[:3, 3] = np.array([x_offset, 0., 0.])
                                obj_T_grasp.append(local_grasp)
                    O_T_obj_init = np.eye(4)
                    O_T_obj_init[:3, :3] = np.array(self.p.getMatrixFromQuaternion(obj_init_quat)).reshape(3, 3)
                    O_T_obj_init[:3, 3] = np.array(obj_init_pos)
                    O_T_grasp_pick = [O_T_obj_init @ local_grasp for local_grasp in obj_T_grasp]
                    O_T_grasp_place = []
                    for x_rot in [0., np.pi / 2, np.pi, -np.pi / 2]:
                        O_T_obj_end = np.eye(4)
                        O_T_obj_end[:3, :3] = np.array(self.p.getMatrixFromQuaternion(tgt_orn)).reshape(3, 3) @ np.array(self.p.getMatrixFromQuaternion([np.sin(x_rot / 2), 0., 0., np.cos(x_rot / 2)])).reshape(3, 3)
                        O_T_obj_end[:3, 3] = np.array(tgt_pos)
                        O_T_grasp_place.append([O_T_obj_end @ local_grasp for local_grasp in obj_T_grasp])
                    q0 = np.array([0.0006290743156705777, -0.6363918264046711, -0.00048377514187155377, -2.498912361135347, -0.000301933506133224, 1.8636677063581644, 0.7857285239452109])
                    valid_grasp_idx = None
                    for i in range(len(O_T_grasp_pick)):
                        print(O_T_grasp_pick[i])
                        quat_pick = mat2quat(O_T_grasp_pick[i][:3, :3])
                        pick_prob_vec, _ = self.p.multiplyTransforms([0., 0., 0.], quat_pick, [0., 0., 1.0], [0., 0., 0., 1.0])
                        pick_prob_vec = np.array(pick_prob_vec)
                        print("pick prob", pick_prob_vec)
                        if not np.dot(pick_prob_vec, np.array([0., 0., -1.0])) > 0.7:
                            continue
                        for sym_idx in range(len(O_T_grasp_place)):
                            place_success = False
                            quat_place = mat2quat(O_T_grasp_place[sym_idx][i][:3, :3])
                            place_prob_vec, _ = self.p.multiplyTransforms([0., 0., 0.], quat_place, [0., 0., 1.0], [0., 0., 0., 1.0])
                            place_prob_vec = np.array(place_prob_vec)
                            print("place prob", place_prob_vec)
                            if np.dot(place_prob_vec, np.array([0., 0., 1.0])) > 0.6 or np.dot(place_prob_vec, np.array([-1.0, 0., 0.])) > 0.6:
                                pass
                            else:
                                place_success = True
                                break
                        if not place_success:
                            continue
                        _, ik_pick_success = self.robot.solve_inverse_kinematics(O_T_grasp_pick[i][:3, 3], quat_pick, q0)
                        _, ik_place_success = self.robot.solve_inverse_kinematics(O_T_grasp_place[sym_idx][i][:3, 3], quat_place, q0)
                        if ik_pick_success and ik_place_success:
                            valid_grasp_idx = i
                            break
                    if valid_grasp_idx is None:
                        print("[ERROR] fail to find grasp")
                        res = -1
                        exit()
                    else:
                        self.robot.reset_primitive(
                            'open', (self.blocks_id[obj_id], -1), 
                            partial(render, robot=self.robot, view_mode="third", width=500, height=500, 
                                    shift_params=(0, 0))
                            )
                        init_eef_pos = self.robot.get_eef_position()
                        init_eef_quat = self.robot.get_eef_orn()
                        attachment = dict(obj_id=self.blocks_id[obj_id], obj_T_grasp=obj_T_grasp[valid_grasp_idx])
                        ee_trajectory = [
                            (O_T_grasp_pick[valid_grasp_idx][:3, 3] - pick_prob_vec * 0.05, quat_pick, None, None),
                            (O_T_grasp_pick[valid_grasp_idx][:3, 3], quat_pick, "close", None),
                            (O_T_grasp_pick[valid_grasp_idx][:3, 3] - pick_prob_vec * 0.1, quat_pick, None, attachment),
                            (np.concatenate([(O_T_grasp_pick[valid_grasp_idx][:3, 3] - pick_prob_vec * 0.1)[:2], [0.3]]), quat_place, None, attachment),
                            (O_T_grasp_place[sym_idx][valid_grasp_idx][:3, 3] - place_prob_vec * 0.05, quat_place, None, attachment),
                            (O_T_grasp_place[sym_idx][valid_grasp_idx][:3, 3], quat_place, "open", attachment),
                            (O_T_grasp_place[sym_idx][valid_grasp_idx][:3, 3] - place_prob_vec * 0.05 + np.array([0., 0., 0.15]), quat_place, None, None),
                            (init_eef_pos, init_eef_quat, None, None)
                        ]
                        for i in range(len(ee_trajectory)):
                            self.robot.change_visual(True)
                            res = self.robot.move_to_ee_pose(ee_trajectory[i][0], ee_trajectory[i][1], ee_trajectory[i][3])
                            if res < 0:
                                print(f"[ERROR] fail in {i}")
                                break
                            if ee_trajectory[i][2] == "close":
                                self.robot.gripper_grasp()
                            elif ee_trajectory[i][2] == "open":
                                self.robot.gripper_move("open", teleport=False)
                        if res < 0:
                            print("[ERROR] move failed")
                            exit()
                        else:
                            print("[SUCCESS]")

                    if res == 0:
                        start_pos_and_rot = [self.p.getBasePositionAndOrientation(self.blocks_id[i]) for i in range(self.n_object)]
                        start_pos, start_rot = zip(*start_pos_and_rot)
                        start_pos, start_rot = map(lambda x: np.asarray(x), [start_pos, start_rot])
                        stable_pos, stable_rot = start_pos, start_rot
                    else:
                        # If low-level fails, do no-op
                        self.p.restoreState(stateId=_state)
                        joint_states = self.p.getJointStates(self.robot._robot, self.robot.motorIndices)
                        servo_angles = [item[0] for item in joint_states]
                        self.p.setJointMotorControlArray(self.robot._robot, self.robot.motorIndices, self.p.POSITION_CONTROL, servo_angles)
                        self.p.stepSimulation()
                    self.p.removeState(_state)
            obs = self._get_obs()
            reward, info = self.compute_reward_and_info()
            done = False
            if not stable:
                reward -= 0.001
            #     # self.p.removeState(state_id)
            return obs, reward, done, info
    
    def _sim_until_stable(self):
        count = 0
        while count < 10 and np.linalg.norm(np.concatenate(
                [self.p.getBaseVelocity(self.blocks_id[i])[0]
                 for i in range(len(self.blocks_id))])) > 1e-3:
            for _ in range(50):
                self.p.stepSimulation()
            count += 1
        print(f"Sim until stable: {count} rounds")
    
    def close(self) -> None:
        if hasattr(self.robot, "video_writer"):
            self.robot.video_writer.close()
        return super().close()


def _create_block(physics_client, halfExtents, pos, orn, mass=0.2, rgba=None, vel=None, vela=None):
    col_id = physics_client.createCollisionShape(physics_client.GEOM_BOX, halfExtents=halfExtents)
    vis_id = physics_client.createVisualShape(physics_client.GEOM_BOX, halfExtents=halfExtents)
    body_id = physics_client.createMultiBody(mass, col_id, vis_id, pos, orn)
    if rgba is not None:
        physics_client.changeVisualShape(body_id, -1, rgbaColor=rgba + [1.])
    if vel is None:
        vel = [0, 0, 0]
    if vela is None:
        vela = [0, 0, 0]
    physics_client.resetBaseVelocity(body_id, vel, vela)
    # restitution = np.random.uniform(*self.restitution_range)
    # friction = np.random.uniform(*self.friction_range)
    # physics_client.changeDynamics(
    #     body_id, -1, mass=1, restitution=restitution, lateralFriction=friction, linearDamping=0.005)
    return body_id


def _in_collision(positions, half_extent):
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if abs(positions[i][0] - positions[j][0]) < 2 * half_extent[0] \
                    and abs(positions[i][1] - positions[j][1]) < 2 * half_extent[1] \
                    and abs(positions[i][2] - positions[j][2]) < 2 * half_extent[2]:
                return True
    return False


def _in_collision_plus(positions, half_extents, eef_pos, eef_halfextent):
    for i in range(len(positions)):
        if abs(positions[i][0] - eef_pos[0]) < half_extents[i][0] + eef_halfextent[0] \
                and abs(positions[i][1] - eef_pos[0]) < half_extents[i][1] + eef_halfextent[1] \
                and abs(positions[i][2] - eef_pos[0]) < half_extents[i][2] + eef_halfextent[2]:
            return True
    return False


class ObsParser(object):
    def __init__(self, robot_dim, obj_dim, goal_dim):
        self.robot_dim = robot_dim + 6
        self.arm_dim = robot_dim
        self.obj_dim = obj_dim
        self.goal_dim = goal_dim

    def forward(self, obs: torch.Tensor):
        assert isinstance(obs, torch.Tensor)
        assert len(obs.shape) == 2
        # robot_dim = env.get_attr("robot_dim")[0]
        # object_dim = env.get_attr("object_dim")[0]
        # goal_dim = env.get_attr("goal")[0].shape[0]
        robot_obs = torch.narrow(obs, dim=1, start=0, length=self.arm_dim)
        achieved_obs = torch.narrow(obs, dim=1, start=obs.shape[1] - 2 * self.goal_dim, length=3)
        goal_obs = torch.narrow(obs, dim=1, start=obs.shape[1] - self.goal_dim, length=3)
        robot_obs = torch.cat([robot_obs, achieved_obs, goal_obs], dim=-1)
        objects_obs = torch.narrow(obs, dim=1, start=self.arm_dim,
                                   length=obs.shape[1] - self.arm_dim - 2 * self.goal_dim)
        objects_obs = torch.reshape(objects_obs, (objects_obs.shape[0], -1, self.obj_dim))
        masks = torch.norm(objects_obs + 1, dim=-1) < 1e-3
        # print("robot obs", robot_obs, "objects obs", objects_obs, "masks", masks)
        # exit()
        return robot_obs, objects_obs, masks


def _quaternion2RotationMatrix(quaternion):
    x, y, z, w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    rot_matrix00 = 1 - 2 * y * y - 2 * z * z
    rot_matrix01 = 2 * x * y - 2 * w * z
    rot_matrix02 = 2 * x * z + 2 * w * y
    rot_matrix10 = 2 * x * y + 2 * w * z
    rot_matrix11 = 1 - 2 * x * x - 2 * z * z
    rot_matrix12 = 2 * y * z - 2 * w * x
    rot_matrix20 = 2 * x * z - 2 * w * y
    rot_matrix21 = 2 * y * z + 2 * w * x
    rot_matrix22 = 1 - 2 * x * x - 2 * y * y
    return np.asarray([
        [rot_matrix00, rot_matrix01, rot_matrix02],
        [rot_matrix10, rot_matrix11, rot_matrix12],
        [rot_matrix20, rot_matrix21, rot_matrix22]
    ], dtype=np.float64)


def convert_symmetric_rotation(q):
    mat1 = np.array([
        # z upward
        np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]),
        np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]]),
        np.array([[-1, 0, 0],
                  [0, -1, 0],
                  [0, 0, 1]]),
        np.array([[0, 1, 0],
                  [-1, 0, 0],
                  [0, 0, 1]]),
    ])
    # -z upward
    mat2 = np.array([mat1[i] @ np.array([[1, 0, 0],
                                         [0, -1, 0],
                                         [0, 0, -1]]) for i in range(4)])
    # x upward
    mat3 = np.array([mat1[i] @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]) for i in range(4)])
    # -x upward
    mat4 = np.array([mat1[i] @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]) for i in range(4)])
    # y upward
    mat5 = np.array([mat1[i] @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) for i in range(4)])
    # -y upward
    mat6 = np.array([mat1[i] @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]) for i in range(4)])
    mat = np.concatenate([mat1, mat2, mat3, mat4, mat5, mat6], axis=0)
    static_quat = [mat2quat(mat[i]) for i in range(mat.shape[0])]
    # mat_q = quat2mat(q)
    symmetrics_quat = np.array([quat_mul(static_quat[i], q) for i in range(mat.shape[0])])
    return symmetrics_quat[np.argmax(symmetrics_quat[:, -1])]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = ArmStack(
        n_object=3, reward_type="sparse", action_dim=7, generate_data=True, primitive=True,
        n_to_stack=np.array([[1, 2, 3]]), name="allow_rotation", robot="panda"
    )
    print("created env")
    obs = env.reset()
    print("reset obs", obs)
    fig, ax = plt.subplots(1, 1)
    os.makedirs("tmp1", exist_ok=True)

    import pickle

    # history data
    _actions_history = []
    _states_history = []

    for i in range(10):
        img = env.render(mode="rgb_array")
        #ax.cla()
        #ax.imshow(img)
        plt.imsave("tmp1/tmp%d.png" % i, img)

        action = env.act()
        action = action.squeeze(dim=0)

        _actions_history.append(action)
        _states_history.append(env.get_state())
        print("[DEBUG] state", _states_history[-1])
        
        # action = env.action_space.sample()
        # action[4:7] = [0, 0, 0.5]
        print("[DEBUG] action", action)
        #action = [0, 0, 0, 0, 0, 0, 0.001]
        obs, reward, done, info = env.step(action)
    _states_history.append(env.get_state())
    pickle.dump({'actions': _actions_history, 'states': _states_history}, open('test_demo_random.pkl', 'wb'))
    print(obs)
