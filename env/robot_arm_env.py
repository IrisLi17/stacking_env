import pybullet as p
from pybullet_utils import bullet_client as bc
import pybullet_data
import gym, os, time
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch
from utils.bullet_rotations import mat2quat, quat2mat, quat_mul


DATAROOT = pybullet_data.getDataPath()
COLOR = [[1.0, 0, 0], [1, 1, 0], [0.2, 0.8, 0.8], [0.8, 0.2, 0.8], [0, 0, 0], [0.0, 0.0, 1.0], [0.5, 0.2, 0.0]]

class ArmGoalEnv(gym.Env):
    def __init__(self, robot="xarm", seed=None, action_dim=4):
        self.seed(seed)
        self._setup_env(robot)
        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32)
        ))
        self.action_space = spaces.Box(-1., 1., shape=(action_dim,))
    
    @property
    def dt(self):
        return 1. / 240

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _setup_env(self, robot, init_qpos=None, base_position=(0, 0, 0)):
        self.p = bc.BulletClient(connection_mode=p.DIRECT)
        self.p.resetSimulation()
        self.p.setTimeStep(self.dt)
        self.p.setGravity(0., 0., -9.8)
        self.p.resetDebugVisualizerCamera(1.0, 40, -20, [0, 0, 0,] )
        plane_id = self.p.loadURDF(os.path.join(DATAROOT, "plane.urdf"), [0, 0, -0.795])
        table_id = self.p.loadURDF(
            os.path.join(DATAROOT, "table/table.urdf"), 
            [0.40000, 0.00000, -.625000], [0.000000, 0.000000, 0.707, 0.707]
        )
        if robot == "xarm":
            from env.robot import XArmRobot
            self.robot = XArmRobot(self.p, init_qpos, base_position)
        elif robot == "panda":
            from env.robot import PandaRobot
            self.robot = PandaRobot(self.p, init_qpos, base_position)
        else:
            raise NotImplementedError
        self._setup_callback()
    
    def reset(self):
        self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs
    
    def step(self, action):
        # action = np.clip(action, -1, 1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        delta_pos = action[:3] * 0.05
        if len(action) == 4:
            eef_orn = (1, 0, 0, 0)
        elif len(action) == 5:
            cur_rpy = self.robot.get_eef_orn(as_type="euler")
            # rotation around z
            new_yaw = cur_rpy[2] + action[4] * np.pi / 9  # +/- 20 degrees
            eef_orn = quat_mul((0, 0, np.sin(new_yaw / 2), np.cos(new_yaw / 2)), (1, 0, 0, 0))
        else:
            raise NotImplementedError
        finger_ctrl = (action[3] + 1) / 2 * (self.robot.finger_range[1] - self.robot.finger_range[0]) + self.robot.finger_range[0]
        self.robot.control(delta_pos, eef_orn, finger_ctrl)
        obs = self._get_obs()
        reward, info = self.compute_reward_and_info()
        done = False
        return obs, reward, done, info
    
    def render(self, mode="rgb_array", width=500, height=500):
        if mode == 'rgb_array':
            view_matrix = self.p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=(0.3, 0, 0.2),
                distance=1.0,
                yaw=60,
                pitch=-10,
                roll=0,
                upAxisIndex=2,
            )
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

    def visualize_goal(self, goal):
        goal_idx = np.argmax(goal[3:])
        if self.body_goal is None:
            vis_id = self.p.createVisualShape(self.p.GEOM_SPHERE, radius=0.025, rgbaColor=COLOR[goal_idx % len(COLOR)] + [0.2])
            self.body_goal = self.p.createMultiBody(0, baseVisualShapeIndex=vis_id, basePosition=goal[:3])
        else:
            self.p.resetBasePositionAndOrientation(self.body_goal, goal[:3], (0, 0, 0, 1))
            self.p.changeVisualShape(self.body_goal, -1, rgbaColor=COLOR[goal_idx % len(COLOR)] + [0.2])
    
    def _reset_sim(self):
        pass

    def _get_obs(self):
        obs = self.robot.get_obs()
        return dict(observation=obs, achieved_goal=self.goal.copy(), desired_goal=self.goal.copy())
    
    def get_obs(self):
        return self._get_obs()

    def compute_reward_and_info(self):
        return None, {}


class ArmPickAndPlace(ArmGoalEnv):
    def __init__(self, robot="xarm", seed=None, n_object=1, reward_type="sparse", action_dim=4):
        self.n_object = n_object
        self.n_active_object = n_object
        self.blocks_id = []
        self.robot_dim = None
        self.object_dim = None
        self.body_goal = None
        self.reward_type = reward_type
        self._previous_distance = None
        self.inactive_xy = (10, 10)
        super(ArmPickAndPlace, self).__init__(robot, seed, action_dim)
    
    def _setup_callback(self):
        for i in range(self.n_object):
            self.blocks_id.append(
                _create_block(
                    self.p, [0.025, 0.025, 0.025], [10, 0, 1 + 0.5 * i], [0, 0, 0, 1], 0.1, COLOR[i % len(COLOR)]
                )
            )
    
    def reset(self):
        obs_dict = super().reset()
        if self.reward_type == "dense":
            self._previous_distance = np.linalg.norm(obs_dict["achieved_goal"][:3] - obs_dict["desired_goal"][:3])
        return obs_dict

    def _sample_goal(self):
        robot_pos = self.robot.base_pos
        x_range = self.robot.x_workspace
        y_range = self.robot.y_workspace
        z_range = (robot_pos[2] + 0.025, robot_pos[2] + 0.425)
        if self.np_random.uniform() < 0.5:
            goal = np.array([self.np_random.uniform(*x_range), self.np_random.uniform(*y_range), z_range[0]])
        else:
            goal = np.array([self.np_random.uniform(*x_range), self.np_random.uniform(*y_range), self.np_random.uniform(*z_range)])
        if self.n_object > 1:
            goal_onehot = np.zeros(self.n_object)
            goal_onehot[self.np_random.randint(self.n_active_object)] = 1
            goal = np.concatenate([goal, goal_onehot])
        goal_idx = 0 if self.n_object == 1 else np.argmax(goal[3:])
        if self.reward_type == "sparse" and goal[2] > z_range[0] and self.np_random.uniform() < 0.5:
            # Let the robot hold the block
            self.robot.control(
                [0, 0, 0], [1, 0, 0, 0],
                (self.robot.finger_range[0] + self.robot.finger_range[1]) / 2,
                relative=True, teleport=True,
            )
            self.p.resetBasePositionAndOrientation(self.blocks_id[goal_idx], self.robot.get_eef_position(), (0, 0, 0, 1))
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
        self.n_active_object = self.np_random.randint(1, self.n_object + 1)
        # Randomize initial position of blocks
        block_positions = np.array([[self.np_random.uniform(*x_range), self.np_random.uniform(*y_range), z_range[0]] 
                                     for _ in range(self.n_active_object)]
                                    + [[eef_pos[0], eef_pos[1], z_range[0]]])
        block_halfextent = np.array(self.p.getCollisionShapeData(self.blocks_id[0], -1)[0][3]) / 2
        while _in_collision(block_positions, block_halfextent):
            block_positions = np.array([[self.np_random.uniform(*x_range), self.np_random.uniform(*y_range), z_range[0]] 
                                         for _ in range(self.n_active_object)]
                                       + [[eef_pos[0], eef_pos[1], z_range[0]]])
        block_positions = block_positions[:-1]
        for n in range(self.n_active_object):
            # randomize orientation if allow rotation
            self.p.resetBasePositionAndOrientation(self.blocks_id[n], block_positions[n], self.gen_obj_quat())
        for n in range(self.n_active_object, self.n_object):
            self.p.resetBasePositionAndOrientation(
                self.blocks_id[n], list(self.inactive_xy) + [z_range[0] + (n - self.n_active_object) * 0.05],
                (0, 0, 0, 1)
            )

    # randomize orientation if allow rotation
    def gen_obj_quat(self):
        if self.action_space.shape[0] == 5:
            angle = self.np_random.uniform(-np.pi, np.pi)
            obj_quat = (0, 0, np.cos(angle / 2), np.sin(angle / 2))
        else:
            obj_quat = (0, 0, 0, 1)
        return obj_quat
    
    def _get_obs(self):
        obs = self.robot.get_obs()
        eef_pos = obs[:3]
        if self.robot_dim is None:
            self.robot_dim = len(obs)
        for i in range(self.n_active_object):
            object_pos, object_quat = self.p.getBasePositionAndOrientation(self.blocks_id[i])
            object_quat = convert_symmetric_rotation(np.array(object_quat))
            object_euler = self.p.getEulerFromQuaternion(object_quat)
            object_velp, object_velr = self.p.getBaseVelocity(self.blocks_id[i])
            object_pos, object_euler, object_velp, object_velr = map(np.array, [object_pos, object_euler, object_velp, object_velr])
            object_velp *= self.dt * self.robot.num_substeps
            object_velr *= self.dt * self.robot.num_substeps
            object_rel_pos = object_pos - eef_pos
            obs = np.concatenate([obs, object_pos, object_rel_pos, object_euler, object_velp, object_velr])
            if self.n_object > 1:
                goal_indicator = (np.argmax(self.goal[3:]) == i)
                obs = np.concatenate([obs, [goal_indicator]])
            if self.object_dim is None:
                self.object_dim = len(obs) - self.robot_dim
        for i in range(self.n_active_object, self.n_object):
            obs = np.concatenate([obs, -np.ones(self.object_dim)])
        achieved_goal = np.concatenate([self._get_achieved_goal(), self.goal[3:]])
        obs_dict = dict(observation=obs, achieved_goal=achieved_goal, desired_goal=self.goal.copy())
        return obs_dict
    
    def _get_achieved_goal(self):
        goal_idx = 0 if self.n_object == 1 else np.argmax(self.goal[3:])
        cur_pos, _ = self.p.getBasePositionAndOrientation(self.blocks_id[goal_idx])
        return np.array(cur_pos)

    def compute_reward_and_info(self):
        goal_pos = self.goal[:3]
        cur_pos = self._get_achieved_goal()
        distance = np.linalg.norm(goal_pos - cur_pos)
        if self.reward_type == "sparse":
            reward = (distance < 0.05)
        elif self.reward_type == "dense":
            reward = self._previous_distance - distance
            self._previous_distance = distance
        else:
            raise NotImplementedError
        is_success = (distance < 0.05)
        info = {'is_success': is_success}
        return reward, info

    def compute_reward(self, obs, goal):
        # For HER
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(goal, torch.Tensor):
            goal = goal.cpu().numpy()
        assert isinstance(obs, np.ndarray) and isinstance(goal, np.ndarray)
        goal_pos = goal[:3]
        goal_idx = np.argmax(goal[3:])
        # TODO
        cur_pos = obs[self.robot_dim + self.object_dim * goal_idx: self.robot_dim + self.object_dim * goal_idx + 3]
        distance = np.linalg.norm(cur_pos - goal_pos)
        # Since it is inconvenient to get previous reward when performing HER, we only accept sparse reward here
        assert self.reward_type == "sparse"
        reward = (distance < 0.05)
        return float(reward), reward

    def relabel_obs(self, obs, goal):
        assert isinstance(obs, np.ndarray) and isinstance(goal, np.ndarray)
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        if len(goal.shape) == 1:
            goal = np.expand_dims(goal, axis=0)
        assert len(obs.shape) == 2 and len(goal.shape) == 2
        obs = obs.copy()
        goal_idx = np.argmax(goal[:, 3:], axis=1)
        goal_dim = goal.shape[-1]
        goal_indicator_idx = np.arange(
            self.robot_dim + self.object_dim - 1, obs.shape[-1] - 2 * goal_dim, self.object_dim
        )
        obs[:, goal_indicator_idx] = 0.
        for idx in range(obs.shape[0]):
            # target obj idx
            obs[idx, self.robot_dim + goal_idx[idx] * self.object_dim + self.object_dim - 1] = 1.
            obs[idx, -2 * goal_dim: -2 * goal_dim + 3] = \
                obs[idx, self.robot_dim + goal_idx[idx] * self.object_dim:
                         self.robot_dim + goal_idx[idx] * self.object_dim + 3]
            obs[idx, -2 * goal_dim + 3: -goal_dim] = goal[idx, 3:]
            obs[idx, -goal_dim:] = goal[idx, :]
        return obs

    def imagine_obs(self, obs, goal, info):
        # imagine a goal has been achieved
        # todo: try to imagine multiple objects movement, ideally with a generative model?
        assert isinstance(obs, np.ndarray) and isinstance(goal, np.ndarray)
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        if len(goal.shape) == 1:
            goal = np.expand_dims(goal, axis=0)
        assert len(obs.shape) == 2 and len(goal.shape) == 2
        obs = obs.copy()
        goal_idx = np.argmax(goal[:, 3:], axis=-1)
        goal_dim = goal.shape[-1]
        ultimate_goal_idx = np.argmax(obs[:, -goal_dim + 3:], axis=-1)

        def is_in_tower(pos: torch.Tensor, goal: torch.Tensor):
            if torch.norm(pos[:2] - goal[:2]) > 0.01:
                return False
            maybe_n_height = (pos[2] - self.robot.z_workspace[0] - 0.025) / 0.05
            if (maybe_n_height - torch.round(maybe_n_height)).abs() > 1e-5:
                return False
            return True

        for idx in range(obs.shape[0]):
            # original objects positions
            max_height = self.robot.base_pos[2] - 0.025 + 0.05 * info["n_base"]
            # n_imagine_to_stack_old = int(torch.round((goal[idx, 2] - max_height_old) / 0.05).item())
            n_imagine_to_stack = \
                int(np.round((goal[idx, 2] - self.robot.base_pos[2] - 0.025) / 0.05)) + 1 - info["n_base"]
            if n_imagine_to_stack > 0:
                # n_base_old = int(torch.round((max_height_old - self.robot.z_workspace[0] - 0.025) / 0.05).item()) + 1
                n_base = info["n_base"]
                move_objects_candidates = list(range(n_base, self.n_object))
                if ultimate_goal_idx[idx] in move_objects_candidates:
                    move_objects_candidates.remove(ultimate_goal_idx[idx])
                if goal_idx[idx] in move_objects_candidates:
                    move_objects_candidates.remove(goal_idx[idx])  # imagined top id
                move_objects_id = list(np.random.choice(move_objects_candidates, size=n_imagine_to_stack - 1, replace=False)) + [goal_idx[idx]]
                assert abs(max_height + len(move_objects_id) * 0.05 - goal[idx, 2]) < 1e-3
                for h_idx in range(len(move_objects_id)):
                    obs[idx, self.robot_dim + move_objects_id[h_idx] * self.object_dim:
                             self.robot_dim + move_objects_id[h_idx] * self.object_dim + 3] = \
                        np.concatenate([goal[idx, :2], [max_height + (h_idx + 1) * 0.05]])
                    obs[idx, self.robot_dim + move_objects_id[h_idx] * self.object_dim + 3:
                             self.robot_dim + move_objects_id[h_idx] * self.object_dim + 6] = \
                        np.concatenate([goal[idx, :2], [max_height + (h_idx + 1) * 0.05]]) - obs[idx, :3]
            else:
                assert abs(goal[idx, 2] - (self.robot.base_pos[2] + 0.025)) < 1e-3
                obs[idx, self.robot_dim + goal_idx[idx] * self.object_dim:
                         self.robot_dim + goal_idx[idx] * self.object_dim + 3] = goal[idx, :3]
                obs[idx, self.robot_dim + goal_idx[idx] * self.object_dim + 3:
                         self.robot_dim + goal_idx[idx] * self.object_dim + 6] = goal[idx, :3] - obs[idx, :3]
            '''
            # object position
            obs[idx, self.robot_dim + goal_idx[idx] * self.object_dim:
                     self.robot_dim + goal_idx[idx] * self.object_dim + 3] = goal[idx, :3]
            # relative position
            obs[idx, self.robot_dim + goal_idx[idx] * self.object_dim + 3:
                     self.robot_dim + goal_idx[idx] * self.object_dim + 6] = goal[idx, :3] - obs[idx, :3]
            '''
            # achieved goal
            obs[idx, -2 * goal_dim: -2 * goal_dim + 3] = \
                obs[idx, self.robot_dim + ultimate_goal_idx[idx] * self.object_dim:
                         self.robot_dim + ultimate_goal_idx[idx] * self.object_dim + 3]
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

    def set_state(self, state_dict):
        self.robot.set_state(state_dict["robot"])
        for i in range(self.n_object):
            qpos = state_dict["objects"]["qpos"][i]
            qvel = state_dict["objects"]["qvel"][i]
            self.p.resetBasePositionAndOrientation(self.blocks_id[i], qpos[:3], qpos[3:])
            self.p.resetBaseVelocity(self.blocks_id[i], qvel[:3], qvel[3:])


class ArmStack(ArmPickAndPlace):
    def __init__(self, *args, n_to_stack=[1], **kwargs):
        self.n_to_stack_choices = n_to_stack
        self.n_to_stack_probs = [1. / len(n_to_stack)] * len(n_to_stack)
        self.n_to_stack = n_to_stack[0]
        self.n_base = 0
        self.base_xy = np.array([0, 0])
        self.cl_ratio = 0
        super(ArmStack, self).__init__(*args, **kwargs)

    def _reset_sim(self):
        # todo: more distracting tasks, e.g. distracting towers
        # self.robot.control([self.robot.base_pos[0] + 0.2, self.robot.base_pos[1] - 0.2, self.robot.base_pos[2] + 0.3],
        #                    (1, 0, 0, 0), 0., relative=False, teleport=True)
        self.robot.set_state(dict(qpos=np.array([-0.35720248, -0.75528038, -0.36600858, -2.77078997, -0.27654494,
                                                 2.02585467,  0.28351196,  0., 0., 0., 0., 0.]),
                                  qvel=np.zeros(12)))
        # print(self.robot.get_eef_position())  # 0.23622709 -0.21536495  0.30265639
        self.n_to_stack = self.np_random.choice(self.n_to_stack_choices, p=self.n_to_stack_probs)
        # self.n_active_object = self.np_random.randint(self.n_to_stack, self.n_object + 1)
        if self.n_to_stack == 1 and self.cl_ratio > 0 and self.np_random.uniform() < self.cl_ratio:
            self.n_active_object = self.np_random.randint(min(4, self.n_object), self.n_object + 1)
        else:
            self.n_active_object = self.n_object
        self.n_base = self.np_random.randint(0, self.n_active_object - self.n_to_stack + 1)
        base_and_other_position = np.array([[self.np_random.uniform(*self.robot.x_workspace),
                                             self.np_random.uniform(*self.robot.y_workspace),
                                             self.robot.base_pos[2] + 0.025]
                                            for _ in range(1 + self.n_active_object - self.n_base)])
        block_halfextent = np.array(self.p.getCollisionShapeData(self.blocks_id[0], -1)[0][3]) / 2
        while _in_collision(base_and_other_position, block_halfextent):
            base_and_other_position = np.array([[self.np_random.uniform(*self.robot.x_workspace),
                                                 self.np_random.uniform(*self.robot.y_workspace),
                                                 self.robot.base_pos[2] + 0.025]
                                                for _ in range(1 + self.n_active_object - self.n_base)])
        self.base_xy = base_and_other_position[0][:2]
        for n in range(self.n_base):
            self.p.resetBasePositionAndOrientation(
                self.blocks_id[n],
                (self.base_xy[0], self.base_xy[1], self.robot.base_pos[2] + 0.025 + 0.05 * n),
                self.gen_obj_quat()
            )
        for n in range(self.n_base, self.n_active_object):
            self.p.resetBasePositionAndOrientation(
                self.blocks_id[n], base_and_other_position[1 + n - self.n_base], self.gen_obj_quat()
            )
        for n in range(self.n_active_object, self.n_object):
            self.p.resetBasePositionAndOrientation(
                self.blocks_id[n],
                list(self.inactive_xy) + [self.robot.base_pos[2] + 0.025 + (n - self.n_active_object) * 0.05],
                (0, 0, 0, 1)
            )
        eef_pos = np.array([self.np_random.uniform(*self.robot.x_workspace),
                            self.np_random.uniform(*self.robot.y_workspace),
                            self.robot.init_eef_height + 0.1])
        while np.linalg.norm(eef_pos[:2] - self.base_xy) < 0.05:
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
        goal = np.array([self.base_xy[0], self.base_xy[1],
                         self.robot.base_pos[2] + 0.025 + (self.n_base + self.n_to_stack - 1) * 0.05])
        goal_onehot = np.zeros(self.n_object)
        goal_idx = self.np_random.randint(self.n_base, self.n_active_object)
        goal_onehot[goal_idx] = 1
        goal = np.concatenate([goal, goal_onehot])
        if self.reward_type == "sparse" and self.n_to_stack == 1 and \
                self.n_base > 0 and self.np_random.uniform() < 0.5:
            self.robot.control(
                [0, 0, 0], [1, 0, 0, 0],
                (self.robot.finger_range[0] + self.robot.finger_range[1]) / 2,
                relative=True, teleport=True
            )
            self.p.resetBasePositionAndOrientation(
                self.blocks_id[goal_idx], self.robot.get_eef_position(), (0, 0, 0, 1))
        if self.n_to_stack > 2 and self.np_random.uniform() < 0.5:
            # generate block position very close to base position
            all_block_positions = [np.array(self.p.getBasePositionAndOrientation(i)[0]) for i in self.blocks_id]
            all_block_positions[goal_idx][0] = self.base_xy[0] + self.np_random.uniform(0.05, 0.1) * self.np_random.choice([1, -1])
            all_block_positions[goal_idx][1] = self.base_xy[1] + self.np_random.uniform(0.05, 0.1) * self.np_random.choice([1, -1])
            block_halfextent = np.array(self.p.getCollisionShapeData(self.blocks_id[0], -1)[0][3]) / 2
            _count = 0
            while _in_collision(all_block_positions[self.n_base: self.n_active_object], block_halfextent) \
                    and _count < 10:
                all_block_positions[goal_idx][0] = \
                    self.base_xy[0] + self.np_random.uniform(0.05, 0.1) * self.np_random.choice([1, -1])
                all_block_positions[goal_idx][1] = \
                    self.base_xy[1] + self.np_random.uniform(0.05, 0.1) * self.np_random.choice([1, -1])
                _count += 1
            self.p.resetBasePositionAndOrientation(
                self.blocks_id[goal_idx], all_block_positions[goal_idx], (0, 0, 0, 1)
            )
        self.visualize_goal(goal)
        return goal

    def set_choice_prob(self, n_to_stack, prob):
        probs = self.n_to_stack_probs
        assert len(n_to_stack) == len(prob)
        visited = [False] * len(probs)
        for i in range(len(n_to_stack)):
            assert n_to_stack[i] in self.n_to_stack_choices
            idx = np.where(np.array(self.n_to_stack_choices) == n_to_stack[i])[0][0]
            visited[idx] = True
            probs[idx] = prob[i]
        for i in range(len(visited)):
            if not visited[i]:
                probs[i] = (1 - sum(prob)) / (len(visited) - len(n_to_stack))
        self.n_to_stack_probs = probs

    def sync_attr(self):
        goal_pos = self.goal[:3]
        if hasattr(self, "goals") and isinstance(self.goals, list) and len(self.goals):
            goal_pos = self.goals[-1][:3]
        n_base, n_active_object = 0, 0
        for n in self.blocks_id:
            pos, orn = self.p.getBasePositionAndOrientation(n)
            if np.linalg.norm(np.array(pos[:2]) - np.array(self.inactive_xy)) > 0.01:
                n_active_object += 1
                if np.linalg.norm(np.array(pos[:2]) - goal_pos[:2]) < 1e-3:
                    n_base += 1
        n_to_stack = int(round((goal_pos[2] - self.robot.base_pos[2] - 0.025) / 0.05)) + 1 - n_base
        self.n_to_stack = n_to_stack
        self.n_active_object = n_active_object
        self.n_base = n_base
        self.base_xy = goal_pos[:2]

    def get_info_from_objects(self, objects_pos, goal):
        if isinstance(objects_pos, torch.Tensor):
            objects_pos = objects_pos.cpu().numpy()
        assert len(objects_pos.shape) == 2

        def in_tower(pos):
            if np.linalg.norm(pos[:2] - goal[:2]) > 1e-2:
                return False
            maybe_n = (pos[2] - self.robot.base_pos[2] - 0.025) / 0.05
            if abs(maybe_n - np.round(maybe_n)) > 1e-2:
                return False
            return True
        in_tower_heights = [objects_pos[i][2] for i in range(objects_pos.shape[0]) if in_tower(objects_pos[i])]
        max_height = np.max(in_tower_heights) if len(in_tower_heights) else self.robot.base_pos[2] - 0.025
        n_base = int(np.round((max_height - (self.robot.base_pos[2] - 0.025)) / 0.05))
        n_to_stack = int(np.round((goal[2] - (self.robot.base_pos[2] - 0.025)) / 0.05)) - n_base
        n_active = int(np.sum(np.linalg.norm(objects_pos[:, :2] + 1, axis=-1) > 1e-2))
        return dict(n_base=n_base, n_to_stack=n_to_stack, n_active=n_active)

    def compute_reward_and_info(self):
        goal_pos = self.goal[:3]
        cur_pos = self._get_achieved_goal()
        goal_distance = np.linalg.norm(goal_pos - cur_pos)
        eef_distance = np.linalg.norm(self.robot.get_eef_position() - goal_pos)
        is_stable = False
        eef_threshold = 0.1 * (1 - self.cl_ratio) if self.n_to_stack == 1 else 0.1
        if goal_distance < 0.03 and eef_distance > eef_threshold:
            # debug_obs1 = self._get_obs()
            state_id = self.p.saveState()
            for _ in range(50):
                self.p.stepSimulation()
            future_pos = self._get_achieved_goal()
            if np.linalg.norm(future_pos - cur_pos) < 1e-3:
                is_stable = True
            self.p.restoreState(stateId=state_id)
            self.p.removeState(state_id)
            # debug_obs2 = self._get_obs()
            # for k in debug_obs1.keys():
            #     assert np.linalg.norm(debug_obs1[k] - debug_obs2[k]) < 1e-5, (k, debug_obs1[k] - debug_obs2[k])

        if self.reward_type == "sparse":
            reward = float(is_stable)
        elif self.reward_type == "dense":
            # TODO
            reward = self._previous_distance - (goal_distance - (goal_distance < 0.05) * eef_distance) + is_stable
            self._previous_distance = goal_distance - (goal_distance < 0.05) * eef_distance
        else:
            raise NotImplementedError
        is_success = is_stable
        info = {'is_success': is_success, "n_to_stack": self.n_to_stack, "n_base": self.n_base}
        return reward, info

    def compute_reward(self, obs, goal):
        # For stacking, it seems meaningless to perform HER
        raise NotImplementedError

    def set_cl_ratio(self, cl_ratio):
        self.cl_ratio = cl_ratio


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
    env = ArmPickAndPlace(n_object=4)
    print("created env")
    obs = env.reset()
    print("reset obs", obs)
    fig, ax = plt.subplots(1, 1)
    for i in range(1000):
        img = env.render(mode="rgb_array")
        ax.cla()
        ax.imshow(img)
        plt.pause(0.1)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    print(obs)