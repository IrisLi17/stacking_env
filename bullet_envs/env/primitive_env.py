import gym
import os
import imageio
import pybullet as p
import pybullet_data
import numpy as np
import time
from bullet_envs.env.bullet_rotations import quat_diff, quat_mul
from bullet_envs.env.robot import PandaRobot
from gym import spaces
from gym.utils import seeding
from pybullet_utils import bullet_client as bc
from typing import Any, Dict, Tuple
from collections import OrderedDict
import pkgutil
egl = pkgutil.get_loader('eglRenderer')


DATAROOT = pybullet_data.getDataPath()

class PrimitiveType:
    MOVE_DIRECT = 0
    GRIPPER_OPEN = 1
    GRIPPER_GRASP = 2


class BasePrimitiveEnv(gym.Env):
    def __init__(self, seed=None) -> None:
        super().__init__()
        self.seed(seed)
        self._setup_env()
        self.goal = self.sample_goal()
        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            OrderedDict([(key, spaces.Box(low=-np.inf, high=np.inf, shape=obs[key].shape)) for key in obs])
        )
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
        self.record_cfg = dict(
            save_video_path=os.path.join(os.path.dirname(__file__), "..", "tmp"),
            fps=10
        )
        self.approach_dist = 0.05

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    @property
    def dt(self):
        return 1. / 240

    def reset(self):
        self._reset_sim()
        if self.robot.get_finger_width() > 0.07:
            gripper_status = "open"
        else:
            gripper_status = "close"
        self.robot.reset_primitive(gripper_status, self._get_graspable_objects(), render)
        # sample goal
        self.goal = self.sample_goal()
        obs = self._get_obs()
        return obs    
    
    def step(self, action) -> Tuple[Any, float, bool, Dict[str, Any]]:
        # Parse primitive
        # Calculate reward and info
        action = action.copy()
        assert action.shape[0] == 1 + 4
        primitive_type = int(np.round(action[0]))
        action[1:] = np.clip(action[1:], -1.0, 1.0)
        # t0 = time.time()
        if primitive_type == PrimitiveType.MOVE_DIRECT:
            # Add neutral value and scale
            eef_pos = action[1:4] * (self.robot_eef_range[1] - self.robot_eef_range[0]) / 2 + np.mean(self.robot_eef_range, axis=0)
            eef_euler = np.array([np.pi, 0., action[4] * np.pi / 2])
            # 2pi modulo
            # if eef_euler[0] > np.pi:
            #     eef_euler[0] -= 2 * np.pi
            eef_quat = self.p.getQuaternionFromEuler(eef_euler)
            self.robot.move_direct_ee_pose(eef_pos, eef_quat)
        elif primitive_type == PrimitiveType.GRIPPER_OPEN:
            self.robot.gripper_move("open")
        elif primitive_type == PrimitiveType.GRIPPER_GRASP:
            self.robot.gripper_grasp()
        else:
            raise NotImplementedError
        # print("step primitive", time.time() - t0)
        new_obs = self._get_obs()
        reward, info = self.compute_reward_and_info()
        done = False
        return new_obs, reward, done, info
    
    def sample_goal(self):
        return {"state": None, "img": None}

    def compute_reward_and_info(self):
        reward = 0
        info = {}
        return reward, info
    
    def render(self, width=500, height=500):
        return render(self.p, width, height)

    def start_rec(self, video_filename):
        assert self.record_cfg

        # make video directory
        if not os.path.exists(self.record_cfg['save_video_path']):
            os.makedirs(self.record_cfg['save_video_path'])

        # close and save existing writer
        if hasattr(self.robot, 'video_writer'):
            self.robot.video_writer.close()

        # initialize writer
        self.robot.video_writer = imageio.get_writer(os.path.join(self.record_cfg['save_video_path'],
                                                            f"{video_filename}.mp4"),
                                               fps=self.record_cfg['fps'],
                                               format='FFMPEG',
                                               codec='h264',)
        self.robot.save_video = True

    def end_rec(self):
        if hasattr(self.robot, 'video_writer'):
            self.robot.video_writer.close()
        self.robot.save_video = False

    def _setup_env(self, init_qpos=None, base_position=(0, 0, 0)):
        self.p = bc.BulletClient(connection_mode=p.DIRECT)
        plugin = self.p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        print("plugin=", plugin)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
        
        self.p.resetSimulation()
        self.p.setTimeStep(self.dt)
        self.p.setGravity(0., 0., -9.8)
        self.p.resetDebugVisualizerCamera(1.0, 40, -20, [0, 0, 0,] )
        plane_id = self.p.loadURDF(os.path.join(DATAROOT, "plane.urdf"), [0, 0, -0.795])
        table_id = self.p.loadURDF(
            os.path.join(DATAROOT, "table/table.urdf"), 
            [0.40000, 0.00000, -.625000], [0.000000, 0.000000, 0.707, 0.707]
        )
        self.robot = PandaRobot(self.p, init_qpos, base_position, is_visible=False)
        _robot_eef_low = np.array([0.3, -0.25, 0.0])
        _robot_eef_high = np.array([0.6, 0.25, 0.3])
        # robot_eef_center = np.array([0.5, 0.0, 0.15])
        self.robot_eef_range = np.stack([
            _robot_eef_low, _robot_eef_high
        ], axis=0)
        self._setup_callback()
    
    def _setup_callback(self):
        pass

    def _reset_sim(self):
        pass

    def _get_obs(self):
        robot_state = self.robot.get_state()
        joint_pos = robot_state["qpos"]
        eef_pos = self.robot.get_eef_position()
        eef_euler = self.robot.get_eef_orn(as_type="euler")
        scene = render(self.p, width=224, height=224).transpose((2, 0, 1))[:3]
        return {"img": scene, "robot_state": np.concatenate([joint_pos, eef_pos, eef_euler]), "goal": self.goal["img"]}
    
    def _get_graspable_objects(self):
        return ()
    
    def _get_ego_view(self, width, height):
        eef_pos = self.robot.get_eef_position().reshape((3, 1))
        eef_rot = np.array(self.p.getMatrixFromQuaternion(self.robot.get_eef_orn())).reshape((3, 3))
        # TODO
        eef_t_cam = np.array([0.06, 0.0, -0.04]).reshape((3, 1))
        eye_position = eef_rot @ eef_t_cam + eef_pos
        target_position = eye_position + eef_rot @ np.array([0, 0, 1]).reshape((3, 1))
        up_vector = eef_rot @ np.array([1, 0, 0])
        # up_vector = np.array([0, 0, -1])
        view_matrix = self.p.computeViewMatrix(eye_position, target_position, up_vector)
        proj_matrix = self.p.computeProjectionMatrixFOV(
            fov=120, aspect=1.0, nearVal=0.01, farVal=10.0
        )
        (_, _, px, _, _) = self.p.getCameraImage(
            width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        )
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (height, width, 4))
        return rgb_array


def render(client: bc.BulletClient, width=256, height=256) -> np.ndarray:
    # TODO: eye on hand
    view_matrix = client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=(0.45, 0, 0.0),
        distance=0.6,
        yaw=-90,
        pitch=-60,
        roll=0,
        upAxisIndex=2,
    )
    proj_matrix = client.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
    )
    (_, _, px, _, _) = client.getCameraImage(
        width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
    )
    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (height, width, 4))

    return rgb_array


# def process_img(color, depth, cam_config):
#     # goal: filter out robot and background, the perspective from robot?
class BoxLidEnv(BasePrimitiveEnv):
    def __init__(self, seed=None, reward_type="sparse") -> None:
        super().__init__(seed)
        self.approach_dist = 0.1
        self.dist_threshold = 0.01
        self.rot_threshold = 0.1
        self.reward_type = reward_type

    def _setup_callback(self):
        # self.drawer_id = self.p.loadURDF(
        #     os.path.join(os.path.dirname(__file__), "assets/drawer.urdf"), 
        #     [0.40000, 0.00000, 0.1], [0.000000, 0.000000, 0.0, 1.0],
        #     globalScaling=0.125
        # )
        # self.drawer_range = np.array([
        #     [self.robot_eef_range[0][0] + 0.05, self.robot_eef_range[0][1] + 0.05, 0.05],
        #     [self.robot_eef_range[1][0], self.robot_eef_range[1][1] - 0.05, 0.05]
        # ])
        # for j in range(self.p.getNumJoints(self.drawer_id)):
        #     joint_info = self.p.getJointInfo(self.drawer_id, j)
        #     if joint_info[2] != self.p.JOINT_FIXED:
        #         self.drawer_joint = joint_info[0]
        #         self.drawer_handle_range = (joint_info[8], joint_info[9])
        #     if joint_info[12] == b'handle_r':
        #         self.drawer_handle_link = joint_info[0]
        #         # self.joint_damping.append(joint_info[6])
        #         break
        self.box_base_id = self.p.loadURDF(
            os.path.join(os.path.dirname(__file__), "assets/box_no_lid.urdf"),
            [0.5, -0.3, 0.08], [0., 0., 0., 1.], useFixedBase=True,
        )
        self.box_lid_id = self.p.loadURDF(
            os.path.join(os.path.dirname(__file__), "assets/box_lid.urdf"),
            [0.5, -0.3, 0.13], [0., 0., 0., 1.]
        )
        self.box_range = np.array([
            [self.robot_eef_range[0][0] + 0.08, self.robot_eef_range[0][1] + 0.08],
            [self.robot_eef_range[1][0] - 0.08, self.robot_eef_range[1][1] - 0.08],
        ])
        for j in range(self.p.getNumJoints(self.box_lid_id)):
            joint_info = self.p.getJointInfo(self.box_lid_id, j)
            if joint_info[12] == b'lid':
                self.box_lid_link = joint_info[0]
                break
        if not hasattr(self, "graspable_objects"):
            self.graspable_objects = ((self.box_lid_id, self.box_lid_link),)
    
    def _reset_sim(self):
        # reset drawer
        # rand_angle = np.random.uniform(-np.pi, 0.)
        # self.p.resetBasePositionAndOrientation(
        #     self.drawer_id, 
        #     np.random.uniform(self.drawer_range[0], self.drawer_range[1]),
        #     # ((self.drawer_range[0][0] + self.drawer_range[1][0]) / 2, 0.1, self.drawer_range[0][2]),
        #     (0., 0., np.sin(rand_angle / 2), np.cos(rand_angle / 2))
        # )
        # # reset drawer joint
        # self.p.resetJointState(self.drawer_id, 0, np.random.uniform(*self.drawer_handle_range))
        
        # reset box
        box_xy = np.random.uniform(
            low=self.box_range[0], high=self.box_range[1],
        )
        self.p.resetBasePositionAndOrientation(
            self.box_base_id, np.concatenate([box_xy, [0.03]]), [0., 0., 0., 1.]
        )
        self.p.resetBasePositionAndOrientation(
            self.box_lid_id, np.concatenate([box_xy , [0.095]]), [0., 0., 0., 1.]
        )
        self.robot.control(
            np.random.uniform(low=self.robot_eef_range[0], high=self.robot_eef_range[1]), 
            np.array([1., 0., 0., 0.]), 0.04, relative=False, teleport=True
        )
        while True:
            self.p.performCollisionDetection()
            is_in_contact = False
            is_in_contact = is_in_contact or (len(self.p.getContactPoints(bodyA=self.robot.id, bodyB=self.box_base_id)) > 0)
            is_in_contact = is_in_contact or (len(self.p.getContactPoints(bodyA=self.robot.id, bodyB=self.box_lid_id)) > 0)
            if not is_in_contact:
                break
            self.robot.control(
                np.random.uniform(low=self.robot_eef_range[0], high=self.robot_eef_range[1]), 
                np.array([1., 0., 0., 0.]), 0.04, relative=False, teleport=True
            )
        
    def _get_graspable_objects(self):
        return self.graspable_objects
    
    def sample_goal(self):
        # store current state for recovery
        robot_state = self.robot.get_state()
        box_lid_pose = self.p.getBasePositionAndOrientation(self.box_lid_id)        
        
        # pair of underlying state and goal image
        box_base_pose = self.p.getBasePositionAndOrientation(self.box_base_id)
        goal_lid_pos = np.concatenate([np.random.uniform(low=self.box_range[0], high=self.box_range[1]), [0.025]])
        while abs(goal_lid_pos[0] - box_base_pose[0][0]) < 0.21 and abs(goal_lid_pos[1] - box_base_pose[0][1]) < 0.16:
            goal_lid_pos = np.concatenate([np.random.uniform(low=self.box_range[0], high=self.box_range[1]), [0.025]])
        goal_lid_quat = box_lid_pose[1]

        # set into the environment to get image
        self.p.resetBasePositionAndOrientation(self.box_lid_id, goal_lid_pos, goal_lid_quat)
        # Need to simulate until valid, or make sure the sampled goal is stable
        self.p.stepSimulation()
        goal_img = render(self.p, width=224, height=224).transpose((2, 0, 1))[:3]
        goal_dict = {'state': (goal_lid_pos, goal_lid_quat), 'img': goal_img}

        # recover state
        self.p.resetBasePositionAndOrientation(self.box_lid_id, box_lid_pose[0], box_lid_pose[1])
        self.robot.set_state(robot_state)
        return goal_dict
    
    def compute_reward_and_info(self):
        cur_lid_pose = self.p.getLinkState(self.box_lid_id, self.box_lid_link)[:2]
        goal_lid_pose = self.goal["state"]
        dist_lid_pos = np.linalg.norm(goal_lid_pose[0] - cur_lid_pose[0])
        dist_lid_ang = 2 * np.arccos(quat_diff(goal_lid_pose[1], cur_lid_pose[1])[3])
        is_success = dist_lid_pos < self.dist_threshold and dist_lid_ang < self.rot_threshold
        reward = float(is_success) if self.reward_type == "sparse" else -dist_lid_pos * self.rew_dist_coef - dist_lid_ang * self.rew_rot_coef
        info = {"state": np.concatenate(cur_lid_pose), "is_success": is_success}
        return reward, info

    def _eef_pos_to_action(self, eef_pos):
        return (eef_pos - np.mean(self.robot_eef_range, axis=0)) / ((self.robot_eef_range[1] - self.robot_eef_range[0]) / 2)

    def oracle_agent(self, task="open_box"):
        if task == "open_box":
            # move up
            action = np.concatenate([
                [PrimitiveType.MOVE_DIRECT], self._eef_pos_to_action(
                    self.robot.get_eef_position() + np.array([0., 0., 0.2])
                ), [0.]])
            self.step(action)
            # move to lid
            lid_pose = self.p.getLinkState(self.box_lid_id, self.box_lid_link)[:2]
            action = np.zeros(5)
            action[0] = PrimitiveType.MOVE_DIRECT
            eef_pos = lid_pose[0]
            action[1:4] = self._eef_pos_to_action(eef_pos)
            self.step(action)
            # grasp
            action = np.concatenate([[PrimitiveType.GRIPPER_GRASP], [0., 0., 0., 0.]])
            self.step(action)
            # lift up
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self._eef_pos_to_action(self.robot.get_eef_position() + np.array([0., 0., 0.1])), [0.]])
            self.step(action)
            # put down
            base_pos = self.p.getBasePositionAndOrientation(self.box_base_id)[0]
            if base_pos[1] > 0:
                new_pos = base_pos + np.array([0., -0.25, 0.02])
            else:
                new_pos = base_pos + np.array([0., 0.25, 0.02])
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self._eef_pos_to_action(new_pos), [0.]])
            self.step(action)
            # release
            action = np.concatenate([[PrimitiveType.GRIPPER_OPEN], [0., 0., 0., 0.]])
            self.step(action)
            # go up
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self._eef_pos_to_action(self.robot.get_eef_position() + np.array([-0.1, -0.2, 0.2])), [0.]])
            self.step(action)
        elif task == "close_box":
            lid_pose = self.p.getLinkState(self.box_lid_id, self.box_lid_link)[:2]
            lid_yaw = self.p.getEulerFromQuaternion(lid_pose[1])[2]
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self._eef_pos_to_action(lid_pose[0]), [lid_yaw / (np.pi / 2)]])
            self.step(action)
            # grasp
            action = np.array([PrimitiveType.GRIPPER_GRASP, 0., 0., 0., 0.])
            self.step(action)
            # lift up
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self._eef_pos_to_action(self.robot.get_eef_position() + np.array([0., 0., 0.2])), [0.]])
            self.step(action)
            # move to box
            box_pos = self.p.getBasePositionAndOrientation(self.box_base_id)[0]
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self._eef_pos_to_action(box_pos + np.array([0., 0., 0.1])), [0.]])
            self.step(action)
            # release
            action = np.concatenate([[PrimitiveType.GRIPPER_OPEN], [0., 0., 0., 0.]])
            self.step(action)
            # move away
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self._eef_pos_to_action(self.robot.get_eef_position() + np.array([0., 0., 0.1])), [0.]])
            self.step(action)
        return 


class DrawerObjEnv(BasePrimitiveEnv):
    def __init__(self, seed=None, reward_type="dense") -> None:
        super().__init__(seed)
        self.approach_dist = 0.1
        self.handle_pos_threshold = 0.01
        self.reward_type = reward_type
    
    def _setup_callback(self):
        self.drawer_id = self.p.loadURDF(
            os.path.join(os.path.dirname(__file__), "assets/drawer.urdf"), 
            [0.40000, 0.00000, 0.1], [0.000000, 0.000000, 0.0, 1.0],
            globalScaling=0.125
        )
        self.drawer_range = np.array([
            [self.robot_eef_range[0][0] + 0.15, self.robot_eef_range[0][1] + 0.05, 0.05],
            [self.robot_eef_range[1][0], self.robot_eef_range[1][1] - 0.05, 0.05]
        ])
        for j in range(self.p.getNumJoints(self.drawer_id)):
            joint_info = self.p.getJointInfo(self.drawer_id, j)
            if joint_info[2] != self.p.JOINT_FIXED:
                self.drawer_joint = joint_info[0]
                self.drawer_handle_range = (joint_info[8], joint_info[9])
            if joint_info[12] == b'handle_r':
                self.drawer_handle_link = joint_info[0]
                # self.joint_damping.append(joint_info[6])
                break
        if not hasattr(self, "graspable_objects"):
            self.graspable_objects = ((self.drawer_id, self.drawer_handle_link),)

    def _reset_sim(self):
        def _randomize_drawer():
            rand_angle = np.random.uniform(-np.pi, 0.)
            # rand_angle = 0
            self.p.resetBasePositionAndOrientation(
                self.drawer_id, 
                np.random.uniform(self.drawer_range[0], self.drawer_range[1]),
                # ((self.drawer_range[0][0] + self.drawer_range[1][0]) / 2, 0.1, self.drawer_range[0][2]),
                (0., 0., np.sin(rand_angle / 2), np.cos(rand_angle / 2))
            )
            # reset drawer joint
            self.p.resetJointState(self.drawer_id, 0, np.random.uniform(*self.drawer_handle_range))
        _randomize_drawer()
        _count = 0
        # check handle position
        handle_position = self.p.getLinkState(self.drawer_id, self.drawer_handle_link)[0]
        while _count < 10 and not (self.robot_eef_range[0][0] <= handle_position[0] <= self.robot_eef_range[1][0] and self.robot_eef_range[0][1] <= handle_position[1] <= self.robot_eef_range[1][1]):
            _randomize_drawer()
            handle_position = self.p.getLinkState(self.drawer_id, self.drawer_handle_link)[0]
            _count += 1
        # reset robot
        self.robot.control(
            np.random.uniform(low=self.robot_eef_range[0], high=self.robot_eef_range[1]), 
            np.array([1., 0., 0., 0.]), 0.04, relative=False, teleport=True
        )
        reset_count = 0
        while reset_count < 10:
            self.p.performCollisionDetection()
            is_in_contact = False
            is_in_contact = is_in_contact or (len(self.p.getContactPoints(bodyA=self.robot.id, bodyB=self.drawer_id)) > 0)
            if not is_in_contact:
                break
            self.robot.control(
                np.random.uniform(low=self.robot_eef_range[0], high=self.robot_eef_range[1]), 
                np.array([1., 0., 0., 0.]), 0.04, relative=False, teleport=True
            )
            reset_count += 1
    
    def _get_graspable_objects(self):
        return self.graspable_objects
    
    def sample_goal(self):
        # store current state for recovery
        robot_state = self.robot.get_state()
        drawer_joint_state = self.p.getJointState(self.drawer_id, self.drawer_joint)        
        
        # pair of underlying state and goal image
        goal_drawer_joint = np.random.uniform(*self.drawer_handle_range)

        # set into the environment to get image
        self.robot.control(np.array([0.4, 0.0, 0.25]), np.array([1., 0., 0., 0.]), 0.04, relative=False, teleport=True)
        self.p.resetJointState(self.drawer_id, self.drawer_joint, goal_drawer_joint, 0.)
        # Need to simulate until valid, or make sure the sampled goal is stable
        self.p.stepSimulation()
        goal_img = render(self.p, width=224, height=224).transpose((2, 0, 1))[:3]
        goal_dict = {'state': (goal_drawer_joint,), 'img': goal_img}

        # recover state
        self.p.resetJointState(self.drawer_id, self.drawer_joint, drawer_joint_state[0], drawer_joint_state[1])
        self.robot.set_state(robot_state)
        return goal_dict
    
    def compute_reward_and_info(self):
        cur_handle_joint = self.p.getJointState(self.drawer_id, self.drawer_joint)[0]
        handle_dist = abs(self.goal["state"][0] - cur_handle_joint)
        is_success = handle_dist < self.handle_pos_threshold
        reward = float(is_success) if self.reward_type == "sparse" else -handle_dist
        info = {'handle_joint': cur_handle_joint, 'is_success': is_success}
        return reward, info
    
    def oracle_agent(self, mode="open_drawer"):
        def eef_pos_to_action(eef_pos):
            return (eef_pos - np.mean(self.robot_eef_range, axis=0)) / ((self.robot_eef_range[1] - self.robot_eef_range[0]) / 2)
        if mode == "open_drawer":
            # lift up
            cur_robot_eef = self.robot.get_eef_position()
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], eef_pos_to_action(cur_robot_eef)[:2], [1.0, 0.0]])
            self.step(action)
            # move to above handle
            handle_pose = self.p.getLinkState(self.drawer_id, self.drawer_handle_link)[:2]
            drawer_center = self.p.getBasePositionAndOrientation(self.drawer_id)
            print("handle_pose", handle_pose, "base pose", self.p.getBasePositionAndOrientation(self.drawer_id))
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], eef_pos_to_action(handle_pose[0])[:2], [1.0, 0.0]])
            self.step(action)
            # offset a little
            handle_pose = self.p.multiplyTransforms(handle_pose[0], handle_pose[1], np.array([0., -0.02, 0.]), np.array([0., 0., 0., 1.]))
            print("offset handle pose", handle_pose)
            action = np.zeros(5)
            action[0] = PrimitiveType.MOVE_DIRECT
            eef_pos = handle_pose[0] + np.array([0.0, 0.0, 0.005])
            action[1:4] = (eef_pos - np.mean(self.robot_eef_range, axis=0)) / ((self.robot_eef_range[1] - self.robot_eef_range[0]) / 2)
            handle_euler = self.p.getEulerFromQuaternion(quat_diff(handle_pose[1], np.array([0., np.sin(1.57 / 2), 0., np.cos(1.57 / 2)])))
            print("handle_euler", handle_euler)
            action[4] = (handle_euler[2] % (np.pi)) / (np.pi / 2)
            if action[4] > 1:
                action[4] -= 2
            self.step(action)
            # grasp
            action = np.array([PrimitiveType.GRIPPER_GRASP, 0., 0., 0., 0.])
            self.step(action)
            # move
            move_dir = (np.array(handle_pose[0]) - np.array(drawer_center[0]))[:2]
            move_dir = move_dir / np.linalg.norm(move_dir)
            eef_pos = eef_pos + 0.2 * np.concatenate([move_dir, [0.]])
            action = np.zeros(5)
            action[0] = PrimitiveType.MOVE_DIRECT
            action[1:4] = (eef_pos - np.mean(self.robot_eef_range, axis=0)) / ((self.robot_eef_range[1] - self.robot_eef_range[0]) / 2)
            action[4] = (handle_euler[2] % np.pi) / (np.pi / 2)
            if action[4] > 1:
                action[4] -= 2
            self.step(action)
            # release
            action = np.array([PrimitiveType.GRIPPER_OPEN, 0., 0., 0., 0.])
            self.step(action)


if __name__ == "__main__":
    env = DrawerObjEnv()
    obs = env.reset()
    cur_img = obs["img"]
    goal_img = obs["goal"]
    goal_state = env.goal["state"]
    import matplotlib.pyplot as plt
    plt.imsave("cur_img.png", cur_img.transpose((1, 2, 0)))
    plt.imsave("goal_img.png", goal_img.transpose((1, 2, 0)))
    print("goal state", goal_state, "robot_state", obs["robot_state"])
    env.start_rec("test")
    env.oracle_agent()
    # env.oracle_agent("open_box")
    # env.oracle_agent("close_box")
    # for i in range(50):
    #     action = np.random.uniform(-1.0, 1.0, size=(5,))
    #     action[0] = np.random.randint(4)
    # #     if i == 0:
    # #     print("Action", action)
    #     env.step(action)
    env.end_rec()
