import gym
import os
import imageio
import pybullet as p
import pybullet_data
import numpy as np
from env.bullet_rotations import quat_diff, quat_mul
from env.robot import PandaRobot
from gym import spaces
from gym.utils import seeding
from pybullet_utils import bullet_client as bc
from typing import Any, Dict, Tuple


DATAROOT = pybullet_data.getDataPath()

class PrimitiveType:
    MOVE_DIRECT = 0
    MOVE_APPROACH = 1
    GRIPPER_OPEN = 2
    GRIPPER_GRASP = 3


class BasePrimitiveEnv(gym.Env):
    def __init__(self, seed=None) -> None:
        super().__init__()
        self.seed(seed)
        self._setup_env()
        self.goal = self.sample_goal()
        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            {key: spaces.Box(low=-np.inf, high=np.inf, shape=obs[key].shape) for key in obs}
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
        if primitive_type in [PrimitiveType.MOVE_DIRECT, PrimitiveType.MOVE_APPROACH]:
            # Add neutral value and scale
            eef_pos = action[1:4] * (self.robot_eef_range[1] - self.robot_eef_range[0]) / 2 + np.mean(self.robot_eef_range, axis=0)
            eef_euler = np.array([np.pi, 0., action[4] * np.pi / 2])
            # 2pi modulo
            # if eef_euler[0] > np.pi:
            #     eef_euler[0] -= 2 * np.pi
            eef_quat = self.p.getQuaternionFromEuler(eef_euler)
        if primitive_type == PrimitiveType.MOVE_DIRECT:
            self.robot.move_direct_ee_pose(eef_pos, eef_quat)
        elif primitive_type == PrimitiveType.MOVE_APPROACH:
            self.robot.move_approach_ee_pose(eef_pos, eef_quat, approach_dist=self.approach_dist)
        elif primitive_type == PrimitiveType.GRIPPER_OPEN:
            self.robot.gripper_move("open")
        elif primitive_type == PrimitiveType.GRIPPER_GRASP:
            self.robot.gripper_grasp()
        else:
            raise NotImplementedError
        new_obs = self._get_obs()
        reward, info = self.compute_reward_and_info()
        done = False
        return new_obs, reward, done, info
    
    def sample_goal(self):
        pass

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
        self.p.resetSimulation()
        self.p.setTimeStep(self.dt)
        self.p.setGravity(0., 0., -9.8)
        self.p.resetDebugVisualizerCamera(1.0, 40, -20, [0, 0, 0,] )
        plane_id = self.p.loadURDF(os.path.join(DATAROOT, "plane.urdf"), [0, 0, -0.795])
        table_id = self.p.loadURDF(
            os.path.join(DATAROOT, "table/table.urdf"), 
            [0.40000, 0.00000, -.625000], [0.000000, 0.000000, 0.707, 0.707]
        )
        self.robot = PandaRobot(self.p, init_qpos, base_position)
        robot_eef_center = np.array([0.5, 0.0, 0.15])
        self.robot_eef_range = np.stack([
            robot_eef_center - np.array([0.2, 0.25, 0.15]),
            robot_eef_center + np.array([0.2, 0.25, 0.15])
        ], axis=0)
        self._setup_callback()
    
    def _setup_callback(self):
        pass

    def _reset_sim(self):
        pass

    def _get_obs(self):
        return {}
    
    def _get_graspable_objects(self):
        return ()


def render(client: bc.BulletClient, width=256, height=256):
    view_matrix = client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=(0.3, 0, 0.2),
        distance=1.0,
        yaw=60,
        pitch=-10,
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


class DrawerObjEnv(BasePrimitiveEnv):
    def __init__(self, seed=None) -> None:
        super().__init__(seed)
        self.approach_dist = 0.1

    def _setup_callback(self):
        self.drawer_id = self.p.loadURDF(
            os.path.join(os.path.dirname(__file__), "assets/drawer.urdf"), 
            [0.40000, 0.00000, 0.1], [0.000000, 0.000000, 0.0, 1.0],
            globalScaling=0.125
        )
        self.drawer_range = np.array([
            [self.robot_eef_range[0][0] + 0.05, self.robot_eef_range[0][1] + 0.05, 0.05],
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
        self.robot.control(
            np.random.uniform(low=self.robot_eef_range[0], high=self.robot_eef_range[1]), 
            np.array([1., 0., 0., 0.]), 0.04, relative=False, teleport=True
        )
        # reset drawer
        rand_angle = np.random.uniform(-np.pi, 0.)
        self.p.resetBasePositionAndOrientation(
            self.drawer_id, 
            np.random.uniform(self.drawer_range[0], self.drawer_range[1]),
            (0., 0., np.sin(rand_angle / 2), np.cos(rand_angle / 2))
        )
        # reset drawer joint
        self.p.resetJointState(self.drawer_id, 0, np.random.uniform(*self.drawer_handle_range))
        
    def _get_graspable_objects(self):
        return self.graspable_objects
    
    def oracle_agent(self):
        handle_pose = self.p.getLinkState(self.drawer_id, self.drawer_handle_link)[:2]
        print("handle_pose", handle_pose, "base pose", self.p.getBasePositionAndOrientation(self.drawer_id))
        # offset a little
        handle_pose = self.p.multiplyTransforms(handle_pose[0], handle_pose[1], np.array([0., -0.02, 0.]), np.array([0., 0., 0., 1.]))
        print("offset handle pose", handle_pose)
        action = np.zeros(5)
        action[0] = PrimitiveType.MOVE_APPROACH
        eef_pos = handle_pose[0]
        action[1:4] = (eef_pos - np.mean(self.robot_eef_range, axis=0)) / ((self.robot_eef_range[1] - self.robot_eef_range[0]) / 2)
        handle_euler = self.p.getEulerFromQuaternion(quat_diff(handle_pose[1], np.array([0., np.sin(1.57 / 2), 0., np.cos(1.57 / 2)])))
        print("handle_euler", handle_euler)
        action[4] = (handle_euler[2] % (np.pi)) / (np.pi / 2)
        if action[4] > 1:
            action[4] -= 2
        return action
        

if __name__ == "__main__":
    env = DrawerObjEnv()
    env.reset()
    env.start_rec("test")
    # reach handle
    action = np.array([PrimitiveType.MOVE_DIRECT, 0., 0., 0.2, 0.0])
    env.step(action)
    action = env.oracle_agent()
    env.step(action)
    # grasp
    action = np.array([PrimitiveType.GRIPPER_GRASP, 0, 0, 0, 0])
    env.step(action)
    # pull
    cur_eef_height = env.robot.get_eef_position()[2]
    action = np.array([PrimitiveType.MOVE_DIRECT, 0., -1., 
        (cur_eef_height - (env.robot_eef_range[0][2] + env.robot_eef_range[1][2]) / 2) / ((env.robot_eef_range[1][2] - env.robot_eef_range[0][2]) / 2), 0.])
    env.step(action)
    # for i in range(10):
    #     action = np.random.uniform(-1.0, 1.0, size=(5,))
    #     action[0] = np.random.randint(4)
    #     if i == 0:
            
    #     print("Action", action)
    #     env.step(action)
    env.end_rec()
