import gym
import os
import imageio
import pybullet as p
import pybullet_data
import numpy as np
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
            eef_pos = action[1:4] * np.array([0.2, 0.25, 0.15]) + np.array([0.5, 0.0, 0.15])
            eef_euler = np.array([np.pi, 0., action[4] * np.pi / 2])
            # 2pi modulo
            # if eef_euler[0] > np.pi:
            #     eef_euler[0] -= 2 * np.pi
            eef_quat = self.p.getQuaternionFromEuler(eef_euler)
        if primitive_type == PrimitiveType.MOVE_DIRECT:
            self.robot.move_direct_ee_pose(eef_pos, eef_quat)
        elif primitive_type == PrimitiveType.MOVE_APPROACH:
            self.robot.move_approach_ee_pose(eef_pos, eef_quat)
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
        # TODO: lets change the end effector to link8 as on real robot?
        self.robot = PandaRobot(self.p, init_qpos, base_position)
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


if __name__ == "__main__":
    env = BasePrimitiveEnv()
    env.reset()
    env.start_rec("test")
    for i in range(10):
        action = np.random.uniform(-1.0, 1.0, size=(5,))
        action[0] = np.random.randint(4)
        print("Action", action)
        env.step(action)
    env.end_rec()

