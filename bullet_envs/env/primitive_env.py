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
from functools import partial
import pkgutil
egl = pkgutil.get_loader('eglRenderer')


DATAROOT = pybullet_data.getDataPath()

class PrimitiveType:
    MOVE_DIRECT = 0
    # GRIPPER_OPEN = 1
    # GRIPPER_GRASP = 2
    MOVE_OPEN = 1
    MOVE_GRASP = 2


class BasePrimitiveEnv(gym.Env):
    def __init__(self, seed=None, view_mode="third", use_gpu_render=True, 
                 shift_params=(0, 0)) -> None:
        super().__init__()
        self.seed(seed)
        self.view_mode = view_mode
        self.use_gpu_render = use_gpu_render
        self.shift_params = shift_params
        self._setup_env()
        self.privilege_dim = None
        self.goal = self.sample_goal()
        obs = self._get_obs()
        if isinstance(obs, dict):
            self.observation_space = spaces.Dict(
                OrderedDict([(key, spaces.Box(low=-np.inf, high=np.inf, shape=obs[key].shape)) for key in obs])
            )
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
        self.record_cfg = dict(
            save_video_path=os.path.join(os.path.dirname(__file__), "..", "tmp"),
            fps=10
        )
        self.approach_dist = 0.05
        self.oracle_step_count = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    @property
    def dt(self):
        return 1. / 240

    def reset(self):
        self.oracle_step_count = 0
        self._reset_sim()
        if self.robot.get_finger_width() > 0.07:
            gripper_status = "open"
        else:
            gripper_status = "close"
        # sample goal
        self.goal = self.sample_goal()
        self.robot.reset_primitive(
            gripper_status, self._get_graspable_objects(), 
            partial(render, robot=self.robot, view_mode=self.view_mode, width=128, height=128, 
                    shift_params=self.shift_params), 
            self.goal["img"].transpose((1, 2, 0)) if self.goal["img"] is not None else None)
        obs = self._get_obs()
        return obs    
    
    def step(self, action) -> Tuple[Any, float, bool, Dict[str, Any]]:
        # Parse primitive
        # Calculate reward and info
        action = action.copy()
        assert action.shape[0] == 1 + 4
        primitive_type = int(np.round(action[0]))
        action[1:] = np.clip(action[1:], -1.0, 1.0)
        # Add neutral value and scale
        eef_pos = action[1:4] * (self.robot_eef_range[1] - self.robot_eef_range[0]) / 2 + np.mean(self.robot_eef_range, axis=0)
        eef_euler = np.array([np.pi, 0., action[4] * np.pi / 2])
        # 2pi modulo
        # if eef_euler[0] > np.pi:
        #     eef_euler[0] -= 2 * np.pi
        eef_quat = self.p.getQuaternionFromEuler(eef_euler)
        self.robot.move_direct_ee_pose(eef_pos, eef_quat)
        if primitive_type == PrimitiveType.MOVE_DIRECT:
            pass
        elif primitive_type == PrimitiveType.MOVE_OPEN:
            self.robot.gripper_move("open")
        elif primitive_type == PrimitiveType.MOVE_GRASP:
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
        return render(self.p, width, height, self.robot, self.view_mode, self.shift_params)

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
        if self.use_gpu_render:
            plugin = self.p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            print("plugin=", plugin)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 0)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
        
        self.p.resetSimulation()
        self.p.setTimeStep(self.dt)
        self.p.setGravity(0., 0., -9.8)
        self.p.resetDebugVisualizerCamera(1.0, 40, -20, [0, 0, 0,] )
        # plane_id = self.p.loadURDF(os.path.join(DATAROOT, "plane.urdf"), [0, 0, -0.795])
        plane_id = self.p.loadURDF(os.path.join(os.path.dirname(__file__), "assets/plane.urdf"), [0, 0, -0.795])
        # get a doormat
        vis_id = self.p.createVisualShape(self.p.GEOM_BOX, halfExtents=[5, 5, 0.02], rgbaColor=[1, 1, 1, 1])
        col_id = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=[5, 5, 0.02])
        self.p.createMultiBody(0.1, col_id, vis_id, [0, 0, -0.775], [0., 0., 0., 1.])
        table_id = self.p.loadURDF(
            os.path.join(DATAROOT, "table/table.urdf"), 
            [0.40000, 0.00000, -.625000], [0.000000, 0.000000, 0.707, 0.707]
        )
        self.robot = PandaRobot(self.p, init_qpos, base_position, is_visible=True)
        _robot_eef_low = np.array([0.3, -0.35, 0.0])
        _robot_eef_high = np.array([0.6, 0.35, 0.3])
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
        scene = render(self.p, width=128, height=128, robot=self.robot, view_mode=self.view_mode,
                       shift_params=self.shift_params).transpose((2, 0, 1))[:3]
        robot_obs = self.robot.get_obs()
        privilege_info = self._get_privilege_info()
        if self.privilege_dim is None:
            self.privilege_dim = privilege_info.shape[0]
        return {"img": scene, "robot_state": robot_obs, 
                "goal": self.goal["img"], "privilege_info": privilege_info}
    
    def _get_graspable_objects(self):
        return ()
    
    def _get_privilege_info(self):
        return np.empty(0)


def render(client: bc.BulletClient, width=256, height=256, robot: PandaRobot = None, view_mode="third",
           shift_params=(0, 0)) -> np.ndarray:
    shift_param = np.random.randint(shift_params[0], shift_params[1] + 1, size=(2,))
    if view_mode == "third":
        view_matrix = client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(0.45, 0, 0.1),
            distance=0.6,
            yaw=90,
            pitch=-45,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = client.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
        )
    elif view_mode == "ego":
        eef_pos = robot.get_eef_position().reshape((3, 1))
        eef_rot = np.array(client.getMatrixFromQuaternion(robot.get_eef_orn())).reshape((3, 3))
        # TODO
        eef_t_cam = np.array([0.05, 0.0, -0.05]).reshape((3, 1))
        eye_position = eef_rot @ eef_t_cam + eef_pos
        target_position = eye_position + eef_rot @ np.array([0, 0, 1]).reshape((3, 1))
        up_vector = eef_rot @ np.array([1, 0, 0])
        # up_vector = np.array([0, 0, -1])
        view_matrix = client.computeViewMatrix(eye_position, target_position, up_vector)
        proj_matrix = client.computeProjectionMatrixFOV(
            fov=90, aspect=1.0, nearVal=0.01, farVal=10.0
        )
    else:
        raise NotImplementedError
    (_, _, px, _, _) = client.getCameraImage(
        width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
    )
    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (height, width, 4))
    if shift_param[0] == 0 and shift_param[1] == 0:
        transformed_img = rgb_array
    else:
        transformed_img = np.zeros_like(rgb_array)
        if shift_param[0] >= 0 and shift_param[1] >=0:
            transformed_img[shift_param[0]:, shift_param[1]:] = rgb_array[:height - shift_param[0], :width - shift_param[1]]
        elif shift_param[0] >= 0 and shift_param[1] < 0:
            transformed_img[shift_param[0]:, :width + shift_param[1]] = rgb_array[:height - shift_param[0], -shift_param[1]:]
        elif shift_param[0] < 0 and shift_param[1] >= 0:
            transformed_img[:height + shift_param[0], shift_param[1]:] = rgb_array[-shift_param[0]:, :width - shift_param[1]]
        else:
            transformed_img[:height + shift_param[0], :width + shift_param[1]] = rgb_array[-shift_param[0]:, -shift_param[1]:]

    return transformed_img


# def process_img(color, depth, cam_config):
#     # goal: filter out robot and background, the perspective from robot?
class BoxLidEnv(BasePrimitiveEnv):
    def __init__(self, seed=None, reward_type="sparse", view_mode="third") -> None:
        super().__init__(seed, view_mode)
        self.approach_dist = 0.1
        self.dist_threshold = 0.02
        self.rot_threshold = 0.2
        self.reward_type = reward_type

    def _setup_callback(self):
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
        # reset box
        box_xy = np.random.uniform(
            low=self.box_range[0], high=self.box_range[1],
        )
        # reset lid
        if np.random.uniform(0, 1) < 0.5:
            lid_xyz = np.concatenate([box_xy, [0.07]])
        else:
            lid_xyz = np.concatenate([np.random.uniform(low=self.box_range[0], high=self.box_range[1]), [0.02]])
            while abs(lid_xyz[0] - box_xy[0]) < 0.21 and abs(lid_xyz[1] - box_xy[1]) < 0.16:
                lid_xyz[:2] = np.random.uniform(low=self.box_range[0], high=self.box_range[1])
        self.p.resetBasePositionAndOrientation(
            self.box_base_id, np.concatenate([box_xy, [0.03]]), [0., 0., 0., 1.]
        )
        self.p.resetBasePositionAndOrientation(
            self.box_lid_id, lid_xyz, [0., 0., 0., 1.]
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
        if np.linalg.norm(np.array(box_lid_pose[0]) - np.array(box_base_pose[0])) > 0.1 and np.random.uniform(0, 1) < 0.5:
            goal_lid_pos = np.concatenate([box_base_pose[0][:2], [0.07]])
        else:
            goal_lid_pos = np.concatenate(
                [np.random.uniform(low=self.box_range[0], high=self.box_range[1]), [0.02]]
            )
            while abs(goal_lid_pos[0] - box_base_pose[0][0]) < 0.21 and abs(goal_lid_pos[1] - box_base_pose[0][1]) < 0.16:
                goal_lid_pos = np.concatenate([np.random.uniform(low=self.box_range[0], high=self.box_range[1]), [0.025]])
        goal_lid_quat = box_lid_pose[1]

        # set into the environment to get image
        self.p.resetBasePositionAndOrientation(self.box_lid_id, goal_lid_pos, goal_lid_quat)
        # Need to simulate until valid, or make sure the sampled goal is stable
        self.p.stepSimulation()
        _robot_xyz = np.array([np.random.uniform(0.35, 0.45), np.random.uniform(-0.1, 0.1), np.random.uniform(0.25, 0.3)])
        self.robot.control(_robot_xyz, np.array([1, 0, 0, 0]), 0.04, relative=False, teleport=True)
        goal_img = render(self.p, width=128, height=128, robot=self.robot, view_mode=self.view_mode,
                          shift_params=self.shift_params).transpose((2, 0, 1))[:3]
        goal_robot_config = self.robot.get_obs()
        goal_lid_pos, goal_lid_quat = self.p.getLinkState(self.box_lid_id, self.box_lid_link)[:2]
        goal_dict = {'state': (np.array(goal_lid_pos), np.array(goal_lid_quat)), 'img': goal_img, 'robot_config': goal_robot_config}

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
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self._eef_pos_to_action(self.robot.get_eef_position() + np.array([0., 0., 0.15])), [0.]])
            self.step(action)
            # move to goal
            new_pos = self.goal["state"][0] + np.array([0., 0., 0.15])
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self._eef_pos_to_action(new_pos), [0.]])
            self.step(action)
            # put down
            new_pos = self.goal["state"][0] + np.array([0, 0, 0.01])
            # base_pos = self.p.getBasePositionAndOrientation(self.box_base_id)[0]
            # if base_pos[1] > 0:
            #     new_pos = base_pos + np.array([0., -0.25, 0.02])
            # else:
            #     new_pos = base_pos + np.array([0., 0.25, 0.02])
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self._eef_pos_to_action(new_pos), [0.]])
            self.step(action)
            # release
            action = np.concatenate([[PrimitiveType.GRIPPER_OPEN], [0., 0., 0., 0.]])
            self.step(action)
            print("lid pose", self.p.getLinkState(self.box_lid_id, self.box_lid_link)[:2])
            print("goal", self.goal["state"])
            # go up
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self._eef_pos_to_action(self.robot.get_eef_position() + np.array([0.0, 0.0, 0.2])), [0.]])
            _, reward, done, info = self.step(action)
            print("reward", reward)
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
        return info


class DrawerObjEnv(BasePrimitiveEnv):
    def __init__(self, seed=None, reward_type="dense", view_mode="third", use_gpu_render=True, 
                 obj_task_ratio=0.5, shift_params=(0, 0)) -> None:
        self.obj_task_ratio = obj_task_ratio
        self.handle_pos_threshold = 0.02
        super().__init__(seed, view_mode, use_gpu_render, shift_params)
        self.approach_dist = 0.1
        self.object_pos_threshold = 0.04
        self.reward_type = reward_type
    
    def _setup_callback(self):
        self.robot_eef_range[1][2] = 0.3
        self.drawer_id = self.p.loadURDF(
            os.path.join(os.path.dirname(__file__), "assets/drawer.urdf"), 
            [0.40000, 0.00000, 0.1], [0.000000, 0.000000, 0.0, 1.0],
            globalScaling=0.125
        )
        visual_shape = self.p.getVisualShapeData(self.drawer_id)
        # drawer_height = 0.1 if visual_shape[2][3][2] < 0.2 else visual_shape[2][3][2] / 2
        drawer_height = visual_shape[2][3][2]
        self.drawer_range = np.array([
            [0.45, 0.1, drawer_height],
            [0.45, 0.1, drawer_height]
        ])
        self.drawer_bottom_height = drawer_height - visual_shape[5][3][2] / 2
        for j in range(self.p.getNumJoints(self.drawer_id)):
            joint_info = self.p.getJointInfo(self.drawer_id, j)
            if joint_info[2] != self.p.JOINT_FIXED:
                self.drawer_joint = joint_info[0]
                self.drawer_handle_range = (joint_info[8], joint_info[9])
            if joint_info[12] == b'handle_r':
                self.drawer_handle_link = joint_info[0]
                # self.joint_damping.append(joint_info[6])
                break
        # self.load_template("YcbBanana", 0.7) # grasp unrealistic, penetration, not good
        # self.load_template("YcbStrawberry") # rolling, not good
        self.load_template("YcbTennisBall", 0.8) # ok
        
        if not hasattr(self, "graspable_objects"):
            self.graspable_objects = ((self.drawer_id, self.drawer_handle_link), (self.object_id, -1))

    def _reset_sim(self):
        def _randomize_drawer():
            # rand_angle = np.random.uniform(-np.pi, 0.)
            rand_angle = 0
            self.p.resetBasePositionAndOrientation(
                self.drawer_id, 
                np.random.uniform(self.drawer_range[0], self.drawer_range[1]),
                # ((self.drawer_range[0][0] + self.drawer_range[1][0]) / 2, 0.1, self.drawer_range[0][2]),
                (0., 0., np.sin(rand_angle / 2), np.cos(rand_angle / 2))
            )
            # reset drawer joint
            self.p.resetJointState(self.drawer_id, self.drawer_joint, np.random.uniform(*self.drawer_handle_range))
        def _randomize_object():
            # if np.random.uniform(0, 1) > 0.5:
            #     if self._is_drawer_open():
            #         object_pos = self._sample_object_inside_drawer()
            #     else:
            #         object_pos = self._sample_object_inside_drawer(initialize=True)
            # else:
            #     object_pos = self._sample_object_outside_drawer()
            if self._is_drawer_open():
                if np.random.uniform(0, 1) > 0.5:
                    object_pos = self._sample_object_inside_drawer()
                else:
                    object_pos = self._sample_object_outside_drawer()
            else:
                object_pos = self._sample_object_outside_drawer()
            rand_angle = np.random.uniform(-np.pi / 6, np.pi / 6)
            object_quat = (0., 0., np.sin(rand_angle / 2), np.cos(rand_angle / 2))
            self.p.resetBasePositionAndOrientation(
                self.object_id, object_pos, object_quat
            )
        self.robot.remove_constraint()
        _randomize_drawer()
        _count = 0
        # check handle position
        handle_position = self.p.getLinkState(self.drawer_id, self.drawer_handle_link)[0]
        while _count < 10 and not (self.robot_eef_range[0][0] <= handle_position[0] <= self.robot_eef_range[1][0] and self.robot_eef_range[0][1] <= handle_position[1] <= self.robot_eef_range[1][1]):
            _randomize_drawer()
            handle_position = self.p.getLinkState(self.drawer_id, self.drawer_handle_link)[0]
            _count += 1
        _randomize_object()
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
    
    def _get_privilege_info(self):
        cur_drawer = self.p.getJointState(self.drawer_id, self.drawer_joint)[0]
        goal_drawer = self.goal["state"][0]
        cur_object_pos = self.p.getBasePositionAndOrientation(self.object_id)[0]
        goal_object_pos = self.goal["state"][2][0]
        return np.concatenate([np.array([cur_drawer, goal_drawer]), cur_object_pos, goal_object_pos])

    def sample_goal(self):
        # store current state for recovery
        robot_state = self.robot.get_state()
        drawer_joint_state = self.p.getJointState(self.drawer_id, self.drawer_joint) 
        object_state = self.p.getBasePositionAndOrientation(self.object_id)       
        
        # check whether object is in drawer, check whether the drawer is open
        is_drawer_open = self._is_drawer_open()
        is_object_inside_drawer = self._is_object_inside_drawer()
        self.is_goal_move_drawer = False
        self.is_goal_move_object_out = False
        self.is_goal_move_object_in = False
        if is_object_inside_drawer and is_drawer_open:
            if np.random.uniform(0, 1) > self.obj_task_ratio:
                self.is_goal_move_drawer = True
            else:
                self.is_goal_move_object_out = True
        elif is_object_inside_drawer and not is_drawer_open:
            raise RuntimeError
            self.is_goal_move_drawer = True
        elif (not is_object_inside_drawer) and is_drawer_open:
            if np.random.uniform(0, 1) > self.obj_task_ratio:
                self.is_goal_move_drawer = True
            elif np.random.uniform(0, 1) > 0.5:
                self.is_goal_move_object_out = True
            else:
                self.is_goal_move_object_in = True
        else:
            if np.random.uniform(0, 1) > self.obj_task_ratio:
                self.is_goal_move_drawer = True
            else:
                self.is_goal_move_object_out = True

        # pair of underlying state and goal image
        if self.is_goal_move_drawer:
            goal_drawer_joint = np.random.uniform(*self.drawer_handle_range)
            while abs(goal_drawer_joint - drawer_joint_state[0]) < self.handle_pos_threshold:
                goal_drawer_joint = np.random.uniform(*self.drawer_handle_range)
            self.p.resetJointState(self.drawer_id, self.drawer_joint, goal_drawer_joint, 0.)
            _handle_pos = self.p.getLinkState(self.drawer_id, self.drawer_handle_link)[0]
            _count = 0
            while _count < 10 and not (self.robot_eef_range[0][0] <= _handle_pos[0] <= self.robot_eef_range[1][0] and self.robot_eef_range[0][1] <= _handle_pos[1] <= self.robot_eef_range[1][1]):
                goal_drawer_joint = np.random.uniform(*self.drawer_handle_range)
                self.p.resetJointState(self.drawer_id, self.drawer_joint, goal_drawer_joint, 0.)
                _handle_pos = self.p.getLinkState(self.drawer_id, self.drawer_handle_link)[0]
                _count += 1
            if is_object_inside_drawer: # object and drawer move together
                goal_object_pos = object_state[0] + np.array([0., goal_drawer_joint - drawer_joint_state[0], 0.])
            else:
                goal_object_pos = object_state[0]
        elif self.is_goal_move_object_out:
            goal_drawer_joint = drawer_joint_state[0]
            goal_object_pos = self._sample_object_outside_drawer()

        elif self.is_goal_move_object_in:
            goal_drawer_joint = drawer_joint_state[0]
            goal_object_pos = self._sample_object_inside_drawer()

        # set into the environment to get image
        self.p.resetJointState(self.drawer_id, self.drawer_joint, goal_drawer_joint, 0.)
        self.p.resetBasePositionAndOrientation(self.object_id, goal_object_pos, object_state[1])
        # Need to simulate until valid, or make sure the sampled goal is stable
        for _ in range(5):
            self.p.stepSimulation()
        self.robot.teleport_joint()
        cur_drawer_joint = self.p.getJointState(self.drawer_id, self.drawer_joint)[0]
        cur_handle_pos = self.p.getLinkState(self.drawer_id, self.drawer_handle_link)[0]
        cur_object_pose = self.p.getBasePositionAndOrientation(self.object_id)
        goal_img = self._get_goal_image()
        goal_dict = {'state': (cur_drawer_joint, cur_handle_pos, cur_object_pose), 'img': goal_img}

        # recover state
        self.p.resetJointState(self.drawer_id, self.drawer_joint, drawer_joint_state[0], drawer_joint_state[1])
        self.p.resetBasePositionAndOrientation(self.object_id, object_state[0], object_state[1])
        self.robot.set_state(robot_state)
        return goal_dict
    
    def _get_goal_image(self):
        goal_img = render(self.p, width=128, height=128, robot=self.robot, view_mode=self.view_mode, 
                          shift_params=self.shift_params).transpose((2, 0, 1))[:3]
        return goal_img
    
    def compute_reward_and_info(self):
        cur_handle_joint = self.p.getJointState(self.drawer_id, self.drawer_joint)[0]
        cur_object_pos = self.p.getBasePositionAndOrientation(self.object_id)[0]
        handle_dist = abs(self.goal["state"][0] - cur_handle_joint)
        object_dist = np.linalg.norm(np.array(self.goal["state"][2][0]) - np.array(cur_object_pos))
        is_success = handle_dist < self.handle_pos_threshold and object_dist < self.object_pos_threshold
        # is_success = object_dist < self.object_pos_threshold if (self.is_goal_move_object_in or self.is_goal_move_object_out) else handle_dist < self.handle_pos_threshold
        if self.reward_type == "sparse":
            reward = float(is_success)
        elif self.reward_type == "dense_stage":
            reward_stages = [-0.2, -0.1, 0.0]
            cur_handle_pos = self.p.getLinkState(self.drawer_id, self.drawer_handle_link)[0]
            eef_dist = np.linalg.norm(self.robot.get_eef_position() - cur_handle_pos)
            if eef_dist > 0.01:
                reward = reward_stages[0] + np.clip(1 - eef_dist, 0.0, 1.0) * (reward_stages[1] - reward_stages[0])
                is_success = False
            elif handle_dist >= self.handle_pos_threshold:
                reward = reward_stages[1] + (1 - handle_dist / (self.drawer_handle_range[1] - self.drawer_handle_range[0])) * (reward_stages[2] - reward_stages[1])
                is_success = False
            else:
                is_success = True
                reward = 1.0
        elif self.reward_type == "dense":
            # only guide acrroding to goal mode
            if self.is_goal_move_drawer:
                cur_handle_pos = self.p.getLinkState(self.drawer_id, self.drawer_handle_link)[0]
                eef_dist = np.linalg.norm(self.robot.get_eef_position() - cur_handle_pos)
                task_dist = handle_dist
            else:
                cur_obj_pos = self.p.getBasePositionAndOrientation(self.object_id)[0]
                eef_dist = np.linalg.norm(self.robot.get_eef_position() - cur_obj_pos)
                task_dist = object_dist
            reward = -0.05 * eef_dist - 0.2 * task_dist + 0 * float(is_success)
        info = {
            'handle_joint': cur_handle_joint, 'object_pos': np.array(cur_object_pos), 
            'is_success': is_success
        }
        return reward, info
    
    def eef_pos_to_action(self, eef_pos):
        return (eef_pos - np.mean(self.robot_eef_range, axis=0)) / ((self.robot_eef_range[1] - self.robot_eef_range[0]) / 2)
    
    def step(self, action) -> Tuple[Any, float, bool, Dict[str, Any]]:
        obs, reward, done, info = super().step(action)
        info["is_goal_move_drawer"] = self.is_goal_move_drawer
        if (self.is_goal_move_object_in or self.is_goal_move_object_out) and self.oracle_step_count > 6:
            done = True
        elif self.is_goal_move_drawer and self.oracle_step_count > 4:
            done = True
        return obs, reward, done, info
    
    def oracle_agent(self, record_traj=False, noise_std=0.0):
        if self.is_goal_move_object_in or self.is_goal_move_object_out:
            # should move object
            action = self.take_out_object(self.goal["state"][2][0], record_traj)
        else:
            # should move drawer
            action = self.perturb_drawer(self.goal["state"][0], record_traj)
        action[1:] = np.clip(action[1:], -1., 1.)
        # TODO: add some noise?
        action[1:] += np.random.normal() * noise_std
        self.oracle_step_count += 1
        return action
    
    def _is_drawer_open(self):
        is_drawer_open = self.p.getJointState(self.drawer_id, self.drawer_joint)[0] < (self.drawer_handle_range[0] + self.drawer_handle_range[1]) / 2
        return is_drawer_open

    def _is_object_inside_drawer(self):
        drawer_joint = self.p.getJointState(self.drawer_id, self.drawer_joint)[0]
        drawer_base_position = self.p.getBasePositionAndOrientation(self.drawer_id)[0]
        object_position = self.p.getBasePositionAndOrientation(self.object_id)[0]
        is_object_inside = (drawer_base_position[0] - 0.75 * 0.125 < object_position[0] < drawer_base_position[0] + 0.75 * 0.125) \
            and (drawer_base_position[1] + drawer_joint - 0.6 * 0.125 < object_position[1] < drawer_base_position[1] + drawer_joint + 0.6 * 0.125) 
        return is_object_inside

    def _outside_drawer_range(self, drawer_base_position, handle_joint):
        outside_drawer_range = np.array([
            [drawer_base_position[0] - 0.75 * 0.125 - 0.1, drawer_base_position[1] - 1.2 * 0.125 + handle_joint - 0.1], 
            [drawer_base_position[0] + 0.75 * 0.125 + 0.1, drawer_base_position[1] + 0.65 * 0.125 + 0.1]
        ])
        return outside_drawer_range
    
    def _inside_drawer_range(self, drawer_base_position, handle_joint):
        return np.array([
            [drawer_base_position[0] - 0.03, drawer_base_position[1] + handle_joint - 0.6 * 0.125 + 0.03, self.drawer_bottom_height + 0.02],
            [drawer_base_position[0] + 0.03, drawer_base_position[1] - 0.6 * 0.125 - 0.03, self.drawer_bottom_height + 0.02]
        ])
                
    def _sample_object_outside_drawer(self):
        drawer_base_position = self.p.getBasePositionAndOrientation(self.drawer_id)[0]
        handle_joint = self.p.getJointState(self.drawer_id, self.drawer_joint)[0]
        outside_drawer_range = self._outside_drawer_range(drawer_base_position, handle_joint)
        outside_position = np.random.uniform(self.object_range[0], self.object_range[1])
        while outside_drawer_range[0][0] < outside_position[0] < outside_drawer_range[1][0] and outside_drawer_range[0][1] < outside_position[1] < outside_drawer_range[1][1]:
            outside_position = np.random.uniform(self.object_range[0], self.object_range[1])
        outside_position[2] = 0.02
        return outside_position
    
    def _sample_object_inside_drawer(self, initialize=False):
        drawer_base_position = self.p.getBasePositionAndOrientation(self.drawer_id)[0]
        handle_joint = self.p.getJointState(self.drawer_id, self.drawer_joint)[0]
        if not initialize:
            inside_drawer_range = self._inside_drawer_range(drawer_base_position, handle_joint)
        else:
            inside_drawer_range = np.array([
                [drawer_base_position[0] - 0.03, drawer_base_position[1] + handle_joint - 0.65 * 0.125 + 0.03, self.drawer_bottom_height + 0.02],
                [drawer_base_position[0] + 0.03, drawer_base_position[1] + handle_joint + 0.65 * 0.125 - 0.03, self.drawer_bottom_height + 0.02]
            ])
        object_position = np.random.uniform(
            inside_drawer_range[0], inside_drawer_range[1]
        )
        return object_position

    @staticmethod
    def _store_transition(traj, obs, action, reward, done):
        traj["obs"].append(obs)
        traj["action"].append(action)
        traj["reward"].append(reward)
        traj["done"].append(done)

    def perturb_drawer(self, drawer_goal, record_traj=False):
        # traj = {"obs": [], "action": [], "reward": [], "done": []}
        # store the first obs
        # obs = self._get_obs()
        # traj["obs"].append(obs)
        if self.oracle_step_count == 0:
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self.eef_pos_to_action(self.robot.get_eef_position() + np.array([0., 0., 0.2])), [0.]])
        # obs, reward, done, info = self.step(action)
        # if record_traj:
        #     self._store_transition(traj, obs, action, reward, done)
        elif self.oracle_step_count == 1:
            handle_position = self.p.getLinkState(self.drawer_id, self.drawer_handle_link)[0]
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self.eef_pos_to_action(handle_position + np.array([0., 0., 0.1])), [0.0]])
        # obs, reward, done, info = self.step(action)
        # if record_traj:
        #     self._store_transition(traj, obs, action, reward, done)
        # _count = 0
        # while _count < 5 and np.linalg.norm(self.robot.get_eef_position() - self.p.getLinkState(self.drawer_id, self.drawer_handle_link)[0] + np.array([0., 0., 0.005])) > 5e-3:
        elif self.oracle_step_count == 2:
            action = np.concatenate([[PrimitiveType.MOVE_GRASP], self.eef_pos_to_action(self.p.getLinkState(self.drawer_id, self.drawer_handle_link)[0] - np.array([0., 0., 0.005])), [0.0]])
        # obs, reward, done, info = self.step(action)
        # if record_traj:
        #     self._store_transition(traj, obs, action, reward, done)
        # _count += 1
        # action = np.concatenate([[PrimitiveType.MOVE_GRASP], self.eef_pos_to_action(self.robot.get_eef_position()), [0.2]])
        # obs, reward, done, info = self.step(action)
        # if record_traj:
        #     self._store_transition(traj, obs, action, reward, done)
        # _count = 0
        # while _count < 5 and abs(self.p.getJointState(self.drawer_id, self.drawer_joint)[0] - drawer_goal) > 5e-3:
        elif self.oracle_step_count == 3:
            drawer_joint = self.p.getJointState(self.drawer_id, self.drawer_joint)[0]
            action = np.concatenate([
                [PrimitiveType.MOVE_OPEN], 
                self.eef_pos_to_action(self.p.getLinkState(self.drawer_id, self.drawer_handle_link)[0] + np.array([0., drawer_goal - drawer_joint, 0.0])), 
                [0.2]])
        # obs, reward, done, info = self.step(action)
        # if record_traj:
        #     self._store_transition(traj, obs, action, reward, done)
        # _count += 1
        # action = np.concatenate([[PrimitiveType.MOVE_OPEN], self.eef_pos_to_action(self.robot.get_eef_position()), [0.2]])
        # obs, reward, done, info = self.step(action)
        # if record_traj:
        #     self._store_transition(traj, obs, action, reward, done)
        elif self.oracle_step_count == 4:
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self.eef_pos_to_action(self.robot.get_eef_position() + np.array([0., 0., 0.2])), [0.]])
        # obs, reward, done, info = self.step(action)
        # if record_traj:
        #     self._store_transition(traj, obs, action, reward, done)
        #     if not info["is_success"]:
        #         traj = {"obs": [], "action": [], "reward": [], "done": []}
        else:
            action = None
        return action
    
    def take_out_object(self, object_goal, record_traj=False):
        # traj = {"obs": [], "action": [], "reward": [], "done": []}
        # obs = self._get_obs()
        # traj["obs"].append(obs)
        if self.oracle_step_count == 0:
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self.eef_pos_to_action(self.robot.get_eef_position() + np.array([0., 0., 0.2])), [0.]])
        # obs, reward, done, info = self.step(action)
        # if record_traj:
        #     self._store_transition(traj, obs, action, reward, done)
        elif self.oracle_step_count == 1:
            object_position, object_quat = self.p.getBasePositionAndOrientation(self.object_id)[:2]
            # orientation?
            zrot = self.p.getEulerFromQuaternion(object_quat)[2]
            zrot = np.pi / 2
            while zrot > np.pi / 2 or zrot < -np.pi / 2:
                if zrot > 0:
                    zrot -= np.pi
                else:
                    zrot += np.pi
            zrot_action = zrot / (np.pi / 2)
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self.eef_pos_to_action(object_position + np.array([0., 0., 0.1])), [zrot_action]])
        # obs, reward, done, info = self.step(action)
        # if record_traj:
        #     self._store_transition(traj, obs, action, reward, done)
        elif self.oracle_step_count == 2:
            object_position = self.p.getBasePositionAndOrientation(self.object_id)[0]
            zrot_action = 1
            action = np.concatenate([[PrimitiveType.MOVE_GRASP], self.eef_pos_to_action(object_position), [zrot_action]])
        # obs, reward, done, info = self.step(action)
        # if record_traj:
        #     self._store_transition(traj, obs, action, reward, done)
        # action = np.concatenate([[PrimitiveType.GRIPPER_GRASP], np.zeros(4)])
        # obs, reward, done, info = self.step(action)
        # if record_traj:
        #     self._store_transition(traj, obs, action, reward, done)
        elif self.oracle_step_count == 3:
            zrot_action = 1
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self.eef_pos_to_action(self.robot.get_eef_position() + np.array([0., 0., 0.25])), [zrot_action]])
        # obs, reward, done, info = self.step(action)
        # if record_traj:
        #     self._store_transition(traj, obs, action, reward, done)
        elif self.oracle_step_count == 4:
            zrot_action = 1
            object_position = self.p.getBasePositionAndOrientation(self.object_id)[0]
            place_eef = self.eef_pos_to_action(self.robot.get_eef_position() + object_goal - object_position)
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], place_eef[:2], [self.robot.get_eef_position()[2]], [zrot_action]])
        # obs, reward, done, info = self.step(action)
        # if record_traj:
        #     self._store_transition(traj, obs, action, reward, done)
        elif self.oracle_step_count == 5:
            zrot_action = 1
            object_position = self.p.getBasePositionAndOrientation(self.object_id)[0]
            place_eef = self.eef_pos_to_action(self.robot.get_eef_position() + object_goal - object_position)
            action = np.concatenate([[PrimitiveType.MOVE_OPEN], place_eef, [zrot_action]])
        # obs, reward, done, info = self.step(action)
        # if record_traj:
        #     self._store_transition(traj, obs, action, reward, done)
        # action = np.concatenate([[PrimitiveType.GRIPPER_OPEN], np.zeros(4)])
        # obs, reward, done, info = self.step(action)
        # if record_traj:
        #     self._store_transition(traj, obs, action, reward, done)
        elif self.oracle_step_count == 6:
            zrot_action = 1
            action = np.concatenate([[PrimitiveType.MOVE_DIRECT], self.eef_pos_to_action(self.robot.get_eef_position() + np.array([0., 0., 0.2])), [zrot_action]])
        # obs, reward, done, info = self.step(action)
        # if record_traj:
        #     self._store_transition(traj, obs, action, reward, done)
        #     if not info["is_success"]:
        #         traj = {"obs": [], "action": [], "reward": [], "done": []}
        #     else:
        #         # find the first done signal
        #         for i in range(len(traj["done"])):
        #             if traj["done"][i]:
        #                 traj["obs"] = traj["obs"][:(i + 1)]
        #                 traj["action"] = traj["action"][:(i + 1)]
        #                 traj["reward"] = traj["reward"][:(i + 1)]
        #                 traj["done"] = traj["done"][:(i + 1)]
        else:
            action = None
        return action

    def play_agent(self):
        # see current status
        # randomly choose a feasible action
        is_drawer_open = self._is_drawer_open()
        is_object_inside = self._is_object_inside_drawer()
        
        if is_drawer_open and is_object_inside:
            # can move the drawer, or take the object outside the drawer
            # object and drawer may move simultaneously
            if np.random.uniform(0, 1) > 0.5:
                drawer_goal = np.random.uniform(*self.drawer_handle_range)
                self.perturb_drawer(drawer_goal)
            else:
                outside_position = self._sample_object_outside_drawer()
                self.take_out_object(outside_position)
            
        elif is_drawer_open and not is_object_inside:
            # can move the drawer, or put the object into the drawer, or move the object on the table
            if np.random.uniform(0, 1) > 0.5:
                drawer_goal = np.random.uniform(*self.drawer_handle_range)
                self.perturb_drawer(drawer_goal)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    # put object into the drawer
                    object_position = self._sample_object_inside_drawer()
                else:
                    # move object on the table
                    object_position = self._sample_object_outside_drawer()
                self.take_out_object(object_position)

        elif (not is_drawer_open) and is_object_inside:
            # move the drawer
            drawer_goal = np.random.uniform(*self.drawer_handle_range)
            self.perturb_drawer(drawer_goal)
        elif (not is_drawer_open) and (not is_object_inside):
            # move the drawer, move object on the table
            if np.random.uniform(0, 1) > 0.5:
                drawer_goal = np.random.uniform(*self.drawer_handle_range)
                self.perturb_drawer(drawer_goal)
            else:
                object_position = self._sample_object_outside_drawer()
                self.take_out_object(object_position)

    def load_template(self, object_name, scaling=1.0):
        # urdf_path = os.path.join(os.path.dirname(__file__), "assets/ycb_objects", object_name, "model.urdf")
        urdf_path = os.path.join(os.path.dirname(__file__), "assets/cube_simple.urdf")
        self.object_range = np.array([
            [0.45 - 0.1, 0.1 - 0.4, self.drawer_range[0][2]], 
            [0.45 + 0.15, 0.1 + 0.2, self.drawer_range[0][2]]])
        # vis_id = self.p.createVisualShape(self.p.GEOM_SPHERE, 0.025, rgbaColor=[0, 1, 0, 1])
        # col_id = self.p.createCollisionShape(self.p.GEOM_SPHERE, 0.025)
        # self.object_id = self.p.createMultiBody(0.1, col_id, vis_id, [0.4, -0.2, self.object_range[0][2]], [0., 0., 0., 1.])
        self.object_id = self.p.loadURDF(urdf_path, [0.4, -0.2, self.object_range[0][2]], [0., 0., 0., 1.], globalScaling=scaling)


class DrawerObjEnvState(DrawerObjEnv):
    def __init__(self, seed=None, reward_type="dense", view_mode="third", use_gpu_render=True, render_goal=False, obj_task_ratio=0.5) -> None:
        self.render_goal = render_goal
        super().__init__(seed, reward_type, view_mode, use_gpu_render, obj_task_ratio)

    def _get_obs(self):
        robot_obs = self.robot.get_obs()
        privilege_info = self._get_privilege_info()
        return np.concatenate([robot_obs, privilege_info])

    def _get_goal_image(self):
        if self.render_goal:
            return super()._get_goal_image()
        else:
            return None

if __name__ == "__main__":
    env = DrawerObjEnv(view_mode="ego", obj_task_ratio=1.0)
    # env = BoxLidEnv()
    is_success = []
    env.start_rec("test")
    for i in range(10):
        success = False
        done = False
        traj = {"obs": [], "action": [], "reward": [], "done": []}
        obs = env.reset()
        traj["obs"].append(obs)
        cur_img = obs["img"]
        goal_img = obs["goal"]
        goal_state = env.goal["state"]
        # import matplotlib.pyplot as plt
        # plt.imsave("cur_img.png", cur_img.transpose((1, 2, 0)))
        # plt.imsave("goal_img.png", goal_img.transpose((1, 2, 0)))
        print("reset obs", obs["privilege_info"])
        print("goal state", goal_state[0], goal_state[2][0])
        # env.start_rec("test")
        while not done:
            action = env.oracle_agent(record_traj=True)
            obs, reward, done, info = env.step(action)
            success = info["is_success"]
            traj["obs"].append(obs)
            traj["action"].append(action)
            traj["reward"].append(reward)
            traj["done"].append(done)
        if not success:
            traj["obs"] = []
            traj["action"] = []
            traj["reward"] = []
            traj["done"] = []
        print("traj", len(traj["obs"]))
        is_success.append(success)
        # env.oracle_agent("open_box")
        # env.oracle_agent("close_box")
        # for i in range(50):
        #     action = np.random.uniform(-1.0, 1.0, size=(5,))
        #     action[0] = np.random.randint(4)
        # #     if i == 0:
        # #     print("Action", action)
        #     env.step(action)
        # env.end_rec()
        # print("is_success", info["is_success"])
        print("info", info)
    env.end_rec()
    print("mean success", np.mean(is_success))